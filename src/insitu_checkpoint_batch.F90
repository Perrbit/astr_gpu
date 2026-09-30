module insitu_checkpoint_batch
  use mpi
  use iso_fortran_env, only: int8,int32,int64,iostat_end
  use iso_c_binding, only: c_int,c_char,c_null_char
  use ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: create_batch,publish_batch,validate_batch,file_fingerprint
  public :: copy_batch_file
  interface
    function make_directory(path,mode) bind(C,name='mkdir') result(status)
      import c_int,c_char
      character(c_char),intent(in) :: path(*)
      integer(c_int),value :: mode
      integer(c_int) :: status
    end function
    function link_file(old,new) bind(C,name='link') result(status)
      import c_int,c_char
      character(c_char),intent(in) :: old(*),new(*)
      integer(c_int) :: status
    end function
    function unlink_file(path) bind(C,name='unlink') result(status)
      import c_int,c_char
      character(c_char),intent(in) :: path(*)
      integer(c_int) :: status
    end function
  end interface
contains
  subroutine copy_batch_file(source,destination,ok)
    character(*),intent(in) :: source,destination
    logical,intent(out) :: ok
    integer(int8) :: buffer(65536)
    integer(int64) :: bytes,crc,copied_bytes,copied_crc,left
    integer :: input,output,status,read_status,write_status,closed,n
    logical :: valid
    ok=.false.
    call file_fingerprint(source,bytes,crc,valid)
    if(.not.valid) return
    open(newunit=input,file=source,status='old',access='stream',form='unformatted',action='read',iostat=status)
    if(status/=0) return
    open(newunit=output,file=destination,status='new',access='stream',form='unformatted',action='write',iostat=status)
    if(status/=0) then
      close(input)
      return
    endif
    left=bytes
    do while(left>0)
      n=int(min(left,int(size(buffer),int64)))
      read(input,iostat=read_status) buffer(:n)
      if(read_status/=0) then
        valid=.false.
        exit
      endif
      write(output,iostat=write_status) buffer(:n)
      if(write_status/=0) then
        valid=.false.
        exit
      endif
      left=left-n
    enddo
    close(input,iostat=closed)
    valid=valid.and.closed==0
    close(output,iostat=closed)
    valid=valid.and.closed==0
    if(.not.valid) return
    call file_fingerprint(destination,copied_bytes,copied_crc,valid)
    ok=valid.and.copied_bytes==bytes.and.copied_crc==crc
  end subroutine

  logical function unanimous(local,comm) result(ok)
    logical,intent(in) :: local
    integer,intent(in) :: comm
    integer :: bad,total,ierr
    bad=merge(0,1,local)
    call MPI_Allreduce(bad,total,1,MPI_INTEGER,MPI_MAX,comm,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(comm,1,ierr)
    ok=total==0
  end function

  subroutine create_batch(path,comm,ok)
    character(*),intent(in) :: path
    integer,intent(in) :: comm
    logical,intent(out) :: ok
    integer :: rank,ierr,status
    character(1024) :: root
    call MPI_Comm_rank(comm,rank,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(comm,1,ierr)
    root=path
    call MPI_Bcast(root,len(root),MPI_CHARACTER,0,comm,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(comm,1,ierr)
    ok=unanimous(len_trim(path)>0.and.len_trim(path)<=len(root).and.root==path,comm)
    if(.not.ok) return
    status=0
    if(rank==0) status=make_directory(trim(path)//c_null_char,int(o'700',c_int))
    ok=unanimous(status==0,comm)
  end subroutine

  subroutine file_fingerprint(path,bytes,crc,ok)
    character(*),intent(in) :: path
    integer(int64),intent(out) :: bytes,crc
    logical,intent(out) :: ok
    integer(int8) :: buffer(65536)
    integer :: unit,status,closed,n,j,bit
    integer(int64) :: remaining
    logical :: exists
    bytes=0
    crc=0
    exists=.false.
    ok=.false.
    inquire(file=path,exist=exists,size=bytes,iostat=status)
    if(status/=0.or..not.exists.or.bytes<0) return
    open(newunit=unit,file=path,status='old',access='stream',form='unformatted',action='read',iostat=status)
    if(status/=0) return
    remaining=bytes
    do while(remaining>0)
      n=int(min(remaining,int(size(buffer),int64)))
      read(unit,iostat=status) buffer(:n)
      if(status/=0) exit
      ! CRC-64/ECMA-182: poly 0x42F0E1EBA9EA3693, init=0, refin=false, xorout=0.
      do j=1,n
        crc=ieor(crc,shiftl(iand(int(buffer(j),int64),255_int64),56))
        do bit=1,8
          if(btest(crc,63)) then
            crc=ieor(shiftl(crc,1),int(z'42F0E1EBA9EA3693',int64))
          else
            crc=shiftl(crc,1)
          endif
        enddo
      enddo
      remaining=remaining-n
    enddo
    if(status==0) then
      read(unit,iostat=status) buffer(1)
      if(status==iostat_end) then
        ok=.true.
      endif
    endif
    close(unit,iostat=closed)
    ok=ok.and.closed==0
  end subroutine

  subroutine inspect_files(path,batch,step,time,topology,config,files,comm,sizes,checksums,ok)
    character(*),intent(in) :: path,batch,config
    character(96),intent(in) :: files(:)
    integer(int64),intent(in) :: step
    real(8),intent(in) :: time
    integer,intent(in) :: topology(3),comm
    integer(int64),intent(out) :: sizes(:),checksums(:)
    logical,intent(out) :: ok
    character(2048) :: description,root
    character(96) :: name
    integer :: rank,ranks,i,j,ierr,nlo,nhi
    logical :: valid,one
    integer(int64) :: local_sizes(size(files)),local_crc(size(files))
    call MPI_Comm_rank(comm,rank,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(comm,1,ierr)
    call MPI_Comm_size(comm,ranks,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(comm,1,ierr)
    call MPI_Allreduce(size(files),nlo,1,MPI_INTEGER,MPI_MIN,comm,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(comm,1,ierr)
    call MPI_Allreduce(size(files),nhi,1,MPI_INTEGER,MPI_MAX,comm,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(comm,1,ierr)
    ok=unanimous(nlo>0.and.nlo==nhi.and.size(sizes)==nlo.and.size(checksums)==nlo,comm)
    if(.not.ok) return
    valid=len_trim(path)>0.and.len_trim(path)<=1024.and.len_trim(batch)>0.and.len_trim(batch)<=64
    valid=valid.and.len_trim(config)>0.and.len_trim(config)<=256.and.step>=0.and.ieee_is_finite(time)
    valid=valid.and.all(topology>0).and.product(int(topology,int64))==ranks
    ok=unanimous(valid,comm)
    if(.not.ok) return
    write(description,*) trim(path),'|',trim(batch),'|',step,time,topology,'|',trim(config)
    root=description
    call MPI_Bcast(root,len(root),MPI_CHARACTER,0,comm,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(comm,1,ierr)
    valid=root==description
    do i=1,size(files)
      name=files(i)
      call MPI_Bcast(name,len(name),MPI_CHARACTER,0,comm,ierr)
      if(ierr/=MPI_SUCCESS) call MPI_Abort(comm,1,ierr)
      valid=valid.and.name==files(i).and.len_trim(name)>0
      valid=valid.and.name/='.'.and.name/='..'.and.name/='manifest.bin'.and.name/='manifest.pending'
      do j=1,len_trim(name)
        valid=valid.and.index('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-',name(j:j))>0
      enddo
      do j=1,i-1
        valid=valid.and.files(i)/=files(j)
      enddo
    enddo
    ok=unanimous(valid,comm)
    if(.not.ok) return
    local_sizes=0
    local_crc=0
    do i=1,size(files)
      if(mod(i-1,ranks)/=rank) cycle
      call file_fingerprint(trim(path)//'/'//trim(files(i)),local_sizes(i),local_crc(i),one)
      valid=valid.and.one
    enddo
    ok=unanimous(valid,comm)
    if(.not.ok) return
    call MPI_Allreduce(local_sizes,sizes,size(files),MPI_INTEGER8,MPI_SUM,comm,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(comm,1,ierr)
    call MPI_Allreduce(local_crc,checksums,size(files),MPI_INTEGER8,MPI_BXOR,comm,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(comm,1,ierr)
  end subroutine

  subroutine publish_batch(path,batch,step,time,topology,config,files,ready,comm,ok)
    character(*),intent(in) :: path,batch,config
    character(96),intent(in) :: files(:)
    integer(int64),intent(in) :: step
    real(8),intent(in) :: time
    integer,intent(in) :: topology(3),comm
    logical,intent(in) :: ready
    logical,intent(out) :: ok
    integer(int64) :: sizes(size(files)),checksums(size(files))
    character(64) :: identity
    character(256) :: configuration
    integer :: rank,ierr,unit,status,closed
    ok=unanimous(ready,comm)
    if(.not.ok) return
    call inspect_files(path,batch,step,time,topology,config,files,comm,sizes,checksums,ok)
    if(.not.ok) return
    identity=batch
    configuration=config
    call MPI_Comm_rank(comm,rank,ierr)
    status=0
    if(rank==0) then
      open(newunit=unit,file=trim(path)//'/manifest.pending',status='new',access='stream', &
        form='unformatted',convert='little_endian',iostat=status)
      if(status==0) then
        write(unit,iostat=status) 'ASTRB001',identity,step,time,int(topology,int32),configuration, &
          int(size(files),int32),files,sizes,checksums
        close(unit,iostat=closed)
        if(closed/=0) status=closed
      endif
      if(status==0) status=link_file(trim(path)//'/manifest.pending'//c_null_char, &
                                    trim(path)//'/manifest.bin'//c_null_char)
      if(status==0) closed=unlink_file(trim(path)//'/manifest.pending'//c_null_char)
    endif
    ok=unanimous(status==0,comm)
  end subroutine

  subroutine validate_batch(path,batch,step,time,topology,config,files,comm,ok)
    character(*),intent(in) :: path,batch,config
    character(96),intent(in) :: files(:)
    integer(int64),intent(in) :: step
    real(8),intent(in) :: time
    integer,intent(in) :: topology(3),comm
    logical,intent(out) :: ok
    integer(int64) :: sizes(size(files)),checksums(size(files)),saved_sizes(size(files)),saved_crc(size(files)),saved_step
    character(96) :: saved_files(size(files))
    character(8) :: magic
    character(64) :: identity
    character(256) :: configuration
    real(8) :: clock
    integer(int32) :: topo(3),count
    integer(int8) :: extra
    integer :: unit,status,closed
    logical :: valid
    call inspect_files(path,batch,step,time,topology,config,files,comm,sizes,checksums,ok)
    if(.not.ok) return
    open(newunit=unit,file=trim(path)//'/manifest.bin',status='old',access='stream', &
      form='unformatted',convert='little_endian',action='read',iostat=status)
    valid=status==0
    if(valid) then
      read(unit,iostat=status) magic,identity,saved_step,clock,topo,configuration,count
      valid=status==0
      if(valid) valid=magic=='ASTRB001'.and.identity==batch.and.saved_step==step.and.clock==time.and. &
        all(topo==topology).and.configuration==config.and.count==size(files)
      if(valid) then
        read(unit,iostat=status) saved_files,saved_sizes,saved_crc
        valid=status==0
        if(valid) valid=all(saved_files==files).and.all(saved_sizes==sizes).and.all(saved_crc==checksums)
      endif
      if(valid) then
        read(unit,iostat=status) extra
        valid=status==iostat_end
      endif
      close(unit,iostat=closed)
      valid=valid.and.closed==0
    endif
    ok=unanimous(valid,comm)
  end subroutine
end module
