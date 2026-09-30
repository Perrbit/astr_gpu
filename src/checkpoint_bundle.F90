module checkpoint_bundle
  use iso_fortran_env, only: int64, iostat_end
  use iso_c_binding, only: c_int,c_char,c_null_char
  use mpi
  use insitu_checkpoint_batch, only: file_fingerprint
  implicit none
  private
  public :: seal_checkpoint_bundle, validate_checkpoint_bundle
  public :: publish_checkpoint_bundle
  public :: checkpoint_retention, publish_retained_checkpoint
  public :: write_checkpoint_resource_refs
  integer, parameter :: max_files=64
  type checkpoint_retention
    private
    character(1024) :: root=''
    character(128) :: names(3)=''
    integer :: count=0,keep=0
    logical :: failed=.false.
  end type
  interface
    function plain_resource(path,name) bind(C,name='astr_checkpoint_plain_resource') result(status)
      import c_int,c_char
      character(c_char),intent(in) :: path(*),name(*)
      integer(c_int) :: status
    end function
    function retire_directory(root,name,entries) bind(C,name='astr_checkpoint_retire') result(status)
      import c_int,c_char
      character(c_char),intent(in) :: root(*),name(*),entries(*)
      integer(c_int) :: status
    end function
#ifdef ASTR_HAS_RENAMEAT2
    function rename_exclusive(olddir,old,newdir,new,flags) bind(C,name='renameat2') result(status)
      import c_int,c_char
      integer(c_int),value :: olddir,newdir,flags
      character(c_char),intent(in) :: old(*),new(*)
      integer(c_int) :: status
    end function
#endif
    function rename_entry(old,new) bind(C,name='rename') result(status)
      import c_int,c_char
      character(c_char),intent(in) :: old(*),new(*)
      integer(c_int) :: status
    end function
  end interface
contains
  subroutine write_checkpoint_resource_refs(path,names,comm,ok)
    character(*),intent(in) :: path,names(:)
    integer,intent(in) :: comm
    logical,intent(out) :: ok
    character(128) :: root_names(max_files)
    integer :: rank,err,bad,total,n,i,j,unit,closed
    integer(int64) :: bytes(max_files),crc(max_files)
    logical :: valid
    call agreement(path,comm,ok,rank)
    if (.not.ok) return
    n=size(names)
    call MPI_Bcast(n,1,MPI_INTEGER,0,comm,err)
    if (err/=MPI_SUCCESS) call MPI_Abort(comm,72,err)
    bad=0
    if (n/=size(names).or.n<1.or.n>max_files) bad=1
    call MPI_Allreduce(bad,total,1,MPI_INTEGER,MPI_MAX,comm,err)
    if (err/=MPI_SUCCESS) call MPI_Abort(comm,72,err)
    ok=total==0
    if (.not.ok) return
    root_names=''
    root_names(:n)=names
    call MPI_Bcast(root_names,128*max_files,MPI_CHARACTER,0,comm,err)
    if (err/=MPI_SUCCESS) call MPI_Abort(comm,72,err)
    do i=1,n
      if (.not.safe_name(names(i)).or.names(i)/=root_names(i)) bad=1
      do j=1,i-1
        if (names(i)==names(j)) bad=1
      enddo
    enddo
    call MPI_Allreduce(bad,total,1,MPI_INTEGER,MPI_MAX,comm,err)
    if (err/=MPI_SUCCESS) call MPI_Abort(comm,72,err)
    ok=total==0
    if (.not.ok) return
    if (rank==0) then
      do i=1,n
        valid=plain_resource(trim(path)//c_null_char,trim(names(i))//c_null_char)==0
        if (valid) call file_fingerprint(trim(path)//'/../../resources/'//trim(names(i)),bytes(i),crc(i),valid)
        if (.not.valid) ok=.false.
      enddo
      if (ok) then
        open(newunit=unit,file=trim(path)//'/RESOURCES',status='new',action='write',iostat=err)
        ok=err==0
        if (ok) then
          write(unit,'(a)',iostat=err) 'ASTR_SHARED_RESOURCES 1'
          if (err==0) write(unit,'(i0)',iostat=err) n
          do i=1,n
            if (err/=0) exit
            write(unit,'(a,1x,i0,1x,z16.16)',iostat=err) trim(names(i)),bytes(i),crc(i)
          enddo
          close(unit,iostat=closed)
          ok=err==0.and.closed==0
        endif
      endif
    endif
    call distribute(ok,comm)
  end subroutine

  logical function valid_resources(path) result(ok)
    character(*),intent(in) :: path
    character(512) :: line,expected
    character(128) :: names(max_files)
    character(16) :: hex
    integer :: unit,err,closed,n,i,j
    integer(int64) :: bytes,crc,wanted_bytes,wanted_crc
    logical :: valid
    ok=.false.
    open(newunit=unit,file=trim(path)//'/RESOURCES',status='old',action='read',iostat=err)
    if (err/=0) return
    read(unit,'(a)',iostat=err) line
    valid=err==0.and.trim(line)=='ASTR_SHARED_RESOURCES 1'
    if (valid) then
      read(unit,'(a)',iostat=err) line
      if (err==0) read(line,*,iostat=err) n
      valid=err==0
      if (valid) valid=n>=1.and.n<=max_files
    endif
    if (valid) then
      write(expected,'(i0)') n
      valid=trim(line)==trim(expected)
    endif
    names=''
    if (valid) then
      do i=1,n
        read(unit,'(a)',iostat=err) line
        if (err==0) read(line,*,iostat=err) names(i),wanted_bytes,hex
        if (err==0) read(hex,'(z16)',iostat=err) wanted_crc
        valid=err==0
        if (.not.valid) exit
        valid=safe_name(names(i)).and.wanted_bytes>=0
        do j=1,i-1
          if (names(i)==names(j)) valid=.false.
        enddo
        write(expected,'(a,1x,i0,1x,z16.16)') trim(names(i)),wanted_bytes,wanted_crc
        valid=valid.and.trim(line)==trim(expected)
        if (.not.valid) exit
        valid=plain_resource(trim(path)//c_null_char,trim(names(i))//c_null_char)==0
        if (.not.valid) exit
        call file_fingerprint(trim(path)//'/../../resources/'//trim(names(i)),bytes,crc,valid)
        valid=valid.and.bytes==wanted_bytes.and.crc==wanted_crc
        if (.not.valid) exit
      enddo
    endif
    if (valid) then
      read(unit,'(a)',iostat=err) line
      valid=err==iostat_end
    endif
    close(unit,iostat=closed)
    ok=valid.and.closed==0
  end function

  subroutine publish_retained_checkpoint(root,name,keep,ledger,comm,ok)
    character(*),intent(in) :: root,name
    integer,intent(in) :: keep,comm
    type(checkpoint_retention),intent(inout) :: ledger
    logical,intent(out) :: ok
    character(128) :: files(max_files),agreed_names(3)
    character(:),allocatable :: entries
    integer :: rank,err,bad,total,count,i,state(2),agreed(2),retired
    call agreement(root,comm,ok,rank)
    if (.not.ok) return
    bad=0
    if (ledger%failed.or.(keep/=1.and.keep/=2)) bad=1
    if (ledger%keep/=0.and.(ledger%keep/=keep.or.ledger%root/=root)) bad=1
    state=[ledger%count,keep]
    agreed=state
    agreed_names=ledger%names
    call MPI_Bcast(agreed,2,MPI_INTEGER,0,comm,err)
    if (err/=MPI_SUCCESS) call MPI_Abort(comm,72,err)
    call MPI_Bcast(agreed_names,384,MPI_CHARACTER,0,comm,err)
    if (err/=MPI_SUCCESS) call MPI_Abort(comm,72,err)
    if (any(state/=agreed).or.any(ledger%names/=agreed_names)) bad=1
    call MPI_Allreduce(bad,total,1,MPI_INTEGER,MPI_MAX,comm,err)
    if (err/=MPI_SUCCESS) call MPI_Abort(comm,72,err)
    ok=total==0
    if (.not.ok) return
    call publish_checkpoint_bundle(root,name,comm,ok)
    if (.not.ok) then
      ledger%failed=.true.
      return
    endif
    ledger%root=root
    ledger%keep=keep
    ledger%count=ledger%count+1
    ledger%names(ledger%count)=name
    if (ledger%count<=keep) return
    ! Only names successfully published by this live ledger can be retired.
    call validate_checkpoint_bundle(trim(root)//'/'//trim(ledger%names(1)),comm,ok,files,count)
    if (ok) then
      retired=0
      if (rank==0) then
        entries='COMPLETE'//achar(10)//'MANIFEST'//achar(10)
        do i=1,count
          entries=entries//trim(files(i))//achar(10)
        enddo
        retired=retire_directory(trim(root)//c_null_char,trim(ledger%names(1))//c_null_char,entries//c_null_char)
      endif
      call MPI_Bcast(retired,1,MPI_INTEGER,0,comm,err)
      if (err/=MPI_SUCCESS) call MPI_Abort(comm,72,err)
      ok=retired==0.or.retired==2
    endif
    if (.not.ok) then
      ledger%failed=.true.
      return
    endif
    ! Explicitly protected batches remain on disk, outside the ordinary keep count.
    ledger%names(1:2)=ledger%names(2:3)
    ledger%names(3)=''
    ledger%count=ledger%count-1
  end subroutine

  subroutine publish_checkpoint_bundle(root,name,comm,ok)
    character(*),intent(in) :: root,name
    integer,intent(in) :: comm
    logical,intent(out) :: ok
    character(128) :: agreed_name
    integer :: rank,err,bad,total,unit,closed
    call agreement(root,comm,ok,rank)
    if (.not.ok) return
    agreed_name=name
    call MPI_Bcast(agreed_name,len(agreed_name),MPI_CHARACTER,0,comm,err)
    if (err/=MPI_SUCCESS) call MPI_Abort(comm,72,err)
    bad=0
    if (.not.safe_name(name).or.name/=agreed_name.or.trim(name)=='LATEST') bad=1
    call MPI_Allreduce(bad,total,1,MPI_INTEGER,MPI_MAX,comm,err)
    if (err/=MPI_SUCCESS) call MPI_Abort(comm,72,err)
    ok=total==0
    if (.not.ok) return
    call validate_checkpoint_bundle(trim(root)//'/'//trim(name)//'.tmp',comm,ok)
    if (.not.ok) return
    if (rank==0) then
      ! Linux RENAME_NOREPLACE is fail-closed on unsupported filesystems.
      ! Never replace an existing destination, even an empty directory/symlink.
#ifdef ASTR_HAS_RENAMEAT2
      err=rename_exclusive(-100_c_int,trim(root)//'/'//trim(name)//'.tmp'//c_null_char, &
        -100_c_int,trim(root)//'/'//trim(name)//c_null_char,1_c_int)
#else
      err=-1
#endif
      ok=err==0
      if (ok) then
        open(newunit=unit,file=trim(root)//'/.LATEST.tmp',status='new',action='write',iostat=err)
        ok=err==0
        if (ok) then
          write(unit,'(a)',iostat=err) trim(name)
          close(unit,iostat=closed)
          ok=err==0.and.closed==0
        endif
      endif
      if (ok) then
        err=rename_entry(trim(root)//'/.LATEST.tmp'//c_null_char,trim(root)//'/LATEST'//c_null_char)
        ok=err==0
      endif
    endif
    call distribute(ok,comm)
  end subroutine

  logical function safe_name(name) result(ok)
    character(*), intent(in) :: name
    integer :: i,c
    ok=len_trim(name)>0.and.len_trim(name)<=128
    if (.not.ok) return
    if (name(1:1)=='.'.or.trim(name)=='MANIFEST'.or.trim(name)=='COMPLETE') then
      ok=.false.
      return
    endif
    do i=1,len_trim(name)
      c=iachar(name(i:i))
      if (.not.((c>=65.and.c<=90).or.(c>=97.and.c<=122).or.(c>=48.and.c<=57).or. &
        name(i:i)=='_'.or.name(i:i)=='-'.or.name(i:i)=='.')) ok=.false.
    enddo
  end function

  subroutine agreement(path,comm,ok,rank)
    character(*), intent(in) :: path
    integer,intent(in) :: comm
    logical,intent(out) :: ok
    integer,intent(out) :: rank
    character(1024) :: root
    integer :: err,bad,total
    call MPI_Comm_rank(comm,rank,err)
    if (err/=MPI_SUCCESS) call MPI_Abort(comm,72,err)
    root=path
    call MPI_Bcast(root,len(root),MPI_CHARACTER,0,comm,err)
    if (err/=MPI_SUCCESS) call MPI_Abort(comm,72,err)
    bad=0
    if (len_trim(path)==0.or.len_trim(path)>len(root).or.path/=root) bad=1
    call MPI_Allreduce(bad,total,1,MPI_INTEGER,MPI_MAX,comm,err)
    if (err/=MPI_SUCCESS) call MPI_Abort(comm,72,err)
    ok=total==0
  end subroutine

  subroutine distribute(ok,comm)
    logical,intent(inout) :: ok
    integer,intent(in) :: comm
    integer :: err
    call MPI_Bcast(ok,1,MPI_LOGICAL,0,comm,err)
    if (err/=MPI_SUCCESS) call MPI_Abort(comm,72,err)
  end subroutine

  subroutine seal_checkpoint_bundle(path,names,comm,ok)
    character(*),intent(in) :: path,names(:)
    integer,intent(in) :: comm
    logical,intent(out) :: ok
    integer :: rank,err,i,j,n,unit,closed,bad,total
    character(128) :: root_names(max_files)
    integer(int64) :: bytes(max_files),crc(max_files),manifest_bytes,manifest_crc
    logical :: valid,has_resources
    call agreement(path,comm,ok,rank)
    if (.not.ok) return
    n=size(names)
    call MPI_Bcast(n,1,MPI_INTEGER,0,comm,err)
    if (err/=MPI_SUCCESS) call MPI_Abort(comm,72,err)
    bad=0
    if (n/=size(names).or.n<1.or.n>max_files) bad=1
    call MPI_Allreduce(bad,total,1,MPI_INTEGER,MPI_MAX,comm,err)
    if (err/=MPI_SUCCESS) call MPI_Abort(comm,72,err)
    ok=total==0
    if (.not.ok) return
    root_names=''
    root_names(:n)=names
    call MPI_Bcast(root_names,128*max_files,MPI_CHARACTER,0,comm,err)
    if (err/=MPI_SUCCESS) call MPI_Abort(comm,72,err)
    do i=1,n
      if (.not.safe_name(names(i)).or.names(i)/=root_names(i)) bad=1
      do j=1,i-1
        if (names(i)==names(j)) bad=1
      enddo
    enddo
    call MPI_Allreduce(bad,total,1,MPI_INTEGER,MPI_MAX,comm,err)
    if (err/=MPI_SUCCESS) call MPI_Abort(comm,72,err)
    ok=total==0
    if (.not.ok) return
    if (rank==0) then
      do i=1,n
        call file_fingerprint(trim(path)//'/'//trim(names(i)),bytes(i),crc(i),valid)
        if (.not.valid) ok=.false.
      enddo
      inquire(file=trim(path)//'/RESOURCES',exist=has_resources,iostat=err)
      if (err/=0) ok=.false.
      if (has_resources.and..not.any(names=='RESOURCES')) ok=.false.
      if (ok) then
        open(newunit=unit,file=trim(path)//'/MANIFEST',status='new',action='write',iostat=err)
        ok=err==0
        if (ok) then
          write(unit,'(a)',iostat=err) 'ASTR_CHECKPOINT_BUNDLE 1'
          if (err==0) write(unit,'(i0)',iostat=err) n
          do i=1,n
            if (err/=0) exit
            write(unit,'(a,1x,i0,1x,z16.16)',iostat=err) trim(names(i)),bytes(i),crc(i)
          enddo
          close(unit,iostat=closed)
          ok=err==0.and.closed==0
        endif
      endif
      if (ok) then
        call file_fingerprint(trim(path)//'/MANIFEST',manifest_bytes,manifest_crc,ok)
      endif
      if (ok) then
        ! The completion record is created only after every data file and manifest close.
        open(newunit=unit,file=trim(path)//'/COMPLETE',status='new',action='write',iostat=err)
        ok=err==0
        if (ok) then
          write(unit,'(a,1x,i0,1x,z16.16)',iostat=err) 'ASTR_COMPLETE_1',manifest_bytes,manifest_crc
          close(unit,iostat=closed)
          ok=err==0.and.closed==0
        endif
      endif
    endif
    call distribute(ok,comm)
  end subroutine

  subroutine validate_checkpoint_bundle(path,comm,ok,files,file_count)
    character(*),intent(in) :: path
    integer,intent(in) :: comm
    logical,intent(out) :: ok
    character(128),optional,intent(out) :: files(max_files)
    integer,optional,intent(out) :: file_count
    integer :: rank,unit,err,closed,n,i,j
    character(512) :: line,expected
    character(128) :: names(max_files),label
    character(16) :: hex
    integer(int64) :: wanted_bytes,wanted_crc,bytes,crc
    logical :: valid
    names=''
    n=0
    call agreement(path,comm,ok,rank)
    if (.not.ok) return
    if (rank==0) call validate_local()
    call distribute(ok,comm)
    if (ok) then
      call MPI_Bcast(n,1,MPI_INTEGER,0,comm,err)
      if (err/=MPI_SUCCESS) call MPI_Abort(comm,72,err)
      call MPI_Bcast(names,128*max_files,MPI_CHARACTER,0,comm,err)
      if (err/=MPI_SUCCESS) call MPI_Abort(comm,72,err)
      if (present(files)) files=names
      if (present(file_count)) file_count=n
    endif
  contains
    subroutine validate_local()
      ok=.false.
      open(newunit=unit,file=trim(path)//'/COMPLETE',status='old',action='read',iostat=err)
      if (err/=0) return
      read(unit,'(a)',iostat=err) line
      if (err==0) read(line,*,iostat=err) label,wanted_bytes,hex
      if (err==0) read(hex,'(z16)',iostat=err) wanted_crc
      if (err==0) then
        write(expected,'(a,1x,i0,1x,z16.16)') 'ASTR_COMPLETE_1',wanted_bytes,wanted_crc
        if (trim(line)/=trim(expected)) err=1
      endif
      valid=err==0
      if (valid) read(unit,'(a)',iostat=err) line
      close(unit,iostat=closed)
      if (.not.valid.or.err/=iostat_end.or.closed/=0) return
      call file_fingerprint(trim(path)//'/MANIFEST',bytes,crc,valid)
      if (.not.valid.or.bytes/=wanted_bytes.or.crc/=wanted_crc) return
      open(newunit=unit,file=trim(path)//'/MANIFEST',status='old',action='read',iostat=err)
      if (err/=0) return
      read(unit,'(a)',iostat=err) line
      valid=err==0.and.trim(line)=='ASTR_CHECKPOINT_BUNDLE 1'
      if (valid) then
        read(unit,'(a)',iostat=err) line
        if (err==0) read(line,*,iostat=err) n
        valid=err==0
        if (valid) valid=n>=1.and.n<=max_files
      endif
      if (valid) then
        write(expected,'(i0)') n
        valid=trim(line)==trim(expected)
      endif
      names=''
      if (valid) then
        do i=1,n
          read(unit,'(a)',iostat=err) line
          if (err==0) read(line,*,iostat=err) names(i),wanted_bytes,hex
          if (err==0) read(hex,'(z16)',iostat=err) wanted_crc
          valid=err==0
          if (.not.valid) exit
          valid=safe_name(names(i)).and.wanted_bytes>=0
          do j=1,i-1
            if (names(j)==names(i)) valid=.false.
          enddo
          write(expected,'(a,1x,i0,1x,z16.16)') trim(names(i)),wanted_bytes,wanted_crc
          valid=valid.and.trim(line)==trim(expected)
          if (.not.valid) exit
          call file_fingerprint(trim(path)//'/'//trim(names(i)),bytes,crc,valid)
          valid=valid.and.bytes==wanted_bytes.and.crc==wanted_crc
          if (.not.valid) exit
        enddo
      endif
      if (valid) then
        read(unit,'(a)',iostat=err) line
        valid=err==iostat_end
      endif
      close(unit,iostat=closed)
      ok=valid.and.closed==0
      if (ok.and.any(names=='RESOURCES')) ok=valid_resources(path)
    end subroutine
  end subroutine
end module
