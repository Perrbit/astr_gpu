module output_lineage
  use iso_fortran_env, only: int64,real64,iostat_end
  use iso_c_binding, only: c_char,c_int,c_int64_t,c_double,c_null_char
  use ieee_arithmetic, only: ieee_is_finite
  use mpi, only: MPI_COMM_SELF
  use insitu_checkpoint_batch, only: file_fingerprint
  use checkpoint_bundle, only: validate_checkpoint_bundle
  implicit none
  private
  public :: prepare_output_lineage,stage_output_lineage
  integer,parameter :: max_segments=128,max_frames=8192
  interface
    function plain_frame(path) bind(C,name='astr_output_plain_frame') result(status)
      import c_char,c_int
      character(c_char),intent(in) :: path(*)
      integer(c_int) :: status
    end function
    function archive_parent(current,parent,relative,capacity) bind(C,name='astr_output_archive_parent') result(status)
      import c_char,c_int
      character(c_char),intent(in) :: current(*),parent(*)
      character(c_char),intent(out) :: relative(*)
      integer(c_int),value :: capacity
      integer(c_int) :: status
    end function
    function catalog(path,steps,capacity,count) bind(C,name='astr_output_frame_catalog') result(status)
      import c_char,c_int,c_int64_t
      character(c_char),intent(in) :: path(*)
      integer(c_int64_t),intent(out) :: steps(*)
      integer(c_int),value :: capacity
      integer(c_int),intent(out) :: count
      integer(c_int) :: status
    end function
    function frame_layout(path,geometry,step,time,components,units,selected,nselected,bytes, &
      shape,axes,indices,nplanes,current_components,current_units,current_selected,current_nselected) &
      bind(C,name='astr_output_frame_layout') result(status)
      import c_char,c_int,c_int64_t,c_double
      character(c_char),intent(in) :: path(*),geometry(*),units(*),current_units(*)
      integer(c_int64_t),value :: step,bytes
      real(c_double),value :: time
      integer(c_int),value :: components,nselected,nplanes,current_components,current_nselected
      integer(c_int),intent(in) :: selected(*),shape(*),axes(*),indices(*),current_selected(*)
      integer(c_int) :: status
    end function
  end interface
contains
  ! Rank zero only: validate immutable ancestors once, never load field arrays.
  subroutine prepare_output_lineage(root,product,shape,axes,indices,components,units,selected,compatible,ok)
    character(*),intent(in) :: root,product,units
    integer,intent(in) :: shape(3),axes(:),indices(:),components,selected(:)
    logical,intent(out) :: compatible,ok
    character(1200) :: paths(max_segments),parent,source,geometry,frame
    character(c_char) :: relative(1200)
    integer(int64) :: origins(max_segments),sizes(max_segments),crcs(max_segments),steps(max_frames)
    real(real64) :: times(max_segments),time,previous_time
    integer(int64) :: parent_id,parent_size,parent_crc,step,previous_step
    integer :: depth,d,i,n,unit,err,closed,status,components_saved,selection(14),nselection
    character(16) :: saved_units
    character(16) :: name
    logical :: valid
    ok=.false.; compatible=.true.; paths=''; paths(1)='.'; depth=1
    do
      if(len_trim(root)+1+len_trim(paths(depth))>1200) return
      source=trim(root)//'/'//trim(paths(depth))
      call read_segment(trim(source),origins(depth),times(depth),sizes(depth),crcs(depth), &
        parent_id,parent_size,parent_crc,parent,valid)
      if(.not.valid) return
      if(parent_id<0) exit
      if(depth==max_segments.or.len_trim(source)+1+len_trim(parent)>1199) return
      source=trim(source)//'/'//trim(parent)
      status=archive_parent(trim(root)//c_null_char,trim(source)//c_null_char,relative,1200_c_int)
      if(status/=0) return
      parent=''
      do i=1,1199
        if(relative(i)==c_null_char) exit
        parent(i:i)=relative(i)
      enddo
      if(any(paths(:depth)==parent)) return
      ! Canonical relative path must end in the same product/segment identity.
      write(name,'("segment",i8.8)') parent_id
      i=len_trim(parent)-len_trim(name)
      if(i<1) return
      if(parent(i:) /= '/'//trim(name)) return
      depth=depth+1; paths(depth)=parent
      call file_fingerprint(trim(root)//'/'//trim(parent)//'/SEGMENT',sizes(depth),crcs(depth),valid)
      if(.not.valid.or.sizes(depth)/=parent_size.or.crcs(depth)/=parent_crc) return
    enddo
    do d=2,depth
      if(origins(d)>origins(d-1).or.times(d)>times(d-1)) return
    enddo
    open(newunit=unit,file=trim(root)//'/lineage.parent',status='new',action='write',iostat=err)
    if(err/=0) return
    write(unit,'(a)',iostat=err) 'ASTR_NATIVE_LINEAGE_1'
    previous_step=-1; previous_time=-1
    do d=depth,2,-1
      if(err/=0) exit
      if(len_trim(root)+1+len_trim(paths(d))+25>1200) then
        err=1; exit
      endif
      source=trim(root)//'/'//trim(paths(d))
      status=catalog(trim(source)//c_null_char,steps,max_frames,n)
      if(status/=0) then
        err=1; exit
      endif
      geometry=trim(source)//'/../resources/data.h5'
      do i=1,n
        if(steps(i)>origins(d-1)) cycle
        write(name,'("step",i12.12)') steps(i)
        frame=trim(source)//'/'//name
        call read_frame(trim(frame),steps(i),step,time,components_saved,saved_units,selection,nselection,valid)
        if(.not.valid) then
          err=1; exit
        endif
        if(step<origins(d).or.time<times(d).or.time>times(d-1).or. &
          step<=previous_step.or.time<=previous_time) then
          err=1; exit
        endif
        block
          integer(int64) :: bytes,download
          call frame_byte_counts(trim(frame),bytes,download,valid)
          if(.not.valid) then
            err=1
          else
            status=frame_layout(trim(frame)//'/data.h5'//c_null_char,trim(geometry)//c_null_char,step,time, &
              components_saved,trim(saved_units)//c_null_char,selection,nselection,bytes,shape,axes,indices,size(axes), &
              components,trim(units)//c_null_char,selected,size(selected))
            if(status==1) err=1
            if(status==2) compatible=.false.
          endif
        end block
        if(err/=0) exit
        if(len_trim(paths(d))+len_trim(name)+10>1200.or.len_trim(paths(d))+25>1200) then
          err=1; exit
        endif
        write(unit,'(i0,1x,es24.16,/,a,/,a)',iostat=err) step,time, &
          trim(paths(d))//'/'//name//'/data.h5',trim(paths(d))//'/../resources/data.h5'
        previous_step=step; previous_time=time
      enddo
    enddo
    close(unit,iostat=closed)
    ok=err==0.and.closed==0
  end subroutine

  subroutine read_segment(path,step,time,bytes,crc,parent_id,parent_bytes,parent_crc,parent,ok)
    character(*),intent(in) :: path
    integer(int64),intent(out) :: step,bytes,crc,parent_id,parent_bytes,parent_crc
    real(real64),intent(out) :: time
    character(*),intent(out) :: parent
    logical,intent(out) :: ok
    integer :: unit,err,closed
    integer(int64) :: restore_bytes,restore_crc,input_bytes,input_crc,actual_bytes,actual_crc
    character(1201) :: rows(6),extra
    character(16) :: hex
    logical :: valid
    ok=.false.
    call file_fingerprint(path//'/SEGMENT',bytes,crc,valid)
    if(.not.valid) return
    open(newunit=unit,file=path//'/SEGMENT',status='old',action='read',iostat=err)
    if(err/=0) return
    do closed=1,6
      read(unit,'(a)',iostat=err) rows(closed)
      if(err/=0) exit
    enddo
    if(err/=0) then
      close(unit)
      return
    endif
    read(unit,'(a)',iostat=err) extra
    close(unit,iostat=closed)
    if(err/=iostat_end.or.closed/=0.or.trim(rows(1))/='ASTR_OUTPUT_SEGMENT_2') return
    if(words(rows(2))/=2.or.words(rows(3))/=2.or.words(rows(4))/=3.or.words(rows(5))/=2) return
    read(rows(2),*,iostat=err) step,time
    if(err/=0) return
    if(step<0.or.step>=1000000000000_int64.or..not.ieee_is_finite(time).or.time<0) return
    read(rows(3),*,iostat=err) restore_bytes,hex
    if(err==0) read(hex,'(z16)',iostat=err) restore_crc
    if(err/=0.or.restore_bytes<0) return
    read(rows(4),*,iostat=err) parent_id,parent_bytes,hex
    if(err==0) read(hex,'(z16)',iostat=err) parent_crc
    if(err/=0) return
    read(rows(5),*,iostat=err) input_bytes,hex
    if(err==0) read(hex,'(z16)',iostat=err) input_crc
    if(err/=0) return
    call file_fingerprint(path//'/input.txt',actual_bytes,actual_crc,valid)
    if(.not.valid.or.actual_bytes/=input_bytes.or.actual_crc/=input_crc) return
    parent=trim(rows(6))
    if(len_trim(rows(6))>len(parent)) return
    if(parent_id==-1) then
      ok=parent=='none'.and.parent_bytes==0.and.parent_crc==0
    else
      ok=parent_id>=0.and.parent_id<=99999999.and.parent_bytes>0.and.restore_bytes>0.and. &
        len_trim(parent)>0
      if(ok) ok=parent(1:1)/='/'.and.index(parent,':')==0
    endif
  end subroutine

  subroutine read_frame(path,expected_step,step,time,components,units,selected,nselected,ok)
    character(*),intent(in) :: path
    integer(int64),intent(in) :: expected_step
    integer(int64),intent(out) :: step
    real(real64),intent(out) :: time
    integer,intent(out) :: components,selected(14),nselected
    character(*),intent(out) :: units
    logical,intent(out) :: ok
    character(256) :: rows(5),extra
    character(128) :: files(64)
    integer :: unit,err,closed,n,i
    logical :: derived,valid
    ok=.false.; selected=0; nselected=0
    if(plain_frame(path//c_null_char)/=0) return
    if(.not.registered_frame_resources(path)) return
    call validate_checkpoint_bundle(path,MPI_COMM_SELF,valid,files,n)
    if(.not.valid) return
    if(n/=4.or..not.all([any(files=='data.h5'),any(files=='data.xdmf'),any(files=='FRAME'),any(files=='RESOURCES')])) return
    open(newunit=unit,file=path//'/FRAME',status='old',action='read',iostat=err)
    if(err/=0) return
    do i=1,4
      read(unit,'(a)',iostat=err) rows(i)
      if(err/=0) exit
    enddo
    if(err/=0) then
      close(unit)
      return
    endif
    derived=rows(1)=='ASTR_DERIVED_FRAME_1'
    if(err==0.and.derived) read(unit,'(a)',iostat=err) rows(5)
    if(err/=0) then
      close(unit)
      return
    endif
    read(unit,'(a)',iostat=err) extra
    close(unit,iostat=closed)
    if(err/=iostat_end.or.closed/=0) return
    if(.not.derived.and.rows(1)/='ASTR_BASIC_FRAME_1') return
    if(words(rows(2))/=2.or.words(rows(3))/=2.or.words(rows(4))/=2) return
    read(rows(2),*,iostat=err) step,time
    if(err/=0) return
    if(step/=expected_step.or..not.ieee_is_finite(time).or.time<0) return
    read(rows(3),*,iostat=err) components,units
    if(err/=0) return
    if(derived) then
      if(words(rows(5))/=14) return
      read(rows(5),*,iostat=err) selected
      if(err/=0) return
      do i=1,14
        if(selected(i)==0) exit
        if(selected(i)<1.or.selected(i)>14) return
        if(i>1) then
          if(selected(i)<=selected(i-1)) return
        endif
        nselected=i
      enddo
      if(nselected==0.or.any(selected(nselected+1:)/=0)) return
    endif
    ok=.true.
  end subroutine

  logical function registered_frame_resources(path) result(ok)
    character(*),intent(in) :: path
    character(512) :: rows(4),extra
    integer :: unit,err,closed,i
    ok=.false.
    open(newunit=unit,file=path//'/RESOURCES',status='old',action='read',iostat=err)
    if(err/=0) return
    do i=1,4
      read(unit,'(a)',iostat=err) rows(i)
      if(err/=0) exit
    enddo
    if(err/=0) then
      close(unit)
      return
    endif
    read(unit,'(a)',iostat=err) extra
    close(unit,iostat=closed)
    ok=err==iostat_end.and.closed==0.and.rows(1)=='ASTR_SHARED_RESOURCES 1'.and.rows(2)=='2'.and. &
      rows(3)(1:8)=='SEGMENT '.and.rows(4)(1:8)=='data.h5 '
  end function

  subroutine frame_byte_counts(path,bytes,download,ok)
    character(*),intent(in) :: path
    integer(int64),intent(out) :: bytes,download
    logical,intent(out) :: ok
    integer :: unit,err,closed,i
    character(256) :: line
    ok=.false.
    open(newunit=unit,file=path//'/FRAME',status='old',action='read',iostat=err)
    if(err/=0) return
    do i=1,4
      read(unit,'(a)',iostat=err) line
      if(err/=0) exit
    enddo
    if(err==0) read(line,*,iostat=err) bytes,download
    close(unit,iostat=closed)
    if(err/=0.or.closed/=0) return
    ok=bytes>=0.and.(download==0.or.download==bytes)
  end subroutine

  integer function words(line) result(n)
    character(*),intent(in) :: line
    integer :: i
    logical :: inword
    n=0; inword=.false.
    do i=1,len_trim(line)
      if(line(i:i)==' '.or.line(i:i)==achar(9)) then
        inword=.false.
      else if(.not.inword) then
        n=n+1; inword=.true.
      endif
    enddo
  end function

  subroutine stage_output_lineage(root,ok)
    character(*),intent(in) :: root
    logical,intent(out) :: ok
    integer :: source,target,err,closed,status,i
    character(1201) :: line
    integer(int64) :: step
    real(real64) :: time
    character(16) :: name
    ok=.false.
    open(newunit=target,file=root//'/lineage.frames.tmp',status='new',action='write',iostat=err)
    if(err/=0) return
    do i=1,2
      if(i==1) then
        open(newunit=source,file=root//'/lineage.parent',status='old',action='read',iostat=err)
      else
        open(newunit=source,file=root//'/series.frames',status='old',action='read',iostat=err)
        if(err==0) read(source,'(a)',iostat=err) line
        if(err==0.and.line/='ASTR_FRAME_SERIES_1') err=1
      endif
      if(err/=0) exit
      do
        read(source,'(a)',iostat=status) line
        if(status==iostat_end) exit
        if(status/=0.or.len_trim(line)>1200) then
          err=1; exit
        endif
        if(i==1) then
          write(target,'(a)',iostat=err) trim(line)
        else
          read(line,*,iostat=err) step,time
          if(err/=0) exit
          write(name,'("step",i12.12)') step
          write(target,'(a,/,a,/,a)',iostat=err) trim(line),name//'/data.h5','../resources/data.h5'
        endif
        if(err/=0) exit
      enddo
      close(source,iostat=closed)
      if(closed/=0) err=closed
      if(err/=0) exit
    enddo
    close(target,iostat=closed)
    ok=err==0.and.closed==0
  end subroutine
end module
