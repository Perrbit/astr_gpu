module insitu_catalog
  use mpi
  use iso_fortran_env, only: int64,real64
  use iso_c_binding, only: c_ptr,c_int,c_int64_t,c_double,c_size_t,c_char,c_null_char,c_associated,c_f_pointer
  use insitu_run_config, only: catalog_product,catalog_field_id,catalog_field_names
  use insitu_product_schedule, only: runtime_catalog_apply,runtime_catalog_count,runtime_catalog_state, &
    runtime_product_state,runtime_catalog_state_file,runtime_active,runtime_bad_config,runtime_waiting, &
    runtime_resource_disabled,runtime_absent
  implicit none
  private
  public :: read_product_catalog,parse_catalog_product,validate_catalog_grid
  public :: configure_catalog_watch,poll_catalog_watch,catalog_watch_counts,rebase_catalog_watch
  public :: catalog_watch_state_file,write_catalog_status
  character(:),allocatable,save :: watched_path,watch_error
  integer(int64),save :: watch_interval=10,watch_next_step=0,watch_reads=0,watch_checks=0
  integer(int64),save :: watch_signature(8)=0
  integer,save :: watch_last_status=-1
  interface
    function json_read(text,n,error,capacity) bind(C,name='astr_catalog_json_read') result(doc)
      import c_ptr,c_char,c_size_t
      character(c_char),intent(in) :: text(*)
      integer(c_size_t),value :: n,capacity
      character(c_char),intent(out) :: error(*)
      type(c_ptr) :: doc
    end function
    subroutine json_free(doc) bind(C,name='astr_catalog_json_free')
      import c_ptr
      type(c_ptr),value :: doc
    end subroutine
    function json_root(doc) bind(C,name='astr_catalog_json_root') result(value)
      import c_ptr
      type(c_ptr),value :: doc
      type(c_ptr) :: value
    end function
    function json_get(obj,key) bind(C,name='astr_catalog_json_get') result(value)
      import c_ptr,c_char
      type(c_ptr),value :: obj
      character(c_char),intent(in) :: key(*)
      type(c_ptr) :: value
    end function
    function json_at(arr,index) bind(C,name='astr_catalog_json_at') result(value)
      import c_ptr,c_size_t
      type(c_ptr),value :: arr
      integer(c_size_t),value :: index
      type(c_ptr) :: value
    end function
    function json_key(obj,index) bind(C,name='astr_catalog_json_key') result(value)
      import c_ptr,c_size_t
      type(c_ptr),value :: obj
      integer(c_size_t),value :: index
      type(c_ptr) :: value
    end function
    integer(c_int) function json_type(value) bind(C,name='astr_catalog_json_type')
      import c_ptr,c_int
      type(c_ptr),value :: value
    end function
    integer(c_size_t) function json_size(value) bind(C,name='astr_catalog_json_size')
      import c_ptr,c_size_t
      type(c_ptr),value :: value
    end function
    function json_string(value,n) bind(C,name='astr_catalog_json_string') result(text)
      import c_ptr,c_size_t
      type(c_ptr),value :: value
      integer(c_size_t),intent(out) :: n
      type(c_ptr) :: text
    end function
    integer(c_int) function json_integer(value,n) bind(C,name='astr_catalog_json_integer')
      import c_ptr,c_int,c_int64_t
      type(c_ptr),value :: value
      integer(c_int64_t),intent(out) :: n
    end function
    integer(c_int) function json_real(value,n) bind(C,name='astr_catalog_json_real')
      import c_ptr,c_int,c_double
      type(c_ptr),value :: value
      real(c_double),intent(out) :: n
    end function
    function catalog_file_read(path,n,identity,error,capacity) bind(C,name='astr_catalog_file_read') result(data)
      import c_ptr,c_char,c_size_t,c_int64_t
      character(c_char),intent(in) :: path(*)
      integer(c_size_t),intent(out) :: n
      integer(c_int64_t),intent(out) :: identity(8)
      character(c_char),intent(out) :: error(*)
      integer(c_size_t),value :: capacity
      type(c_ptr) :: data
    end function
    integer(c_int) function catalog_file_signature(path,identity,error,capacity) &
        bind(C,name='astr_catalog_file_signature') result(status)
      import c_char,c_size_t,c_int64_t,c_int
      character(c_char),intent(in) :: path(*)
      integer(c_int64_t),intent(out) :: identity(8)
      character(c_char),intent(out) :: error(*)
      integer(c_size_t),value :: capacity
    end function
    subroutine catalog_release(data) bind(C,name='astr_catalog_release')
      import c_ptr
      type(c_ptr),value :: data
    end subroutine
    function json_quote(text,n,length) bind(C,name='astr_catalog_json_quote') result(encoded)
      import c_char,c_size_t,c_ptr
      character(c_char),intent(in) :: text(*)
      integer(c_size_t),value :: n
      integer(c_size_t),intent(out) :: length
      type(c_ptr) :: encoded
    end function
    integer(c_int) function publish_text(path,text,n,error,capacity) bind(C,name='astr_catalog_publish_text') result(status)
      import c_char,c_size_t,c_int
      character(c_char),intent(in) :: path(*),text(*)
      integer(c_size_t),value :: n,capacity
      character(c_char),intent(out) :: error(*)
    end function
  end interface
contains
  subroutine catalog_watch_state_file(unit,writing,step,time,comm,ok)
    integer,intent(in) :: unit,comm
    logical,intent(in) :: writing
    integer(int64),intent(in) :: step
    real(real64),intent(in) :: time
    logical,intent(out) :: ok
    character(8) :: magic
    integer(int64) :: interval,next_step
    integer :: status
    magic='ASTRCW01';ok=.false.
    if(writing) then
      write(unit,iostat=status) magic,watch_interval,watch_next_step
    else
      read(unit,iostat=status) magic,interval,next_step
      if(status/=0) return
      if(magic/='ASTRCW01'.or.interval<1.or.next_step<0) return
    endif
    if(status/=0) return
    call runtime_catalog_state_file(unit,writing,step,time,ok)
    if(.not.ok.or.writing) return
    watch_interval=interval;watch_next_step=next_step
    call rebase_catalog_watch(comm)
  end subroutine

  subroutine quote_catalog_text(text,encoded,ok)
    character(*),intent(in) :: text
    character(:),allocatable,intent(out) :: encoded
    logical,intent(out) :: ok
    type(c_ptr) :: address
    integer(c_size_t) :: n
    character(c_char),pointer :: bytes(:)
    integer :: i
    address=json_quote(text,len(text,kind=c_size_t),n)
    ok=c_associated(address)
    encoded=''
    if(.not.ok) return
    call c_f_pointer(address,bytes,[int(n)])
    encoded=repeat(' ',int(n))
    do i=1,int(n)
      encoded(i:i)=bytes(i)
    enddo
    call catalog_release(address)
  end subroutine

  function integer_text(n) result(text)
    integer(int64),intent(in) :: n
    character(:),allocatable :: text
    character(32) :: buffer
    write(buffer,'(i0)') n
    text=trim(buffer)
  end function

  function real_text(n) result(text)
    real(real64),intent(in) :: n
    character(:),allocatable :: text
    character(32) :: buffer
    write(buffer,'(es26.17e3)') n
    text=trim(adjustl(buffer))
  end function

  subroutine write_catalog_status(path,step,time,comm,message,ok)
    character(*),intent(in) :: path,message
    integer(int64),intent(in) :: step
    real(real64),intent(in) :: time
    integer,intent(in) :: comm
    logical,intent(out) :: ok
    type(runtime_product_state) :: state
    character(:),allocatable :: output,encoded,id,reason,label
    character(c_char) :: errors(512)
    integer :: rank,err,status,i
    logical :: quoted
    call MPI_Comm_rank(comm,rank,err)
    call require_catalog_mpi(err,comm)
    status=0
    if(rank==0) then
      if(len(message)>0.or..not.allocated(watch_error)) then
        call quote_catalog_text(message,encoded,quoted)
      else
        call quote_catalog_text(watch_error,encoded,quoted)
      endif
      if(.not.quoted) status=-1
      output='{"schema":"astr-insitu-products-status-v1","step":'//integer_text(step)// &
        ',"time":'//real_text(time)//',"catalog_error":'//encoded//',"products":['
      do i=1,runtime_catalog_count()
        if(status/=0) exit
        call runtime_catalog_state(i,state)
        call quote_catalog_text(state%definition%id,id,quoted)
        if(.not.quoted) then
          status=-1;exit
        endif
        call quote_catalog_text(state%reason,reason,quoted)
        if(.not.quoted) then
          status=-1;exit
        endif
        select case(state%status)
        case(runtime_active)
          label='active'
        case(runtime_bad_config)
          label='invalid-configuration'
        case(runtime_waiting)
          label='waiting-for-statistics'
        case(runtime_resource_disabled)
          label='resource-disabled'
        case(runtime_absent)
          label='not-in-catalog'
        end select
        if(i>1) output=output//','
        output=output//'{"id":'//id//',"version":'//integer_text(state%version)// &
          ',"state":"'//label//'","effective_step":'//integer_text(state%applied_step)// &
          ',"effective_time":'//real_text(state%applied_time)// &
          ',"last_output_step":'//integer_text(state%last_step)// &
          ',"last_output_time":'//real_text(state%last_time)// &
          ',"consumed_enable_request":'//integer_text(state%consumed_request)// &
          ',"reason":'//reason//'}'
      enddo
      output=output//']}'
      if(status==0) status=publish_text(path//c_null_char,output,len(output,kind=c_size_t),errors,size(errors,kind=c_size_t))
      if(status/=0) then
        if(status<0) then
          print '(a)','ASTR_CATALOG_STATUS_WRITE_FAILED cannot encode JSON status'
        else
          print '(a)','ASTR_CATALOG_STATUS_WRITE_FAILED '//error_text(errors)
        endif
      endif
    endif
    call MPI_Bcast(status,1,MPI_INTEGER,0,comm,err)
    call require_catalog_mpi(err,comm)
    ok=status==0
  end subroutine

  subroutine configure_catalog_watch(path,interval,origin_step,comm,restoring)
    character(*),intent(in) :: path
    integer(int64),intent(in) :: interval,origin_step
    integer,intent(in) :: comm
    logical,optional,intent(in) :: restoring
    watched_path=path;watch_interval=interval;watch_next_step=origin_step
    watch_error=''
    watch_reads=0;watch_checks=0;watch_last_status=-1;watch_signature=0
    if(interval<=0) error stop 'catalog polling interval must be positive'
    if(present(restoring)) then
      if(restoring) call rebase_catalog_watch(comm)
    endif
  end subroutine

  subroutine rebase_catalog_watch(comm)
    integer,intent(in) :: comm
    integer :: rank,err
    character(c_char) :: error(512)
    call MPI_Comm_rank(comm,rank,err)
    call require_catalog_mpi(err,comm)
    if(rank/=0) return
    watch_checks=watch_checks+1
    watch_last_status=catalog_file_signature(watched_path//c_null_char,watch_signature,error,size(error,kind=c_size_t))
  end subroutine

  subroutine poll_catalog_watch(step,time,extent,numq,comm,updated,message)
    integer(int64),intent(in) :: step
    real(real64),intent(in) :: time
    integer,intent(in) :: extent(3),numq,comm
    logical,intent(out) :: updated
    character(:),allocatable,intent(out) :: message
    type(catalog_product),allocatable :: entries(:)
    integer(int64) :: observed(8),opened(8)
    character(c_char) :: error(512)
    logical :: accepted
    integer :: rank,err,status,event,n,i
    updated=.false.;message=''
    if(step<watch_next_step) return
    if(step<=huge(step)-watch_interval) then
      watch_next_step=step+watch_interval
    else
      watch_next_step=huge(step)
    endif
    call MPI_Comm_rank(comm,rank,err)
    call require_catalog_mpi(err,comm)
    event=0
    if(rank==0) then
      watch_checks=watch_checks+1
      status=catalog_file_signature(watched_path//c_null_char,observed,error,size(error,kind=c_size_t))
      if(status/=watch_last_status.or.any(observed/=watch_signature)) then
        watch_last_status=status;watch_signature=observed
        if(status/=0) then
          event=2;message=error_text(error)
        else
          watch_reads=watch_reads+1
          call read_product_catalog(watched_path,entries,accepted,message,opened)
          ! This is the opened descriptor's identity, not a later stat of the path.
          if(any(opened/=0)) watch_signature=opened
          if(accepted) then
            event=1
            do i=1,size(entries)
              call validate_catalog_grid(entries(i),extent,numq)
            enddo
          else
            event=2
          endif
        endif
      endif
    endif
    call MPI_Bcast(event,1,MPI_INTEGER,0,comm,err)
    call require_catalog_mpi(err,comm)
    if(event==0) return
    call broadcast_catalog_text(message,rank,comm)
    if(event==2) then
      watch_error=message
      return
    endif
    watch_error=''
    n=0
    if(rank==0) n=size(entries)
    call MPI_Bcast(n,1,MPI_INTEGER,0,comm,err)
    call require_catalog_mpi(err,comm)
    if(rank/=0) allocate(entries(n))
    do i=1,n
      call broadcast_catalog_product(entries(i),rank,comm)
    enddo
    call runtime_catalog_apply(entries,step,time)
    updated=.true.
  end subroutine

  subroutine require_catalog_mpi(status,comm)
    integer,intent(in) :: status,comm
    integer :: err
    if(status==MPI_SUCCESS) return
    call MPI_Abort(comm,status,err)
    error stop 'runtime catalog MPI communication failed'
  end subroutine

  subroutine broadcast_catalog_text(text,rank,comm)
    character(:),allocatable,intent(inout) :: text
    integer,intent(in) :: rank,comm
    integer :: n,err
    n=0
    if(rank==0) then
      if(allocated(text)) n=len(text)
    endif
    call MPI_Bcast(n,1,MPI_INTEGER,0,comm,err)
    call require_catalog_mpi(err,comm)
    if(rank/=0) text=repeat(' ',n)
    if(n==0) return
    call MPI_Bcast(text,n,MPI_CHARACTER,0,comm,err)
    call require_catalog_mpi(err,comm)
  end subroutine

  subroutine broadcast_catalog_names(names,rank,comm)
    character(:),allocatable,intent(inout) :: names(:)
    integer,intent(in) :: rank,comm
    character(:),allocatable :: text
    integer :: shape(2),i,err
    shape=[0,1]
    if(rank==0) then
      if(allocated(names)) shape=[size(names),len(names)]
    endif
    call MPI_Bcast(shape,2,MPI_INTEGER,0,comm,err)
    call require_catalog_mpi(err,comm)
    if(rank/=0) allocate(character(shape(2))::names(shape(1)))
    do i=1,shape(1)
      if(rank==0) text=names(i)
      call broadcast_catalog_text(text,rank,comm)
      if(rank/=0) names(i)=text
    enddo
  end subroutine

  subroutine broadcast_catalog_product(entry,rank,comm)
    type(catalog_product),intent(inout) :: entry
    integer,intent(in) :: rank,comm
    integer :: header(4),err
    integer(int64) :: words(10)
    real(real64) :: periods(2)
    if(rank==0) then
      header=[merge(1,0,entry%valid),entry%line,entry%axis,size(entry%fields)]
      words=[entry%index,reshape(entry%ranges,[6]),entry%enable_request,entry%interval_steps,entry%dense_interval_steps]
      periods=[entry%interval_time,entry%dense_interval_time]
    endif
    call MPI_Bcast(header,4,MPI_INTEGER,0,comm,err)
    call require_catalog_mpi(err,comm)
    call MPI_Bcast(words,10,MPI_INTEGER8,0,comm,err)
    call require_catalog_mpi(err,comm)
    call MPI_Bcast(periods,2,MPI_DOUBLE_PRECISION,0,comm,err)
    call require_catalog_mpi(err,comm)
    if(rank/=0) then
      entry%valid=header(1)==1;entry%line=header(2);entry%axis=header(3)
      allocate(entry%fields(header(4)))
      entry%index=words(1);entry%ranges=reshape(words(2:7),[2,3]);entry%enable_request=words(8)
      entry%interval_steps=words(9);entry%dense_interval_steps=words(10)
      entry%interval_time=periods(1);entry%dense_interval_time=periods(2)
    endif
    call MPI_Bcast(entry%fields,header(4),MPI_INTEGER,0,comm,err)
    call require_catalog_mpi(err,comm)
    call broadcast_catalog_text(entry%id,rank,comm)
    call broadcast_catalog_text(entry%source,rank,comm)
    call broadcast_catalog_text(entry%statistics_source,rank,comm)
    call broadcast_catalog_text(entry%record,rank,comm)
    call broadcast_catalog_text(entry%error,rank,comm)
    call broadcast_catalog_names(entry%event_ids,rank,comm)
    call broadcast_catalog_names(entry%window_ids,rank,comm)
  end subroutine

  subroutine catalog_watch_counts(reads,checks)
    integer(int64),intent(out) :: reads,checks
    reads=watch_reads;checks=watch_checks
  end subroutine

  function text_value(value) result(text)
    type(c_ptr),intent(in) :: value
    character(:),allocatable :: text
    character(c_char),pointer :: bytes(:)
    type(c_ptr) :: address
    integer(c_size_t) :: n
    integer :: i
    text=''
    if(json_type(value)/=3) return
    address=json_string(value,n)
    if(n==0) return
    call c_f_pointer(address,bytes,[int(n)])
    text=repeat(' ',int(n))
    do i=1,int(n)
      text(i:i)=bytes(i)
    enddo
  end function

  logical function keys_allowed(obj,names,error) result(ok)
    type(c_ptr),intent(in) :: obj
    character(*),intent(in) :: names(:)
    character(:),allocatable,intent(out) :: error
    character(:),allocatable :: key
    integer(c_size_t) :: i
    ok=.false.; error='expected JSON object'
    if(json_type(obj)/=1) return
    do i=0,json_size(obj)-1
      key=text_value(json_key(obj,i))
      if(any(names==key)) cycle
      error='unknown parameter: '//key
      return
    enddo
    ok=.true.; error=''
  end function

  logical function read_text(obj,key,text,error) result(ok)
    type(c_ptr),intent(in) :: obj
    character(*),intent(in) :: key
    character(:),allocatable,intent(out) :: text,error
    type(c_ptr) :: value
    value=json_get(obj,key//c_null_char)
    ok=.false.; text=''; error=key//' must be a nonempty string'
    if(json_type(value)/=3) return
    text=text_value(value)
    if(len(text)==0.or.index(text,achar(0))>0) return
    ok=.true.; error=''
  end function

  logical function read_integer(obj,key,n,error,minimum) result(ok)
    type(c_ptr),intent(in) :: obj
    character(*),intent(in) :: key
    integer(int64),intent(out) :: n
    character(:),allocatable,intent(out) :: error
    integer(int64),intent(in) :: minimum
    integer(c_int64_t) :: parsed
    ok=.false.; n=0; error=key//' must be a representable integer'
    if(json_integer(json_get(obj,key//c_null_char),parsed)==0) return
    if(parsed<minimum) then
      error=key//' is below its allowed minimum'
      return
    endif
    n=parsed; ok=.true.; error=''
  end function

  logical function read_positive_real(obj,key,n,error) result(ok)
    type(c_ptr),intent(in) :: obj
    character(*),intent(in) :: key
    real(real64),intent(out) :: n
    character(:),allocatable,intent(out) :: error
    real(c_double) :: parsed
    ok=.false.; n=0; error=key//' must be a finite positive number'
    if(json_real(json_get(obj,key//c_null_char),parsed)==0) return
    if(parsed<=0) return
    n=parsed; ok=.true.; error=''
  end function

  logical function read_names(obj,key,names,error) result(ok)
    type(c_ptr),intent(in) :: obj
    character(*),intent(in) :: key
    character(:),allocatable,intent(out) :: names(:),error
    type(c_ptr) :: array,item
    integer :: i,n,width
    character(:),allocatable :: name
    ok=.false.; error=key//' must be an array of nonempty strings'
    array=json_get(obj,key//c_null_char)
    if(.not.c_associated(array)) then
      allocate(character(1)::names(0))
      ok=.true.; error=''
      return
    endif
    if(json_type(array)/=2) return
    n=int(json_size(array)); width=1
    do i=1,n
      item=json_at(array,int(i-1,c_size_t))
      if(json_type(item)/=3) return
      name=text_value(item)
      if(len(name)==0.or.index(name,achar(0))>0) return
      width=max(width,len(name))
    enddo
    allocate(character(width)::names(n))
    do i=1,n
      names(i)=text_value(json_at(array,int(i-1,c_size_t)))
    enddo
    ok=.true.; error=''
  end function

  subroutine parse_catalog_product(line,entry)
    character(*),intent(in) :: line
    type(catalog_product),intent(out) :: entry
    type(c_ptr) :: doc,root,sampling,ranges,array,item,output,schedule
    character(c_char) :: errors(512)
    character(:),allocatable :: value,error
    character(1),parameter :: axes(3)=['i','j','k']
    logical :: selected(size(catalog_field_names)),good
    integer :: i,j,id
    integer(c_int64_t) :: endpoints(2)
    entry%id=''; entry%source=''; entry%statistics_source=''; entry%error=''; entry%record=line
    allocate(entry%fields(0))
    doc=json_read(line,len(line,kind=c_size_t),errors,size(errors,kind=c_size_t))
    if(.not.c_associated(doc)) then
      entry%error=error_text(errors)
      return
    endif
    root=json_root(doc)
    ! Retain an identifiable ID even when another parameter is invalid.
    entry%id=text_value(json_get(root,'id'//c_null_char))
    validation: block
      if(.not.keys_allowed(root,[character(24)::'id','kind','enable_request','source', &
          'statistics_source','fields','sampling','output','schedule'],error)) exit validation
      if(.not.read_text(root,'id',entry%id,error)) exit validation
      if(entry%id=='.'.or.entry%id=='..'.or.index(entry%id,'/')>0.or.index(entry%id,achar(92))>0) then
        error='id must be a single product name, not a path'
        exit validation
      endif
      do i=1,len(entry%id)
        if(iachar(entry%id(i:i))>=32.and.iachar(entry%id(i:i))/=127) cycle
        error='id must not contain control characters'
        exit validation
      enddo
      if(.not.read_text(root,'kind',value,error)) exit validation
      if(value/='slice') then
        error='unsupported product kind: '//value
        exit validation
      endif
      if(.not.read_integer(root,'enable_request',entry%enable_request,error,1_int64)) exit validation
      if(.not.read_text(root,'source',entry%source,error)) exit validation
      select case(entry%source)
      case('instantaneous','reynolds','favre')
      case default
        error='unsupported source: '//entry%source
        exit validation
      end select
      if(c_associated(json_get(root,'statistics_source'//c_null_char))) then
        if(.not.read_text(root,'statistics_source',entry%statistics_source,error)) exit validation
        if(entry%source=='instantaneous') then
          error='instantaneous source does not use statistics_source'
          exit validation
        endif
      endif
      array=json_get(root,'fields'//c_null_char)
      error='fields must be a nonempty array of registered names'
      if(json_type(array)/=2) exit validation
      if(json_size(array)==0) exit validation
      selected=.false.
      do i=1,int(json_size(array))
        item=json_at(array,int(i-1,c_size_t))
        if(json_type(item)/=3) exit validation
        value=text_value(item); id=catalog_field_id(value)
        if(id==0) then
          error='unknown field: '//value
          exit validation
        endif
        selected(id)=.true.
      enddo
      entry%fields=pack([(i,i=1,size(selected))],selected)
      sampling=json_get(root,'sampling'//c_null_char)
      if(.not.keys_allowed(sampling,[character(16)::'mode','axis','index','index_ranges'],error)) exit validation
      if(.not.read_text(sampling,'mode',value,error)) exit validation
      if(value/='grid_index') then
        error='data slices require sampling mode grid_index'
        exit validation
      endif
      if(.not.read_text(sampling,'axis',value,error)) exit validation
      entry%axis=0
      do i=1,3
        if(value==axes(i)) entry%axis=i
      enddo
      if(entry%axis==0) then
        error='axis must be i, j or k'
        exit validation
      endif
      if(.not.read_integer(sampling,'index',entry%index,error,0_int64)) exit validation
      ranges=json_get(sampling,'index_ranges'//c_null_char)
      if(.not.keys_allowed(ranges,axes,error)) exit validation
      error='index_ranges must contain exactly the two varying global axes'
      if(json_size(ranges)/=2) exit validation
      do i=1,3
        array=json_get(ranges,axes(i)//c_null_char)
        if(i==entry%axis) then
          if(c_associated(array)) exit validation
          entry%ranges(:,i)=entry%index
          cycle
        endif
        if(json_type(array)/=2) exit validation
        if(json_size(array)/=2) exit validation
        do j=1,2
          if(json_integer(json_at(array,int(j-1,c_size_t)),endpoints(j))==0) exit validation
        enddo
        if(endpoints(1)<0.or.endpoints(2)<endpoints(1)) exit validation
        entry%ranges(:,i)=endpoints
      enddo
      output=json_get(root,'output'//c_null_char)
      if(.not.keys_allowed(output,[character(8)::'channel','format'],error)) exit validation
      if(.not.read_text(output,'channel',value,error)) exit validation
      if(value/='data') then
        error='unsupported slice channel in numerical archive: '//value
        exit validation
      endif
      if(.not.read_text(output,'format',value,error)) exit validation
      if(value/='hdf5-xdmf') then
        error='data format must be hdf5-xdmf'
        exit validation
      endif
      schedule=json_get(root,'schedule'//c_null_char)
      if(.not.keys_allowed(schedule,[character(16)::'mode','interval','events','windows','dense_interval'],error)) &
        exit validation
      if(.not.read_text(schedule,'mode',value,error)) exit validation
      select case(value)
      case('steps')
        good=read_integer(schedule,'interval',entry%interval_steps,error,1_int64)
      case('time')
        good=read_positive_real(schedule,'interval',entry%interval_time,error)
      case default
        error='schedule mode must be steps or time'
        exit validation
      end select
      if(.not.good) exit validation
      if(.not.read_names(schedule,'events',entry%event_ids,error)) exit validation
      if(.not.read_names(schedule,'windows',entry%window_ids,error)) exit validation
      if(size(entry%event_ids)+size(entry%window_ids)>0) then
        if(entry%interval_steps>0) then
          if(.not.read_integer(schedule,'dense_interval',entry%dense_interval_steps,error,1_int64)) exit validation
          if(entry%dense_interval_steps>=entry%interval_steps) then
            error='dense_interval must be smaller than interval'
            exit validation
          endif
        else
          if(.not.read_positive_real(schedule,'dense_interval',entry%dense_interval_time,error)) exit validation
          if(entry%dense_interval_time>=entry%interval_time) then
            error='dense_interval must be smaller than interval'
            exit validation
          endif
        endif
      elseif(c_associated(json_get(schedule,'dense_interval'//c_null_char))) then
        error='dense_interval requires at least one event or window'
        exit validation
      endif
      error=''
    end block validation
    entry%error=error; entry%valid=len(error)==0
    call json_free(doc)
  end subroutine

  function error_text(bytes) result(text)
    character(c_char),intent(in) :: bytes(:)
    character(:),allocatable :: text
    integer :: i
    text=''
    do i=1,size(bytes)
      if(bytes(i)==c_null_char) exit
      text=text//bytes(i)
    enddo
  end function

  subroutine validate_catalog_grid(entry,global_extent,numq)
    type(catalog_product),intent(inout) :: entry
    integer,intent(in) :: global_extent(3),numq
    if(.not.entry%valid) return
    if(any(entry%ranges(2,:)>int(global_extent,int64))) then
      entry%error='slice indices exceed the global grid'
    elseif(numq/=5.and.numq/=11) then
      entry%error='numerical slice fields require an available five-variable or AIR5 layout'
    elseif(numq==5.and.any(entry%fields>=7.and.entry%fields<=12)) then
      entry%error='requested AIR5 field is not present in the five-variable source'
    else
      return
    endif
    entry%valid=.false.
  end subroutine

  subroutine read_product_catalog(path,products,ok,message,signature)
    character(*),intent(in) :: path
    type(catalog_product),allocatable,intent(out) :: products(:)
    logical,intent(out) :: ok
    character(:),allocatable,intent(out) :: message
    integer(int64),optional,intent(out) :: signature(8)
    type(catalog_product) :: entry
    type(catalog_product),allocatable :: expanded(:)
    type(c_ptr) :: doc,root,data
    character(c_char),pointer :: bytes(:)
    character(c_char) :: errors(512)
    character(:),allocatable :: line,value,contents
    integer(c_size_t) :: length
    integer(c_int64_t) :: identity(8)
    logical :: header
    integer :: line_number,i,j,first,last,offset
    allocate(products(0))
    ok=.false.; message=''; header=.false.; line_number=0
    data=catalog_file_read(path//c_null_char,length,identity,errors,size(errors,kind=c_size_t))
    if(present(signature)) signature=identity
    if(.not.c_associated(data)) then
      message=error_text(errors)
      return
    endif
    if(length>int(huge(i),c_size_t)) then
      message='catalog text length is not representable by this Fortran interface'
      call catalog_release(data)
      return
    endif
    contents=repeat(' ',int(length))
    if(length>0) then
      call c_f_pointer(data,bytes,[int(length)])
      do i=1,int(length)
        contents(i:i)=bytes(i)
      enddo
    endif
    call catalog_release(data)
    first=1
    do while(first<=len(contents))
      offset=index(contents(first:),achar(10))
      last=len(contents)
      if(offset>0) last=first+offset-2
      line=contents(first:last)
      first=last+2
      line_number=line_number+1
      if(verify(line,' '//achar(9)//achar(13))==0) cycle
      if(.not.header) then
        doc=json_read(line,len(line,kind=c_size_t),errors,size(errors,kind=c_size_t))
        if(.not.c_associated(doc)) then
          message='invalid catalog header: '//error_text(errors)
          exit
        endif
        root=json_root(doc)
        header=keys_allowed(root,[character(16)::'catalog_schema'],message)
        if(header) header=read_text(root,'catalog_schema',value,message)
        if(header) then
          header=value=='astr-insitu-products-v1'
          if(.not.header) message='unsupported catalog_schema'
        endif
        call json_free(doc)
        if(.not.header) exit
        cycle
      endif
      call parse_catalog_product(line,entry)
      entry%line=line_number
      allocate(expanded(size(products)+1))
      expanded(:size(products))=products
      expanded(size(expanded))=entry
      call move_alloc(expanded,products)
    enddo
    if(.not.header.and.len(message)==0) message='missing catalog header'
    ok=header.and.len(message)==0
    if(.not.ok) then
      deallocate(products)
      allocate(products(0))
      return
    endif
    ! A repeated ID has no implicit first/last winner; unrelated entries survive.
    do i=1,size(products)
      if(len(products(i)%id)==0) cycle
      do j=i+1,size(products)
        if(products(i)%id/=products(j)%id) cycle
        products(i)%valid=.false.; products(j)%valid=.false.
        products(i)%error='duplicate product id'; products(j)%error='duplicate product id'
      enddo
    enddo
  end subroutine
end module
