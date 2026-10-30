module insitu_product_schedule
  use astr_precision_io, only: read_source_reals
  use iso_fortran_env, only: int64,real64
  use astr_precision, only: real_kind
  use iso_c_binding, only: c_int,c_char,c_null_char,c_int64_t,c_double
  use ieee_arithmetic, only: ieee_is_finite
  use insitu_run_config, only: insitu_options,insitu_max_products,catalog_product,catalog_definition_equal,catalog_field_names, &
    pack_catalog_render,unpack_catalog_render
  use insitu_schedule, only: sample_schedule,configure_schedule,poll_schedule,write_schedule_state, &
    restore_schedule_state,resume_schedule,schedule_next_target
  use adaptive_output, only: adaptive_clock,adaptive_binding_equal,adaptive_dependencies_reset, &
    configure_adaptive_clock,poll_adaptive_clock,report_adaptive_clock,adaptive_clock_file,adaptive_shared_config, &
    adaptive_binding,adaptive_level,validate_adaptive_binding
  implicit none
  private
  public :: configure_product_schedules,poll_product_schedules,product_scene_due,product_due
  public :: product_results_complete,product_state_file,product_schedules_active
  public :: product_schedule_equal,product_failures_present
  public :: runtime_product_state,runtime_catalog_apply,runtime_catalog_count,runtime_catalog_state
  public :: runtime_catalog_due,runtime_catalog_success,runtime_catalog_disable,runtime_catalog_wait
  public :: runtime_catalog_state_file
  public :: runtime_catalog_reject
  public :: runtime_catalog_image_work
  integer,parameter,public :: runtime_active=1,runtime_bad_config=2,runtime_waiting=3, &
    runtime_resource_disabled=4,runtime_absent=5
  type :: runtime_product_state
    type(catalog_product) :: definition
    type(adaptive_binding) :: binding
    integer(int64) :: version=0,consumed_request=0,last_step=-1,attempt_step=-1,applied_step=0
    real(real64) :: last_time=0,attempt_time=0,applied_time=0
    integer :: status=runtime_absent
    logical :: pending_first=.false.,dense=.false.
    character(:),allocatable :: reason
  end type
  type(runtime_product_state),allocatable,save :: runtime_states(:)
  type :: product_state
    type(sample_schedule) :: clock
    type(adaptive_clock) :: adaptive
    integer(int64) :: origin_step=0
    real(real_kind) :: origin_time=0.e0_real_kind
    integer(int64) :: attempt=-1,success=-1,missing=-1
    real(real_kind) :: attempt_time=-1.e0_real_kind,success_time=-1.e0_real_kind,missing_time=-1.e0_real_kind
    integer :: code=0,phase=0,missing_code=0,missing_phase=0
    logical :: due=.false.,reported=.true.
  end type
  type(insitu_options),save :: configuration
  type(product_state),save :: states(insitu_max_products)
  logical,save :: configured=.false.
  character(64),save :: disabled_products(128)=''
  integer,save :: disabled_count=0
contains
  logical function runtime_catalog_image_work() result(active)
    integer :: i
    active=.false.
    do i=1,runtime_catalog_count()
      if(runtime_states(i)%definition%view%channel/='image'.and.runtime_states(i)%definition%view%channel/='geometry') cycle
      if(runtime_states(i)%status/=runtime_active.and.runtime_states(i)%status/=runtime_waiting) cycle
      active=.true.;return
    enddo
  end function

  integer(c_int) function runtime_count_c() bind(C,name='astr_insitu_runtime_product_count') result(n)
    n=runtime_catalog_count()
  end function

  integer(c_int) function runtime_identity_c(index,id,capacity,metadata,clock) &
      bind(C,name='astr_insitu_runtime_product_identity') result(length)
    integer(c_int),value :: index,capacity
    character(c_char),intent(out) :: id(*)
    integer(c_int64_t),intent(out) :: metadata(5)
    real(c_double),intent(out) :: clock
    integer :: i,n
    length=0;metadata=0;clock=0
    if(index<1.or.index>runtime_catalog_count()) return
    n=len(runtime_states(index)%definition%id);length=n+1
    metadata=[runtime_states(index)%version,int(runtime_states(index)%status,c_int64_t), &
      int(merge(1,0,runtime_states(index)%definition%view%channel=='image'.or. &
        runtime_states(index)%definition%view%channel=='geometry'),c_int64_t), &
      runtime_states(index)%consumed_request,runtime_states(index)%applied_step]
    clock=runtime_states(index)%applied_time
    if(capacity<n+1) return
    do i=1,n
      id(i)=runtime_states(index)%definition%id(i:i)
    enddo
    id(n+1)=c_null_char
  end function

  integer(c_int) function runtime_fields_c(index,ids,names,capacity) &
      bind(C,name='astr_insitu_runtime_product_fields') result(n)
    integer(c_int),value :: index,capacity
    integer(c_int),intent(out) :: ids(*)
    character(c_char),intent(out) :: names(*)
    integer :: field,j,id
    n=0
    if(index<1.or.index>runtime_catalog_count()) return
    n=size(runtime_states(index)%definition%fields)
    if(capacity<n) return
    do field=1,n
      id=runtime_states(index)%definition%fields(field);ids(field)=id
      do j=1,len(catalog_field_names)
        names((field-1)*len(catalog_field_names)+j)=catalog_field_names(id)(j:j)
      enddo
    enddo
  end function

  subroutine runtime_catalog_state_file(unit,writing,step,time,ok)
    integer,intent(in) :: unit
    logical,intent(in) :: writing
    integer(int64),intent(in) :: step
    real(real64),intent(in) :: time
    logical,intent(out) :: ok
    type(runtime_product_state),allocatable :: restored(:)
    character(8) :: magic
    integer(int64) :: n,bytes,position
    integer :: i,status
    ok=.false.;status=0;magic='ASTRRC04'
    if(writing) then
      n=runtime_catalog_count()
      write(unit,iostat=status) magic,n
    else
      read(unit,iostat=status) magic,n
      if(status/=0) return
      if(magic/='ASTRRC04'.or.n<0.or.n>huge(i)) return
      inquire(unit=unit,size=bytes,pos=position,iostat=status)
      if(status/=0) return
      ! Bound allocations by the actual remaining control bytes, not an arbitrary product cap.
      if(n>(bytes-position+1)/92_int64) return
      allocate(restored(int(n)))
    endif
    if(status/=0) return
    do i=1,int(n)
      if(writing) then
        call runtime_record_file(unit,.true.,runtime_states(i),status)
      else
        call runtime_record_file(unit,.false.,restored(i),status)
      endif
      if(status/=0) return
      if(.not.writing) then
        if(.not.runtime_record_valid(restored(i),step,time)) return
      endif
    enddo
    if(.not.writing) call move_alloc(restored,runtime_states)
    ok=.true.
  end subroutine

  subroutine runtime_record_file(unit,writing,state,status)
    integer,intent(in) :: unit
    logical,intent(in) :: writing
    type(runtime_product_state),intent(inout) :: state
    integer,intent(out) :: status
    integer :: flags(3)
    integer(int64) :: words(5)
    real(real64) :: times(3)
    status=0
    call runtime_text_file(unit,writing,state%definition%id,status)
    if(status/=0) return
    if(writing) then
      flags=[state%status,merge(1,0,state%pending_first),merge(1,0,state%dense)]
      words=[state%version,state%consumed_request,state%last_step,state%attempt_step,state%applied_step]
      times=[state%last_time,state%attempt_time,state%applied_time]
      write(unit,iostat=status) flags,words,times
    else
      read(unit,iostat=status) flags,words,times
      if(status/=0) return
      if(flags(1)<runtime_active.or.flags(1)>runtime_absent.or.any(flags(2:3)<0).or.any(flags(2:3)>1)) then
        status=1;return
      endif
      state%status=flags(1);state%pending_first=flags(2)==1;state%dense=flags(3)==1
      state%version=words(1);state%consumed_request=words(2);state%last_step=words(3)
      state%attempt_step=words(4);state%applied_step=words(5)
      state%last_time=times(1);state%attempt_time=times(2);state%applied_time=times(3)
    endif
    if(status/=0) return
    call runtime_text_file(unit,writing,state%reason,status)
    if(status/=0) return
    if(state%version>0) call runtime_definition_file(unit,writing,state%definition,status)
  end subroutine

  subroutine runtime_definition_file(unit,writing,entry,status)
    integer,intent(in) :: unit
    logical,intent(in) :: writing
    type(catalog_product),intent(inout) :: entry
    integer,intent(out) :: status
    integer :: header(3)
    character(32) :: labels(11)
    integer :: view_words(7)
    real(real64) :: view_reals(31)
    integer(int64) :: words(10)
    real(real64) :: periods(2)
    call runtime_text_file(unit,writing,entry%source,status)
    if(status/=0) return
    call runtime_text_file(unit,writing,entry%statistics_source,status)
    if(status/=0) return
    call runtime_text_file(unit,writing,entry%record,status)
    if(status/=0) return
    if(writing) then
      header=[entry%axis,entry%line,size(entry%fields)]
      words=[entry%index,reshape(entry%ranges,[6]),entry%enable_request,entry%interval_steps,entry%dense_interval_steps]
      periods=[entry%interval_time,entry%dense_interval_time]
      write(unit,iostat=status) header,words,periods
    else
      read(unit,iostat=status) header,words,periods
      if(status/=0) return
      if(header(1)<0.or.header(1)>3.or.header(3)<0.or.header(3)>size(catalog_field_names)) then
        status=1;return
      endif
      entry%axis=header(1);entry%line=header(2);allocate(entry%fields(header(3)))
      entry%index=words(1);entry%ranges=reshape(words(2:7),[2,3]);entry%enable_request=words(8)
      entry%interval_steps=words(9);entry%dense_interval_steps=words(10)
      entry%interval_time=periods(1);entry%dense_interval_time=periods(2)
      entry%valid=.true.;entry%error=''
    endif
    if(status/=0) return
    if(writing) then
      write(unit,iostat=status) entry%fields
    else
      read(unit,iostat=status) entry%fields
    endif
    if(status/=0) return
    if(writing) then
      call pack_catalog_render(entry%view,labels,view_words,view_reals)
      write(unit,iostat=status) labels,view_words,view_reals
    else
      read(unit,iostat=status) labels,view_words,view_reals
      if(status/=0) return
      if(view_words(6)<0.or.view_words(6)>1.or.view_words(7)<0.or..not.all(ieee_is_finite(view_reals))) then
        status=1;return
      endif
      call unpack_catalog_render(entry%view,labels,view_words,view_reals)
    endif
    if(status/=0) return
    call runtime_names_file(unit,writing,entry%event_ids,status)
    if(status/=0) return
    call runtime_names_file(unit,writing,entry%window_ids,status)
  end subroutine

  subroutine runtime_text_file(unit,writing,text,status)
    integer,intent(in) :: unit
    logical,intent(in) :: writing
    character(:),allocatable,intent(inout) :: text
    integer,intent(out) :: status
    integer(int64) :: n,bytes,position
    if(writing) then
      n=0
      if(allocated(text)) n=len(text,kind=int64)
      write(unit,iostat=status) n
    else
      read(unit,iostat=status) n
      if(status/=0) return
      inquire(unit=unit,size=bytes,pos=position,iostat=status)
      if(status/=0) return
      if(n<0.or.n>bytes-position+1.or.n>huge(1)) then
        status=1;return
      endif
      text=repeat(' ',int(n))
    endif
    if(status/=0.or.n==0) return
    if(writing) then
      write(unit,iostat=status) text
    else
      read(unit,iostat=status) text
    endif
  end subroutine

  subroutine runtime_names_file(unit,writing,names,status)
    integer,intent(in) :: unit
    logical,intent(in) :: writing
    character(:),allocatable,intent(inout) :: names(:)
    integer,intent(out) :: status
    integer(int64) :: dims(2),bytes,position
    dims=[0_int64,1_int64]
    if(writing) then
      if(allocated(names)) dims=[size(names,kind=int64),len(names,kind=int64)]
      write(unit,iostat=status) dims
    else
      read(unit,iostat=status) dims
      if(status/=0) return
      inquire(unit=unit,size=bytes,pos=position,iostat=status)
      if(status/=0) return
      if(any(dims<0).or.any(dims>huge(1))) then
        status=1;return
      endif
      if(dims(1)>0) then
        if(dims(2)==0) then
          status=1;return
        endif
        if(dims(1)>(bytes-position+1)/dims(2)) then
          status=1;return
        endif
      endif
      allocate(character(int(dims(2)))::names(int(dims(1))))
    endif
    if(status/=0.or.dims(1)==0) return
    if(writing) then
      write(unit,iostat=status) names
    else
      read(unit,iostat=status) names
    endif
  end subroutine

  logical function runtime_record_valid(state,step,time) result(ok)
    type(runtime_product_state),intent(inout) :: state
    integer(int64),intent(in) :: step
    real(real64),intent(in) :: time
    integer :: i
    ok=.false.
    if(len(state%definition%id)==0.or.state%version<0.or.state%consumed_request<0) return
    if(min(state%last_step,state%attempt_step)<-1.or.state%applied_step<0) return
    if(max(state%last_step,state%attempt_step,state%applied_step)>step) return
    if(.not.all(ieee_is_finite([state%last_time,state%attempt_time,state%applied_time]))) return
    if(min(state%last_time,state%attempt_time,state%applied_time)<0) return
    if(max(state%last_time,state%attempt_time,state%applied_time)>time) return
    if(state%version==0) then
      ok=state%status==runtime_bad_config.or.state%status==runtime_absent
      return
    endif
    if(state%definition%enable_request<1.or.state%definition%enable_request>state%consumed_request) return
    if(state%definition%index<0.or.any(state%definition%ranges<0)) return
    if(any(state%definition%ranges(1,:)>state%definition%ranges(2,:))) return
    select case(trim(state%definition%view%sampling))
    case('grid_index')
      if(state%definition%axis<1.or.state%definition%axis>3) return
      if(any(state%definition%ranges(:,state%definition%axis)/=state%definition%index)) return
    case('grid_node')
      if(any(state%definition%ranges(1,:)/=state%definition%ranges(2,:))) return
    case('physical_plane','volume')
      if(state%definition%axis/=0) return
    case('wall','wall_midspan')
      if(state%definition%axis<1.or.state%definition%axis>3) return
    case default
      return
    end select
    if(size(state%definition%fields)==0.and.state%definition%view%kind/='wall_curve') return
    if(any(state%definition%fields<1).or.any(state%definition%fields>size(catalog_field_names))) return
    do i=2,size(state%definition%fields)
      if(state%definition%fields(i)<=state%definition%fields(i-1)) return
    enddo
    if(state%definition%interval_steps<0.or.state%definition%interval_time<0) return
    if(.not.ieee_is_finite(state%definition%interval_time)) return
    if((state%definition%interval_steps>0).eqv.(state%definition%interval_time>0)) return
    call runtime_binding(state%definition,state%binding,ok)
  end function

  subroutine runtime_catalog_apply(entries,step,time)
    type(catalog_product),intent(in) :: entries(:)
    integer(int64),intent(in) :: step
    real(real64),intent(in) :: time
    type(runtime_product_state),allocatable :: expanded(:)
    type(catalog_product) :: entry
    type(adaptive_binding) :: binding
    logical,allocatable :: seen(:)
    logical :: fresh,changed,new_request,valid
    integer :: i,j,index,old_status
    if(.not.allocated(runtime_states)) allocate(runtime_states(0))
    allocate(seen(size(runtime_states)+size(entries)))
    seen=.false.
    do i=1,size(entries)
      entry=entries(i)
      if(len(entry%id)==0) cycle
      index=0
      do j=1,size(runtime_states)
        if(runtime_states(j)%definition%id==entry%id) then
          index=j
          exit
        endif
      enddo
      if(index==0) then
        index=size(runtime_states)+1
        allocate(expanded(index))
        expanded(:index-1)=runtime_states
        expanded(index)%definition=entry
        expanded(index)%reason=''
        call move_alloc(expanded,runtime_states)
      endif
      seen(index)=.true.
      if(entry%valid) then
        call runtime_binding(entry,binding,valid)
        if(.not.valid) then
          entry%valid=.false.
          entry%error='adaptive references or interval are not available in the shared configuration'
        endif
      endif
      if(.not.entry%valid) then
        runtime_states(index)%status=runtime_bad_config
        runtime_states(index)%reason=entry%error
        runtime_states(index)%applied_step=step
        runtime_states(index)%applied_time=time
        cycle
      endif
      old_status=runtime_states(index)%status
      fresh=runtime_states(index)%version==0
      if(.not.fresh) fresh=.not.catalog_definition_equal(runtime_states(index)%definition,entry)
      changed=fresh
      if(.not.fresh) changed=.not.runtime_cadence_equal(runtime_states(index)%definition,entry)
      new_request=entry%enable_request>runtime_states(index)%consumed_request
      if(fresh) then
        runtime_states(index)%version=runtime_states(index)%version+1
        runtime_states(index)%last_step=-1
        runtime_states(index)%last_time=0
        runtime_states(index)%pending_first=.true.
      endif
      runtime_states(index)%definition=entry
      runtime_states(index)%binding=binding
      runtime_states(index)%consumed_request=max(runtime_states(index)%consumed_request,entry%enable_request)
      if(old_status==runtime_bad_config.or.old_status==runtime_absent.or.new_request) then
        if(old_status/=runtime_active.and.old_status/=runtime_waiting) runtime_states(index)%pending_first=.true.
        if(old_status/=runtime_waiting) runtime_states(index)%status=runtime_active
      endif
      if(fresh.and.old_status/=runtime_resource_disabled) runtime_states(index)%status=runtime_active
      if(runtime_states(index)%status==runtime_active) runtime_states(index)%reason=''
      if(changed.or.new_request.or.old_status==runtime_bad_config.or.old_status==runtime_absent) then
        runtime_states(index)%applied_step=step
        runtime_states(index)%applied_time=time
      endif
    enddo
    ! Retain inactive IDs and version counters so removing/re-adding cannot rewind them.
    do i=1,size(runtime_states)
      if(seen(i).or.runtime_states(i)%status==runtime_absent) cycle
      runtime_states(i)%status=runtime_absent
      runtime_states(i)%reason='not present in the current catalog'
      runtime_states(i)%applied_step=step
      runtime_states(i)%applied_time=time
    enddo
  end subroutine

  logical function runtime_cadence_equal(a,b) result(same)
    type(catalog_product),intent(in) :: a,b
    same=.false.
    if(a%interval_steps/=b%interval_steps.or.a%interval_time/=b%interval_time.or. &
      a%dense_interval_steps/=b%dense_interval_steps.or.a%dense_interval_time/=b%dense_interval_time) return
    if(size(a%event_ids)/=size(b%event_ids).or.size(a%window_ids)/=size(b%window_ids)) return
    same=all(a%event_ids==b%event_ids).and.all(a%window_ids==b%window_ids)
  end function

  subroutine runtime_binding(entry,binding,ok)
    type(catalog_product),intent(in) :: entry
    type(adaptive_binding),intent(out) :: binding
    logical,intent(out) :: ok
    character(5) :: mode
    integer :: i,j
    binding=adaptive_binding()
    ok=.true.
    binding%enabled=size(entry%event_ids)+size(entry%window_ids)>0
    if(.not.binding%enabled) return
    do i=1,size(entry%event_ids)
      j=adaptive_name_index(adaptive_shared_config%event_ids,entry%event_ids(i))
      if(j==0) then
        ok=.false.; return
      endif
      binding%events(j)=entry%event_ids(i)
    enddo
    do i=1,size(entry%window_ids)
      j=adaptive_name_index(adaptive_shared_config%window_ids,entry%window_ids(i))
      if(j==0) then
        ok=.false.; return
      endif
      binding%windows(j)=entry%window_ids(i)
    enddo
    binding%dense_steps=entry%dense_interval_steps
    binding%dense_time=real(entry%dense_interval_time,real_kind)
    mode='time'
    if(entry%interval_steps>0) mode='steps'
    call validate_adaptive_binding(binding,trim(mode),entry%interval_steps, &
      real(entry%interval_time,real_kind),adaptive_shared_config,ok)
  end subroutine

  integer function adaptive_name_index(names,name) result(index)
    character(*),intent(in) :: names(:),name
    integer :: i
    ! NVHPC's FINDLOC does not handle the JSON/fixed-width character pair reliably.
    index=0
    do i=1,size(names)
      if(names(i)/=name) cycle
      index=i;return
    enddo
  end function

  integer function runtime_catalog_count() result(n)
    n=0
    if(allocated(runtime_states)) n=size(runtime_states)
  end function

  subroutine runtime_catalog_state(index,state)
    integer,intent(in) :: index
    type(runtime_product_state),intent(out) :: state
    state=runtime_states(index)
  end subroutine

  logical function runtime_catalog_due(index,step,time) result(due)
    integer,intent(in) :: index
    integer(int64),intent(in) :: step
    real(real64),intent(in) :: time
    logical :: dense,ok,entry
    integer(int64) :: interval
    real(real64) :: period
    due=.false.
    if(runtime_states(index)%status/=runtime_active) return
    if(runtime_states(index)%last_step==step.or.runtime_states(index)%attempt_step==step) return
    call adaptive_level(runtime_states(index)%binding,dense,ok)
    if(.not.ok) then
      runtime_states(index)%status=runtime_bad_config
      runtime_states(index)%reason='shared adaptive dependency is unavailable'
      return
    endif
    entry=dense.and..not.runtime_states(index)%dense
    runtime_states(index)%dense=dense
    if(runtime_states(index)%pending_first) then
      due=.true.
    elseif(runtime_states(index)%definition%interval_steps>0) then
      interval=runtime_states(index)%definition%interval_steps
      if(dense) interval=runtime_states(index)%definition%dense_interval_steps
      due=step-runtime_states(index)%last_step>=interval.or.entry
    else
      period=runtime_states(index)%definition%interval_time
      if(dense) period=runtime_states(index)%definition%dense_interval_time
      due=time-runtime_states(index)%last_time>=period.or.entry
    endif
  end function

  subroutine runtime_catalog_success(index,step,time)
    integer,intent(in) :: index
    integer(int64),intent(in) :: step
    real(real64),intent(in) :: time
    runtime_states(index)%last_step=step
    runtime_states(index)%last_time=time
    runtime_states(index)%attempt_step=step
    runtime_states(index)%attempt_time=time
    runtime_states(index)%pending_first=.false.
  end subroutine

  subroutine runtime_catalog_disable(index,step,time,reason)
    integer,intent(in) :: index
    integer(int64),intent(in) :: step
    real(real64),intent(in) :: time
    character(*),intent(in) :: reason
    runtime_states(index)%status=runtime_resource_disabled
    runtime_states(index)%reason=reason
    runtime_states(index)%attempt_step=step
    runtime_states(index)%attempt_time=time
  end subroutine

  subroutine runtime_catalog_reject(index,reason)
    integer,intent(in) :: index
    character(*),intent(in) :: reason
    runtime_states(index)%status=runtime_bad_config
    runtime_states(index)%reason=reason
  end subroutine

  subroutine runtime_catalog_wait(index,waiting)
    integer,intent(in) :: index
    logical,intent(in) :: waiting
    if(waiting.and.runtime_states(index)%status==runtime_active) then
      runtime_states(index)%status=runtime_waiting
      runtime_states(index)%reason='waiting for existing statistics coverage'
    elseif(.not.waiting.and.runtime_states(index)%status==runtime_waiting) then
      runtime_states(index)%status=runtime_active
      runtime_states(index)%reason=''
    endif
  end subroutine

  logical function product_failures_present()
    product_failures_present=disabled_count>0
  end function

  logical function product_schedules_active()
    product_schedules_active=configured.and.configuration%product_count>0
  end function

  logical function m12_family_active(scene) result(active)
    character(*),intent(in) :: scene
    character(64) :: names(10)
    integer :: i,n
    active=.false.
    if(configuration%products/='m12_diagnostics') return
    if(disabled_count==0) return
    names=''
    select case(scene)
    case('q_surface')
      n=2; names(1)='q_surface.image'; names(2)='q_surface.local.image'
    case('velocity_slice')
      n=10
      names(1)='velocity_slice.image'; names(2)='temperature_slice.image'
      names(3)='spanwise_slice.image'; names(4)='spanwise_slice_weak.image'
      names(5)='velocity_slice.x0_100.image'; names(6)='velocity_slice.x100_200.image'
      names(7)='velocity_slice.x200_300.image'; names(8)='temperature_slice.x0_100.image'
      names(9)='temperature_slice.x100_200.image'; names(10)='temperature_slice.x200_300.image'
    case default
      return
    end select
    do i=1,n
      if(.not.any(disabled_products(1:disabled_count)==names(i))) active=.true.
    enddo
  end function

  logical function active_family_clock(id) result(active)
    character(*),intent(in) :: id
    integer :: suffix
    suffix=index(id,'.image')
    active=.false.
    if(suffix>1) active=m12_family_active(id(:suffix-1))
  end function

  subroutine configure_product_schedules(options,origin_step,origin_time,ok)
    type(insitu_options),intent(in) :: options
    integer(int64),intent(in) :: origin_step
    real(real_kind),intent(in) :: origin_time
    logical,intent(out) :: ok
    integer :: i
    configuration=options; states=product_state(); configured=.false.; ok=.true.
    disabled_products=''; disabled_count=0
    do i=1,options%product_count
      states(i)%origin_step=origin_step; states(i)%origin_time=origin_time
      call configure_schedule(states(i)%clock,trim(options%product_modes(i)),options%product_steps(i), &
        options%product_times(i),origin_step,origin_time,options%initial_frame,options%final_frame,ok)
      if(.not.ok) return
      if(options%product_adaptive(i)%enabled) then
        call configure_adaptive_clock(states(i)%adaptive,options%product_adaptive(i),options%product_modes(i), &
          options%product_steps(i),options%product_times(i),origin_step,origin_time, &
          options%initial_frame,options%final_frame,ok)
        if(.not.ok) return
      endif
    enddo
    configured=.true.
  end subroutine

  subroutine poll_product_schedules(step,time,final,emit,ok)
    integer(int64),intent(in) :: step
    real(real_kind),intent(in) :: time
    logical,intent(in) :: final
    logical,intent(out) :: emit,ok
    integer(int64) :: crossed
    integer :: i
    emit=.false.; ok=configured
    if(.not.ok) return
    do i=1,configuration%product_count
      if(.not.states(i)%reported) then
        ok=.false.; return
      endif
      if(any(disabled_products(1:disabled_count)==configuration%product_ids(i)).and. &
        .not.active_family_clock(configuration%product_ids(i))) then
        states(i)%due=.false.; cycle
      endif
      if(configuration%product_adaptive(i)%enabled) then
        call poll_adaptive_clock(states(i)%adaptive,configuration%product_adaptive(i),step,time,final,states(i)%due,ok)
      else
        call poll_schedule(states(i)%clock,step,time,final,states(i)%due,crossed,ok)
      endif
      if(.not.ok) return
      if(states(i)%due) then
        states(i)%attempt=step; states(i)%attempt_time=time
        states(i)%reported=.false.
        if(any(disabled_products(1:disabled_count)==configuration%product_ids(i))) then
          ! A failed main image can still supply healthy derived views on its clock.
          states(i)%reported=.true.; states(i)%code=-1; states(i)%phase=0
          if(configuration%product_adaptive(i)%enabled) then
            call report_adaptive_clock(states(i)%adaptive,-1,0,ok)
            if(.not.ok) return
          endif
        endif
        emit=.true.
      endif
    enddo
  end subroutine

  logical function product_due(id) result(due)
    character(*),intent(in) :: id
    integer :: i
    due=.false.
    if(any(disabled_products(1:disabled_count)==id)) return
    due=.not.product_schedules_active()
    if(due) return
    do i=1,configuration%product_count
      if(configuration%product_ids(i)==id) then
        due=states(i)%due; return
      endif
    enddo
  end function

  logical function product_scene_due(scene) result(due)
    character(*),intent(in) :: scene
    integer :: i
    if(m12_family_active(scene)) then
      due=.not.product_schedules_active()
      if(due) return
      do i=1,configuration%product_count
        if(configuration%product_ids(i)/=trim(scene)//'.image') cycle
        due=states(i)%due; return
      enddo
      return
    endif
    due=product_due(trim(scene)//'.image')
    if(configuration%rendering_pipeline=='compatible') due=due.or.product_due(trim(scene)//'.geometry')
  end function

  integer(c_int) function product_clock_active_c() bind(C,name='astr_insitu_product_clocks_active') result(active)
    active=merge(1_c_int,0_c_int,product_schedules_active())
  end function

  function c_string(value) result(text)
    character(c_char),intent(in) :: value(*)
    character(64) :: text
    integer :: i
    text=''
    do i=1,len(text)
      if(value(i)==c_null_char) exit
      text(i:i)=value(i)
    enddo
  end function

  integer(c_int) function product_due_c(id) bind(C,name='astr_insitu_product_due') result(due)
    character(c_char),intent(in) :: id(*)
    due=merge(1_c_int,0_c_int,product_due(trim(c_string(id))))
  end function

  integer(c_int) function product_disable_c(id) bind(C,name='astr_insitu_product_disable') result(status)
    character(c_char),intent(in) :: id(*)
    character(64) :: name
    integer :: i
    logical :: ok
    status=1; name=c_string(id)
    if(len_trim(name)==0) return
    if(any(disabled_products(1:disabled_count)==name)) then
      status=0; return
    endif
    if(disabled_count==size(disabled_products)) return
    do i=1,configuration%product_count
      if(configuration%product_ids(i)/=name) cycle
      if(.not.states(i)%reported) then
        if(configuration%product_adaptive(i)%enabled) then
          call report_adaptive_clock(states(i)%adaptive,-1,0,ok)
          if(.not.ok) return
        endif
        states(i)%reported=.true.; states(i)%code=-1; states(i)%phase=0
      endif
      if(.not.active_family_clock(name)) states(i)%due=.false.
    enddo
    disabled_count=disabled_count+1; disabled_products(disabled_count)=name
    status=0
  end function

  integer(c_int) function product_enabled_c(id) bind(C,name='astr_insitu_product_enabled') result(enabled)
    character(c_char),intent(in) :: id(*)
    enabled=merge(0_c_int,1_c_int,any(disabled_products(1:disabled_count)==c_string(id)))
  end function

  integer(c_int) function product_scene_due_c(id) bind(C,name='astr_insitu_product_scene_due') result(due)
    character(c_char),intent(in) :: id(*)
    due=merge(1_c_int,0_c_int,product_scene_due(trim(c_string(id))))
  end function

  integer(c_int) function adaptive_info_c(id,integers,times) bind(C,name='astr_insitu_product_adaptive_info') result(active)
    character(c_char),intent(in) :: id(*)
    integer(c_int64_t),intent(out) :: integers(8)
    real(c_double),intent(out) :: times(4)
    character(64) :: name
    integer :: i
    active=0; integers=0; times=0
    if(.not.product_schedules_active()) return
    name=c_string(id)
    do i=1,configuration%product_count
      if(configuration%product_ids(i)/=name.or..not.configuration%product_adaptive(i)%enabled) cycle
      active=1
      integers=[merge(1_int64,0_int64,states(i)%adaptive%dense),int(states(i)%adaptive%reason,int64), &
        states(i)%adaptive%requested_step,states(i)%adaptive%next_step,states(i)%adaptive%success, &
        states(i)%adaptive%attempt,int(states(i)%adaptive%code,int64),int(states(i)%adaptive%phase,int64)]
      times=[states(i)%adaptive%requested_time,states(i)%adaptive%next_time, &
        states(i)%adaptive%success_time,states(i)%adaptive%attempt_time]
      return
    enddo
  end function

  integer(c_int) function product_report_c(id,code,phase) bind(C,name='astr_insitu_product_report') result(status)
    character(c_char),intent(in) :: id(*)
    integer(c_int),value :: code,phase
    character(64) :: name
    integer :: i
    logical :: ok
    status=0
    if(.not.product_schedules_active()) return
    status=1; name=c_string(id)
    do i=1,configuration%product_count
      if(configuration%product_ids(i)/=name) cycle
      if(.not.states(i)%due.or.states(i)%reported.or.code< -1.or.phase<0.or.phase>4) return
      if(code>0.and.index(name,'.geometry')>0) return
      if(code>0.and.code/=5.and.code/=13.and.code/=28) return
      if((code>0).neqv.(phase>0)) return
      if(configuration%product_adaptive(i)%enabled) then
        call report_adaptive_clock(states(i)%adaptive,int(code),int(phase),ok)
        if(.not.ok) return
      endif
      states(i)%reported=.true.; states(i)%code=code; states(i)%phase=phase
      if(code==0) then
        states(i)%success=states(i)%attempt; states(i)%success_time=states(i)%attempt_time
      else if(code>0) then
        states(i)%missing=states(i)%attempt; states(i)%missing_time=states(i)%attempt_time
        states(i)%missing_code=code; states(i)%missing_phase=phase
      endif
      status=0; return
    enddo
  end function

  logical function product_results_complete() result(ok)
    ok=.true.
    if(product_schedules_active()) ok=all(states(1:configuration%product_count)%reported)
  end function

  logical function product_schedule_equal(a,b) result(same)
    type(insitu_options),intent(in) :: a,b
    integer :: i
    same=a%product_count==b%product_count.and.all(a%product_ids==b%product_ids).and. &
      all(a%product_modes==b%product_modes).and.all(a%product_steps==b%product_steps).and. &
      all(a%product_times==b%product_times)
    do i=1,insitu_max_products
      same=same.and.adaptive_binding_equal(a%product_adaptive(i),b%product_adaptive(i))
    enddo
  end function

  subroutine product_state_file(unit,writing,step,time,same,ok,override,source_bytes)
    integer,intent(in) :: unit
    logical,intent(in) :: writing
    integer(int64),intent(in) :: step
    real(real_kind),intent(in) :: time
    logical,intent(out) :: same,ok
    logical,optional,intent(in) :: override
    integer,optional,intent(in) :: source_bytes
    type(insitu_options) :: saved
    type(product_state) :: restored(insitu_max_products)
    integer(int64) :: next_step,expected_step,history(7),count_saved,flags(2)
    real(real_kind) :: next_time,expected_time,times(3)
    character(8) :: magic
    logical :: initial,final,allow,entry_same(insitu_max_products),clock_same,compatible
    integer :: i,j,status,saved_disabled_count
    character(64) :: saved_disabled(128)
    ok=.false.; same=.false.
    allow=.false.
    if(present(override)) allow=override
    if(writing) then
      if(.not.product_results_complete()) return
      flags=[merge(1_int64,0_int64,configuration%initial_frame),merge(1_int64,0_int64,configuration%final_frame)]
      magic='ASTRPF01'
      if(any(configuration%product_adaptive%enabled)) magic='ASTRPF02'
      if(disabled_count>0) then
        magic='ASTRPF03'
        if(any(configuration%product_adaptive%enabled)) magic='ASTRPF04'
      endif
      write(unit,iostat=status) magic,int(configuration%product_count,int64),flags
      if(status/=0) return
      do i=1,configuration%product_count
        call schedule_next_target(states(i)%clock,next_step,next_time,ok)
        if(.not.ok) return
        write(unit,iostat=status) configuration%product_ids(i),configuration%product_modes(i), &
          configuration%product_steps(i),configuration%product_times(i),states(i)%attempt,states(i)%success, &
          states(i)%missing,int(states(i)%code,int64),int(states(i)%phase,int64), &
          int(states(i)%missing_code,int64),int(states(i)%missing_phase,int64), &
          states(i)%attempt_time,states(i)%success_time,states(i)%missing_time,next_step,next_time
        if(status/=0) return
        write(unit,iostat=status) states(i)%origin_step,states(i)%origin_time
        if(status/=0) return
        call write_schedule_state(unit,states(i)%clock,configuration%product_ids(i),step,time,ok)
        if(.not.ok) return
      enddo
      if(magic=='ASTRPF02'.or.magic=='ASTRPF04') then
        do i=1,configuration%product_count
          write(unit,iostat=status) configuration%product_adaptive(i)%enabled
          if(status/=0) return
          if(.not.configuration%product_adaptive(i)%enabled) cycle
          call adaptive_clock_file(unit,.true.,states(i)%adaptive,configuration%product_adaptive(i), &
            step,time,clock_same,ok)
          if(.not.ok) return
        enddo
      endif
      if(disabled_count>0) then
        write(unit,iostat=status) disabled_count,disabled_products(1:disabled_count)
        if(status/=0) return
      endif
      same=.true.; ok=.true.; return
    endif
    read(unit,iostat=status) magic,count_saved,flags
    if(status/=0.or.all(magic/=['ASTRPF01','ASTRPF02','ASTRPF03','ASTRPF04']).or. &
      count_saved<0.or.count_saved>insitu_max_products) return
    if(any(flags<0).or.any(flags>1)) return
    initial=flags(1)==1; final=flags(2)==1
    saved%product_count=int(count_saved); saved%initial_frame=initial; saved%final_frame=final
    do i=1,saved%product_count
      read(unit,iostat=status) saved%product_ids(i),saved%product_modes(i),saved%product_steps(i)
      if(status==0) call read_source_reals(unit,saved%product_times(i),source_bytes,status)
      if(status==0) read(unit,iostat=status) history
      if(status==0) call read_source_reals(unit,times,source_bytes,status)
      if(status==0) read(unit,iostat=status) next_step
      if(status==0) call read_source_reals(unit,next_time,source_bytes,status)
      if(status/=0) return
      if(i>1) then
        if(saved%product_ids(i)<=saved%product_ids(i-1)) return
      endif
      if(any(history(1:3)< -1).or.any(history(1:3)>step).or.history(2)>history(1).or. &
        history(3)>history(1).or.history(4)< -1.or.history(4)>28.or.history(5)<0.or.history(5)>4.or. &
        history(6)<0.or.history(6)>28.or.history(7)<0.or.history(7)>4) return
      if(history(4)>0.and.all(history(4)/=[5_int64,13_int64,28_int64])) return
      if(history(6)>0.and.all(history(6)/=[5_int64,13_int64,28_int64])) return
      if((history(4)>0).neqv.(history(5)>0)) return
      if((history(6)>0).neqv.(history(7)>0)) return
      if((history(3)>=0).neqv.(history(6)>0)) return
      if(history(4)==0.and.history(1)/=history(2)) return
      if(history(4)>0.and.history(1)/=history(3)) return
      if(.not.all(ieee_is_finite(times)).or.any(times< -1.e0_real_kind).or.any(times>time)) return
      do status=1,3
        if((history(status)==-1).neqv.(times(status)==-1.e0_real_kind)) return
      enddo
      ! The saved schedule carries its origin and independently validates history.
      call restore_product_clock(unit,restored(i),saved,i,step,time,ok,source_bytes)
      if(.not.ok) return
      call schedule_next_target(restored(i)%clock,expected_step,expected_time,ok)
      if(.not.ok.or.next_step/=expected_step.or.next_time/=expected_time) return
      restored(i)%attempt=history(1); restored(i)%success=history(2); restored(i)%missing=history(3)
      restored(i)%code=int(history(4)); restored(i)%phase=int(history(5))
      restored(i)%missing_code=int(history(6)); restored(i)%missing_phase=int(history(7))
      restored(i)%attempt_time=times(1); restored(i)%success_time=times(2); restored(i)%missing_time=times(3)
      call resume_schedule(restored(i)%clock,ok)
      if(.not.ok) return
    enddo
    entry_same=.true.
    if(magic=='ASTRPF02'.or.magic=='ASTRPF04') then
      do i=1,saved%product_count
        read(unit,iostat=status) saved%product_adaptive(i)%enabled
        if(status/=0) return
        j=findloc(configuration%product_ids,saved%product_ids(i),dim=1)
        if(saved%product_adaptive(i)%enabled) then
          if(j>0) restored(i)%adaptive=states(j)%adaptive
          if(j==0) j=insitu_max_products
          call adaptive_clock_file(unit,.false.,restored(i)%adaptive,configuration%product_adaptive(j), &
            step,time,clock_same,ok,source_bytes)
          if(.not.ok) return
          entry_same(i)=clock_same
          ! Detailed binding identity is validated by ASTRAC01, not the legacy PF header.
          saved%product_adaptive(i)=configuration%product_adaptive(j)
        else if(j>0) then
          entry_same(i)=.not.configuration%product_adaptive(j)%enabled
        endif
      enddo
    endif
    saved_disabled=''; saved_disabled_count=0
    if(magic=='ASTRPF03'.or.magic=='ASTRPF04') then
      read(unit,iostat=status) saved_disabled_count
      if(status/=0.or.saved_disabled_count<1.or.saved_disabled_count>size(saved_disabled)) return
      read(unit,iostat=status) saved_disabled(1:saved_disabled_count)
      if(status/=0.or.any(saved_disabled(1:saved_disabled_count)=='')) return
    endif
    same=product_schedule_equal(configuration,saved).and.(initial.eqv.configuration%initial_frame).and. &
      (final.eqv.configuration%final_frame).and.all(entry_same)
    if(same) then
      do i=1,configuration%product_count
        if(adaptive_dependencies_reset(configuration%product_adaptive(i))) same=.false.
      enddo
    endif
    if(.not.same.and..not.allow) then
      ok=.true.; return
    endif
    disabled_products=''; disabled_count=0
    if(same) then
      disabled_products=saved_disabled; disabled_count=saved_disabled_count
    endif
    do j=1,configuration%product_count
      i=findloc(saved%product_ids,configuration%product_ids(j),dim=1)
      compatible=.false.
      if(i>0) compatible=entry_same(i).and.saved%product_modes(i)==configuration%product_modes(j).and. &
        saved%product_steps(i)==configuration%product_steps(j).and.saved%product_times(i)==configuration%product_times(j).and. &
        (initial.eqv.configuration%initial_frame).and.(final.eqv.configuration%final_frame).and. &
        (saved%product_adaptive(i)%enabled.eqv.configuration%product_adaptive(j)%enabled).and. &
        .not.adaptive_dependencies_reset(configuration%product_adaptive(j))
      ! Fixed-only overrides retain their pre-AP whole-registry reset contract.
      if(.not.same.and..not.adaptive_shared_config%enabled.and. &
        (magic=='ASTRPF01'.or.magic=='ASTRPF03')) compatible=.false.
      if(compatible) then
        states(j)=restored(i)
        if(.not.same.and.any(saved_disabled==configuration%product_ids(j))) then
          disabled_count=disabled_count+1
          disabled_products(disabled_count)=configuration%product_ids(j)
        endif
      else
        states(j)=product_state(); states(j)%origin_step=step; states(j)%origin_time=time
        if(adaptive_shared_config%enabled) write(*,'(3a)') 'ASTR_AP_RESET product ',trim(configuration%product_ids(j)), &
          ' reason=product_configuration_or_dependency_change'
        call configure_schedule(states(j)%clock,configuration%product_modes(j),configuration%product_steps(j), &
          configuration%product_times(j),step,time,.false.,configuration%final_frame,ok)
        if(.not.ok) return
        if(configuration%product_adaptive(j)%enabled) then
          call configure_adaptive_clock(states(j)%adaptive,configuration%product_adaptive(j),configuration%product_modes(j), &
            configuration%product_steps(j),configuration%product_times(j),step,time,.false.,configuration%final_frame,ok)
          if(.not.ok) return
        endif
      endif
    enddo
    ok=.true.
  end subroutine

  subroutine restore_product_clock(unit,state,saved,i,step,time,ok,source_bytes)
    integer,intent(in) :: unit,i
    type(product_state),intent(out) :: state
    type(insitu_options),intent(in) :: saved
    integer(int64),intent(in) :: step
    real(real_kind),intent(in) :: time
    logical,intent(out) :: ok
    integer,optional,intent(in) :: source_bytes
    integer(int64) :: origin_step
    real(real_kind) :: origin_time
    integer :: status
    ! Origin is stored alongside the normal schedule; never infer it from restart time.
    read(unit,iostat=status) origin_step
    if(status==0) call read_source_reals(unit,origin_time,source_bytes,status)
    ok=status==0
    if(.not.ok) return
    state%origin_step=origin_step; state%origin_time=origin_time
    call configure_schedule(state%clock,trim(saved%product_modes(i)),saved%product_steps(i),saved%product_times(i), &
      origin_step,origin_time,saved%initial_frame,saved%final_frame,ok)
    if(ok) call restore_schedule_state(unit,state%clock,saved%product_ids(i),step,time,ok,source_bytes)
  end subroutine
end module
