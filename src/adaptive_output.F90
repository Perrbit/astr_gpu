module adaptive_output
  use iso_fortran_env, only: int64,real64
  use ieee_arithmetic, only: ieee_is_finite
  use insitu_schedule, only: sample_schedule,configure_schedule,poll_schedule, &
    write_schedule_state,restore_schedule_state,resume_schedule
  implicit none
  private
  integer,parameter,public :: adaptive_capacity=8
  public :: adaptive_config,adaptive_binding,adaptive_state,adaptive_clock
  public :: validate_adaptive_config,validate_adaptive_binding,configure_adaptive
  public :: adaptive_monitor_due,observe_adaptive,feed_adaptive_signal,adaptive_level
  public :: configure_adaptive_clock,poll_adaptive_clock,report_adaptive_clock
  public :: adaptive_clock_file,adaptive_shared_file,adaptive_binding_equal,adaptive_dependencies_reset
  public :: adaptive_shared_config,adaptive_shared_state
  public :: adaptive_config_equal
  public :: canonical_adaptive_config,canonical_adaptive_binding
  type :: adaptive_config
    logical :: enabled=.false.
    character(32) :: indicator='tgv_kinetic_energy'
    character(16) :: monitor_mode='steps'
    integer(int64) :: monitor_steps=0
    real(real64) :: monitor_time=0.d0
    character(64) :: event_ids(adaptive_capacity)='',window_ids(adaptive_capacity)=''
    real(real64) :: s_ref(adaptive_capacity)=0,t_ref(adaptive_capacity)=0
    real(real64) :: r_on(adaptive_capacity)=0,r_off(adaptive_capacity)=0,hold_time(adaptive_capacity)=0
    character(16) :: window_modes(adaptive_capacity)=''
    integer(int64) :: window_steps(2,adaptive_capacity)=0
    real(real64) :: window_times(2,adaptive_capacity)=0
  end type
  type :: adaptive_binding
    logical :: enabled=.false.
    integer(int64) :: dense_steps=0
    real(real64) :: dense_time=0.d0
    character(64) :: events(adaptive_capacity)='',windows(adaptive_capacity)=''
  end type
  type :: adaptive_state
    type(sample_schedule) :: monitor
    logical :: has_previous=.false.,event_previous(adaptive_capacity)=.false.
    logical :: active(adaptive_capacity)=.false.
    real(real64) :: previous_s=0,previous_time=0,r(adaptive_capacity)=0,entry_time(adaptive_capacity)=0
    integer(int64) :: last_step=-1,window_phase(adaptive_capacity)=0
    real(real64) :: last_time=0
    integer(int64) :: origin_step=0
    real(real64) :: origin_time=0
  end type
  type :: adaptive_clock
    character(16) :: mode='steps'
    integer(int64) :: normal_steps=0,origin_step=0,success=-1,attempt=-1,seen=-1,next_step=-1
    real(real64) :: normal_time=0,origin_time=0,success_time=-1,attempt_time=-1,seen_time=0,next_time=-1
    integer(int64) :: dense_steps=0,requested_step=-1
    real(real64) :: dense_time=0,requested_time=-1
    logical :: dense=.false.,initial=.false.,final=.false.,pending=.false.
    integer :: reason=0,code=0,phase=0
  end type
  type(adaptive_config),save :: adaptive_shared_config
  type(adaptive_state),save :: adaptive_shared_state
  logical,save :: reset_events(adaptive_capacity)=.false.,reset_windows(adaptive_capacity)=.false.
contains
  subroutine canonical_adaptive_binding(b)
    type(adaptive_binding),intent(inout) :: b
    call canonical_ids(b%events)
    call canonical_ids(b%windows)
  end subroutine

  subroutine canonical_ids(ids)
    character(64),intent(inout) :: ids(adaptive_capacity)
    character(64) :: compact(adaptive_capacity),tmp
    integer :: i,j,n
    n=count(ids/=''); compact=''; compact(:n)=pack(ids,ids/=''); ids=compact
    do i=2,n
      j=i
      do while(j>1)
        if(ids(j)>=ids(j-1)) exit
        tmp=ids(j); ids(j)=ids(j-1); ids(j-1)=tmp; j=j-1
      enddo
    enddo
  end subroutine

  subroutine canonical_adaptive_config(c)
    type(adaptive_config),intent(inout) :: c
    type(adaptive_config) :: tmp
    integer :: i,j
    do i=2,count(c%event_ids/='')
      j=i
      do while(j>1)
        if(c%event_ids(j)>=c%event_ids(j-1)) exit
        tmp=c
        c%event_ids(j-1:j)=[tmp%event_ids(j),tmp%event_ids(j-1)]
        c%s_ref(j-1:j)=[tmp%s_ref(j),tmp%s_ref(j-1)]
        c%t_ref(j-1:j)=[tmp%t_ref(j),tmp%t_ref(j-1)]
        c%r_on(j-1:j)=[tmp%r_on(j),tmp%r_on(j-1)]
        c%r_off(j-1:j)=[tmp%r_off(j),tmp%r_off(j-1)]
        c%hold_time(j-1:j)=[tmp%hold_time(j),tmp%hold_time(j-1)]
        j=j-1
      enddo
    enddo
    do i=2,count(c%window_ids/='')
      j=i
      do while(j>1)
        if(c%window_ids(j)>=c%window_ids(j-1)) exit
        tmp=c
        c%window_ids(j-1:j)=[tmp%window_ids(j),tmp%window_ids(j-1)]
        c%window_modes(j-1:j)=[tmp%window_modes(j),tmp%window_modes(j-1)]
        c%window_steps(:,j-1)=tmp%window_steps(:,j); c%window_steps(:,j)=tmp%window_steps(:,j-1)
        c%window_times(:,j-1)=tmp%window_times(:,j); c%window_times(:,j)=tmp%window_times(:,j-1)
        j=j-1
      enddo
    enddo
  end subroutine

  logical function valid_id(id)
    character(*),intent(in) :: id
    integer :: i,c
    valid_id=.false.
    if(len_trim(id)==0.or.len_trim(id)==len(id).or.id/=adjustl(id)) return
    do i=1,len_trim(id)
      c=iachar(id(i:i))
      if(index('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-',id(i:i))==0) return
    enddo
    valid_id=.true.
  end function

  subroutine validate_adaptive_config(c,ok,message)
    type(adaptive_config),intent(in) :: c
    logical,intent(out) :: ok
    character(*),intent(out) :: message
    integer :: i,j,ne,nw
    ok=.false.; message='adaptive: invalid event/window IDs or unused entries'
    ne=count(c%event_ids/=''); nw=count(c%window_ids/='')
    if(any(c%event_ids(1:ne)=='').or.any(c%window_ids(1:nw)=='')) return
    do i=1,ne
      if(.not.valid_id(c%event_ids(i))) return
      do j=1,i-1
        if(c%event_ids(i)==c%event_ids(j)) return
      enddo
    enddo
    do i=1,nw
      if(.not.valid_id(c%window_ids(i))) return
      do j=1,i-1
        if(c%window_ids(i)==c%window_ids(j)) return
      enddo
    enddo
    if(any(c%s_ref(ne+1:)/=0).or.any(c%t_ref(ne+1:)/=0).or.any(c%r_on(ne+1:)/=0).or. &
      any(c%r_off(ne+1:)/=0).or.any(c%hold_time(ne+1:)/=0).or.any(c%window_modes(nw+1:)/='').or. &
      any(c%window_steps(:,nw+1:)/=0).or.any(c%window_times(:,nw+1:)/=0)) return
    message='adaptive: disabled configuration must not define events/windows or monitoring'
    if(.not.c%enabled) then
      if(ne+nw>0.or.c%monitor_steps/=0.or.c%monitor_time/=0) return
      ok=.true.; message=''; return
    endif
    message='adaptive: only tgv_kinetic_energy is registered; at least one event/window is required'
    if(c%indicator/='tgv_kinetic_energy'.or.ne+nw==0) return
    message='adaptive: monitoring requires exactly one finite positive steps/time interval'
    if(.not.ieee_is_finite(c%monitor_time)) return
    if(ne>0) then
      select case(c%monitor_mode)
      case('steps')
        if(c%monitor_steps<=0.or.c%monitor_time/=0) return
      case('time')
        if(c%monitor_steps/=0.or.c%monitor_time<=0) return
      case default
        return
      end select
    else
      if(c%monitor_steps/=0.or.c%monitor_time/=0) return
    endif
    message='adaptive: positive finite scales, 0 <= r_off < r_on, nonnegative hold are required'
    do i=1,ne
      if(.not.all(ieee_is_finite([c%s_ref(i),c%t_ref(i),c%r_on(i),c%r_off(i),c%hold_time(i)]))) return
      if(min(c%s_ref(i),c%t_ref(i))<=0.or.c%r_off(i)<0.or.c%r_off(i)>=c%r_on(i).or.c%hold_time(i)<0) return
      if(.not.ieee_is_finite(c%t_ref(i)/c%s_ref(i))) return
    enddo
    message='adaptive: windows require increasing nonnegative [start,end) and one steps/time mode'
    do i=1,nw
      select case(c%window_modes(i))
      case('steps')
        if(any(c%window_times(:,i)/=0).or.c%window_steps(1,i)<0.or. &
          c%window_steps(2,i)<=c%window_steps(1,i)) return
      case('time')
        if(any(c%window_steps(:,i)/=0).or..not.all(ieee_is_finite(c%window_times(:,i)))) return
        if(c%window_times(1,i)<0.or.c%window_times(2,i)<=c%window_times(1,i)) return
      case default
        return
      end select
    enddo
    ok=.true.; message=''
  end subroutine

  subroutine validate_adaptive_binding(b,mode,steps,time,c,ok)
    type(adaptive_binding),intent(in) :: b
    character(*),intent(in) :: mode
    integer(int64),intent(in) :: steps
    real(real64),intent(in) :: time
    type(adaptive_config),intent(in),optional :: c
    logical,intent(out) :: ok
    integer :: i,j
    ok=.false.
    if(.not.ieee_is_finite(b%dense_time)) return
    if(.not.b%enabled) then
      ok=b%dense_steps==0.and.b%dense_time==0.and.all(b%events=='').and.all(b%windows==''); return
    endif
    if(all(b%events=='').and.all(b%windows=='')) return
    select case(mode)
    case('steps')
      if(b%dense_steps<=0.or.b%dense_steps>=steps.or.b%dense_time/=0) return
    case('time')
      if(b%dense_steps/=0.or.b%dense_time<=0.or.b%dense_time>=time) return
    case default
      return
    end select
    do i=1,adaptive_capacity
      if(b%events(i)/='') then
        if(.not.valid_id(b%events(i))) return
        if(present(c)) then
          if(.not.c%enabled.or..not.any(c%event_ids==b%events(i))) return
        endif
        do j=1,i-1
          if(b%events(i)==b%events(j)) return
        enddo
      endif
      if(b%windows(i)/='') then
        if(.not.valid_id(b%windows(i))) return
        if(present(c)) then
          if(.not.c%enabled.or..not.any(c%window_ids==b%windows(i))) return
        endif
        do j=1,i-1
          if(b%windows(i)==b%windows(j)) return
        enddo
      endif
    enddo
    ok=.true.
  end subroutine

  subroutine configure_adaptive(c,step,time,ok)
    type(adaptive_config),intent(in) :: c
    integer(int64),intent(in) :: step
    real(real64),intent(in) :: time
    logical,intent(out) :: ok
    character(256) :: message
    call validate_adaptive_config(c,ok,message)
    if(.not.ok) return
    if(step<0.or..not.ieee_is_finite(time).or.time<0) then
      ok=.false.; return
    endif
    adaptive_shared_config=c; adaptive_shared_state=adaptive_state()
    adaptive_shared_state%origin_step=step; adaptive_shared_state%origin_time=time
    adaptive_shared_state%last_time=time
    reset_events=.false.; reset_windows=.false.
    if(c%enabled.and.any(c%event_ids/='')) call configure_schedule(adaptive_shared_state%monitor, &
      c%monitor_mode,c%monitor_steps,c%monitor_time,step,time,.true.,.false.,ok)
  end subroutine

  subroutine adaptive_monitor_due(step,time,due,ok)
    integer(int64),intent(in) :: step
    real(real64),intent(in) :: time
    logical,intent(out) :: due,ok
    integer(int64) :: crossed
    due=.false.; ok=.true.
    if(.not.adaptive_shared_config%enabled.or.all(adaptive_shared_config%event_ids=='')) return
    call poll_schedule(adaptive_shared_state%monitor,step,time,.false.,due,crossed,ok)
  end subroutine

  subroutine observe_adaptive(step,time,ok)
    integer(int64),intent(in) :: step
    real(real64),intent(in) :: time
    logical,intent(out) :: ok
    integer :: i
    logical :: inside,before
    ok=.false.
    if(.not.ieee_is_finite(time).or.step<adaptive_shared_state%last_step.or. &
      time<adaptive_shared_state%last_time) return
    do i=1,adaptive_capacity
      if(adaptive_shared_config%window_ids(i)=='') cycle
      if(adaptive_shared_config%window_modes(i)=='steps') then
        inside=step>=adaptive_shared_config%window_steps(1,i).and.step<adaptive_shared_config%window_steps(2,i)
        before=step<adaptive_shared_config%window_steps(1,i)
      else
        inside=time>=adaptive_shared_config%window_times(1,i).and.time<adaptive_shared_config%window_times(2,i)
        before=time<adaptive_shared_config%window_times(1,i)
      endif
      if(inside) then
        adaptive_shared_state%window_phase(i)=1
      else if(.not.before.and.adaptive_shared_state%window_phase(i)<2) then
        adaptive_shared_state%window_phase(i)=merge(2_int64,3_int64,adaptive_shared_state%window_phase(i)==1)
      endif
    enddo
    adaptive_shared_state%last_step=step; adaptive_shared_state%last_time=time; ok=.true.
  end subroutine

  subroutine feed_adaptive_signal(value,time,ok)
    real(real64),intent(in) :: value,time
    logical,intent(out) :: ok
    real(real64) :: difference,width,rate,scale,r,elapsed,expiry
    integer :: i
    ok=.false.
    if(.not.all(ieee_is_finite([value,time])).or.value<0.or.time/=adaptive_shared_state%last_time) return
    if(adaptive_shared_state%has_previous) then
      width=time-adaptive_shared_state%previous_time
      if(.not.ieee_is_finite(width).or.width<=0) return
      difference=value-adaptive_shared_state%previous_s
      if(.not.ieee_is_finite(difference)) return
      rate=abs(difference)/width
      if(.not.ieee_is_finite(rate)) return
      do i=1,adaptive_capacity
        if(adaptive_shared_config%event_ids(i)=='') cycle
        if(.not.adaptive_shared_state%event_previous(i)) cycle
        scale=adaptive_shared_config%t_ref(i)/adaptive_shared_config%s_ref(i)
        r=scale*rate
        if(.not.ieee_is_finite(r)) return
        adaptive_shared_state%r(i)=r
        if(adaptive_shared_state%active(i)) then
          elapsed=time-adaptive_shared_state%entry_time(i)
          if(.not.ieee_is_finite(elapsed).or.elapsed<0) return
          if(elapsed>=adaptive_shared_config%hold_time(i).and.r<=adaptive_shared_config%r_off(i)) &
            adaptive_shared_state%active(i)=.false.
        else if(r>=adaptive_shared_config%r_on(i)) then
          expiry=time+adaptive_shared_config%hold_time(i)
          if(.not.ieee_is_finite(expiry)) return
          adaptive_shared_state%active(i)=.true.; adaptive_shared_state%entry_time(i)=time
        endif
      enddo
    endif
    adaptive_shared_state%previous_s=value; adaptive_shared_state%previous_time=time
    adaptive_shared_state%has_previous=.true.
    adaptive_shared_state%event_previous=adaptive_shared_config%event_ids/=''
    ok=.true.
  end subroutine

  subroutine adaptive_level(b,dense,ok)
    type(adaptive_binding),intent(in) :: b
    logical,intent(out) :: dense,ok
    integer :: i,j
    dense=.false.; ok=.true.
    if(.not.b%enabled) return
    do i=1,adaptive_capacity
      if(b%events(i)/='') then
        j=findloc(adaptive_shared_config%event_ids,b%events(i),dim=1)
        if(j==0) then
          ok=.false.; return
        endif
        dense=dense.or.adaptive_shared_state%active(j)
      endif
      if(b%windows(i)/='') then
        j=findloc(adaptive_shared_config%window_ids,b%windows(i),dim=1)
        if(j==0) then
          ok=.false.; return
        endif
        dense=dense.or.adaptive_shared_state%window_phase(j)==1
      endif
    enddo
  end subroutine

  subroutine configure_adaptive_clock(c,b,mode,steps,time,origin_step,origin_time,initial,final,ok)
    type(adaptive_clock),intent(out) :: c
    type(adaptive_binding),intent(in) :: b
    character(*),intent(in) :: mode
    integer(int64),intent(in) :: steps,origin_step
    real(real64),intent(in) :: time,origin_time
    logical,intent(in) :: initial,final
    logical,intent(out) :: ok
    call validate_adaptive_binding(b,mode,steps,time,adaptive_shared_config,ok)
    if(.not.ok.or..not.b%enabled) return
    c=adaptive_clock(); c%mode=mode; c%normal_steps=steps; c%normal_time=time
    c%dense_steps=b%dense_steps; c%dense_time=b%dense_time
    c%origin_step=origin_step; c%origin_time=origin_time
    c%initial=initial; c%final=final
    ok=origin_step>=0.and.ieee_is_finite(origin_time).and.origin_time>=0
    if(ok) call next_adaptive_target(c,ok)
  end subroutine

  subroutine poll_adaptive_clock(c,b,step,time,final,emit,ok)
    type(adaptive_clock),intent(inout) :: c
    type(adaptive_binding),intent(in) :: b
    integer(int64),intent(in) :: step
    real(real64),intent(in) :: time
    logical,intent(in) :: final
    logical,intent(out) :: emit,ok
    logical :: dense,entry,due
    emit=.false.; ok=.false.
    if(c%pending.or..not.ieee_is_finite(time).or.step<c%seen.or.time<c%seen_time.or.step<c%origin_step) return
    call adaptive_level(b,dense,ok)
    if(.not.ok) return
    entry=dense.and..not.c%dense
    c%dense=dense; c%seen=step; c%seen_time=time
    call next_adaptive_target(c,ok)
    if(.not.ok) return
    due=step>=c%next_step
    if(c%mode=='time') due=time>=c%next_time
    if(c%attempt>c%success.and.c%code>0) entry=entry.and.due
    emit=(entry.or.due.or.(c%initial.and.step==c%origin_step).or.(final.and.c%final)).and.step/=c%attempt
    c%reason=0
    if(due) c%reason=1
    if(entry) c%reason=2
    if(c%initial.and.step==c%origin_step) c%reason=3
    if(final.and.c%final) c%reason=4
    if(emit) then
      c%requested_step=c%next_step; c%requested_time=c%next_time
      if(c%reason>=2) then
        c%requested_step=step; c%requested_time=time
      endif
    endif
    c%pending=emit
  end subroutine

  subroutine next_adaptive_target(c,ok)
    type(adaptive_clock),intent(inout) :: c
    logical,intent(out) :: ok
    integer(int64) :: anchor,interval
    real(real64) :: anchor_time,period
    anchor=c%origin_step; anchor_time=c%origin_time
    if(c%success>=0) then
      anchor=c%success; anchor_time=c%success_time
    endif
    if(c%attempt>c%success) then
      anchor=c%attempt; anchor_time=c%attempt_time
    endif
    if(c%mode=='steps') then
      interval=merge(c%dense_steps,c%normal_steps,c%dense)
      ok=interval>0.and.anchor<=huge(anchor)-interval
      if(.not.ok) return
      c%next_step=anchor+interval; c%next_time=-1
    else
      period=merge(c%dense_time,c%normal_time,c%dense)
      c%next_time=anchor_time+period; c%next_step=-1
      ok=period>0.and.ieee_is_finite(c%next_time).and.c%next_time>anchor_time
    endif
  end subroutine

  subroutine report_adaptive_clock(c,code,phase,ok)
    type(adaptive_clock),intent(inout) :: c
    integer,intent(in) :: code,phase
    logical,intent(out) :: ok
    ok=c%pending.and.(code==0.or.code==-1.or.code==5.or.code==13.or.code==28).and.phase>=0.and.phase<=4
    if(.not.ok) return
    ok=(code>0).eqv.(phase>0)
    if(.not.ok) return
    c%attempt=c%seen; c%attempt_time=c%seen_time; c%code=code; c%phase=phase; c%pending=.false.
    if(code==0) then
      c%success=c%seen; c%success_time=c%seen_time
    endif
    call next_adaptive_target(c,ok)
  end subroutine

  logical function adaptive_binding_equal(a,b) result(same)
    type(adaptive_binding),intent(in) :: a,b
    same=(a%enabled.eqv.b%enabled).and.a%dense_steps==b%dense_steps.and.a%dense_time==b%dense_time.and. &
      all(a%events==b%events).and.all(a%windows==b%windows)
  end function

  logical function adaptive_config_equal(a,b) result(same)
    type(adaptive_config),intent(in) :: a,b
    same=(a%enabled.eqv.b%enabled).and.a%indicator==b%indicator.and.a%monitor_mode==b%monitor_mode.and. &
      a%monitor_steps==b%monitor_steps.and.a%monitor_time==b%monitor_time.and.all(a%event_ids==b%event_ids).and. &
      all(a%window_ids==b%window_ids).and.all(a%s_ref==b%s_ref).and.all(a%t_ref==b%t_ref).and. &
      all(a%r_on==b%r_on).and.all(a%r_off==b%r_off).and.all(a%hold_time==b%hold_time).and. &
      all(a%window_modes==b%window_modes).and.all(a%window_steps==b%window_steps).and.all(a%window_times==b%window_times)
  end function

  logical function adaptive_dependencies_reset(b) result(reset)
    type(adaptive_binding),intent(in) :: b
    integer :: i,j
    reset=.false.
    do i=1,adaptive_capacity
      if(b%events(i)/='') then
        j=findloc(adaptive_shared_config%event_ids,b%events(i),dim=1)
        if(j>0) reset=reset.or.reset_events(j)
      endif
      if(b%windows(i)/='') then
        j=findloc(adaptive_shared_config%window_ids,b%windows(i),dim=1)
        if(j>0) reset=reset.or.reset_windows(j)
      endif
    enddo
  end function

  subroutine adaptive_clock_file(unit,writing,c,b,step,time,same,ok)
    integer,intent(in) :: unit
    logical,intent(in) :: writing
    type(adaptive_clock),intent(inout) :: c
    type(adaptive_binding),intent(in) :: b
    integer(int64),intent(in) :: step
    real(real64),intent(in) :: time
    logical,intent(out) :: same,ok
    type(adaptive_clock) :: saved,checked
    type(adaptive_binding) :: binding
    character(8) :: magic
    integer :: err
    logical :: valid
    if(writing) then
      ok=.not.c%pending; same=.true.
      if(.not.ok) return
      write(unit,iostat=err) 'ASTRAC01',b%enabled,b%dense_steps,b%dense_time,b%events,b%windows, &
        c%mode,c%normal_steps,c%normal_time,c%origin_step,c%origin_time,c%success,c%success_time, &
        c%attempt,c%attempt_time,c%seen,c%seen_time,c%next_step,c%next_time,c%dense,c%initial,c%final, &
        c%reason,c%code,c%phase,c%requested_step,c%requested_time
      ok=err==0; return
    endif
    read(unit,iostat=err) magic,binding%enabled,binding%dense_steps,binding%dense_time,binding%events,binding%windows, &
      saved%mode,saved%normal_steps,saved%normal_time,saved%origin_step,saved%origin_time,saved%success,saved%success_time, &
      saved%attempt,saved%attempt_time,saved%seen,saved%seen_time,saved%next_step,saved%next_time, &
      saved%dense,saved%initial,saved%final,saved%reason,saved%code,saved%phase,saved%requested_step,saved%requested_time
    same=.false.; ok=.false.
    if(err/=0.or.magic/='ASTRAC01') return
    if(.not.all(ieee_is_finite([saved%normal_time,saved%origin_time,saved%success_time,saved%attempt_time, &
      saved%seen_time,saved%next_time,binding%dense_time,saved%requested_time]))) return
    if(saved%origin_step<0.or.saved%origin_step>step.or.saved%origin_time<0.or.saved%origin_time>time.or. &
      saved%success< -1.or.saved%success>saved%attempt.or.saved%attempt>step.or.saved%seen>step.or. &
      saved%seen_time>time.or.saved%success_time>saved%attempt_time.or.saved%attempt_time>time) return
    saved%dense_steps=binding%dense_steps; saved%dense_time=binding%dense_time
    call validate_adaptive_binding(binding,saved%mode,saved%normal_steps,saved%normal_time,ok=valid)
    if(.not.valid.or.saved%attempt< -1.or.saved%seen< -1.or.saved%origin_time<0.or.saved%success_time< -1.or. &
      saved%attempt_time< -1.or.saved%seen_time<0.or.saved%reason<0.or.saved%reason>4.or. &
      saved%phase<0.or.saved%phase>4.or.all(saved%code/=[-1,0,5,13,28])) return
    if((saved%success<0).neqv.(saved%success_time==-1)) return
    if((saved%attempt<0).neqv.(saved%attempt_time==-1)) return
    if((saved%code>0).neqv.(saved%phase>0)) return
    checked=saved
    call next_adaptive_target(checked,valid)
    if(.not.valid.or.saved%next_step/=checked%next_step.or.saved%next_time/=checked%next_time) return
    same=adaptive_binding_equal(b,binding).and.c%mode==saved%mode.and.c%normal_steps==saved%normal_steps.and. &
      c%normal_time==saved%normal_time.and.(c%final.eqv.saved%final).and. &
      .not.adaptive_dependencies_reset(b)
    if(same) c=saved
    ok=.true.
  end subroutine

  subroutine config_file(unit,writing,c,err)
    integer,intent(in) :: unit
    logical,intent(in) :: writing
    type(adaptive_config),intent(inout) :: c
    integer,intent(out) :: err
    if(writing) then
      write(unit,iostat=err) c%enabled,c%indicator,c%monitor_mode,c%monitor_steps,c%monitor_time, &
        c%event_ids,c%s_ref,c%t_ref,c%r_on,c%r_off,c%hold_time,c%window_ids,c%window_modes,c%window_steps,c%window_times
    else
      read(unit,iostat=err) c%enabled,c%indicator,c%monitor_mode,c%monitor_steps,c%monitor_time, &
        c%event_ids,c%s_ref,c%t_ref,c%r_on,c%r_off,c%hold_time,c%window_ids,c%window_modes,c%window_steps,c%window_times
    endif
  end subroutine

  subroutine adaptive_shared_file(unit,writing,step,time,override,ok)
    integer,intent(in) :: unit
    logical,intent(in) :: writing,override
    integer(int64),intent(in) :: step
    real(real64),intent(in) :: time
    logical,intent(out) :: ok
    type(adaptive_config) :: old
    type(adaptive_state) :: saved,restored
    character(8) :: magic
    character(256) :: message
    integer :: err,i,j
    logical :: same_monitor,same_event,same_window,valid,same_scale
    ok=.false.
    if(writing) then
      write(unit,iostat=err) 'ASTRAP01'
      if(err/=0) return
      call config_file(unit,.true.,adaptive_shared_config,err)
      if(err/=0) return
      saved=adaptive_shared_state
      write(unit,iostat=err) saved%has_previous,saved%event_previous,saved%active,saved%previous_s,saved%previous_time, &
        saved%r,saved%entry_time,saved%last_step,saved%last_time,saved%window_phase,saved%origin_step,saved%origin_time
      if(err/=0) return
      if(adaptive_shared_config%enabled.and.any(adaptive_shared_config%event_ids/='')) then
        call write_schedule_state(unit,saved%monitor,'adaptive-monitor',step,time,ok)
      else
        ok=.true.
      endif
      return
    endif
    read(unit,iostat=err) magic
    if(err/=0.or.magic/='ASTRAP01') return
    call config_file(unit,.false.,old,err)
    if(err/=0) return
    call validate_adaptive_config(old,valid,message)
    if(.not.valid) return
    read(unit,iostat=err) saved%has_previous,saved%event_previous,saved%active,saved%previous_s,saved%previous_time, &
      saved%r,saved%entry_time,saved%last_step,saved%last_time,saved%window_phase,saved%origin_step,saved%origin_time
    if(err/=0) return
    if(.not.all(ieee_is_finite([saved%previous_s,saved%previous_time,saved%r,saved%entry_time, &
      saved%last_time,saved%origin_time]))) return
    if(saved%origin_step<0.or.saved%origin_step>step.or.saved%origin_time<0.or.saved%origin_time>time.or. &
      saved%last_step< -1.or.saved%last_step>step.or.saved%last_time<saved%origin_time.or.saved%last_time>time.or. &
      saved%previous_time<0.or.saved%previous_time>saved%last_time.or.saved%previous_s<0.or. &
      any(saved%r<0).or.any(saved%window_phase<0).or.any(saved%window_phase>3).or. &
      any(saved%entry_time<0).or.any(saved%entry_time>saved%last_time)) return
    if(any(saved%event_previous.and.old%event_ids=='').or.any(saved%active.and.old%event_ids=='').or. &
      any(saved%window_phase/=0.and.old%window_ids=='')) return
    if(.not.saved%has_previous.and.any(saved%event_previous.or.saved%active)) return
    if(saved%has_previous.and.saved%previous_time<saved%origin_time) return
    if(old%enabled.and.any(old%event_ids/='')) then
      call configure_schedule(saved%monitor,old%monitor_mode,old%monitor_steps,old%monitor_time, &
        saved%origin_step,saved%origin_time,.true.,.false.,valid)
      if(valid) call restore_schedule_state(unit,saved%monitor,'adaptive-monitor',step,time,valid)
      if(.not.valid) return
      call resume_schedule(saved%monitor,valid)
      if(.not.valid) return
    endif
    same_monitor=(old%enabled.eqv.adaptive_shared_config%enabled).and. &
      old%indicator==adaptive_shared_config%indicator.and.old%monitor_mode==adaptive_shared_config%monitor_mode.and. &
      old%monitor_steps==adaptive_shared_config%monitor_steps.and.old%monitor_time==adaptive_shared_config%monitor_time
    restored=saved
    if(.not.same_monitor) then
      if(.not.override) return
      call configure_adaptive(adaptive_shared_config,step,time,valid)
      if(.not.valid) return
      restored=adaptive_shared_state; restored%last_step=step
      write(*,'(a)') 'ASTR_AP_RESET monitor reason=monitor_configuration_change'
    endif
    reset_events=.false.; reset_windows=.false.
    restored%active=.false.; restored%event_previous=.false.; restored%r=0; restored%entry_time=0
    restored%window_phase=0
    do i=1,adaptive_capacity
      if(adaptive_shared_config%event_ids(i)=='') cycle
      j=findloc(old%event_ids,adaptive_shared_config%event_ids(i),dim=1)
      same_event=.false.
      if(j>0) same_event=same_monitor.and.old%s_ref(j)==adaptive_shared_config%s_ref(i).and. &
        old%t_ref(j)==adaptive_shared_config%t_ref(i).and.old%r_on(j)==adaptive_shared_config%r_on(i).and. &
        old%r_off(j)==adaptive_shared_config%r_off(i).and.old%hold_time(j)==adaptive_shared_config%hold_time(i)
      if(.not.same_event.and..not.override) return
      reset_events(i)=.not.same_event
      same_scale=.false.
      if(j>0) same_scale=same_monitor.and.old%s_ref(j)==adaptive_shared_config%s_ref(i).and. &
        old%t_ref(j)==adaptive_shared_config%t_ref(i)
      if(same_scale) restored%event_previous(i)=saved%event_previous(j)
      if(same_event) then
        restored%active(i)=saved%active(j); restored%event_previous(i)=saved%event_previous(j)
        restored%r(i)=saved%r(j); restored%entry_time(i)=saved%entry_time(j)
      endif
      if(reset_events(i)) then
        message='threshold_or_hold_change'
        if(.not.same_scale) message='scale_or_new_event'
        if(.not.same_monitor) message='monitor_configuration_change'
        write(*,'(4a)') 'ASTR_AP_RESET event ',trim(adaptive_shared_config%event_ids(i)), &
          ' reason=',trim(message)
      endif
    enddo
    do i=1,adaptive_capacity
      if(adaptive_shared_config%window_ids(i)=='') cycle
      j=findloc(old%window_ids,adaptive_shared_config%window_ids(i),dim=1)
      same_window=.false.
      if(j>0) same_window=old%window_modes(j)==adaptive_shared_config%window_modes(i).and. &
        all(old%window_steps(:,j)==adaptive_shared_config%window_steps(:,i)).and. &
        all(old%window_times(:,j)==adaptive_shared_config%window_times(:,i))
      if(.not.same_window.and..not.override) return
      reset_windows(i)=.not.same_window
      if(same_window) restored%window_phase(i)=saved%window_phase(j)
      if(reset_windows(i)) write(*,'(3a)') 'ASTR_AP_RESET window ',trim(adaptive_shared_config%window_ids(i)), &
        ' reason=window_definition_change'
    enddo
    if(.not.override) then
      if(count(old%event_ids/='')/=count(adaptive_shared_config%event_ids/='').or. &
        count(old%window_ids/='')/=count(adaptive_shared_config%window_ids/='')) return
    endif
    adaptive_shared_state=restored; ok=.true.
  end subroutine
end module

module adaptive_output_collective
  use mpi
  use adaptive_output
  use iso_fortran_env, only: int64,real64
  implicit none
  private
  public :: agree_adaptive_config,agree_adaptive_bindings
contains
  subroutine agree_adaptive_config(c,comm,ok)
    type(adaptive_config),intent(in) :: c
    integer,intent(in) :: comm
    logical,intent(out) :: ok
    integer(int64) :: integers(1+2*adaptive_capacity)
    real(real64) :: values(1+7*adaptive_capacity)
    character(64) :: labels(2+3*adaptive_capacity)
    type(adaptive_config) :: root
    logical :: enabled,same
    integer :: status,n
    n=adaptive_capacity; enabled=c%enabled
    integers=[c%monitor_steps,reshape(c%window_steps,[2*n])]
    values=[c%monitor_time,c%s_ref,c%t_ref,c%r_on,c%r_off,c%hold_time,reshape(c%window_times,[2*n])]
    labels=[character(64) :: c%indicator,c%monitor_mode,c%event_ids,c%window_ids,c%window_modes]
    call MPI_Bcast(enabled,1,MPI_LOGICAL,0,comm,status)
    ok=status==MPI_SUCCESS
    if(.not.ok) return
    call MPI_Bcast(integers,size(integers),MPI_INTEGER8,0,comm,status)
    ok=status==MPI_SUCCESS
    if(.not.ok) return
    call MPI_Bcast(values,size(values),MPI_DOUBLE_PRECISION,0,comm,status)
    ok=status==MPI_SUCCESS
    if(.not.ok) return
    call MPI_Bcast(labels,size(labels)*len(labels),MPI_CHARACTER,0,comm,status)
    ok=status==MPI_SUCCESS
    if(.not.ok) return
    root%enabled=enabled; root%monitor_steps=integers(1)
    root%window_steps=reshape(integers(2:),[2,n])
    root%monitor_time=values(1); root%s_ref=values(2:n+1); root%t_ref=values(n+2:2*n+1)
    root%r_on=values(2*n+2:3*n+1); root%r_off=values(3*n+2:4*n+1)
    root%hold_time=values(4*n+2:5*n+1); root%window_times=reshape(values(5*n+2:),[2,n])
    root%indicator=labels(1); root%monitor_mode=labels(2); root%event_ids=labels(3:n+2)
    root%window_ids=labels(n+3:2*n+2); root%window_modes=labels(2*n+3:)
    same=adaptive_config_equal(c,root)
    call MPI_Allreduce(same,ok,1,MPI_LOGICAL,MPI_LAND,comm,status)
    ok=ok.and.status==MPI_SUCCESS
  end subroutine

  subroutine agree_adaptive_bindings(bindings,comm,ok)
    type(adaptive_binding),intent(in) :: bindings(:)
    integer,intent(in) :: comm
    logical,intent(out) :: ok
    logical :: enabled(size(bindings)),same
    integer(int64) :: steps(size(bindings))
    real(real64) :: times(size(bindings))
    character(64) :: names(2*adaptive_capacity,size(bindings))
    type(adaptive_binding) :: root
    integer :: i,status,n
    n=adaptive_capacity; enabled=bindings%enabled; steps=bindings%dense_steps; times=bindings%dense_time
    do i=1,size(bindings)
      names(:,i)=[bindings(i)%events,bindings(i)%windows]
    enddo
    call MPI_Bcast(enabled,size(enabled),MPI_LOGICAL,0,comm,status)
    ok=status==MPI_SUCCESS
    if(.not.ok) return
    call MPI_Bcast(steps,size(steps),MPI_INTEGER8,0,comm,status)
    ok=status==MPI_SUCCESS
    if(.not.ok) return
    call MPI_Bcast(times,size(times),MPI_DOUBLE_PRECISION,0,comm,status)
    ok=status==MPI_SUCCESS
    if(.not.ok) return
    call MPI_Bcast(names,size(names)*len(names),MPI_CHARACTER,0,comm,status)
    ok=status==MPI_SUCCESS
    if(.not.ok) return
    same=.true.
    do i=1,size(bindings)
      root%enabled=enabled(i); root%dense_steps=steps(i); root%dense_time=times(i)
      root%events=names(:n,i); root%windows=names(n+1:,i)
      same=same.and.adaptive_binding_equal(bindings(i),root)
    enddo
    call MPI_Allreduce(same,ok,1,MPI_LOGICAL,MPI_LAND,comm,status)
    ok=ok.and.status==MPI_SUCCESS
  end subroutine
end module
