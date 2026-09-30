module insitu_schedule
  use iso_fortran_env, only: int64,int32,real64
  use ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: sample_schedule,configure_schedule,poll_schedule
  public :: write_schedule_state,restore_schedule_state
  type :: sample_schedule
    private
    character(5) :: mode=''
    integer(int64) :: interval=0,origin_step=0,seen_step=0,index=0,emitted_step=-1
    real(real64) :: period=0,origin_time=0,seen_time=0
    logical :: initial=.false.,final=.false.,ready=.false.,ended=.false.
  end type
contains
  subroutine write_schedule_state(unit,s,batch,step,time,ok)
    integer,intent(in) :: unit
    type(sample_schedule),intent(in) :: s
    character(*),intent(in) :: batch
    integer(int64),intent(in) :: step
    real(real64),intent(in) :: time
    logical,intent(out) :: ok
    character(64) :: identity
    integer(int32) :: flags(3)
    integer :: ios
    ok=.false.
    if(.not.s%ready.or.len_trim(batch)==0.or.len_trim(batch)>64) return
    if(step<s%seen_step.or..not.ieee_is_finite(time)) return
    if(time<s%seen_time) return
    identity=batch
    flags=[merge(1_int32,0_int32,s%initial),merge(1_int32,0_int32,s%final), &
           merge(1_int32,0_int32,s%ended)]
    write(unit,iostat=ios) 'ASTRSC01',identity,step,time,s%mode,s%interval, &
      s%period,s%origin_step,s%origin_time,s%seen_step,s%seen_time,s%index,s%emitted_step,flags
    ok=ios==0
  end subroutine

  subroutine restore_schedule_state(unit,s,batch,step,time,ok)
    integer,intent(in) :: unit
    type(sample_schedule),intent(inout) :: s
    character(*),intent(in) :: batch
    integer(int64),intent(in) :: step
    real(real64),intent(in) :: time
    logical,intent(out) :: ok
    type(sample_schedule) :: candidate,check
    character(8) :: magic
    character(64) :: identity
    integer(int32) :: flags(3)
    integer(int64) :: saved_step,crossed
    real(real64) :: saved_time
    logical :: valid,emit
    integer :: ios
    ok=.false.
    if(.not.s%ready.or.len_trim(batch)==0.or.len_trim(batch)>64) return
    if(step<0.or..not.ieee_is_finite(time)) return
    read(unit,iostat=ios) magic,identity,saved_step,saved_time,candidate%mode, &
      candidate%interval,candidate%period,candidate%origin_step,candidate%origin_time, &
      candidate%seen_step,candidate%seen_time,candidate%index,candidate%emitted_step,flags
    if(ios/=0) return
    if(magic/='ASTRSC01'.or.identity/=batch.or.saved_step/=step.or.saved_time/=time) return
    if(any(flags<0).or.any(flags>1)) return
    if(candidate%mode/=s%mode.or.candidate%interval/=s%interval.or.candidate%period/=s%period) return
    if(candidate%origin_step/=s%origin_step.or.candidate%origin_time/=s%origin_time) return
    if((flags(1)==1).neqv.s%initial) return
    if((flags(2)==1).neqv.s%final) return
    if(candidate%seen_step>step.or..not.ieee_is_finite(candidate%seen_time)) return
    if(candidate%seen_time>time) return
    if(candidate%emitted_step/=-1) then
      if(candidate%emitted_step<candidate%origin_step.or.candidate%emitted_step>candidate%seen_step) return
    endif
    ! Recompute the target index from the frozen configuration, not the serialized index.
    call configure_schedule(check,s%mode,s%interval,s%period,s%origin_step,s%origin_time, &
      s%initial,s%final,valid)
    if(.not.valid) return
    call poll_schedule(check,candidate%seen_step,candidate%seen_time,.false.,emit,crossed,valid)
    if(.not.valid.or.check%index/=candidate%index) return
    candidate%initial=s%initial
    candidate%final=s%final
    candidate%ready=.true.
    candidate%ended=flags(3)==1
    s=candidate
    ok=.true.
  end subroutine

  subroutine configure_schedule(s,mode,step_interval,time_interval,origin_step,origin_time,initial,final,ok)
    type(sample_schedule),intent(out) :: s
    character(*),intent(in) :: mode
    integer(int64),intent(in) :: step_interval,origin_step
    real(real64),intent(in) :: time_interval,origin_time
    logical,intent(in) :: initial,final
    logical,intent(out) :: ok
    ok=.false.
    if(origin_step<0.or..not.ieee_is_finite(origin_time)) return
    if(.not.ieee_is_finite(time_interval)) return
    select case(mode)
    case('steps')
      if(step_interval<=0.or.time_interval/=0) return
    case('time')
      if(step_interval/=0.or.time_interval<=0) return
      if(.not.ieee_is_finite(origin_time+time_interval)) return
      if(origin_time+time_interval<=origin_time) return
    case default
      return
    end select
    s%mode=mode
    s%interval=step_interval
    s%period=time_interval
    s%origin_step=origin_step
    s%seen_step=origin_step
    s%origin_time=origin_time
    s%seen_time=origin_time
    s%initial=initial
    s%final=final
    s%ready=.true.
    ok=.true.
  end subroutine

  subroutine poll_schedule(s,step,t,is_final,emit,crossed,ok)
    type(sample_schedule),intent(inout) :: s
    integer(int64),intent(in) :: step
    real(real64),intent(in) :: t
    logical,intent(in) :: is_final
    logical,intent(out) :: emit,ok
    integer(int64),intent(out) :: crossed
    integer(int64) :: index
    real(real64) :: ratio,target,next
    emit=.false.
    ok=.false.
    crossed=0
    if(.not.s%ready.or.s%ended.or..not.ieee_is_finite(t)) return
    if(step<s%seen_step.or.t<s%seen_time) return
    if((step==s%seen_step).neqv.(t==s%seen_time)) return
    if(s%mode=='steps') then
      index=(step-s%origin_step)/s%interval
    else
      ratio=(t-s%origin_time)/s%period
      if(.not.ieee_is_finite(ratio)) return
      ! Bound the index so integer-to-FP64 conversion remains exact.
      if(ratio<0.or.ratio>=2.d0**52-2.d0) return
      index=floor(ratio,kind=int64)
      target=s%origin_time+real(index,real64)*s%period
      if(.not.ieee_is_finite(target)) return
      if(target>t) index=index-1
      next=s%origin_time+real(index+1,real64)*s%period
      if(.not.ieee_is_finite(next)) return
      if(next<=t) index=index+1
      target=s%origin_time+real(index,real64)*s%period
      next=s%origin_time+real(index+1,real64)*s%period
      ! Compare representable target times strictly; never emit early via epsilon.
      if(.not.ieee_is_finite(next).or.next<=target.or.target>t.or.next<=t) return
    endif
    if(index<s%index) return
    crossed=index-s%index
    emit=crossed>0.or.(s%initial.and.step==s%origin_step).or.(s%final.and.is_final)
    emit=emit.and.step/=s%emitted_step
    if(emit) s%emitted_step=step
    s%index=index
    s%seen_step=step
    s%seen_time=t
    s%ended=is_final
    ok=.true.
  end subroutine
end module
