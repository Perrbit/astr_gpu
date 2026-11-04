module mean_statistics_schedule
  use iso_fortran_env, only: int64
  use astr_precision, only: real_kind
  use astr_precision_io, only: read_source_reals
  use run_control_config, only: statistics_options,run_control_options,validate_run_control
  use insitu_schedule, only: sample_schedule,configure_schedule,poll_schedule, &
    write_schedule_state,restore_schedule_state
  implicit none
  private
  public :: mean_schedule,reset_mean_schedule,mean_schedule_poll,mean_schedule_file

  type :: mean_schedule
    private
    type(statistics_options) :: configuration
    type(sample_schedule) :: sampling,writing
    integer(int64) :: origin_step(2)=0
    real(real_kind) :: origin_time(2)=0
    logical :: initialized=.false.,sample_initial=.false.
  end type
contains
  subroutine reset_mean_schedule(state)
    type(mean_schedule),intent(out) :: state
    state=mean_schedule()
  end subroutine

  pure logical function sample_equal(a,b) result(same)
    type(statistics_options),intent(in) :: a,b
    same=a%sample_clock==b%sample_clock
    if(a%sample_clock=='steps') then
      same=same.and.a%sample_steps==b%sample_steps
    else
      same=same.and.a%sample_time==b%sample_time
    endif
  end function

  pure logical function write_equal(a,b) result(same)
    type(statistics_options),intent(in) :: a,b
    same=a%write_clock==b%write_clock
    if(a%write_clock=='steps') then
      same=same.and.a%write_steps==b%write_steps
    else
      same=same.and.a%write_time==b%write_time
    endif
  end function

  pure logical function writes_enabled(options) result(enabled)
    type(statistics_options),intent(in) :: options
    enabled=options%write_steps>0
    if(options%write_clock=='time') enabled=options%write_time>0
  end function

  subroutine configure_clock(clock,mode,steps,period,step,time,initial,ok)
    type(sample_schedule),intent(out) :: clock
    character(*),intent(in) :: mode
    integer(int64),intent(in) :: steps,step
    real(real_kind),intent(in) :: period,time
    logical,intent(in) :: initial
    logical,intent(out) :: ok
    call configure_schedule(clock,mode,merge(steps,0_int64,mode=='steps'), &
      merge(period,0.0_real_kind,mode=='time'),step,time,initial,.false.,ok)
  end subroutine

  subroutine mean_schedule_poll(state,options,step,time,sample,write_data,new_window,changed,ok)
    type(mean_schedule),intent(inout) :: state
    type(statistics_options),intent(in) :: options
    integer(int64),intent(in) :: step
    real(real_kind),intent(in) :: time
    logical,intent(out) :: sample,write_data,new_window,changed,ok
    logical :: new_sample,new_write
    integer(int64) :: crossed
    sample=.false.;write_data=.false.;new_window=.false.;changed=.false.;ok=.true.
    new_window=options%enabled.and.(.not.state%initialized.or..not.state%configuration%enabled)
    new_sample=new_window.or..not.sample_equal(options,state%configuration)
    new_write=new_window.or..not.write_equal(options,state%configuration)
    changed=.not.state%initialized.or.(options%enabled.neqv.state%configuration%enabled).or. &
      .not.sample_equal(options,state%configuration).or..not.write_equal(options,state%configuration)
    if(options%enabled) then
      if(new_sample) then
        state%origin_step(1)=step;state%origin_time(1)=time
        state%sample_initial=new_window
        call configure_clock(state%sampling,options%sample_clock,options%sample_steps, &
          options%sample_time,step,time,new_window,ok)
        if(.not.ok) return
      endif
      if(new_write.and.writes_enabled(options)) then
        state%origin_step(2)=step;state%origin_time(2)=time
        call configure_clock(state%writing,options%write_clock,options%write_steps, &
          options%write_time,step,time,.false.,ok)
        if(.not.ok) return
      endif
      call poll_schedule(state%sampling,step,time,.false.,sample,crossed,ok)
      if(.not.ok) return
      if(writes_enabled(options)) call poll_schedule(state%writing,step,time,.false.,write_data,crossed,ok)
      if(.not.ok) return
    endif
    state%configuration=options
    state%initialized=.true.
  end subroutine

  subroutine mean_schedule_file(unit,writing,state,step,time,ok,source_bytes)
    integer,intent(in) :: unit
    logical,intent(in) :: writing
    type(mean_schedule),intent(inout) :: state
    integer(int64),intent(in) :: step
    real(real_kind),intent(in) :: time
    logical,intent(out) :: ok
    integer,optional,intent(in) :: source_bytes
    type(mean_schedule) :: candidate
    type(run_control_options) :: check
    integer :: status,flags(3)
    character(8) :: tag
    character(512) :: message
    ok=.false.
    if(writing) then
      if(.not.state%initialized) return
      candidate=state
      flags=[1,merge(1,0,candidate%configuration%enabled),merge(1,0,candidate%sample_initial)]
      write(unit,iostat=status) 'ASTRMS01',flags,candidate%configuration%sample_clock, &
        candidate%configuration%write_clock,candidate%configuration%sample_steps, &
        candidate%configuration%write_steps,candidate%configuration%sample_time, &
        candidate%configuration%write_time,candidate%origin_step,candidate%origin_time
    else
      read(unit,iostat=status) tag,flags,candidate%configuration%sample_clock, &
        candidate%configuration%write_clock,candidate%configuration%sample_steps, &
        candidate%configuration%write_steps
      if(status==0) call read_source_reals(unit,candidate%configuration%sample_time,source_bytes,status)
      if(status==0) call read_source_reals(unit,candidate%configuration%write_time,source_bytes,status)
      if(status==0) read(unit,iostat=status) candidate%origin_step
      if(status==0) call read_source_reals(unit,candidate%origin_time,source_bytes,status)
      if(status/=0) return
      if(tag/='ASTRMS01'.or.flags(1)/=1.or.any(flags(2:3)<0).or.any(flags(2:3)>1)) return
      candidate%configuration%enabled=flags(2)==1
      candidate%sample_initial=flags(3)==1
      candidate%initialized=.true.
      check%max_steps=0;check%statistics=candidate%configuration
      call validate_run_control(check,ok,message)
      if(.not.ok) return
    endif
    if(status/=0) return
    if(candidate%configuration%enabled) then
      if(writing) then
        call write_schedule_state(unit,candidate%sampling,'mean-sample',step,time,ok)
      else
        call configure_clock(candidate%sampling,candidate%configuration%sample_clock, &
          candidate%configuration%sample_steps,candidate%configuration%sample_time, &
          candidate%origin_step(1),candidate%origin_time(1),candidate%sample_initial,ok)
        if(ok) call restore_schedule_state(unit,candidate%sampling,'mean-sample',step,time,ok,source_bytes)
      endif
      if(.not.ok) return
      if(writes_enabled(candidate%configuration)) then
        if(writing) then
          call write_schedule_state(unit,candidate%writing,'mean-write',step,time,ok)
        else
          call configure_clock(candidate%writing,candidate%configuration%write_clock, &
            candidate%configuration%write_steps,candidate%configuration%write_time, &
            candidate%origin_step(2),candidate%origin_time(2),.false.,ok)
          if(ok) call restore_schedule_state(unit,candidate%writing,'mean-write',step,time,ok,source_bytes)
        endif
        if(.not.ok) return
      endif
    endif
    if(.not.writing) state=candidate
    ok=.true.
  end subroutine

end module
