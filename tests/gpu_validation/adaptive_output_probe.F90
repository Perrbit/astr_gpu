program adaptive_output_probe
  use iso_fortran_env, only: int64,real64
  use ieee_arithmetic, only: ieee_value,ieee_quiet_nan
  use adaptive_output
  implicit none
  type(adaptive_config) :: cfg
  type(adaptive_binding) :: binding
  type(adaptive_clock) :: clock,other
  integer :: step,unit
  character(1024) :: filename,scenario
  character(256) :: message
  logical :: ok,due,dense,same
  real(real64),parameter :: values(0:12)=[0.d0,0.d0,2.d0,3.d0,3.d0,3.d0,3.d0,3.d0, &
    3.d0,5.d0,5.d0,5.d0,5.d0]
  call get_command_argument(1,filename)
  call get_command_argument(2,scenario)
  if(len_trim(scenario)>0) then
    call edge_case(trim(scenario))
    print '(2a)', 'PASS adaptive ',trim(scenario)
    stop
  endif
  cfg%enabled=.true.; cfg%monitor_steps=1
  cfg%event_ids(1)='pulse'; cfg%s_ref(1)=1; cfg%t_ref(1)=1
  cfg%r_on(1)=2; cfg%r_off(1)=1; cfg%hold_time(1)=2
  cfg%window_ids(1:2)=['window ','crossed']
  cfg%window_modes(1:2)='steps'; cfg%window_steps(:,1)=[6_int64,8_int64]
  cfg%window_steps(:,2)=[30_int64,32_int64]
  binding%enabled=.true.; binding%dense_steps=1
  binding%events(1)='pulse'; binding%windows(1)='window'
  call configure_adaptive(cfg,0_int64,0.d0,ok); call require(ok,'configure')
  call configure_adaptive_clock(clock,binding,'steps',5_int64,0.d0,0_int64,0.d0,.false.,.false.,ok)
  call require(ok,'clock')
  do step=0,12
    call observe_adaptive(int(step,int64),real(step,real64),ok); call require(ok,'observe')
    call adaptive_monitor_due(int(step,int64),real(step,real64),due,ok); call require(ok.and.due,'monitor')
    call feed_adaptive_signal(values(step),real(step,real64),ok); call require(ok,'signal')
    call poll_adaptive_clock(clock,binding,int(step,int64),real(step,real64),.false.,due,ok)
    call require(ok,'poll')
    call require(due.eqv.any(step==[2,3,6,7,9,10]),'identity')
    if(due) then
      call report_adaptive_clock(clock,merge(28,0,step==3),merge(1,0,step==3),ok)
      call require(ok,'publication')
    endif
    if(step==3) then
      call require(clock%success==2.and.clock%attempt==3,'missing preserves success')
      open(newunit=unit,file=filename,status='replace',access='stream',form='unformatted')
      call adaptive_shared_file(unit,.true.,3_int64,3.d0,.false.,ok); call require(ok,'save shared')
      call adaptive_clock_file(unit,.true.,clock,binding,3_int64,3.d0,same,ok); call require(ok,'save clock')
      close(unit)
      call configure_adaptive(cfg,0_int64,0.d0,ok); call require(ok,'reset')
      call configure_adaptive_clock(other,binding,'steps',5_int64,0.d0,0_int64,0.d0,.false.,.false.,ok)
      open(newunit=unit,file=filename,status='old',access='stream',form='unformatted')
      call adaptive_shared_file(unit,.false.,3_int64,3.d0,.false.,ok); call require(ok,'restore shared')
      call adaptive_clock_file(unit,.false.,other,binding,3_int64,3.d0,same,ok)
      call require(ok.and.same,'restore clock'); close(unit)
      call require(other%success==clock%success.and.other%attempt==clock%attempt.and. &
        (other%dense.eqv.clock%dense),'restored identity')
      clock=other
    endif
  enddo
  call observe_adaptive(40_int64,40.d0,ok); call require(ok,'cross')
  call require(adaptive_shared_state%window_phase(2)==3,'crossed whole window')
  call feed_adaptive_signal(ieee_value(0.d0,ieee_quiet_nan),40.d0,ok)
  call require(.not.ok,'nonfinite rejected')
  call feed_adaptive_signal(3.d0,40.d0,ok); call require(ok,'finite after rejected signal')
  call feed_adaptive_signal(3.d0,40.d0,ok); call require(.not.ok,'zero dt rejected')
  cfg%r_off(1)=cfg%r_on(1)
  call validate_adaptive_config(cfg,ok,message); call require(.not.ok,'invalid hysteresis')
  binding%dense_steps=5
  call validate_adaptive_binding(binding,'steps',5_int64,0.d0,ok=ok)
  call require(.not.ok,'dense not faster')
  print '(a)', 'PASS adaptive events/windows/clocks/missing/exact restart/nonfinite'
contains
  subroutine edge_case(name)
    character(*),intent(in) :: name
    integer :: s
    cfg%enabled=.true.; cfg%monitor_steps=2
    cfg%event_ids(1)='pulse'; cfg%s_ref(1)=1; cfg%t_ref(1)=1
    cfg%r_on(1)=2; cfg%r_off(1)=1; cfg%hold_time(1)=3
    binding%enabled=.true.; binding%dense_steps=1; binding%events(1)='pulse'
    select case(name)
    case('sparse_hold')
      call configure_adaptive(cfg,0_int64,0.d0,ok); call require(ok,'configure sparse')
      do s=0,6
        call observe_adaptive(int(s,int64),real(s,8),ok); call require(ok,'sparse observe')
        call adaptive_monitor_due(int(s,int64),real(s,8),due,ok)
        call require(ok.and.(due.eqv.(mod(s,2)==0)),'sparse due')
        if(due) then
          call feed_adaptive_signal(merge(0.d0,4.d0,s==0),real(s,8),ok); call require(ok,'sparse feed')
        endif
        call require(adaptive_shared_state%active(1).eqv.(s>=2.and.s<6),'no stale hold exit')
      enddo
    case('overflow')
      cfg%monitor_steps=1; cfg%t_ref(1)=2
      call configure_adaptive(cfg,0_int64,0.d0,ok); call require(ok,'overflow config')
      call observe_adaptive(0_int64,0.d0,ok)
      call adaptive_monitor_due(0_int64,0.d0,due,ok)
      call feed_adaptive_signal(0.d0,0.d0,ok); call require(ok,'overflow baseline')
      call observe_adaptive(1_int64,1.d-308,ok)
      call adaptive_monitor_due(1_int64,1.d-308,due,ok)
      call feed_adaptive_signal(1.d0,1.d-308,ok); call require(.not.ok,'intermediate multiply overflow')
      cfg%t_ref(1)=1.d308; cfg%s_ref(1)=1.d-308
      call validate_adaptive_config(cfg,ok,message); call require(.not.ok,'scale division overflow')
    case('step_overflow')
      call configure_adaptive(cfg,0_int64,0.d0,ok)
      call configure_adaptive_clock(clock,binding,'steps',5_int64,0.d0,huge(0_int64)-1,0.d0,.false.,.false.,ok)
      call require(.not.ok,'next step overflow')
    case('time_and_crossing')
      cfg%monitor_mode='time'; cfg%monitor_steps=0; cfg%monitor_time=1.5
      cfg%window_ids(1)='tiny'; cfg%window_modes(1)='time'; cfg%window_times(:,1)=[.5d0,1.5d0]
      call configure_adaptive(cfg,0_int64,0.d0,ok); call require(ok,'time config')
      call observe_adaptive(0_int64,0.d0,ok)
      call adaptive_monitor_due(0_int64,0.d0,due,ok); call require(ok.and.due,'time initial')
      call feed_adaptive_signal(0.d0,0.d0,ok)
      call observe_adaptive(1_int64,2.d0,ok); call require(ok,'time jump')
      call adaptive_monitor_due(1_int64,2.d0,due,ok); call require(ok.and.due,'time overshoot once')
      call feed_adaptive_signal(4.d0,2.d0,ok); call require(ok,'actual dt')
      call require(adaptive_shared_state%r(1)==2.and.adaptive_shared_state%window_phase(1)==3,'time rate and skip')
      binding%dense_steps=0; binding%dense_time=.25d0
      call configure_adaptive_clock(clock,binding,'time',0_int64,1.d0,0_int64,0.d0,.false.,.false.,ok)
      call require(ok,'time product')
      call poll_adaptive_clock(clock,binding,1_int64,2.d0,.false.,due,ok); call require(ok.and.due,'time entry')
      call report_adaptive_clock(clock,0,0,ok); call require(ok,'time report')
      call require(clock%next_time==2.25d0,'actual output anchors next')
    case('missing_wait')
      cfg%window_ids(1)='later'; cfg%window_modes(1)='steps'; cfg%window_steps(:,1)=[2_int64,4_int64]
      binding%events=''; binding%windows(1)='later'; binding%dense_steps=2
      call configure_adaptive(cfg,0_int64,0.d0,ok)
      call configure_adaptive_clock(clock,binding,'steps',5_int64,0.d0,0_int64,0.d0,.true.,.false.,ok)
      call observe_adaptive(0_int64,0.d0,ok)
      call poll_adaptive_clock(clock,binding,0_int64,0.d0,.false.,due,ok); call require(ok.and.due,'initial')
      call report_adaptive_clock(clock,13,1,ok); call require(ok,'missing')
      call observe_adaptive(1_int64,1.d0,ok)
      call poll_adaptive_clock(clock,binding,1_int64,1.d0,.false.,due,ok); call require(ok.and..not.due,'normal wait')
      call observe_adaptive(2_int64,2.d0,ok)
      call poll_adaptive_clock(clock,binding,2_int64,2.d0,.false.,due,ok); call require(ok.and.due,'dense wait due')
      call report_adaptive_clock(clock,28,1,ok)
      call observe_adaptive(3_int64,3.d0,ok)
      call poll_adaptive_clock(clock,binding,3_int64,3.d0,.false.,due,ok); call require(ok.and..not.due,'no busy retry')
    case('selective_override')
      cfg%event_ids(2)='other'; cfg%s_ref(2)=1; cfg%t_ref(2)=1
      cfg%r_on(2)=2; cfg%r_off(2)=1; cfg%hold_time(2)=3
      call configure_adaptive(cfg,0_int64,0.d0,ok)
      call observe_adaptive(0_int64,0.d0,ok)
      call adaptive_monitor_due(0_int64,0.d0,due,ok)
      call feed_adaptive_signal(0.d0,0.d0,ok)
      call observe_adaptive(2_int64,2.d0,ok)
      call adaptive_monitor_due(2_int64,2.d0,due,ok)
      call feed_adaptive_signal(4.d0,2.d0,ok)
      open(newunit=unit,file=filename,status='replace',access='stream',form='unformatted')
      call adaptive_shared_file(unit,.true.,2_int64,2.d0,.false.,ok); call require(ok,'override save')
      close(unit)
      cfg%r_on(1)=3
      call configure_adaptive(cfg,0_int64,0.d0,ok)
      open(newunit=unit,file=filename,status='old',access='stream',form='unformatted')
      call adaptive_shared_file(unit,.false.,2_int64,2.d0,.false.,ok); call require(.not.ok,'unapproved change')
      close(unit)
      open(newunit=unit,file=filename,status='old',access='stream',form='unformatted')
      call adaptive_shared_file(unit,.false.,2_int64,2.d0,.true.,ok); call require(ok,'approved change')
      close(unit)
      call require(.not.adaptive_shared_state%active(1).and.adaptive_shared_state%active(2),'selective latch')
      call require(adaptive_dependencies_reset(binding),'dependent reset')
      binding%events(1)='other'
      call require(.not.adaptive_dependencies_reset(binding),'unrelated preserve')
    case('corrupt_shared')
      call configure_adaptive(cfg,0_int64,0.d0,ok)
      call observe_adaptive(0_int64,0.d0,ok)
      call adaptive_monitor_due(0_int64,0.d0,due,ok)
      call feed_adaptive_signal(1.d0,0.d0,ok)
      adaptive_shared_state%r(1)=-1
      open(newunit=unit,file=filename,status='replace',access='stream',form='unformatted')
      call adaptive_shared_file(unit,.true.,0_int64,0.d0,.false.,ok); call require(ok,'corruption fixture')
      close(unit)
      call configure_adaptive(cfg,0_int64,0.d0,ok)
      open(newunit=unit,file=filename,status='old',access='stream',form='unformatted')
      call adaptive_shared_file(unit,.false.,0_int64,0.d0,.false.,ok)
      call require(.not.ok,'negative stored rate rejected'); close(unit)
    case('overlap')
      cfg%event_ids=''; cfg%s_ref=0; cfg%t_ref=0; cfg%r_on=0; cfg%r_off=0; cfg%hold_time=0
      cfg%monitor_steps=0
      cfg%window_ids(1:2)=['first ','second']; cfg%window_modes(1:2)='steps'
      cfg%window_steps(:,1)=[2_int64,5_int64]; cfg%window_steps(:,2)=[4_int64,7_int64]
      binding%events=''; binding%windows(1:2)=cfg%window_ids(1:2); binding%dense_steps=2
      call configure_adaptive(cfg,0_int64,0.d0,ok); call require(ok,'overlap config')
      call configure_adaptive_clock(clock,binding,'steps',5_int64,0.d0,0_int64,0.d0,.false.,.false.,ok)
      do s=0,9
        call observe_adaptive(int(s,int64),real(s,8),ok)
        call poll_adaptive_clock(clock,binding,int(s,int64),real(s,8),.false.,due,ok)
        call require(ok.and.(due.eqv.any(s==[2,4,6])),'union no extra entry frame')
        if(due) call report_adaptive_clock(clock,0,0,ok)
      enddo
    case default
      call require(.false.,'unknown test case')
    end select
  end subroutine

  subroutine require(valid,name)
    logical,intent(in) :: valid
    character(*),intent(in) :: name
    if(valid) return
    print '(2a)', 'FAIL ',name
    stop 1
  end subroutine
end program
