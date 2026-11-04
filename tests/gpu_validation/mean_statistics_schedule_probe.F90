program mean_statistics_schedule_probe
  use iso_fortran_env, only: int64
  use astr_precision, only: real_kind
  use run_control_config, only: statistics_options
  use mean_statistics_schedule, only: mean_schedule,reset_mean_schedule,mean_schedule_poll,mean_schedule_file
  implicit none
  type(mean_schedule) :: continuous,resumed,restored
  type(statistics_options) :: options
  real(real_kind),parameter :: times(0:11)=[0.0_real_kind,0.125_real_kind,0.375_real_kind, &
    0.75_real_kind,1.0_real_kind,1.125_real_kind,1.5_real_kind,1.75_real_kind, &
    2.0_real_kind,2.5_real_kind,3.0_real_kind,3.125_real_kind]
  logical :: sample,write_data,new_window,changed,ok,reference(4,0:11),observed(4)
  integer :: step,unit,restart_at,case_index
  integer,parameter :: restarts(3)=[0,5,8]
  do step=0,11
    call configuration(step,options)
    call mean_schedule_poll(continuous,options,int(step,int64),times(step),sample,write_data,new_window,changed,ok)
    if(.not.ok) error stop 'continuous mean schedule failed'
    reference(:,step)=[sample,write_data,new_window,changed]
  enddo
  if(any(reference(1,:).neqv.[.true.,.false.,.true.,.true.,.true.,.false., &
    .false.,.true.,.false.,.true.,.true.,.false.])) error stop 'actual-time sample sequence'
  if(any(reference(2,:).neqv.[.false.,.false.,.false.,.true.,.true.,.false., &
    .false.,.false.,.false.,.false.,.false.,.false.])) error stop 'independent write sequence'
  if(any(reference(3,:).neqv.[.true.,.false.,.false.,.false.,.false.,.false., &
    .false.,.false.,.false.,.true.,.false.,.false.])) error stop 'window reset sequence'
  do case_index=1,size(restarts)
    restart_at=restarts(case_index)
    call reset_mean_schedule(resumed)
    do step=0,11
      call configuration(step,options)
      call mean_schedule_poll(resumed,options,int(step,int64),times(step),sample,write_data,new_window,changed,ok)
      if(.not.ok) error stop 'resumed mean schedule failed'
      observed=[sample,write_data,new_window,changed]
      if(any(observed.neqv.reference(:,step))) error stop 'mean schedule restart sequence mismatch'
      if(step==restart_at) then
        open(newunit=unit,status='scratch',access='stream',form='unformatted')
        call mean_schedule_file(unit,.true.,resumed,int(step,int64),times(step),ok)
        if(.not.ok) error stop 'mean schedule write failed'
        rewind(unit)
        call reset_mean_schedule(restored)
        call mean_schedule_file(unit,.false.,restored,int(step,int64),times(step),ok)
        if(.not.ok) error stop 'mean schedule restore failed'
        resumed=restored
        close(unit)
        ! Re-entering the same complete step cannot create a duplicate endpoint or write.
        call mean_schedule_poll(resumed,options,int(step,int64),times(step),sample,write_data,new_window,changed,ok)
        if(.not.ok.or.sample.or.write_data.or.new_window.or.changed) error stop 'duplicate restart event'
      endif
    enddo
  enddo
  print *, 'MEAN_STATISTICS_SCHEDULE_PASS'
contains
  subroutine configuration(step,options)
    integer,intent(in) :: step
    type(statistics_options),intent(out) :: options
    options=statistics_options()
    options%enabled=.true.
    options%sample_clock='time';options%sample_time=0.25_real_kind
    options%write_clock='time';options%write_time=0.5_real_kind
    if(step>=5.and.step<=8) then
      options%sample_clock='steps';options%sample_steps=2
      options%write_clock='steps';options%write_steps=3
    endif
    if(step==8) options%enabled=.false.
    if(step>=9) options%write_time=1.0_real_kind
  end subroutine
end program
