program run_control_collective_probe
  use iso_fortran_env, only: int64
  use mpi
  use astr_precision, only: real_kind,numerical_precision
  use timestep_control, only: timestep_options,timestep_fixed,choose_timestep,reduce_directional_rates
  use ieee_arithmetic, only: ieee_value,ieee_quiet_nan
  use run_control_config, only: run_control_options,read_run_control_collective,same_run_control
  use run_control_runtime, only: active_run,initialize_run_control,poll_run_control, &
    select_run_timestep,run_control_history,poll_mean_statistics
  implicit none
  type(run_control_options) :: options,expected,before
  type(timestep_options) :: automatic
  integer :: ierr,rank,unit,history_status,collect_count=0
  logical :: ok,changed,sample,writing,new_window,schedule_changed
  real(real_kind) :: local_rates(3),global_rates(3),dt,previous
  character(2048) :: filename,read_path
  character(512) :: message

  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)
  call get_command_argument(1,filename)
  read_path=filename
  if(rank/=0) read_path=trim(filename)//'.must_not_be_read'
  options%max_steps=int(rank+5,int64)
  if(rank==0) then
    open(newunit=unit,file=trim(filename),status='replace',action='write')
    write(unit,'(A)') '&run_control max_steps=20, stop_time=0.75, reload_steps=1 /'
    write(unit,'(A)') '&timestep mode="fixed", fixed_dt=0.02, cfl_target=0.3,'
    write(unit,'(A)') ' growth_factor=1.2, max_dt=0.001, initial_dt=0.0001 /'
    write(unit,'(A)') '&logging list_steps=0 /'
    write(unit,'(A)') '&repair crashfix=.true. /'
    write(unit,'(A)') '&statistics enabled=.true., sample_clock="time", sample_time=0.125, sample_steps=3,'
    write(unit,'(A)') ' write_clock="time", write_time=0.5, write_steps=7 /'
    close(unit)
  endif
  call read_run_control_collective(trim(read_path),MPI_COMM_WORLD,options,ok,message)
  call require(ok,'collective read: '//trim(message))
  expected=run_control_options()
  expected%max_steps=20;expected%stop_time=0.75_real_kind;expected%reload_steps=1
  expected%list_steps=0;expected%crashfix=.true.
  expected%timestep%mode=timestep_fixed;expected%timestep%fixed_dt=0.02_real_kind
  expected%timestep%cfl_target=0.3_real_kind;expected%timestep%growth_factor=1.2_real_kind
  expected%timestep%max_dt=0.001_real_kind;expected%timestep%initial_dt=0.0001_real_kind
  expected%statistics%enabled=.true.;expected%statistics%sample_clock='time'
  expected%statistics%sample_steps=3;expected%statistics%write_steps=7
  expected%statistics%sample_time=0.125_real_kind;expected%statistics%write_clock='time'
  expected%statistics%write_time=0.5_real_kind
  call require(same_run_control(options,expected),'broadcast must include every named field')
  ! Do not let the production packer also be the only oracle for its own fields.
  call require(all([options%max_steps,options%reload_steps,options%list_steps, &
    options%statistics%sample_steps,options%statistics%write_steps]==[20_int64,1_int64,0_int64,3_int64,7_int64]), &
    'named integer payload')
  call require(options%timestep%mode==timestep_fixed.and.options%crashfix.and.options%statistics%enabled, &
    'named mode and switch payload')
  call require(trim(options%statistics%sample_clock)=='time'.and.trim(options%statistics%write_clock)=='time', &
    'named clock payload')
  call require(all([options%stop_time,options%timestep%cfl_target,options%timestep%growth_factor, &
    options%timestep%max_dt,options%timestep%initial_dt,options%timestep%fixed_dt, &
    options%statistics%sample_time,options%statistics%write_time]== &
    [0.75_real_kind,0.3_real_kind,1.2_real_kind,0.001_real_kind,0.0001_real_kind, &
     0.02_real_kind,0.125_real_kind,0.5_real_kind]),'named real payload')
  before=options

  if(rank==0) then
    open(newunit=unit,file=trim(filename),status='replace',action='write')
    write(unit,'(A)') '&run_control max_steps=999, reload_steps=0 /'
    close(unit)
  endif
  call read_run_control_collective(trim(read_path),MPI_COMM_WORLD,options,ok,message)
  call require(.not.ok.and.len_trim(message)>0,'all ranks must see invalid update')
  call require(same_run_control(options,before),'invalid update must retain every rank state')

  if(rank==0) then
    open(newunit=unit,file=trim(filename),status='old')
    close(unit,status='delete')
  endif
  call read_run_control_collective(trim(read_path),MPI_COMM_WORLD,options,ok,message)
  call require(.not.ok.and.same_run_control(options,before),'missing file must not reset ranks')

  local_rates=[4.0_real_kind,1.0_real_kind,1.0_real_kind]
  if(rank/=0) local_rates=[1.0_real_kind,5.0_real_kind,1.0_real_kind]
  call reduce_directional_rates(local_rates,.true.,MPI_COMM_WORLD,global_rates,ok)
  call require(ok,'valid global wave bound')
  call require(all(global_rates==[4.0_real_kind,5.0_real_kind,1.0_real_kind]), &
    'each directional maximum must be reduced separately')
  call choose_timestep(automatic,global_rates,0.0_real_kind,.true.,dt,ok,message)
  call require(ok,'global automatic choice')
  call require(abs(dt-0.05_real_kind)<=64.0_real_kind*epsilon(dt),'sum(max) must not become max(sum)')
  call reduce_directional_rates(local_rates,rank==0,MPI_COMM_WORLD,global_rates,ok)
  call require(.not.ok,'invalid cell status must propagate from any rank')
  if(rank/=0) local_rates(1)=ieee_value(0.0_real_kind,ieee_quiet_nan)
  call reduce_directional_rates(local_rates,.true.,MPI_COMM_WORLD,global_rates,ok)
  call require(.not.ok,'MPI_MAX must not hide a nonfinite local rate')
  call write_runtime('&timestep cfl_target=0.3 /')
  call initialize_run_control(trim(read_path),MPI_COMM_WORLD,ok,message)
  call require(ok,'runtime startup')
  call select_run_timestep(collect,MPI_COMM_WORLD,0.0_real_kind,dt,ok,message)
  call require(ok.and.abs(dt-0.03_real_kind)<64*epsilon(dt),'runtime initial global bound')

  if(rank==0) then
    open(newunit=unit,file=trim(filename),status='replace')
    write(unit,'(A)') '&run_control max_steps=20, reload_steps=0 /'
    close(unit)
  endif
  call poll_run_control(1_int64,1.0_real_kind,MPI_COMM_WORLD,changed,ok,message,accept_configuration)
  call require(ok.and..not.changed,'poll must honor independent reload cadence')
  call poll_run_control(2_int64,1.0_real_kind,MPI_COMM_WORLD,changed,ok,message,accept_configuration)
  call require(.not.ok.and..not.changed,'invalid runtime update must reject atomically')
  call require(active_run%timestep%cfl_target==0.3_real_kind,'rejection must retain active control')
  call write_runtime('&timestep mode="fixed", fixed_dt=0.02, max_dt=0.001 /')
  call poll_run_control(4_int64,1.0_real_kind,MPI_COMM_WORLD,changed,ok,message,accept_configuration)
  call require(ok.and.changed,'fixed runtime update')
  previous=dt
  call select_run_timestep(collect,MPI_COMM_WORLD,previous,dt,ok,message)
  call require(ok.and.dt==0.02_real_kind.and.collect_count==1,'fixed mode must skip wave scan and caps')
  call write_runtime('&timestep cfl_target=0.3 /')
  call poll_run_control(6_int64,1.0_real_kind,MPI_COMM_WORLD,changed,ok,message,accept_configuration)
  previous=dt
  call select_run_timestep(collect,MPI_COMM_WORLD,previous,dt,ok,message)
  call require(ok.and.changed.and.abs(dt-0.022_real_kind)<64*epsilon(dt),'fixed to auto growth history')
  if(rank==0) then
    open(newunit=unit,file=trim(filename)//'.history',status='replace',access='stream',form='unformatted')
    call run_control_history(unit,.true.,.true.,6_int64,history_status)
    call require(history_status==0,'write adaptive history')
    close(unit)
  endif
  call MPI_Barrier(MPI_COMM_WORLD,ierr)
  ! A fresh startup file with defaults must not erase the saved effective CFL.
  if(rank==0) then
    open(newunit=unit,file=trim(filename),status='replace')
    write(unit,'(A)') '&run_control max_steps=30, reload_steps=2 /'
    write(unit,'(A)') '&logging list_steps=7 /'
    close(unit)
  endif
  call initialize_run_control(trim(read_path),MPI_COMM_WORLD,ok,message)
  open(newunit=unit,file=trim(filename)//'.history',status='old',access='stream',form='unformatted')
  call run_control_history(unit,.false.,.true.,6_int64,history_status)
  close(unit)
  call require(history_status==0.and.active_run%timestep%cfl_target==0.3_real_kind,'restore effective CFL')
  call require(active_run%max_steps==30.and.active_run%list_steps==7,'restart may adjust stop and logging only')
  collect_count=0
  previous=dt
  call select_run_timestep(collect,MPI_COMM_WORLD,previous,dt,ok,message)
  call require(ok.and.dt==previous.and.collect_count==0,'exact first dt_next must not be re-estimated')
  call poll_run_control(8_int64,1.0_real_kind,MPI_COMM_WORLD,changed,ok,message,accept_configuration)
  call require(ok.and..not.changed.and.active_run%timestep%cfl_target==0.3_real_kind, &
    'unchanged startup file must not overwrite exact history')
  if(rank==0) then
    open(newunit=unit,file=trim(filename),status='replace')
    write(unit,'(A)') '&run_control max_steps=21, reload_steps=2 /'
    close(unit)
  endif
  call poll_run_control(10_int64,1.0_real_kind,MPI_COMM_WORLD,changed,ok,message,accept_configuration)
  call require(ok.and.changed.and.active_run%timestep%cfl_target==0.5_real_kind,'new full snapshot uses defaults')
  open(newunit=unit,file=trim(filename)//'.history',status='old',access='stream',form='unformatted')
  call run_control_history(unit,.false.,.false.,6_int64,history_status)
  close(unit)
  call select_run_timestep(collect,MPI_COMM_WORLD,0.0001_real_kind,dt,ok,message)
  call require(history_status==0.and.ok.and.abs(dt-0.05_real_kind)<64*epsilon(dt), &
    'non-exact first dt must discard old growth and CFL history')
  open(newunit=unit,status='scratch',access='stream',form='unformatted')
  call run_control_history(unit,.false.,.false.,6_int64,history_status)
  call require(history_status==0,'old empty control tail permits non-exact migration')
  rewind(unit)
  call run_control_history(unit,.false.,.true.,6_int64,history_status)
  call require(history_status/=0,'old empty control tail cannot claim exact continuation')
  close(unit)
  before=active_run
  if(rank==0) then
    open(newunit=unit,file=trim(filename),status='replace')
    write(unit,'(A)') '&run_control max_steps=99, reload_steps=2 /'
    write(unit,'(A)') '&timestep cfl_target=0.9 /'
    write(unit,'(A)') '&statistics enabled=t,sample_clock="time",sample_time=1e-30 /'
    close(unit)
  endif
  call poll_run_control(12_int64,1.0_real_kind,MPI_COMM_WORLD,changed,ok,message,accept_configuration)
  call require(.not.ok.and..not.changed.and.same_run_control(active_run,before), &
    'unrepresentable statistics interval must reject the whole candidate before commit')
  if(rank==0) then
    open(newunit=unit,file=trim(filename),status='replace')
    write(unit,'(A)') '&run_control max_steps=99, reload_steps=2 /'
    write(unit,'(A)') '&statistics enabled=t,sample_clock="time",sample_time=0.25,'
    write(unit,'(A)') 'write_clock="time",write_time=1e-30 /'
    close(unit)
  endif
  call poll_run_control(14_int64,1.0_real_kind,MPI_COMM_WORLD,changed,ok,message,accept_configuration)
  call require(.not.ok.and..not.changed.and.same_run_control(active_run,before), &
    'unrepresentable writing interval must not commit another candidate field')
  if(rank==0) then
    open(newunit=unit,file=trim(filename),status='replace')
    write(unit,'(A)') '&run_control max_steps=99, reload_steps=2 /'
    write(unit,'(A)') '&statistics enabled=t,sample_clock="time",sample_time=0.25 /'
    close(unit)
  endif
  call poll_run_control(16_int64,1.0_real_kind,MPI_COMM_WORLD,changed,ok,message,accept_configuration)
  call require(ok.and.changed.and.active_run%max_steps==99,'representable candidate must be accepted')
  call poll_mean_statistics(16_int64,1.0_real_kind,sample,writing,new_window,schedule_changed,ok)
  call require(ok.and.sample.and.new_window,'preflight must not consume the initial sample')
  call poll_mean_statistics(18_int64,1.25_real_kind,sample,writing,new_window,schedule_changed,ok)
  call require(ok.and.sample.and..not.new_window,'accepted time clock must emit its next real sample')
  print '(A,I0,2A)', 'RUN_CONTROL_COLLECTIVE_PASS rank=',rank,' ',numerical_precision
  call MPI_Finalize(ierr)
contains
  subroutine accept_configuration(candidate,ok,message)
    type(run_control_options),intent(in) :: candidate
    logical,intent(out) :: ok
    character(*),intent(out) :: message
    ! This probe owns no solver/statistics provider; only the control contract is under test.
    ok=.true.;message=''
  end subroutine

  subroutine write_runtime(timestep_line)
    character(*),intent(in) :: timestep_line
    if(rank==0) then
      open(newunit=unit,file=trim(filename),status='replace')
      write(unit,'(A)') '&run_control max_steps=20, reload_steps=2 /'
      write(unit,'(A)') timestep_line
      close(unit)
    endif
  end subroutine

  subroutine collect(rates,valid)
    real(real_kind),intent(out) :: rates(3)
    logical,intent(out) :: valid
    collect_count=collect_count+1
    rates=[4.0_real_kind,1.0_real_kind,1.0_real_kind]
    if(rank/=0) rates=[1.0_real_kind,5.0_real_kind,1.0_real_kind]
    valid=.true.
  end subroutine

  subroutine require(condition,reason)
    logical,intent(in) :: condition
    character(*),intent(in) :: reason
    integer :: abort_status
    if(condition) return
    print '(A,I0,2A)', 'rank=',rank,' ',trim(reason)
    call MPI_Abort(MPI_COMM_WORLD,1,abort_status)
  end subroutine
end program run_control_collective_probe
