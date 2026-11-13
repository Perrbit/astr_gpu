module run_control_runtime
  use iso_fortran_env, only: int64
  use astr_precision, only: real_kind,clock_kind
  use run_control_config, only: run_control_options,read_run_control_collective,same_run_control, &
    run_finished,pack_control,unpack_control,validate_run_control
  use timestep_control, only: timestep_automatic,choose_timestep,reduce_directional_rates
  use mean_statistics_schedule, only: mean_schedule,reset_mean_schedule,mean_schedule_poll,mean_schedule_file
  implicit none
  private
  public :: active_run,initialize_run_control,poll_run_control,select_run_timestep,run_is_finished
  public :: run_control_history
  public :: report_run_control_timing
  public :: poll_mean_statistics,mean_statistics_history,reset_mean_statistics_history,mean_history_available
  type(run_control_options),save :: active_run,observed_file
  character(:),allocatable,save :: run_file
  integer(int64),save :: last_reload_step=0
  logical,save :: first_choice=.true.,saved_next_dt=.false.
  real(clock_kind),save :: selection_seconds=0.0_clock_kind,reload_seconds=0.0_clock_kind
  real(clock_kind),save :: initial_selection_seconds=0.0_clock_kind
  integer(int64),save :: selection_count=0,reload_count=0
  type(mean_schedule),save :: mean_clock
  logical,save :: mean_history_available=.false.
  abstract interface
    subroutine run_update_validator(candidate,ok,message)
      import :: run_control_options
      type(run_control_options),intent(in) :: candidate
      logical,intent(out) :: ok
      character(*),intent(out) :: message
    end subroutine
    subroutine wave_collector(rates,valid)
      import :: real_kind
      real(real_kind),intent(out) :: rates(3)
      logical,intent(out) :: valid
    end subroutine
  end interface
contains
  subroutine initialize_run_control(filename,communicator,ok,message)
    character(*),intent(in) :: filename
    integer,intent(in) :: communicator
    logical,intent(out) :: ok
    character(*),intent(out) :: message
    call read_run_control_collective(filename,communicator,active_run,ok,message)
    if(.not.ok) return
    run_file=filename
    observed_file=active_run
    last_reload_step=0
    first_choice=.true.
    saved_next_dt=.false.
    selection_seconds=0.0_clock_kind;reload_seconds=0.0_clock_kind
    initial_selection_seconds=0.0_clock_kind
    selection_count=0;reload_count=0
    call reset_mean_statistics_history()
  end subroutine

  subroutine reset_mean_statistics_history()
    call reset_mean_schedule(mean_clock)
    mean_history_available=.false.
  end subroutine

  subroutine poll_mean_statistics(step,time,sample,writing,new_window,changed,ok)
    integer(int64),intent(in) :: step
    real(real_kind),intent(in) :: time
    logical,intent(out) :: sample,writing,new_window,changed,ok
    call mean_schedule_poll(mean_clock,active_run%statistics,step,time,sample,writing,new_window,changed,ok)
  end subroutine

  subroutine mean_statistics_history(unit,writing,exact,step,time,status,source_bytes)
    integer,intent(in) :: unit
    logical,intent(in) :: writing,exact
    integer(int64),intent(in) :: step
    real(real_kind),intent(in) :: time
    integer,intent(out) :: status
    integer,optional,intent(in) :: source_bytes
    integer(int64) :: position
    character(8) :: tag
    logical :: ok
    if(.not.writing) then
      inquire(unit,pos=position)
      tag=''
      read(unit,iostat=status) tag
      read(unit,pos=position)
      if(status/=0.or.tag/='ASTRMS01') then
        ! Approved migration of pre-adaptive batches starts a new statistics window.
        call reset_mean_statistics_history()
        status=merge(1,0,exact)
        return
      endif
    endif
    call mean_schedule_file(unit,writing,mean_clock,step,time,ok,source_bytes)
    status=merge(0,1,ok)
    if(.not.writing) mean_history_available=ok
  end subroutine

  subroutine poll_run_control(step,time,communicator,changed,ok,message,validate)
    use mpi, only: MPI_Wtime
    integer(int64),intent(in) :: step
    real(real_kind),intent(in) :: time
    integer,intent(in) :: communicator
    logical,intent(out) :: changed,ok
    character(*),intent(out) :: message
    procedure(run_update_validator) :: validate
    type(run_control_options) :: candidate
    type(mean_schedule) :: candidate_mean_clock
    logical :: sample,writing,new_window,schedule_changed
    real(clock_kind) :: started
    changed=.false.;ok=.true.;message=''
    if(step-last_reload_step<active_run%reload_steps) return
    last_reload_step=step
    started=MPI_Wtime()
    call read_run_control_collective(run_file,communicator,candidate,ok,message)
    reload_seconds=reload_seconds+(MPI_Wtime()-started)
    reload_count=reload_count+1
    if(.not.ok) return
    ! An unchanged file cannot overwrite exact-restored effective settings.
    if(same_run_control(candidate,observed_file)) return
    call validate(candidate,ok,message)
    if(.not.ok) return
    candidate_mean_clock=mean_clock
    call mean_schedule_poll(candidate_mean_clock,candidate%statistics,step,time, &
      sample,writing,new_window,schedule_changed,ok)
    if(.not.ok) then
      message='statistics schedule cannot represent the requested interval at the current complete-step time'
      return
    endif
    active_run=candidate
    observed_file=candidate
    changed=.true.
  end subroutine

  subroutine select_run_timestep(collect,communicator,previous_dt,dt,ok,message)
    use mpi, only: MPI_Wtime
    procedure(wave_collector) :: collect
    integer,intent(in) :: communicator
    real(real_kind),intent(in) :: previous_dt
    real(real_kind),intent(out) :: dt
    logical,intent(out) :: ok
    character(*),intent(out) :: message
    real(real_kind) :: local(3),global(3)
    logical :: valid
    real(clock_kind) :: started,elapsed
    if(saved_next_dt) then
      dt=previous_dt;ok=.true.;message=''
      saved_next_dt=.false.
      return
    endif
    started=MPI_Wtime()
    global=0.0_real_kind
    if(active_run%timestep%mode==timestep_automatic) then
      call collect(local,valid)
      call reduce_directional_rates(local,valid,communicator,global,ok)
      if(.not.ok) then
        message='invalid physical state or metric in timestep wave collector'
        return
      endif
    endif
    call choose_timestep(active_run%timestep,global,previous_dt,first_choice,dt,ok,message)
    elapsed=MPI_Wtime()-started
    if(first_choice) then
      initial_selection_seconds=elapsed
    else
      selection_seconds=selection_seconds+elapsed
    endif
    selection_count=selection_count+1
    if(ok) first_choice=.false.
  end subroutine

  subroutine report_run_control_timing(communicator)
    use mpi
    integer,intent(in) :: communicator
    real(clock_kind) :: local(3),slowest(3)
    integer :: rank,status,ignored
    local=[initial_selection_seconds,selection_seconds,reload_seconds]
    call MPI_Reduce(local,slowest,3,MPI_DOUBLE_PRECISION,MPI_MAX,0,communicator,status)
    if(status/=MPI_SUCCESS) call MPI_Abort(communicator,status,ignored)
    call MPI_Comm_rank(communicator,rank,status)
    if(status/=MPI_SUCCESS) call MPI_Abort(communicator,status,ignored)
    if(rank==0) write(*,'(A,I0,A,I0,3(A,ES24.16E3))') &
      'ASTR_RUN_CONTROL_TIMING choices=',selection_count,' polls=',reload_count, &
      ' initial_selection_max_seconds=',slowest(1),' next_selection_max_seconds=',slowest(2), &
      ' reload_max_seconds=',slowest(3)
  end subroutine

  logical function run_is_finished(step,time) result(finished)
    integer(int64),intent(in) :: step
    real(real_kind),intent(in) :: time
    finished=run_finished(active_run,step,time)
  end function

  subroutine run_control_history(unit,writing,exact,step,status,source_bytes)
    use astr_precision_io, only: read_source_reals,pack_logical32,unpack_logical32
    use iso_fortran_env, only: int32
    integer,intent(in) :: unit
    logical,intent(in) :: writing,exact
    integer(int64),intent(in) :: step
    integer,intent(out) :: status
    integer,optional,intent(in) :: source_bytes
    integer(int64) :: integers(8,2),reload_step,position
    real(real_kind) :: reals(8,2)
    character(32) :: clocks(2,2)
    character(8) :: tag
    logical :: first,valid
    integer(int32) :: wire_first
    type(run_control_options) :: effective,snapshot
    character(512) :: message
    status=0
    if(writing) then
      call pack_control(active_run,integers(:,1),reals(:,1),clocks(:,1))
      call pack_control(observed_file,integers(:,2),reals(:,2),clocks(:,2))
      write(unit,iostat=status) 'ASTRDT01',integers,reals,clocks,last_reload_step,pack_logical32(first_choice)
      return
    endif
    inquire(unit,pos=position)
    tag=''
    read(unit,iostat=status) tag
    if(status/=0.or.tag/='ASTRDT01') then
      ! Only non-exact migration may start a new history from an older batch.
      read(unit,pos=position,iostat=status)
      if(exact) status=1
      last_reload_step=step
      first_choice=.true.
      return
    endif
    read(unit,iostat=status) integers
    if(status==0) call read_source_reals(unit,reals(:,1),source_bytes,status)
    if(status==0) call read_source_reals(unit,reals(:,2),source_bytes,status)
    if(status==0) read(unit,iostat=status) clocks,reload_step,wire_first
    if(status==0) first=unpack_logical32(wire_first)
    if(status/=0) return
    if(.not.exact) then
      last_reload_step=step
      first_choice=.true.
      return
    endif
    call unpack_control(integers(:,1),reals(:,1),clocks(:,1),effective)
    call unpack_control(integers(:,2),reals(:,2),clocks(:,2),snapshot)
    call validate_run_control(effective,valid,message)
    if(.not.valid.or.reload_step<0.or.reload_step>step) then
      status=1
      return
    endif
    call validate_run_control(snapshot,valid,message)
    if(.not.valid) then
      status=1
      return
    endif
    active_run=effective
    ! Stop targets and ordinary logging may change without changing the trajectory.
    active_run%max_steps=observed_file%max_steps
    active_run%stop_time=observed_file%stop_time
    active_run%list_steps=observed_file%list_steps
    ! Keep the startup file as the polling baseline, not the restored settings.
    last_reload_step=reload_step
    first_choice=first
    saved_next_dt=.true.
  end subroutine
end module run_control_runtime
