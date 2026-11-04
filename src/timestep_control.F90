module timestep_control
  use astr_precision, only: real_kind
  use ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: timestep_options,timestep_automatic,timestep_fixed
  public :: validate_timestep_options,choose_timestep
  public :: reduce_directional_rates

  integer,parameter :: timestep_automatic=1,timestep_fixed=2
  type :: timestep_options
    integer :: mode=timestep_automatic
    real(real_kind) :: cfl_target=0.5_real_kind
    real(real_kind) :: growth_factor=1.1_real_kind
    real(real_kind) :: max_dt=0.0_real_kind
    real(real_kind) :: initial_dt=0.0_real_kind
    real(real_kind) :: fixed_dt=0.0_real_kind
  end type
contains
  pure subroutine validate_timestep_options(options,ok,message)
    type(timestep_options),intent(in) :: options
    logical,intent(out) :: ok
    character(*),intent(out) :: message
    ok=.false.
    message=''
    select case(options%mode)
    case(timestep_fixed)
      if(.not.ieee_is_finite(options%fixed_dt)) then
        message='fixed_dt must be finite'
        return
      endif
      if(options%fixed_dt<=0.0_real_kind) then
        message='fixed_dt must be positive in fixed mode'
        return
      endif
    case(timestep_automatic)
      if(.not.all(ieee_is_finite([options%cfl_target,options%growth_factor, &
          options%max_dt,options%initial_dt]))) then
        message='automatic timestep parameters must be finite'
        return
      endif
      if(options%cfl_target<=0.0_real_kind) then
        message='cfl_target must be positive'
        return
      endif
      if(options%growth_factor<1.0_real_kind) then
        message='growth_factor must be at least one'
        return
      endif
      if(options%max_dt<0.0_real_kind.or.options%initial_dt<0.0_real_kind) then
        message='max_dt and initial_dt must be zero or positive'
        return
      endif
    case default
      message='unknown timestep mode'
      return
    end select
    ok=.true.
  end subroutine validate_timestep_options

  ! Options are validated once when a configuration is accepted.
  pure subroutine choose_timestep(options,rates,previous_dt,first_automatic,dt,ok,message)
    type(timestep_options),intent(in) :: options
    real(real_kind),intent(in) :: rates(3),previous_dt
    logical,intent(in) :: first_automatic
    real(real_kind),intent(out) :: dt
    logical,intent(out) :: ok
    character(*),intent(out) :: message
    real(real_kind) :: rate_sum

    dt=0.0_real_kind
    ok=.false.
    message=''
    select case(options%mode)
    case(timestep_fixed)
      dt=options%fixed_dt
      ok=.true.
      return
    case(timestep_automatic)
      continue
    case default
      message='unknown timestep mode'
      return
    end select
    if(.not.all(ieee_is_finite(rates))) then
      message='nonfinite directional wave rate'
      return
    endif
    if(any(rates<0.0_real_kind)) then
      message='negative directional wave rate'
      return
    endif
    rate_sum=sum(rates)
    if(.not.ieee_is_finite(rate_sum)) then
      message='directional wave-rate sum is nonfinite'
      return
    endif
    if(rate_sum<=0.0_real_kind) then
      message='directional wave-rate sum must be positive'
      return
    endif

    dt=options%cfl_target/rate_sum
    if(options%max_dt>0.0_real_kind) dt=min(dt,options%max_dt)
    if(first_automatic) then
      if(options%initial_dt>0.0_real_kind) dt=min(dt,options%initial_dt)
    else
      if(.not.ieee_is_finite(previous_dt)) then
        message='previous timestep is nonfinite'
        return
      endif
      if(previous_dt<=0.0_real_kind) then
        message='previous timestep must be positive'
        return
      endif
      ! A nonbinding growth product need not be formed and could overflow.
      if(dt/options%growth_factor>previous_dt) dt=options%growth_factor*previous_dt
    endif
    if(.not.ieee_is_finite(dt)) then
      message='selected timestep is nonfinite'
      return
    endif
    if(dt<=0.0_real_kind) then
      message='selected timestep is not representable as a positive value'
      return
    endif
    ok=.true.
  end subroutine choose_timestep

  subroutine reduce_directional_rates(local_rates,local_valid,communicator,rates,ok)
    use mpi
    use astr_precision_mpi, only: mpi_real_type
    real(real_kind),intent(in) :: local_rates(3)
    logical,intent(in) :: local_valid
    integer,intent(in) :: communicator
    real(real_kind),intent(out) :: rates(3)
    logical,intent(out) :: ok
    real(real_kind) :: packet(4),global(4)
    integer :: status,abort_status
    packet=0.0_real_kind
    packet(4)=1.0_real_kind
    if(local_valid) then
      if(all(ieee_is_finite(local_rates))) then
        if(all(local_rates>=0.0_real_kind)) then
          packet(1:3)=local_rates
          packet(4)=0.0_real_kind
        endif
      endif
    endif
    ! One small reduction carries all direction maxima and the invalid-state flag.
    call MPI_Allreduce(packet,global,4,mpi_real_type,MPI_MAX,communicator,status)
    if(status/=MPI_SUCCESS) call MPI_Abort(communicator,status,abort_status)
    rates=global(1:3)
    ok=global(4)==0.0_real_kind
  end subroutine reduce_directional_rates
end module timestep_control
