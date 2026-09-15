module chemistry_ros2
  use iso_fortran_env, only: real64
  use ieee_arithmetic, only: ieee_is_finite
  use chemistry_air5_data, only: air5_num_species
  use chemistry_model, only: chemistry_status_ok, chemistry_status_out_of_domain, &
    chemistry_status_nonfinite, chemistry_status_invalid_timestep, &
    chemistry_status_invalid_tolerance, chemistry_status_step_limit, &
    air5_validate_physical_species_state, air5_pressure_is_in_domain
  use chemistry_thermo, only: air5_temperature_from_q5, air5_tv_from_ev, air5_pressure
  use chemistry_source, only: air5_instantaneous_source, &
    air5_instantaneous_source_jacobian, air5_source_mode_coupled
  use chemistry_linear6, only: air5_lu_factor_6, air5_lu_solve_6
  implicit none
  private

  real(real64), parameter, public :: air5_ros2_gamma = &
    1.0_real64+1.0_real64/sqrt(2.0_real64)
  real(real64), parameter, public :: air5_ros2_a21 = 1.0_real64/air5_ros2_gamma
  real(real64), parameter, public :: air5_ros2_c21 = -2.0_real64/air5_ros2_gamma
  real(real64), parameter, public :: air5_ros2_m1 = &
    3.0_real64/(2.0_real64*air5_ros2_gamma)
  real(real64), parameter, public :: air5_ros2_m2 = &
    1.0_real64/(2.0_real64*air5_ros2_gamma)
  real(real64), parameter, public :: air5_ros2_e1 = air5_ros2_m2
  real(real64), parameter, public :: air5_ros2_e2 = air5_ros2_m2

  public :: air5_ros2_fixed_step
  public :: air5_ros2_advance

contains

  pure subroutine air5_ros2_fixed_step(rho, momentum, q5, state, step, &
      candidate, error_estimate, status, retryable, rhs_evaluations, &
      jacobian_evaluations, source_mode)
    real(real64), intent(in) :: rho, momentum(3), q5, state(6), step
    real(real64), intent(out) :: candidate(6), error_estimate(6)
    integer, intent(out) :: status
    logical, intent(out), optional :: retryable
    integer, intent(out), optional :: rhs_evaluations, jacobian_evaluations
    integer, intent(in), optional :: source_mode
    real(real64) :: source(6), jacobian(6,6), matrix(6,6), lu(6,6)
    real(real64) :: stage_state(6), stage_source(6), k1(6), k2(6), rhs(6)
    integer :: pivots(6), index, active_mode

    candidate = state
    error_estimate = 0.0_real64
    if (present(retryable)) retryable = .false.
    if (present(rhs_evaluations)) rhs_evaluations = 0
    if (present(jacobian_evaluations)) jacobian_evaluations = 0
    active_mode = air5_source_mode_coupled
    if (present(source_mode)) active_mode = source_mode
    if (.not. ieee_is_finite(step)) then
      status = chemistry_status_invalid_timestep
      return
    end if
    if (step <= 0.0_real64) then
      status = chemistry_status_invalid_timestep
      return
    end if
    call air5_instantaneous_source_jacobian(rho,momentum,q5,state,source, &
      jacobian,status,active_mode)
    if (present(rhs_evaluations)) rhs_evaluations = 1
    if (present(jacobian_evaluations)) jacobian_evaluations = 1
    if (status /= chemistry_status_ok) return
    if (present(retryable)) retryable = .true.

    matrix = -step*air5_ros2_gamma*jacobian
    do index = 1, 6
      matrix(index,index) = matrix(index,index)+1.0_real64
    end do
    call air5_lu_factor_6(matrix,lu,pivots,status)
    if (status /= chemistry_status_ok) return

    rhs = step*air5_ros2_gamma*source
    call air5_lu_solve_6(lu,pivots,rhs,k1,status)
    if (status /= chemistry_status_ok) return
    stage_state = state+air5_ros2_a21*k1
    call air5_instantaneous_source(rho,momentum,q5,stage_state,stage_source,status, &
      active_mode)
    if (present(rhs_evaluations)) rhs_evaluations = 2
    if (status /= chemistry_status_ok) return

    rhs = step*air5_ros2_gamma*stage_source + &
      air5_ros2_gamma*air5_ros2_c21*k1
    call air5_lu_solve_6(lu,pivots,rhs,k2,status)
    if (status /= chemistry_status_ok) return
    candidate = state+air5_ros2_m1*k1+air5_ros2_m2*k2
    error_estimate = air5_ros2_e1*k1+air5_ros2_e2*k2
    if (.not. all(ieee_is_finite(candidate)) .or. &
        .not. all(ieee_is_finite(error_estimate))) then
      candidate = state
      error_estimate = 0.0_real64
      status = chemistry_status_nonfinite
      return
    end if
    call air5_validate_ros2_candidate(rho,momentum,q5,candidate,status)
    if (status /= chemistry_status_ok) then
      candidate = state
      error_estimate = 0.0_real64
    end if
  end subroutine air5_ros2_fixed_step

  pure subroutine air5_ros2_advance(rho, momentum, q5, state, duration, &
      initial_step, rtol, atol, max_attempts, final_state, suggested_step, &
      accepted_steps, rejected_steps, rhs_evaluations, jacobian_evaluations, status, &
      source_mode)
    real(real64), intent(in) :: rho, momentum(3), q5, state(6), duration
    real(real64), intent(in) :: initial_step, rtol, atol(6)
    integer, intent(in) :: max_attempts
    real(real64), intent(out) :: final_state(6), suggested_step
    integer, intent(out) :: accepted_steps, rejected_steps
    integer, intent(out) :: rhs_evaluations, jacobian_evaluations, status
    integer, intent(in), optional :: source_mode
    real(real64), parameter :: safety = 0.9_real64
    real(real64), parameter :: minimum_factor = 0.2_real64
    real(real64), parameter :: maximum_factor = 5.0_real64
    real(real64) :: current(6), candidate(6), error_estimate(6)
    real(real64) :: elapsed, step, remaining, error_norm, factor
    integer :: attempt, attempt_rhs, attempt_jac, attempt_status, active_mode
    logical :: retryable, previous_rejected

    final_state = state
    suggested_step = 0.0_real64
    accepted_steps = 0
    rejected_steps = 0
    rhs_evaluations = 0
    jacobian_evaluations = 0
    status = chemistry_status_ok
    active_mode = air5_source_mode_coupled
    if (present(source_mode)) active_mode = source_mode
    if (.not. ieee_is_finite(duration)) then
      status = chemistry_status_invalid_timestep
      return
    end if
    if (.not. ieee_is_finite(initial_step)) then
      status = chemistry_status_invalid_timestep
      return
    end if
    if (duration <= 0.0_real64 .or. initial_step <= 0.0_real64) then
      status = chemistry_status_invalid_timestep
      return
    end if
    if (.not. ieee_is_finite(rtol)) then
      status = chemistry_status_invalid_tolerance
      return
    end if
    if (.not. all(ieee_is_finite(atol))) then
      status = chemistry_status_invalid_tolerance
      return
    end if
    if (rtol <= 0.0_real64 .or. any(atol <= 0.0_real64)) then
      status = chemistry_status_invalid_tolerance
      return
    end if
    if (max_attempts <= 0) then
      status = chemistry_status_step_limit
      return
    end if
    call air5_validate_ros2_candidate(rho,momentum,q5,state,status)
    if (status /= chemistry_status_ok) return

    current = state
    elapsed = 0.0_real64
    step = min(initial_step,duration)
    previous_rejected = .false.
    do attempt = 1, max_attempts
      remaining = duration-elapsed
      step = min(step,remaining)
      if (step <= 0.0_real64 .or. elapsed+step == elapsed) then
        final_state = state
        suggested_step = step
        status = chemistry_status_step_limit
        return
      end if
      call air5_ros2_fixed_step(rho,momentum,q5,current,step,candidate, &
        error_estimate,attempt_status,retryable,attempt_rhs,attempt_jac,active_mode)
      rhs_evaluations = rhs_evaluations+attempt_rhs
      jacobian_evaluations = jacobian_evaluations+attempt_jac
      if (attempt_status == chemistry_status_ok) then
        call air5_ros2_weighted_error_norm(current,candidate,error_estimate, &
          rtol,atol,error_norm)
        if (error_norm <= 1.0_real64) then
          current = candidate
          elapsed = elapsed+step
          accepted_steps = accepted_steps+1
          if (error_norm == 0.0_real64) then
            factor = maximum_factor
          else
            factor = min(maximum_factor,max(minimum_factor, &
              safety/sqrt(error_norm)))
          end if
          if (previous_rejected) factor = min(1.0_real64,factor)
          suggested_step = step*factor
          previous_rejected = .false.
          if (elapsed >= duration) then
            final_state = current
            status = chemistry_status_ok
            return
          end if
          step = min(suggested_step,duration-elapsed)
          cycle
        end if
        if (ieee_is_finite(error_norm)) then
          factor = max(minimum_factor,min(1.0_real64,safety/sqrt(error_norm)))
        else
          factor = minimum_factor
        end if
        step = step*factor
      else if (retryable) then
        step = 0.5_real64*step
      else
        final_state = state
        suggested_step = step
        status = attempt_status
        return
      end if
      rejected_steps = rejected_steps+1
      suggested_step = step
      previous_rejected = .true.
    end do

    final_state = state
    status = chemistry_status_step_limit
  end subroutine air5_ros2_advance

  pure subroutine air5_ros2_weighted_error_norm(state, candidate, estimate, &
      rtol, atol, error_norm)
    real(real64), intent(in) :: state(6), candidate(6), estimate(6), rtol, atol(6)
    real(real64), intent(out) :: error_norm
    real(real64) :: scaled(6), maximum_scaled

    scaled = abs(estimate)/(atol+rtol*max(abs(state),abs(candidate)))
    maximum_scaled = maxval(scaled)
    if (.not. ieee_is_finite(maximum_scaled)) then
      error_norm = huge(1.0_real64)
    else if (maximum_scaled == 0.0_real64) then
      error_norm = 0.0_real64
    else
      error_norm = maximum_scaled*sqrt(sum((scaled/maximum_scaled)**2)/6.0_real64)
    end if
  end subroutine air5_ros2_weighted_error_norm

  pure subroutine air5_validate_ros2_candidate(rho, momentum, q5, state, status)
    real(real64), intent(in) :: rho, momentum(3), q5, state(6)
    integer, intent(out) :: status
    real(real64) :: temperature, tv, pressure

    call air5_validate_physical_species_state(rho,state(1:air5_num_species),status)
    if (status /= chemistry_status_ok) return
    if (.not. ieee_is_finite(state(6))) then
      status = chemistry_status_nonfinite
      return
    end if
    if (state(6) < 0.0_real64) then
      status = chemistry_status_out_of_domain
      return
    end if
    call air5_temperature_from_q5(rho,momentum,state(1:air5_num_species), &
      state(6),q5,temperature,status)
    if (status /= chemistry_status_ok) return
    call air5_tv_from_ev(state(1:air5_num_species),state(6),tv,status)
    if (status /= chemistry_status_ok) return
    call air5_pressure(state(1:air5_num_species),temperature,pressure,status)
    if (status /= chemistry_status_ok) return
    if (.not. air5_pressure_is_in_domain(pressure)) then
      status = chemistry_status_out_of_domain
    else
      status = chemistry_status_ok
    end if
  end subroutine air5_validate_ros2_candidate

end module chemistry_ros2
