program chemistry_ros2_probe
  use iso_fortran_env, only: real64
  use ieee_arithmetic, only: ieee_value, ieee_quiet_nan
  use chemistry_air5_data, only: air5_num_species, air5_atom_count, air5_molar_mass, &
    air5_ru, air5_cv_tr_factor, air5_formation_energy
  use chemistry_model, only: chemistry_status_ok, chemistry_status_linear_failure, &
    chemistry_status_invalid_timestep, chemistry_status_invalid_tolerance, &
    chemistry_status_step_limit, chemistry_status_out_of_domain
  use chemistry_thermo, only: air5_ev_from_tv, air5_q5_from_state
  use chemistry_source, only: air5_source_mode_chemical, air5_source_mode_vt, &
    air5_instantaneous_source
  use chemistry_linear6, only: air5_lu_factor_6, air5_lu_solve_6
  use chemistry_ros2, only: air5_ros2_gamma, air5_ros2_a21, air5_ros2_c21, &
    air5_ros2_m1, air5_ros2_m2, air5_ros2_e1, air5_ros2_e2, &
    air5_ros2_fixed_step, air5_ros2_advance
  implicit none

  character(len=32) :: mode

  call get_command_argument(1, mode)
  select case (trim(mode))
  case ('coefficients')
    call coefficient_probe
  case ('linear')
    call linear_probe
  case ('scaled_linear')
    call scaled_linear_probe
  case ('fixed_step')
    call fixed_step_probe
  case ('update_terms')
    call update_terms_probe
  case ('compensation_transaction')
    call compensation_transaction_probe
  case ('invalid_step')
    call invalid_step_probe
  case ('adaptive')
    call adaptive_probe
  case ('attempt_limit')
    call attempt_limit_probe
  case ('invalid_control')
    call invalid_control_probe
  case ('fixed_trajectory')
    call fixed_trajectory_probe
  case ('adaptive_trajectory')
    call adaptive_trajectory_probe(.false.)
  case ('compensated_trajectory')
    call adaptive_trajectory_probe(.true.)
  case ('component_trajectory')
    call component_trajectory_probe
  case ('matrix_trajectory')
    call matrix_trajectory_probe
  case ('invalid_state_matrix')
    call invalid_state_matrix_probe
  case ('domain_endpoint_source')
    call domain_endpoint_source_probe
  case ('stiff_fixed_trajectory')
    call stiff_fixed_trajectory_probe
  case default
    error stop 'unknown chemistry ROS2 probe mode'
  end select

contains

  subroutine compensation_transaction_probe
    real(real64) :: rho,momentum(3),q5,state(6),result(6),carry(6),seed(6),atol(6),h
    real(real64) :: candidate(6),estimate(6),terms(6,2),adjusted(6),updated(6),expected(6),low(6)
    integer :: status,accepted,rejected,rhs,jac,part
    call make_state(rho,momentum,q5,state)
    seed = 0.125_real64*epsilon(rho)*state
    atol(1:5) = 1.0e-13_real64*rho
    atol(6) = 1.0e-13_real64*max(abs(state(6)),1.0_real64)
    call air5_ros2_fixed_step(rho,momentum,q5,state,1.0e-15_real64, &
      candidate,estimate,status,update_terms=terms)
    if (status /= 0) error stop 'uncompensated fixture invalid'
    low = seed
    expected = state
    do part = 1,2
      adjusted = terms(:,part)-low
      updated = expected+adjusted
      low = (updated-expected)-adjusted
      expected = updated
    end do
    carry = seed
    call air5_ros2_fixed_step(rho,momentum,q5,state,1.0e-15_real64, &
      candidate,estimate,status,compensation=carry)
    if (status /= 0 .or. any(candidate /= expected) .or. any(carry /= low)) &
      error stop 'fixed-step compensation disagrees with increment reference'
    carry = seed
    call air5_ros2_advance(rho,momentum,q5,state,2.0e-10_real64, &
      2.0e-10_real64,1.0e-8_real64,atol,1,result,h,accepted,rejected,rhs,jac,status, &
      compensation=carry)
    if (status /= chemistry_status_step_limit .or. rejected /= 1 .or. &
        any(result /= state) .or. any(carry /= seed)) error stop 'rejected transaction rollback'
    call air5_ros2_advance(rho,momentum,q5,state,2.0e-10_real64, &
      1.0e-15_real64,1.0e-8_real64,atol,1,result,h,accepted,rejected,rhs,jac,status, &
      compensation=carry)
    if (status /= chemistry_status_step_limit .or. accepted /= 1 .or. &
        any(result /= state) .or. any(carry /= seed)) error stop 'accepted prefix rollback'
    call air5_ros2_advance(rho,momentum,q5,state,2.0e-10_real64, &
      2.0e-10_real64,1.0e-8_real64,atol,200000,result,h,accepted,rejected,rhs,jac,status, &
      compensation=carry)
    if (status /= 0 .or. accepted < 1 .or. rejected < 1) error stop 'adaptive rejection recovery'
    write(*,*) 'COMPENSATION_TRANSACTION_PASS'
  end subroutine compensation_transaction_probe

  subroutine update_terms_probe
    real(real64) :: rho, momentum(3), q5, state(6), plain(6), observed(6)
    real(real64) :: error_plain(6), error_observed(6), terms(6,2), difference
    integer :: status, observed_status
    call make_state(rho,momentum,q5,state)
    call air5_ros2_fixed_step(rho,momentum,q5,state,1.0e-13_real64, &
      plain,error_plain,status)
    call air5_ros2_fixed_step(rho,momentum,q5,state,1.0e-13_real64, &
      observed,error_observed,observed_status,update_terms=terms)
    if (status /= chemistry_status_ok .or. observed_status /= status) &
      error stop 'update terms successful-step status mismatch'
    difference = maxval(abs((state+terms(:,1)+terms(:,2))-observed))
    write(*,'(3(es24.16,1x))') maxval(abs(plain-observed)), &
      maxval(abs(error_plain-error_observed)),difference
    terms = 1.0_real64
    call air5_ros2_fixed_step(rho,momentum,q5,state,-1.0_real64, &
      observed,error_observed,observed_status,update_terms=terms)
    if (observed_status /= chemistry_status_invalid_timestep) &
      error stop 'update terms invalid-step status mismatch'
    write(*,'(2(es24.16,1x))') maxval(abs(terms)),maxval(abs(observed-state))
  end subroutine update_terms_probe

  subroutine coefficient_probe
    write(*,'(7(es24.16,1x))') air5_ros2_gamma, air5_ros2_a21, &
      air5_ros2_c21, air5_ros2_m1, air5_ros2_m2, air5_ros2_e1, air5_ros2_e2
  end subroutine coefficient_probe

  subroutine linear_probe
    real(real64) :: matrix(6,6), original(6,6), lu(6,6), rhs(6), solution(6)
    real(real64) :: expected(6), solution_error, residual
    integer :: pivots(6), status, singular_status, row

    matrix = 0.0_real64
    do row = 1, 6
      matrix(row,row) = real(row+1,real64)
    end do
    matrix(1,1) = 0.0_real64
    matrix(2,1) = 3.0_real64
    matrix(1,2) = -2.0_real64
    matrix(3,2) = 0.5_real64
    matrix(4,3) = -0.75_real64
    matrix(5,4) = 1.25_real64
    matrix(6,5) = -1.5_real64
    original = matrix
    expected = [1.0_real64, -2.0_real64, 3.0_real64, -4.0_real64, &
      5.0_real64, -6.0_real64]
    rhs = matmul(original,expected)
    call air5_lu_factor_6(matrix,lu,pivots,status)
    if (status == chemistry_status_ok) then
      call air5_lu_solve_6(lu,pivots,rhs,solution,status)
    else
      solution = 0.0_real64
    end if
    solution_error = maxval(abs(solution-expected))/maxval(abs(expected))
    residual = maxval(abs(matmul(original,solution)-rhs))/max(maxval(abs(rhs)),1.0_real64)

    matrix = 0.0_real64
    call air5_lu_factor_6(matrix,lu,pivots,singular_status)
    write(*,'(2(es24.16,1x),3(i0,1x))') solution_error, residual, status, &
      singular_status, chemistry_status_linear_failure
  end subroutine linear_probe

  subroutine scaled_linear_probe
    real(real64) :: matrix(6,6), original(6,6), lu(6,6), rhs(6), solution(6)
    real(real64) :: expected(6), solution_error, residual
    integer :: pivots(6), status, row

    matrix = 0.0_real64
    matrix(1,1) = 1.0e20_real64
    do row = 2, 6
      matrix(row,row) = real(row-1,real64)
    end do
    original = matrix
    expected = [1.0_real64,-2.0_real64,3.0_real64,-4.0_real64, &
      5.0_real64,-6.0_real64]
    rhs = matmul(original,expected)
    call air5_lu_factor_6(matrix,lu,pivots,status)
    if (status == chemistry_status_ok) then
      call air5_lu_solve_6(lu,pivots,rhs,solution,status)
    else
      solution = 0.0_real64
    end if
    solution_error = maxval(abs(solution-expected))/maxval(abs(expected))
    residual = maxval(abs(matmul(original,solution)-rhs))/max(maxval(abs(rhs)),1.0_real64)
    write(*,'(2(es24.16,1x),i0)') solution_error,residual,status
  end subroutine scaled_linear_probe

  subroutine make_state(rho, momentum, q5, state)
    real(real64), intent(out) :: rho, momentum(3), q5, state(6)
    real(real64), parameter :: composition(air5_num_species) = [ &
      0.55_real64, 0.15_real64, 0.10_real64, 0.12_real64, 0.08_real64]
    integer :: status

    rho = 0.05_real64
    momentum = rho*[40.0_real64, -5.0_real64, 2.0_real64]
    state(1:air5_num_species) = rho*composition
    call air5_ev_from_tv(state(1:air5_num_species),1000.0_real64,state(6),status)
    if (status /= chemistry_status_ok) error stop 'state Ev construction failed'
    call air5_q5_from_state(rho,momentum,state(1:air5_num_species),state(6), &
      6000.0_real64,q5,status)
    if (status /= chemistry_status_ok) error stop 'state q5 construction failed'
  end subroutine make_state

  subroutine fixed_step_probe
    real(real64), parameter :: step = 1.0e-13_real64
    real(real64) :: rho, momentum(3), q5, state(6), candidate(6), estimate(6)
    real(real64) :: mass_drift, nitrogen_drift, oxygen_drift
    integer :: status

    call make_state(rho,momentum,q5,state)
    call air5_ros2_fixed_step(rho,momentum,q5,state,step,candidate,estimate,status)
    mass_drift = abs(sum(candidate(1:5))-sum(state(1:5)))/rho
    call element_drift(state,candidate,1,nitrogen_drift)
    call element_drift(state,candidate,2,oxygen_drift)
    write(*,'(i0,6(1x,es24.16))') status, minval(candidate), mass_drift, &
      nitrogen_drift, oxygen_drift, maxval(abs(candidate-state)), &
      maxval(abs(estimate))
  end subroutine fixed_step_probe

  subroutine element_drift(initial, final, element, drift)
    real(real64), intent(in) :: initial(6), final(6)
    integer, intent(in) :: element
    real(real64), intent(out) :: drift
    real(real64) :: initial_amount, final_amount
    integer :: species

    initial_amount = 0.0_real64
    final_amount = 0.0_real64
    do species = 1, air5_num_species
      initial_amount = initial_amount + real(air5_atom_count(element,species),real64)* &
        initial(species)/air5_molar_mass(species)
      final_amount = final_amount + real(air5_atom_count(element,species),real64)* &
        final(species)/air5_molar_mass(species)
    end do
    drift = abs(final_amount-initial_amount)/max(abs(initial_amount),1.0_real64)
  end subroutine element_drift

  subroutine invalid_step_probe
    real(real64) :: rho, momentum(3), q5, state(6), candidate(6), estimate(6), nan
    integer :: statuses(3)

    call make_state(rho,momentum,q5,state)
    nan = ieee_value(0.0_real64,ieee_quiet_nan)
    call air5_ros2_fixed_step(rho,momentum,q5,state,0.0_real64, &
      candidate,estimate,statuses(1))
    call air5_ros2_fixed_step(rho,momentum,q5,state,-1.0_real64, &
      candidate,estimate,statuses(2))
    call air5_ros2_fixed_step(rho,momentum,q5,state,nan, &
      candidate,estimate,statuses(3))
    write(*,'(3(i0,1x))') statuses
    write(*,'(3(i0,1x))') chemistry_status_invalid_timestep, &
      chemistry_status_invalid_timestep, chemistry_status_invalid_timestep
  end subroutine invalid_step_probe

  subroutine adaptive_probe
    real(real64) :: rho, momentum(3), q5, state(6), final_state(6), atol(6)
    real(real64) :: suggested_step, mass_drift, nitrogen_drift, oxygen_drift
    integer :: accepted, rejected, rhs_calls, jacobian_calls, status

    call make_state(rho,momentum,q5,state)
    atol(1:5) = rho*1.0e-12_real64
    atol(6) = max(abs(state(6)),1.0_real64)*1.0e-12_real64
    call air5_ros2_advance(rho,momentum,q5,state,2.0e-13_real64, &
      2.0e-13_real64,1.0e-6_real64,atol,10000,final_state,suggested_step, &
      accepted,rejected,rhs_calls,jacobian_calls,status)
    mass_drift = abs(sum(final_state(1:5))-sum(state(1:5)))/rho
    call element_drift(state,final_state,1,nitrogen_drift)
    call element_drift(state,final_state,2,oxygen_drift)
    write(*,'(5(i0,1x),6(es24.16,1x))') status, accepted, rejected, rhs_calls, &
      jacobian_calls, suggested_step, minval(final_state), mass_drift, &
      nitrogen_drift, oxygen_drift, maxval(abs(final_state-state))
  end subroutine adaptive_probe

  subroutine attempt_limit_probe
    real(real64) :: rho, momentum(3), q5, state(6), final_state(6), atol(6)
    real(real64) :: suggested_step
    integer :: accepted, rejected, rhs_calls, jacobian_calls, status

    call make_state(rho,momentum,q5,state)
    atol = 1.0e-30_real64
    call air5_ros2_advance(rho,momentum,q5,state,1.0e-13_real64, &
      1.0e-13_real64,1.0e-15_real64,atol,1,final_state,suggested_step, &
      accepted,rejected,rhs_calls,jacobian_calls,status)
    write(*,'(4(i0,1x),es24.16)') status, chemistry_status_step_limit, &
      accepted, rejected, maxval(abs(final_state-state))
  end subroutine attempt_limit_probe

  subroutine invalid_control_probe
    real(real64) :: rho, momentum(3), q5, state(6), final_state(6), atol(6)
    real(real64) :: suggested_step, nan
    integer :: accepted, rejected, rhs_calls, jacobian_calls, statuses(5)

    call make_state(rho,momentum,q5,state)
    atol = 1.0e-12_real64
    nan = ieee_value(0.0_real64,ieee_quiet_nan)
    call air5_ros2_advance(rho,momentum,q5,state,0.0_real64,1.0e-13_real64, &
      1.0e-6_real64,atol,100,final_state,suggested_step,accepted,rejected, &
      rhs_calls,jacobian_calls,statuses(1))
    call air5_ros2_advance(rho,momentum,q5,state,1.0e-13_real64,0.0_real64, &
      1.0e-6_real64,atol,100,final_state,suggested_step,accepted,rejected, &
      rhs_calls,jacobian_calls,statuses(2))
    call air5_ros2_advance(rho,momentum,q5,state,1.0e-13_real64,1.0e-13_real64, &
      0.0_real64,atol,100,final_state,suggested_step,accepted,rejected, &
      rhs_calls,jacobian_calls,statuses(3))
    atol(1) = nan
    call air5_ros2_advance(rho,momentum,q5,state,1.0e-13_real64,1.0e-13_real64, &
      1.0e-6_real64,atol,100,final_state,suggested_step,accepted,rejected, &
      rhs_calls,jacobian_calls,statuses(4))
    atol = 1.0e-12_real64
    call air5_ros2_advance(rho,momentum,q5,state,1.0e-13_real64,1.0e-13_real64, &
      1.0e-6_real64,atol,0,final_state,suggested_step,accepted,rejected, &
      rhs_calls,jacobian_calls,statuses(5))
    write(*,'(5(i0,1x))') statuses
    write(*,'(5(i0,1x))') chemistry_status_invalid_timestep, &
      chemistry_status_invalid_timestep, chemistry_status_invalid_tolerance, &
      chemistry_status_invalid_tolerance, chemistry_status_step_limit
  end subroutine invalid_control_probe

  subroutine fixed_trajectory_probe
    real(real64), parameter :: duration = 1.0e-3_real64
    real(real64) :: rho, momentum(3), q5, state(6), candidate(6), estimate(6)
    real(real64) :: step
    character(len=32) :: argument
    integer :: count, index, status

    call get_command_argument(2,argument)
    read(argument,*) count
    if (count <= 0) error stop 'fixed trajectory step count must be positive'
    rho = 0.01_real64
    momentum = rho*[40.0_real64,-5.0_real64,2.0_real64]
    state(1:5) = rho*[0.7653_real64,0.2347_real64,0.0_real64,0.0_real64,0.0_real64]
    call air5_ev_from_tv(state(1:5),4000.0_real64,state(6),status)
    if (status /= chemistry_status_ok) error stop 'trajectory Ev construction failed'
    call air5_q5_from_state(rho,momentum,state(1:5),state(6), &
      4000.0_real64,q5,status)
    if (status /= chemistry_status_ok) error stop 'trajectory q5 construction failed'
    step = duration/real(count,real64)
    do index = 1, count
      call air5_ros2_fixed_step(rho,momentum,q5,state,step,candidate,estimate,status)
      if (status /= chemistry_status_ok) exit
      state = candidate
    end do
    write(*,'(i0,1x,6(es24.16,1x))') status,state
  end subroutine fixed_trajectory_probe

  subroutine adaptive_trajectory_probe(compensated)
    logical, intent(in) :: compensated
    real(real64) :: rho, momentum(3), q5, state(6), final_state(6), atol(6)
    real(real64) :: suggested_step,carry(6)
    integer :: accepted, rejected, rhs_calls, jacobian_calls, status

    call make_state(rho,momentum,q5,state)
    atol(1:5) = rho*1.0e-13_real64
    atol(6) = max(abs(state(6)),1.0_real64)*1.0e-13_real64
    if (compensated) then
      carry = 0.0_real64
      call air5_ros2_advance(rho,momentum,q5,state,2.0e-8_real64, &
        2.0e-8_real64,1.0e-9_real64,atol,100000,final_state,suggested_step, &
        accepted,rejected,rhs_calls,jacobian_calls,status,compensation=carry)
    else
      call air5_ros2_advance(rho,momentum,q5,state,2.0e-8_real64, &
        2.0e-8_real64,1.0e-9_real64,atol,100000,final_state,suggested_step, &
        accepted,rejected,rhs_calls,jacobian_calls,status)
    end if
    write(*,'(5(i0,1x))') status,accepted,rejected,rhs_calls,jacobian_calls
    write(*,'(6(es24.16,1x))') final_state
  end subroutine adaptive_trajectory_probe

  subroutine component_trajectory_probe
    real(real64) :: rho, momentum(3), q5, state(6), final_state(6), atol(6)
    real(real64) :: suggested_step
    character(len=32) :: argument
    integer :: accepted, rejected, rhs_calls, jacobian_calls, status, source_mode

    call get_command_argument(2,argument)
    select case (trim(argument))
    case ('chemical')
      source_mode = air5_source_mode_chemical
    case ('vt')
      source_mode = air5_source_mode_vt
    case default
      error stop 'unknown component trajectory source mode'
    end select
    call make_state(rho,momentum,q5,state)
    atol(1:5) = rho*1.0e-13_real64
    atol(6) = max(abs(state(6)),1.0_real64)*1.0e-13_real64
    call air5_ros2_advance(rho,momentum,q5,state,2.0e-8_real64, &
      2.0e-8_real64,1.0e-9_real64,atol,100000,final_state,suggested_step, &
      accepted,rejected,rhs_calls,jacobian_calls,status,source_mode)
    write(*,'(5(i0,1x))') status,accepted,rejected,rhs_calls,jacobian_calls
    write(*,'(6(es24.16,1x))') final_state
  end subroutine component_trajectory_probe

  subroutine make_parameter_state(temperature, tv, pressure, composition_index, &
      rho, momentum, q5, state)
    real(real64), intent(in) :: temperature, tv, pressure
    integer, intent(in) :: composition_index
    real(real64), intent(out) :: rho, momentum(3), q5, state(6)
    real(real64), parameter :: composition(air5_num_species,2) = reshape([ &
      0.7653_real64,0.2347_real64,0.0_real64,0.0_real64,0.0_real64, &
      0.55_real64,0.15_real64,0.10_real64,0.12_real64,0.08_real64], &
      [air5_num_species,2])
    real(real64) :: mixture_gas_constant
    integer :: species, local_status

    if (composition_index < 1 .or. composition_index > 2) then
      error stop 'invalid matrix composition index'
    end if
    mixture_gas_constant = 0.0_real64
    do species = 1, air5_num_species
      mixture_gas_constant = mixture_gas_constant + &
        composition(species,composition_index)*air5_ru/air5_molar_mass(species)
    end do
    rho = pressure/(mixture_gas_constant*temperature)
    momentum = rho*[40.0_real64,-5.0_real64,2.0_real64]
    state(1:air5_num_species) = rho*composition(:,composition_index)
    call air5_ev_from_tv(state(1:air5_num_species),tv,state(6),local_status)
    if (local_status /= chemistry_status_ok) error stop 'matrix Ev construction failed'
    call air5_q5_from_state(rho,momentum,state(1:air5_num_species),state(6), &
      temperature,q5,local_status)
    if (local_status /= chemistry_status_ok) error stop 'matrix q5 construction failed'
  end subroutine make_parameter_state

  subroutine matrix_trajectory_probe
    real(real64) :: temperature, tv, pressure, duration
    real(real64) :: rho, momentum(3), q5, state(6), final_state(6), atol(6)
    real(real64) :: suggested_step
    character(len=32) :: argument
    integer :: composition_index, accepted, rejected, rhs_calls, jacobian_calls, status

    call get_command_argument(2,argument)
    read(argument,*) temperature
    call get_command_argument(3,argument)
    read(argument,*) tv
    call get_command_argument(4,argument)
    read(argument,*) pressure
    call get_command_argument(5,argument)
    read(argument,*) duration
    call get_command_argument(6,argument)
    read(argument,*) composition_index
    call make_parameter_state(temperature,tv,pressure,composition_index, &
      rho,momentum,q5,state)
    atol(1:5) = rho*1.0e-13_real64
    atol(6) = max(abs(state(6)),1.0_real64)*1.0e-13_real64
    call air5_ros2_advance(rho,momentum,q5,state,duration,duration, &
      1.0e-9_real64,atol,200000,final_state,suggested_step,accepted,rejected, &
      rhs_calls,jacobian_calls,status)
    write(*,'(5(i0,1x),5(es26.16e3,1x))') status,accepted,rejected,rhs_calls, &
      jacobian_calls,rho,momentum,q5
    write(*,'(6(es26.16e3,1x))') state
    write(*,'(6(es26.16e3,1x))') final_state
  end subroutine matrix_trajectory_probe

  pure function raw_q5(rho,momentum,state,temperature) result(q5)
    real(real64), intent(in) :: rho, momentum(3), state(6), temperature
    real(real64) :: q5, cv_density, formation_density
    integer :: species

    cv_density = 0.0_real64
    formation_density = 0.0_real64
    do species = 1, air5_num_species
      cv_density = cv_density + state(species)*air5_cv_tr_factor(species)* &
        air5_ru/air5_molar_mass(species)
      formation_density = formation_density + &
        state(species)*air5_formation_energy(species)
    end do
    q5 = dot_product(momentum,momentum)/(2.0_real64*rho) + &
      cv_density*temperature + state(6) + formation_density
  end function raw_q5

  subroutine invalid_state_matrix_probe
    real(real64), parameter :: invalid_temperature(4) = [ &
      200.0_real64,10000.0_real64,1000.0_real64,1000.0_real64]
    real(real64), parameter :: invalid_pressure(4) = [ &
      1.0e5_real64,1.0e5_real64,1.0e2_real64,1.0e7_real64]
    real(real64) :: rho, momentum(3), q5, state(6), final_state(6), atol(6)
    real(real64) :: suggested_step, mixture_gas_constant
    real(real64), parameter :: composition(air5_num_species) = [ &
      0.55_real64,0.15_real64,0.10_real64,0.12_real64,0.08_real64]
    integer :: case_index, species, accepted, rejected, rhs_calls, jacobian_calls
    integer :: status, setup_status

    mixture_gas_constant = 0.0_real64
    do species = 1, air5_num_species
      mixture_gas_constant = mixture_gas_constant + &
        composition(species)*air5_ru/air5_molar_mass(species)
    end do
    do case_index = 1, 4
      rho = invalid_pressure(case_index)/(mixture_gas_constant* &
        invalid_temperature(case_index))
      momentum = rho*[40.0_real64,-5.0_real64,2.0_real64]
      state(1:5) = rho*composition
      call air5_ev_from_tv(state(1:5),1000.0_real64,state(6),setup_status)
      if (setup_status /= chemistry_status_ok) error stop 'invalid state Ev setup failed'
      q5 = raw_q5(rho,momentum,state,invalid_temperature(case_index))
      atol(1:5) = rho*1.0e-13_real64
      atol(6) = max(abs(state(6)),1.0_real64)*1.0e-13_real64
      call air5_ros2_advance(rho,momentum,q5,state,1.0e-10_real64, &
        1.0e-10_real64,1.0e-9_real64,atol,100,final_state,suggested_step, &
        accepted,rejected,rhs_calls,jacobian_calls,status)
      write(*,'(7(i0,1x),es24.16)') status,chemistry_status_out_of_domain, &
        accepted,rejected,rhs_calls,jacobian_calls,case_index, &
        maxval(abs(final_state-state))
    end do
  end subroutine invalid_state_matrix_probe

  subroutine domain_endpoint_source_probe
    real(real64), parameter :: temperature(4) = [ &
      300.0_real64,8000.0_real64,3000.0_real64,8000.0_real64]
    real(real64), parameter :: tv(4) = [ &
      300.0_real64,1000.0_real64,6000.0_real64,8000.0_real64]
    real(real64), parameter :: pressure(4) = [ &
      1.0e3_real64,1.0e6_real64,1.0e6_real64,1.0e6_real64]
    real(real64) :: rho, momentum(3), q5, state(6), source(6)
    integer :: case_index, composition_index, status

    do case_index = 1, 4
      composition_index = merge(1,2,case_index == 1)
      call make_parameter_state(temperature(case_index),tv(case_index), &
        pressure(case_index),composition_index,rho,momentum,q5,state)
      call air5_instantaneous_source(rho,momentum,q5,state,source,status)
      write(*,'(2(i0,1x),es26.16e3)') status,case_index,maxval(abs(source))
    end do
  end subroutine domain_endpoint_source_probe

  subroutine stiff_fixed_trajectory_probe
    real(real64), parameter :: duration = 2.0e-8_real64
    real(real64) :: rho, momentum(3), q5, state(6), candidate(6), estimate(6), step
    character(len=32) :: argument
    integer :: count, index, status

    call get_command_argument(2,argument)
    read(argument,*) count
    if (count <= 0) error stop 'stiff fixed trajectory step count must be positive'
    call make_state(rho,momentum,q5,state)
    step = duration/real(count,real64)
    do index = 1, count
      call air5_ros2_fixed_step(rho,momentum,q5,state,step,candidate,estimate,status)
      if (status /= chemistry_status_ok) exit
      state = candidate
    end do
    write(*,'(i0,1x,6(es26.16e3,1x))') status,state
  end subroutine stiff_fixed_trajectory_probe

end program chemistry_ros2_probe
