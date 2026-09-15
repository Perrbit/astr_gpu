program chemistry_source_probe
  use iso_fortran_env, only: real64, error_unit
  use ieee_arithmetic, only: ieee_is_finite, ieee_value, ieee_quiet_nan
  use chemistry_air5_data, only: air5_num_species
  use chemistry_model, only: chemistry_status_ok, chemistry_status_invalid_composition, &
    chemistry_status_out_of_domain, chemistry_status_nonfinite, &
    chemistry_status_invalid_source_mode
  use chemistry_thermo, only: air5_ev_from_tv, air5_q5_from_state
  use chemistry_relaxation, only: air5_vt_relaxation_source
  use chemistry_source, only: air5_instantaneous_source, &
    air5_instantaneous_source_jacobian, air5_centered_difference_jacobian, &
    air5_admissible_difference_jacobian, air5_source_conservation_residuals, &
    air5_source_mode_coupled, air5_source_mode_chemical, air5_source_mode_vt
  implicit none

  character(len=32) :: mode

  call get_command_argument(1, mode)
  select case (trim(mode))
  case ('sources')
    call source_probe
  case ('jacobian')
    call jacobian_probe
  case ('boundary_jacobian')
    call boundary_jacobian_probe
  case ('invalid')
    call invalid_probe
  case ('components')
    call component_probe
  case ('invalid_mode')
    call invalid_mode_probe
  case default
    error stop 'unknown chemistry source probe mode'
  end select

contains

  subroutine make_state(case_index, rho, momentum, q5, state, positive_floor)
    integer, intent(in) :: case_index
    real(real64), intent(out) :: rho, momentum(3), q5, state(6)
    logical, intent(in), optional :: positive_floor
    real(real64), parameter :: density(6) = [1.2_real64, 0.01_real64, &
      0.02_real64, 0.05_real64, 0.01_real64, 0.012_real64]
    real(real64), parameter :: temperature(6) = [500.0_real64, 4000.0_real64, &
      7000.0_real64, 6000.0_real64, 2926.98_real64, 7600.0_real64]
    real(real64), parameter :: tv(6) = [500.0_real64, 4000.0_real64, &
      6000.0_real64, 1000.0_real64, 2926.98_real64, 5000.0_real64]
    real(real64), parameter :: composition(air5_num_species, 6) = reshape([ &
      0.7653_real64, 0.2347_real64, 0.0_real64, 0.0_real64, 0.0_real64, &
      0.7653_real64, 0.2347_real64, 0.0_real64, 0.0_real64, 0.0_real64, &
      0.40_real64, 0.10_real64, 0.20_real64, 0.20_real64, 0.10_real64, &
      0.55_real64, 0.15_real64, 0.10_real64, 0.12_real64, 0.08_real64, &
      0.747076_real64, 0.152197_real64, 1.27988e-5_real64, &
      0.0616899_real64, 0.0390241_real64, &
      0.25_real64, 0.10_real64, 0.25_real64, 0.25_real64, 0.15_real64], &
      [air5_num_species, 6])
    real(real64) :: local_composition(air5_num_species)
    integer :: status

    rho = density(case_index)
    local_composition = composition(:,case_index)
    if (present(positive_floor)) then
      if (positive_floor) local_composition = max(local_composition, 1.0e-4_real64)
    end if
    state(1:5) = rho*local_composition/sum(local_composition)
    momentum = rho*[40.0_real64, -5.0_real64, 2.0_real64]
    call air5_ev_from_tv(state(1:5), tv(case_index), state(6), status)
    if (status /= chemistry_status_ok) error stop 'state Ev construction failed'
    call air5_q5_from_state(rho, momentum, state(1:5), state(6), &
      temperature(case_index), q5, status)
    if (status /= chemistry_status_ok) error stop 'state q5 construction failed'
  end subroutine make_state

  subroutine source_probe
    real(real64) :: rho, momentum(3), q5, state(6), source(6), vt_source
    real(real64) :: mass_residual, nitrogen_residual, oxygen_residual
    real(real64) :: maximum_conservation, maximum_source, equilibrium_vt
    integer :: case_index, status

    maximum_conservation = 0.0_real64
    maximum_source = 0.0_real64
    equilibrium_vt = 0.0_real64
    do case_index = 1, 6
      call make_state(case_index, rho, momentum, q5, state)
      call air5_instantaneous_source(rho, momentum, q5, state, source, status)
      if (status /= chemistry_status_ok) then
        write(error_unit,'(a,i0,a,i0)') 'source case ', case_index, ' status ', status
        error stop 'instantaneous source failed'
      end if
      if (.not. all(ieee_is_finite(source))) error stop 'nonfinite source'
      call air5_source_conservation_residuals(source, mass_residual, &
        nitrogen_residual, oxygen_residual)
      maximum_conservation = max(maximum_conservation, mass_residual, &
        nitrogen_residual, oxygen_residual)
      maximum_source = max(maximum_source, maxval(abs(source)))
      if (case_index == 1 .or. case_index == 2 .or. case_index == 5) then
        call air5_vt_relaxation_source(state(1:5), &
          merge(500.0_real64, merge(4000.0_real64, 2926.98_real64, case_index == 2), &
                case_index == 1), &
          merge(500.0_real64, merge(4000.0_real64, 2926.98_real64, case_index == 2), &
                case_index == 1), vt_source, status)
        if (status /= chemistry_status_ok) error stop 'equilibrium VT source failed'
        equilibrium_vt = max(equilibrium_vt, abs(vt_source))
      end if
    end do
    write(*,'(3(es24.16,1x))') maximum_conservation, maximum_source, equilibrium_vt
  end subroutine source_probe

  subroutine jacobian_probe
    real(real64), parameter :: rtol = 1.0e-6_real64, atol = 1.0e-8_real64
    real(real64) :: rho, momentum(3), q5, state(6), source(6)
    real(real64) :: analytic(6,6), numeric(6,6), scaled_error, maximum_scaled_error
    real(real64) :: worst_analytic, worst_numeric
    integer :: case_index, row, column, status, worst_case, worst_row, worst_column

    maximum_scaled_error = 0.0_real64
    worst_case = 0
    worst_row = 0
    worst_column = 0
    worst_analytic = 0.0_real64
    worst_numeric = 0.0_real64
    do case_index = 1, 6
      call make_state(case_index, rho, momentum, q5, state, positive_floor=.true.)
      call air5_instantaneous_source_jacobian(rho, momentum, q5, state, &
        source, analytic, status)
      if (status /= chemistry_status_ok) error stop 'analytic Jacobian failed'
      call air5_centered_difference_jacobian(rho, momentum, q5, state, numeric, status)
      if (status /= chemistry_status_ok) error stop 'numeric Jacobian failed'
      do column = 1, 6
        do row = 1, 6
          scaled_error = abs(analytic(row,column)-numeric(row,column))/ &
            (atol+rtol*abs(numeric(row,column)))
          if (scaled_error > maximum_scaled_error) then
            maximum_scaled_error = scaled_error
            worst_case = case_index
            worst_row = row
            worst_column = column
            worst_analytic = analytic(row,column)
            worst_numeric = numeric(row,column)
          end if
        end do
      end do
    end do
    write(error_unit,'(a,3(i0,1x),2(es24.16,1x))') 'worst ', worst_case, &
      worst_row, worst_column, worst_analytic, worst_numeric
    write(*,'(es24.16)') maximum_scaled_error
  end subroutine jacobian_probe

  subroutine boundary_jacobian_probe
    real(real64), parameter :: rtol = 1.0e-6_real64, atol = 1.0e-8_real64
    real(real64) :: rho, momentum(3), q5, state(6), source(6)
    real(real64) :: analytic(6,6), numeric(6,6), scaled_error, maximum_scaled_error
    real(real64) :: worst_analytic, worst_numeric
    integer :: case_index, row, column, status, worst_case, worst_row, worst_column

    maximum_scaled_error = 0.0_real64
    worst_case = 0
    worst_row = 0
    worst_column = 0
    worst_analytic = 0.0_real64
    worst_numeric = 0.0_real64
    do case_index = 1, 2
      call make_state(case_index, rho, momentum, q5, state)
      call air5_instantaneous_source_jacobian(rho, momentum, q5, state, &
        source, analytic, status)
      if (status /= chemistry_status_ok) error stop 'boundary analytic Jacobian failed'
      call air5_admissible_difference_jacobian(rho, momentum, q5, state, numeric, status)
      if (status /= chemistry_status_ok) error stop 'boundary numeric Jacobian failed'
      do column = 1, 6
        do row = 1, 6
          scaled_error = abs(analytic(row,column)-numeric(row,column))/ &
            (atol+rtol*abs(numeric(row,column)))
          if (scaled_error > maximum_scaled_error) then
            maximum_scaled_error = scaled_error
            worst_case = case_index
            worst_row = row
            worst_column = column
            worst_analytic = analytic(row,column)
            worst_numeric = numeric(row,column)
          end if
        end do
      end do
    end do
    write(error_unit,'(a,3(i0,1x),2(es24.16,1x))') 'boundary worst ', &
      worst_case, worst_row, worst_column, worst_analytic, worst_numeric
    write(*,'(es24.16)') maximum_scaled_error
  end subroutine boundary_jacobian_probe

  subroutine invalid_probe
    real(real64) :: rho, momentum(3), q5, state(6), source(6)
    real(real64) :: vt_source
    integer :: statuses(5), status

    call make_state(2, rho, momentum, q5, state)
    state(3) = -1.0e-8_real64
    call air5_instantaneous_source(rho, momentum, q5, state, source, statuses(1))
    call make_state(2, rho, momentum, q5, state)
    rho = 1.0e-8_real64
    state(1:5) = rho*[0.7653_real64, 0.2347_real64, 0.0_real64, &
      0.0_real64, 0.0_real64]
    momentum = 0.0_real64
    call air5_ev_from_tv(state(1:5), 4000.0_real64, state(6), status)
    if (status /= chemistry_status_ok) error stop 'low pressure Ev setup failed'
    call air5_q5_from_state(rho, momentum, state(1:5), state(6), &
      4000.0_real64, q5, status)
    if (status /= chemistry_status_ok) error stop 'low pressure q5 setup failed'
    call air5_instantaneous_source(rho, momentum, q5, state, source, statuses(2))
    call make_state(2, rho, momentum, q5, state)
    q5 = q5 + 1.0e12_real64
    call air5_instantaneous_source(rho, momentum, q5, state, source, statuses(3))
    call air5_vt_relaxation_source(state(1:5), 4000.0_real64, 200.0_real64, &
      vt_source, statuses(4))
    call make_state(2, rho, momentum, q5, state)
    state(6) = ieee_value(0.0_real64, ieee_quiet_nan)
    call air5_instantaneous_source(rho, momentum, q5, state, source, statuses(5))
    write(*,'(5(i0,1x))') statuses
    write(*,'(3(i0,1x))') chemistry_status_invalid_composition, &
      chemistry_status_out_of_domain, chemistry_status_nonfinite
  end subroutine invalid_probe

  subroutine component_probe
    real(real64) :: rho, momentum(3), q5, state(6)
    real(real64) :: coupled(6), chemical(6), vt(6), scale, decomposition
    integer :: status

    call make_state(4,rho,momentum,q5,state)
    call air5_instantaneous_source(rho,momentum,q5,state,coupled,status, &
      air5_source_mode_coupled)
    if (status /= chemistry_status_ok) error stop 'coupled source failed'
    call air5_instantaneous_source(rho,momentum,q5,state,chemical,status, &
      air5_source_mode_chemical)
    if (status /= chemistry_status_ok) error stop 'chemical source failed'
    call air5_instantaneous_source(rho,momentum,q5,state,vt,status,air5_source_mode_vt)
    if (status /= chemistry_status_ok) error stop 'VT source failed'
    scale = max(maxval(abs(coupled)),1.0_real64)
    decomposition = maxval(abs(coupled-chemical-vt))/scale

    call make_state(2,rho,momentum,q5,state)
    call air5_instantaneous_source(rho,momentum,q5,state,vt,status,air5_source_mode_vt)
    if (status /= chemistry_status_ok) error stop 'equilibrium VT source failed'
    write(*,'(3(es24.16,1x))') decomposition,maxval(abs(vt(1:5))),abs(vt(6))
  end subroutine component_probe

  subroutine invalid_mode_probe
    real(real64) :: rho, momentum(3), q5, state(6), source(6)
    integer :: status

    call make_state(4,rho,momentum,q5,state)
    call air5_instantaneous_source(rho,momentum,q5,state,source,status,99)
    write(*,'(2(i0,1x))') status,chemistry_status_invalid_source_mode
  end subroutine invalid_mode_probe

end program chemistry_source_probe
