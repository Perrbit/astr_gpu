program chemistry_thermo_probe
  use iso_fortran_env, only: real64
  use ieee_arithmetic, only: ieee_value, ieee_quiet_nan
  use chemistry_air5_data, only: air5_num_species
  use chemistry_model, only: chemistry_status_ok, chemistry_status_invalid_mechanism, &
    chemistry_status_invalid_density, chemistry_status_invalid_composition, &
    chemistry_status_out_of_domain, chemistry_status_no_vibrational_capacity, &
    chemistry_status_nonfinite, air5_validate_mechanism_id, &
    air5_validate_physical_species_state
  use chemistry_thermo, only: air5_ev_from_tv, air5_tv_from_ev, &
    air5_q5_from_state, air5_temperature_from_q5, air5_temperature_derivatives, &
    air5_tv_derivatives
  implicit none

  character(len=32) :: mode

  call get_command_argument(1, mode)
  select case (trim(mode))
  case ('roundtrip')
    call roundtrip_probe
  case ('invalid')
    call invalid_probe
  case ('mechanism')
    call mechanism_probe
  case ('derivatives')
    call derivative_probe
  case default
    error stop 'unknown chemistry thermo probe mode'
  end select

contains

  subroutine mechanism_probe
    integer :: statuses(2)

    call air5_validate_mechanism_id('air5_kimjo12', statuses(1))
    call air5_validate_mechanism_id('wrong', statuses(2))
    write(*,'(2(i0,1x))') statuses
    write(*,'(2(i0,1x))') chemistry_status_ok, chemistry_status_invalid_mechanism
  end subroutine mechanism_probe

  subroutine roundtrip_probe
    real(real64), parameter :: temperatures(5) = [300.0_real64, 500.0_real64, &
      2200.0_real64, 5000.0_real64, 8000.0_real64]
    real(real64), parameter :: vibrational_temperatures(5) = [300.0_real64, &
      1200.0_real64, 4000.0_real64, 7600.0_real64, 8000.0_real64]
    real(real64), parameter :: densities(5) = [1.2_real64, 0.4_real64, &
      0.01_real64, 0.08_real64, 2.0_real64]
    real(real64), parameter :: compositions(air5_num_species, 5) = reshape([ &
      0.7653_real64, 0.2347_real64, 0.0_real64, 0.0_real64, 0.0_real64, &
      0.60_real64, 0.20_real64, 0.08_real64, 0.06_real64, 0.06_real64, &
      0.40_real64, 0.10_real64, 0.20_real64, 0.20_real64, 0.10_real64, &
      0.15_real64, 0.10_real64, 0.30_real64, 0.30_real64, 0.15_real64, &
      0.05_real64, 0.05_real64, 0.35_real64, 0.35_real64, 0.20_real64], &
      [air5_num_species, 5])
    real(real64) :: rho_species(air5_num_species), momentum(3)
    real(real64) :: ev, ev_back, tv_back, q5, q5_back, temperature_back
    real(real64) :: max_ev_residual, max_tv_residual, max_q5_residual
    integer :: case_index, status

    max_ev_residual = 0.0_real64
    max_tv_residual = 0.0_real64
    max_q5_residual = 0.0_real64
    do case_index = 1, size(temperatures)
      rho_species = densities(case_index)*compositions(:, case_index)
      momentum = densities(case_index)*[32.0_real64*case_index, &
        -7.0_real64*case_index, 2.5_real64*case_index]
      call air5_ev_from_tv(rho_species, vibrational_temperatures(case_index), ev, status)
      if (status /= chemistry_status_ok) error stop 'ev_from_tv failed'
      call air5_tv_from_ev(rho_species, ev, tv_back, status)
      if (status /= chemistry_status_ok) error stop 'tv_from_ev failed'
      call air5_ev_from_tv(rho_species, tv_back, ev_back, status)
      if (status /= chemistry_status_ok) error stop 'ev_from_tv back failed'
      call air5_q5_from_state(densities(case_index), momentum, rho_species, ev, &
        temperatures(case_index), q5, status)
      if (status /= chemistry_status_ok) error stop 'q5_from_state failed'
      call air5_temperature_from_q5(densities(case_index), momentum, rho_species, ev, &
        q5, temperature_back, status)
      if (status /= chemistry_status_ok) error stop 'temperature_from_q5 failed'
      call air5_q5_from_state(densities(case_index), momentum, rho_species, ev, &
        temperature_back, q5_back, status)
      if (status /= chemistry_status_ok) error stop 'q5_from_state back failed'

      max_ev_residual = max(max_ev_residual, abs(ev_back - ev)/max(abs(ev), 1.0_real64))
      max_tv_residual = max(max_tv_residual, &
        abs(tv_back-vibrational_temperatures(case_index))/vibrational_temperatures(case_index))
      max_q5_residual = max(max_q5_residual, abs(q5_back-q5)/max(abs(q5), 1.0_real64))
    end do
    write(*,'(3(es24.16,1x))') max_ev_residual, max_tv_residual, max_q5_residual
  end subroutine roundtrip_probe

  subroutine invalid_probe
    real(real64) :: rho_species(air5_num_species), ev, tv, q5, temperature
    real(real64) :: momentum(3)
    integer :: statuses(9), status

    rho_species = [0.7653_real64, 0.2347_real64, 0.0_real64, 0.0_real64, 0.0_real64]
    momentum = 0.0_real64
    call air5_validate_mechanism_id('wrong', statuses(1))
    call air5_validate_physical_species_state(0.0_real64, rho_species, statuses(2))
    rho_species(1) = -1.0_real64
    call air5_validate_physical_species_state(1.0_real64, rho_species, statuses(3))
    rho_species = [0.7653_real64, 0.2347_real64, 0.0_real64, 0.0_real64, 0.0_real64]
    call air5_ev_from_tv(rho_species, 299.0_real64, ev, statuses(4))
    call air5_ev_from_tv(rho_species, 4000.0_real64, ev, status)
    if (status /= chemistry_status_ok) error stop 'valid ev state failed'
    call air5_tv_from_ev(rho_species, -1.0_real64, tv, statuses(5))
    rho_species = [0.0_real64, 0.0_real64, 0.6_real64, 0.4_real64, 0.0_real64]
    call air5_tv_from_ev(rho_species, 0.0_real64, tv, statuses(6))
    rho_species = [0.7653_real64, 0.2347_real64, 0.0_real64, 0.0_real64, 0.0_real64]
    call air5_q5_from_state(1.0_real64, momentum, rho_species, ev, 4000.0_real64, q5, status)
    if (status /= chemistry_status_ok) error stop 'valid q5 state failed'
    call air5_temperature_from_q5(1.0_real64, momentum, rho_species, ev, &
      q5-1.0e9_real64, temperature, statuses(7))
    rho_species(1) = rho_species(1) + 0.1_real64
    call air5_validate_physical_species_state(1.0_real64, rho_species, statuses(8))
    rho_species = [0.7653_real64, 0.2347_real64, 0.0_real64, 0.0_real64, 0.0_real64]
    rho_species(1) = ieee_value(0.0_real64, ieee_quiet_nan)
    call air5_validate_physical_species_state(1.0_real64, rho_species, statuses(9))
    write(*,'(9(i0,1x))') statuses
    write(*,'(7(i0,1x))') chemistry_status_invalid_mechanism, &
      chemistry_status_invalid_density, chemistry_status_invalid_composition, &
      chemistry_status_out_of_domain, chemistry_status_no_vibrational_capacity, &
      chemistry_status_nonfinite, chemistry_status_ok
  end subroutine invalid_probe

  subroutine derivative_probe
    real(real64), parameter :: rho = 0.09_real64
    real(real64), parameter :: h_scale = 5.0e-6_real64
    real(real64) :: rho_species(air5_num_species), momentum(3)
    real(real64) :: ev, q5, temperature, tv, dt_dz(air5_num_species), dt_dev
    real(real64) :: dtv_dz(air5_num_species), dtv_dev
    real(real64) :: plus_species(air5_num_species), minus_species(air5_num_species)
    real(real64) :: plus, minus, h, numeric, maximum_relative_error
    integer :: index, status

    rho_species = rho*[0.42_real64, 0.12_real64, 0.18_real64, 0.18_real64, 0.10_real64]
    momentum = rho*[180.0_real64, -20.0_real64, 11.0_real64]
    temperature = 5100.0_real64
    tv = 3600.0_real64
    call air5_ev_from_tv(rho_species, tv, ev, status)
    if (status /= chemistry_status_ok) error stop 'ev setup failed'
    call air5_q5_from_state(rho, momentum, rho_species, ev, temperature, q5, status)
    if (status /= chemistry_status_ok) error stop 'q5 setup failed'
    call air5_temperature_derivatives(rho_species, temperature, dt_dz, dt_dev, status)
    if (status /= chemistry_status_ok) error stop 'temperature derivatives failed'
    call air5_tv_derivatives(rho_species, tv, dtv_dz, dtv_dev, status)
    if (status /= chemistry_status_ok) error stop 'tv derivatives failed'

    maximum_relative_error = 0.0_real64
    do index = 1, air5_num_species
      h = h_scale*max(rho_species(index), 1.0e-4_real64)
      plus_species = rho_species
      minus_species = rho_species
      plus_species(index) = plus_species(index) + h
      minus_species(index) = minus_species(index) - h
      call air5_temperature_from_q5(rho, momentum, plus_species, ev, q5, plus, status)
      if (status /= chemistry_status_ok) error stop 'temperature plus failed'
      call air5_temperature_from_q5(rho, momentum, minus_species, ev, q5, minus, status)
      if (status /= chemistry_status_ok) error stop 'temperature minus failed'
      numeric = (plus-minus)/(2.0_real64*h)
      maximum_relative_error = max(maximum_relative_error, &
        abs(numeric-dt_dz(index))/max(abs(numeric), 1.0_real64))
      call air5_tv_from_ev(plus_species, ev, plus, status)
      if (status /= chemistry_status_ok) error stop 'tv plus failed'
      call air5_tv_from_ev(minus_species, ev, minus, status)
      if (status /= chemistry_status_ok) error stop 'tv minus failed'
      numeric = (plus-minus)/(2.0_real64*h)
      maximum_relative_error = max(maximum_relative_error, &
        abs(numeric-dtv_dz(index))/max(abs(numeric), 1.0_real64))
    end do
    h = h_scale*max(ev, 1.0_real64)
    call air5_temperature_from_q5(rho, momentum, rho_species, ev+h, q5, plus, status)
    if (status /= chemistry_status_ok) error stop 'temperature ev plus failed'
    call air5_temperature_from_q5(rho, momentum, rho_species, ev-h, q5, minus, status)
    if (status /= chemistry_status_ok) error stop 'temperature ev minus failed'
    numeric = (plus-minus)/(2.0_real64*h)
    maximum_relative_error = max(maximum_relative_error, &
      abs(numeric-dt_dev)/max(abs(numeric), 1.0_real64))
    call air5_tv_from_ev(rho_species, ev+h, plus, status)
    if (status /= chemistry_status_ok) error stop 'tv ev plus failed'
    call air5_tv_from_ev(rho_species, ev-h, minus, status)
    if (status /= chemistry_status_ok) error stop 'tv ev minus failed'
    numeric = (plus-minus)/(2.0_real64*h)
    maximum_relative_error = max(maximum_relative_error, &
      abs(numeric-dtv_dev)/max(abs(numeric), 1.0_real64))
    write(*,'(es24.16)') maximum_relative_error
  end subroutine derivative_probe

end program chemistry_thermo_probe
