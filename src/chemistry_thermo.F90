module chemistry_thermo
  use iso_fortran_env, only: real64
  use ieee_arithmetic, only: ieee_is_finite
  use chemistry_air5_data, only: air5_num_species, air5_molar_mass, &
    air5_formation_energy, air5_cv_tr_factor, air5_theta_v, air5_ru, &
    air5_temperature_min_k, air5_temperature_max_k
  use chemistry_model, only: chemistry_status_ok, chemistry_status_invalid_density, &
    chemistry_status_invalid_composition, chemistry_status_out_of_domain, &
    chemistry_status_no_vibrational_capacity, chemistry_status_nonconverged, &
    chemistry_status_nonfinite, air5_validate_partial_densities
  implicit none
  private

  public :: air5_species_gas_constant
  public :: air5_species_cv_tr
  public :: air5_species_vibrational_energy
  public :: air5_species_vibrational_cv
  public :: air5_ev_from_tv
  public :: air5_tv_from_ev
  public :: air5_q5_from_state
  public :: air5_temperature_from_q5
  public :: air5_temperature_derivatives
  public :: air5_tv_derivatives
  public :: air5_pressure

contains

  pure function air5_species_gas_constant(species_index) result(gas_constant)
    integer, intent(in) :: species_index
    real(real64) :: gas_constant

    gas_constant = air5_ru/air5_molar_mass(species_index)
  end function air5_species_gas_constant

  pure function air5_species_cv_tr(species_index) result(cv_tr)
    integer, intent(in) :: species_index
    real(real64) :: cv_tr

    cv_tr = air5_cv_tr_factor(species_index)*air5_species_gas_constant(species_index)
  end function air5_species_cv_tr

  pure function air5_species_vibrational_energy(species_index, tv) result(energy)
    integer, intent(in) :: species_index
    real(real64), intent(in) :: tv
    real(real64) :: energy, theta

    theta = air5_theta_v(species_index)
    if (theta > 0.0_real64) then
      energy = air5_species_gas_constant(species_index)*theta/(exp(theta/tv)-1.0_real64)
    else
      energy = 0.0_real64
    end if
  end function air5_species_vibrational_energy

  pure function air5_species_vibrational_cv(species_index, tv) result(cv_v)
    integer, intent(in) :: species_index
    real(real64), intent(in) :: tv
    real(real64) :: cv_v, theta, x, exponential

    theta = air5_theta_v(species_index)
    if (theta > 0.0_real64) then
      x = theta/tv
      exponential = exp(x)
      cv_v = air5_species_gas_constant(species_index)*x*x*exponential/ &
        ((exponential-1.0_real64)*(exponential-1.0_real64))
    else
      cv_v = 0.0_real64
    end if
  end function air5_species_vibrational_cv

  pure subroutine air5_ev_from_tv(rho_species, tv, ev, status)
    real(real64), intent(in) :: rho_species(air5_num_species), tv
    real(real64), intent(out) :: ev
    integer, intent(out) :: status
    integer :: species_index

    ev = 0.0_real64
    call air5_validate_partial_densities(rho_species, status)
    if (status /= chemistry_status_ok) return
    if (.not. ieee_is_finite(tv)) then
      status = chemistry_status_nonfinite
      return
    end if
    if (tv < air5_temperature_min_k .or. tv > air5_temperature_max_k) then
      status = chemistry_status_out_of_domain
      return
    end if
    if (sum(rho_species, mask=air5_theta_v > 0.0_real64) <= tiny(1.0_real64)) then
      status = chemistry_status_no_vibrational_capacity
      return
    end if
    do species_index = 1, air5_num_species
      ev = ev + rho_species(species_index)* &
        air5_species_vibrational_energy(species_index, tv)
    end do
  end subroutine air5_ev_from_tv

  pure subroutine air5_tv_from_ev(rho_species, ev, tv, status)
    real(real64), intent(in) :: rho_species(air5_num_species), ev
    real(real64), intent(out) :: tv
    integer, intent(out) :: status
    real(real64) :: ev_lower, ev_upper, ev_trial, lower, upper, residual_tolerance
    integer :: iteration, trial_status

    tv = air5_temperature_min_k
    call air5_validate_partial_densities(rho_species, status)
    if (status /= chemistry_status_ok) return
    if (.not. ieee_is_finite(ev)) then
      status = chemistry_status_nonfinite
      return
    end if
    if (sum(rho_species, mask=air5_theta_v > 0.0_real64) <= tiny(1.0_real64)) then
      status = chemistry_status_no_vibrational_capacity
      return
    end if
    call air5_ev_from_tv(rho_species, air5_temperature_min_k, ev_lower, trial_status)
    if (trial_status /= chemistry_status_ok) then
      status = trial_status
      return
    end if
    call air5_ev_from_tv(rho_species, air5_temperature_max_k, ev_upper, trial_status)
    if (trial_status /= chemistry_status_ok) then
      status = trial_status
      return
    end if
    residual_tolerance = 8.0_real64*epsilon(1.0_real64)* &
      max(abs(ev_lower), abs(ev), 1.0_real64)
    if (ev < ev_lower-residual_tolerance .or. ev > ev_upper+residual_tolerance) then
      status = chemistry_status_out_of_domain
      return
    end if
    if (abs(ev-ev_lower) <= residual_tolerance) then
      tv = air5_temperature_min_k
      status = chemistry_status_ok
      return
    end if
    if (abs(ev-ev_upper) <= residual_tolerance) then
      tv = air5_temperature_max_k
      status = chemistry_status_ok
      return
    end if

    lower = air5_temperature_min_k
    upper = air5_temperature_max_k
    do iteration = 1, 96
      tv = 0.5_real64*(lower+upper)
      call air5_ev_from_tv(rho_species, tv, ev_trial, trial_status)
      if (trial_status /= chemistry_status_ok) then
        status = trial_status
        return
      end if
      if (abs(ev_trial-ev) <= residual_tolerance) then
        status = chemistry_status_ok
        return
      end if
      if (ev_trial < ev) then
        lower = tv
      else
        upper = tv
      end if
      if (upper-lower <= 8.0_real64*epsilon(1.0_real64)*max(tv, 1.0_real64)) then
        status = chemistry_status_ok
        return
      end if
    end do
    status = chemistry_status_nonconverged
  end subroutine air5_tv_from_ev

  pure subroutine air5_q5_from_state(rho, momentum, rho_species, ev, temperature, q5, status)
    real(real64), intent(in) :: rho, momentum(3), rho_species(air5_num_species)
    real(real64), intent(in) :: ev, temperature
    real(real64), intent(out) :: q5
    integer, intent(out) :: status
    real(real64) :: cv_density, formation_density, kinetic_density
    integer :: species_index

    q5 = 0.0_real64
    if (.not. ieee_is_finite(rho) .or. .not. all(ieee_is_finite(momentum)) .or. &
        .not. ieee_is_finite(ev) .or. .not. ieee_is_finite(temperature)) then
      status = chemistry_status_nonfinite
      return
    end if
    if (rho <= 0.0_real64) then
      status = chemistry_status_invalid_density
      return
    end if
    call air5_validate_partial_densities(rho_species, status)
    if (status /= chemistry_status_ok) return
    if (ev < 0.0_real64 .or. temperature < air5_temperature_min_k .or. &
        temperature > air5_temperature_max_k) then
      status = chemistry_status_out_of_domain
      return
    end if
    cv_density = 0.0_real64
    formation_density = 0.0_real64
    do species_index = 1, air5_num_species
      cv_density = cv_density + rho_species(species_index)* &
        air5_species_cv_tr(species_index)
      formation_density = formation_density + &
        rho_species(species_index)*air5_formation_energy(species_index)
    end do
    kinetic_density = dot_product(momentum, momentum)/(2.0_real64*rho)
    q5 = kinetic_density + cv_density*temperature + ev + formation_density
    status = chemistry_status_ok
  end subroutine air5_q5_from_state

  pure subroutine air5_temperature_from_q5(rho, momentum, rho_species, ev, q5, &
      temperature, status)
    real(real64), intent(in) :: rho, momentum(3), rho_species(air5_num_species)
    real(real64), intent(in) :: ev, q5
    real(real64), intent(out) :: temperature
    integer, intent(out) :: status
    real(real64) :: cv_density, formation_density, kinetic_density, bound_tolerance
    integer :: species_index

    temperature = air5_temperature_min_k
    if (.not. ieee_is_finite(rho) .or. .not. all(ieee_is_finite(momentum)) .or. &
        .not. ieee_is_finite(ev) .or. .not. ieee_is_finite(q5)) then
      status = chemistry_status_nonfinite
      return
    end if
    if (rho <= 0.0_real64) then
      status = chemistry_status_invalid_density
      return
    end if
    call air5_validate_partial_densities(rho_species, status)
    if (status /= chemistry_status_ok) return
    if (ev < 0.0_real64) then
      status = chemistry_status_out_of_domain
      return
    end if
    cv_density = 0.0_real64
    formation_density = 0.0_real64
    do species_index = 1, air5_num_species
      cv_density = cv_density + rho_species(species_index)* &
        air5_species_cv_tr(species_index)
      formation_density = formation_density + &
        rho_species(species_index)*air5_formation_energy(species_index)
    end do
    if (cv_density <= tiny(1.0_real64)) then
      status = chemistry_status_invalid_composition
      return
    end if
    kinetic_density = dot_product(momentum, momentum)/(2.0_real64*rho)
    temperature = (q5-kinetic_density-ev-formation_density)/cv_density
    bound_tolerance = 64.0_real64*epsilon(1.0_real64)*air5_temperature_max_k
    if (.not. ieee_is_finite(temperature)) then
      status = chemistry_status_nonfinite
    else if (temperature < air5_temperature_min_k-bound_tolerance .or. &
             temperature > air5_temperature_max_k+bound_tolerance) then
      status = chemistry_status_out_of_domain
    else
      temperature = min(max(temperature,air5_temperature_min_k),air5_temperature_max_k)
      status = chemistry_status_ok
    end if
  end subroutine air5_temperature_from_q5

  pure subroutine air5_temperature_derivatives(rho_species, temperature, &
      dt_dz, dt_dev, status)
    real(real64), intent(in) :: rho_species(air5_num_species), temperature
    real(real64), intent(out) :: dt_dz(air5_num_species), dt_dev
    integer, intent(out) :: status
    real(real64) :: cv_density
    integer :: species_index

    dt_dz = 0.0_real64
    dt_dev = 0.0_real64
    call air5_validate_partial_densities(rho_species, status)
    if (status /= chemistry_status_ok) return
    if (.not. ieee_is_finite(temperature)) then
      status = chemistry_status_nonfinite
      return
    end if
    if (temperature < air5_temperature_min_k .or. &
        temperature > air5_temperature_max_k) then
      status = chemistry_status_out_of_domain
      return
    end if
    cv_density = 0.0_real64
    do species_index = 1, air5_num_species
      cv_density = cv_density + rho_species(species_index)* &
        air5_species_cv_tr(species_index)
    end do
    if (cv_density <= tiny(1.0_real64)) then
      status = chemistry_status_invalid_composition
      return
    end if
    do species_index = 1, air5_num_species
      dt_dz(species_index) = -(air5_formation_energy(species_index) + &
        air5_species_cv_tr(species_index)*temperature)/cv_density
    end do
    dt_dev = -1.0_real64/cv_density
    status = chemistry_status_ok
  end subroutine air5_temperature_derivatives

  pure subroutine air5_tv_derivatives(rho_species, tv, dtv_dz, dtv_dev, status)
    real(real64), intent(in) :: rho_species(air5_num_species), tv
    real(real64), intent(out) :: dtv_dz(air5_num_species), dtv_dev
    integer, intent(out) :: status
    real(real64) :: cv_v_density
    integer :: species_index

    dtv_dz = 0.0_real64
    dtv_dev = 0.0_real64
    call air5_validate_partial_densities(rho_species, status)
    if (status /= chemistry_status_ok) return
    if (.not. ieee_is_finite(tv)) then
      status = chemistry_status_nonfinite
      return
    end if
    if (tv < air5_temperature_min_k .or. tv > air5_temperature_max_k) then
      status = chemistry_status_out_of_domain
      return
    end if
    cv_v_density = 0.0_real64
    do species_index = 1, air5_num_species
      cv_v_density = cv_v_density + rho_species(species_index)* &
        air5_species_vibrational_cv(species_index, tv)
    end do
    if (cv_v_density <= tiny(1.0_real64)) then
      status = chemistry_status_no_vibrational_capacity
      return
    end if
    do species_index = 1, air5_num_species
      dtv_dz(species_index) = -air5_species_vibrational_energy(species_index, tv)/ &
        cv_v_density
    end do
    dtv_dev = 1.0_real64/cv_v_density
    status = chemistry_status_ok
  end subroutine air5_tv_derivatives

  pure subroutine air5_pressure(rho_species, temperature, pressure, status)
    real(real64), intent(in) :: rho_species(air5_num_species), temperature
    real(real64), intent(out) :: pressure
    integer, intent(out) :: status
    integer :: species_index

    pressure = 0.0_real64
    call air5_validate_partial_densities(rho_species, status)
    if (status /= chemistry_status_ok) return
    if (.not. ieee_is_finite(temperature)) then
      status = chemistry_status_nonfinite
      return
    end if
    if (temperature < air5_temperature_min_k .or. &
        temperature > air5_temperature_max_k) then
      status = chemistry_status_out_of_domain
      return
    end if
    do species_index = 1, air5_num_species
      pressure = pressure + rho_species(species_index)* &
        air5_species_gas_constant(species_index)*temperature
    end do
    status = chemistry_status_ok
  end subroutine air5_pressure

end module chemistry_thermo
