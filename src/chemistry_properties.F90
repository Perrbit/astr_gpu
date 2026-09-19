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

module chemistry_flow_state
  use iso_fortran_env, only: real64
  use ieee_arithmetic, only: ieee_is_finite
  use chemistry_air5_data, only: air5_num_species
  use chemistry_model, only: chemistry_status_ok, chemistry_status_invalid_density, &
    chemistry_status_invalid_composition, chemistry_status_nonfinite, &
    air5_validate_physical_species_state
  use chemistry_state_layout, only: air5_idx_density, air5_idx_momentum_first, &
    air5_idx_momentum_last, air5_idx_total_energy, air5_idx_species_first, &
    air5_idx_species_last, air5_idx_ev, air5_num_conservative
  use chemistry_thermo, only: air5_ev_from_tv, air5_q5_from_state, &
    air5_temperature_from_q5, air5_tv_from_ev, air5_pressure
  implicit none
  private

  public :: air5_primitive_to_conservative
  public :: air5_conservative_to_primitive

contains

  pure subroutine air5_primitive_to_conservative(rho,velocity,temperature, &
      mass_fraction,tv,q,status)
    real(real64), intent(in) :: rho,velocity(3),temperature,tv
    real(real64), intent(in) :: mass_fraction(air5_num_species)
    real(real64), intent(out) :: q(air5_num_conservative)
    integer, intent(out) :: status
    real(real64) :: rho_species(air5_num_species),momentum(3),ev,tolerance

    q=0.0_real64
    if(.not.ieee_is_finite(rho) .or. .not.all(ieee_is_finite(velocity)) .or. &
       .not.ieee_is_finite(temperature) .or. .not.ieee_is_finite(tv) .or. &
       .not.all(ieee_is_finite(mass_fraction))) then
      status=chemistry_status_nonfinite
      return
    endif
    if(rho<=0.0_real64) then
      status=chemistry_status_invalid_density
      return
    endif
    tolerance=64.0_real64*epsilon(1.0_real64)
    if(any(mass_fraction<0.0_real64) .or. &
       abs(sum(mass_fraction)-1.0_real64)>tolerance) then
      status=chemistry_status_invalid_composition
      return
    endif

    rho_species=rho*mass_fraction
    momentum=rho*velocity
    call air5_validate_physical_species_state(rho,rho_species,status)
    if(status/=chemistry_status_ok) return
    call air5_ev_from_tv(rho_species,tv,ev,status)
    if(status/=chemistry_status_ok) return
    call air5_q5_from_state(rho,momentum,rho_species,ev,temperature, &
      q(air5_idx_total_energy),status)
    if(status/=chemistry_status_ok) return

    q(air5_idx_density)=rho
    q(air5_idx_momentum_first:air5_idx_momentum_last)=momentum
    q(air5_idx_species_first:air5_idx_species_last)=rho_species
    q(air5_idx_ev)=ev
  end subroutine air5_primitive_to_conservative

  pure subroutine air5_conservative_to_primitive(q,rho,velocity,temperature, &
      mass_fraction,tv,pressure,status)
    real(real64), intent(in) :: q(air5_num_conservative)
    real(real64), intent(out) :: rho,velocity(3),temperature
    real(real64), intent(out) :: mass_fraction(air5_num_species),tv,pressure
    integer, intent(out) :: status
    real(real64) :: rho_species(air5_num_species),momentum(3),ev

    rho=0.0_real64
    velocity=0.0_real64
    temperature=0.0_real64
    mass_fraction=0.0_real64
    tv=0.0_real64
    pressure=0.0_real64
    if(.not.all(ieee_is_finite(q))) then
      status=chemistry_status_nonfinite
      return
    endif

    rho=q(air5_idx_density)
    momentum=q(air5_idx_momentum_first:air5_idx_momentum_last)
    rho_species=q(air5_idx_species_first:air5_idx_species_last)
    ev=q(air5_idx_ev)
    call air5_validate_physical_species_state(rho,rho_species,status)
    if(status/=chemistry_status_ok) return
    velocity=momentum/rho
    mass_fraction=rho_species/rho
    call air5_temperature_from_q5(rho,momentum,rho_species,ev, &
      q(air5_idx_total_energy),temperature,status)
    if(status/=chemistry_status_ok) return
    call air5_tv_from_ev(rho_species,ev,tv,status)
    if(status/=chemistry_status_ok) return
    call air5_pressure(rho_species,temperature,pressure,status)
  end subroutine air5_conservative_to_primitive

end module chemistry_flow_state

module chemistry_transport
  use iso_fortran_env, only: real64
  use ieee_arithmetic, only: ieee_is_finite
  use chemistry_air5_data, only: air5_num_species,air5_molar_mass, &
    air5_blottner,air5_binary_diffusion,air5_atom_count,air5_formation_energy
  use chemistry_model, only: chemistry_status_ok,chemistry_status_invalid_composition, &
    chemistry_status_nonfinite,chemistry_status_out_of_domain, &
    air5_pressure_is_in_domain
  use chemistry_thermo, only: air5_species_gas_constant,air5_species_cv_tr, &
    air5_species_vibrational_energy,air5_species_vibrational_cv
  implicit none
  private

  public :: air5_transport_properties
  public :: air5_diffusive_flux

contains

  pure subroutine air5_transport_properties(temperature,tv,pressure,mass_fraction, &
      viscosity,conductivity_tr,conductivity_v,species_viscosity,binary_diffusion, &
      mixture_diffusion,status)
    real(real64), intent(in) :: temperature,tv,pressure
    real(real64), intent(in) :: mass_fraction(air5_num_species)
    real(real64), intent(out) :: viscosity,conductivity_tr,conductivity_v
    real(real64), intent(out) :: species_viscosity(air5_num_species)
    real(real64), intent(out) :: binary_diffusion(air5_num_species,air5_num_species)
    real(real64), intent(out) :: mixture_diffusion(air5_num_species)
    integer, intent(out) :: status
    real(real64) :: mole_fraction(air5_num_species),phi(air5_num_species,air5_num_species)
    real(real64) :: species_conductivity(air5_num_species), &
      species_vibrational_conductivity(air5_num_species)
    real(real64) :: log_temperature,mole_sum,denominator,numerator
    integer :: species,partner

    viscosity=0.0_real64
    conductivity_tr=0.0_real64
    conductivity_v=0.0_real64
    species_viscosity=0.0_real64
    binary_diffusion=0.0_real64
    mixture_diffusion=0.0_real64
    if(.not.ieee_is_finite(temperature) .or. .not.ieee_is_finite(tv) .or. &
       .not.ieee_is_finite(pressure) .or. .not.all(ieee_is_finite(mass_fraction))) then
      status=chemistry_status_nonfinite
      return
    endif
    if(temperature<=0.0_real64 .or. tv<=0.0_real64 .or. &
       .not.air5_pressure_is_in_domain(pressure)) then
      status=chemistry_status_out_of_domain
      return
    endif
    if(any(mass_fraction<0.0_real64) .or. &
       abs(sum(mass_fraction)-1.0_real64)>64.0_real64*epsilon(1.0_real64)) then
      status=chemistry_status_invalid_composition
      return
    endif

    mole_sum=sum(mass_fraction/air5_molar_mass)
    if(mole_sum<=0.0_real64) then
      status=chemistry_status_invalid_composition
      return
    endif
    mole_fraction=(mass_fraction/air5_molar_mass)/mole_sum
    log_temperature=log(temperature)
    do species=1,air5_num_species
      species_viscosity(species)=0.1_real64*exp( &
        air5_blottner(1,species)*log_temperature*log_temperature+ &
        air5_blottner(2,species)*log_temperature+air5_blottner(3,species))
      if(sum(air5_atom_count(:,species))==1) then
        species_conductivity(species)=3.75_real64*species_viscosity(species)* &
          air5_species_gas_constant(species)
      else
        species_conductivity(species)=4.75_real64*species_viscosity(species)* &
          air5_species_gas_constant(species)
      endif
      species_vibrational_conductivity(species)=species_viscosity(species)* &
        air5_species_vibrational_cv(species,tv)
    enddo

    do species=1,air5_num_species
      do partner=1,air5_num_species
        phi(species,partner)=(1.0_real64+sqrt(species_viscosity(species)/ &
          species_viscosity(partner))*(air5_molar_mass(partner)/ &
          air5_molar_mass(species))**0.25_real64)**2/ &
          sqrt(8.0_real64*(1.0_real64+air5_molar_mass(species)/ &
          air5_molar_mass(partner)))
      enddo
      denominator=sum(mole_fraction*phi(species,:))
      viscosity=viscosity+mole_fraction(species)*species_viscosity(species)/denominator
      conductivity_tr=conductivity_tr+ &
        mole_fraction(species)*species_conductivity(species)/denominator
      conductivity_v=conductivity_v+ &
        mole_fraction(species)*species_vibrational_conductivity(species)/denominator
    enddo

    do species=1,air5_num_species
      do partner=1,air5_num_species
        binary_diffusion(species,partner)=10.1325_real64/pressure* &
          exp(air5_binary_diffusion(4,species,partner))* &
          temperature**(air5_binary_diffusion(1,species,partner)* &
          log_temperature*log_temperature+air5_binary_diffusion(2,species,partner)* &
          log_temperature+air5_binary_diffusion(3,species,partner))
      enddo
    enddo

    do species=1,air5_num_species
      numerator=sum(mole_fraction,mask=[(partner/=species,partner=1,air5_num_species)])
      denominator=0.0_real64
      do partner=1,air5_num_species
        if(partner/=species) denominator=denominator+ &
          mole_fraction(partner)/binary_diffusion(species,partner)
      enddo
      if(numerator==0.0_real64) then
        mixture_diffusion(species)=0.0_real64
      elseif(denominator>0.0_real64) then
        mixture_diffusion(species)=numerator/denominator
      else
        status=chemistry_status_invalid_composition
        return
      endif
    enddo
    if(.not.ieee_is_finite(viscosity) .or. .not.ieee_is_finite(conductivity_tr) .or. &
       .not.ieee_is_finite(conductivity_v) .or. &
       .not.all(ieee_is_finite(binary_diffusion)) .or. &
       .not.all(ieee_is_finite(mixture_diffusion))) then
      status=chemistry_status_nonfinite
    else
      status=chemistry_status_ok
    endif
  end subroutine air5_transport_properties

  pure subroutine air5_diffusive_flux(rho,velocity,temperature,tv,pressure, &
      mass_fraction,grad_velocity,grad_temperature,grad_tv,grad_mass_fraction, &
      momentum_flux,species_flux,energy_flux,ev_flux,status)
    real(real64), intent(in) :: rho,velocity(3),temperature,tv,pressure
    real(real64), intent(in) :: mass_fraction(air5_num_species)
    real(real64), intent(in) :: grad_velocity(3,3),grad_temperature(3),grad_tv(3)
    real(real64), intent(in) :: grad_mass_fraction(air5_num_species,3)
    real(real64), intent(out) :: momentum_flux(3,3)
    real(real64), intent(out) :: species_flux(air5_num_species,3)
    real(real64), intent(out) :: energy_flux(3),ev_flux(3)
    integer, intent(out) :: status
    real(real64) :: viscosity,conductivity_tr,conductivity_v, &
      species_viscosity(air5_num_species),binary_diffusion(air5_num_species,air5_num_species), &
      mixture_diffusion(air5_num_species),raw_flux(air5_num_species)
    real(real64) :: divergence,trace_correction,enthalpy(air5_num_species), &
      vibrational_energy(air5_num_species)
    integer :: direction,species,i,j

    momentum_flux=0.0_real64
    species_flux=0.0_real64
    energy_flux=0.0_real64
    ev_flux=0.0_real64
    if(.not.ieee_is_finite(rho) .or. rho<=0.0_real64 .or. &
       .not.all(ieee_is_finite(velocity)) .or. &
       .not.all(ieee_is_finite(grad_velocity)) .or. &
       .not.all(ieee_is_finite(grad_temperature)) .or. &
       .not.all(ieee_is_finite(grad_tv)) .or. &
       .not.all(ieee_is_finite(grad_mass_fraction))) then
      status=chemistry_status_nonfinite
      return
    endif
    call air5_transport_properties(temperature,tv,pressure,mass_fraction,viscosity, &
      conductivity_tr,conductivity_v,species_viscosity,binary_diffusion, &
      mixture_diffusion,status)
    if(status/=chemistry_status_ok) return

    divergence=grad_velocity(1,1)+grad_velocity(2,2)+grad_velocity(3,3)
    do j=1,3
      do i=1,3
        momentum_flux(i,j)=viscosity*(grad_velocity(i,j)+grad_velocity(j,i))
        if(i==j) momentum_flux(i,j)=momentum_flux(i,j)- &
          (2.0_real64/3.0_real64)*viscosity*divergence
      enddo
    enddo

    do species=1,air5_num_species
      vibrational_energy(species)=air5_species_vibrational_energy(species,tv)
      enthalpy(species)=air5_species_cv_tr(species)*temperature+ &
        vibrational_energy(species)+air5_formation_energy(species)+ &
        air5_species_gas_constant(species)*temperature
    enddo
    do direction=1,3
      raw_flux=-rho*mixture_diffusion*grad_mass_fraction(:,direction)
      trace_correction=sum(raw_flux)
      species_flux(:,direction)=raw_flux-mass_fraction*trace_correction
      energy_flux(direction)=dot_product(momentum_flux(:,direction),velocity)+ &
        conductivity_tr*grad_temperature(direction)+conductivity_v*grad_tv(direction)- &
        dot_product(enthalpy,species_flux(:,direction))
      ev_flux(direction)=conductivity_v*grad_tv(direction)- &
        dot_product(vibrational_energy,species_flux(:,direction))
    enddo
    if(.not.all(ieee_is_finite(momentum_flux)) .or. &
       .not.all(ieee_is_finite(species_flux)) .or. &
       .not.all(ieee_is_finite(energy_flux)) .or. &
       .not.all(ieee_is_finite(ev_flux))) status=chemistry_status_nonfinite
  end subroutine air5_diffusive_flux

end module chemistry_transport

module chemistry_relaxation
  use iso_fortran_env, only: real64
  use ieee_arithmetic, only: ieee_is_finite
  use chemistry_air5_data, only: air5_num_species, air5_molar_mass, air5_theta_v, &
    air5_mw_a, air5_mw_b, air5_collision_sigma0, air5_collision_sigma_power, &
    air5_ru, air5_temperature_min_k, air5_temperature_max_k, &
    air5_relaxation_uses_relaxing_species_density
  use chemistry_model, only: chemistry_status_ok, chemistry_status_out_of_domain, &
    chemistry_status_nonfinite, air5_validate_partial_densities, &
    air5_pressure_is_in_domain
  use chemistry_thermo, only: air5_species_gas_constant, &
    air5_species_vibrational_energy, air5_species_vibrational_cv, air5_pressure
  implicit none
  private

  real(real64), parameter :: avogadro = 6.02214076e23_real64
  real(real64), parameter :: pressure_atmosphere_pa = 101325.0_real64
  real(real64), parameter :: pi = 3.1415926535897932384626433832795_real64

  public :: air5_vt_relaxation_source
  public :: air5_vt_relaxation_source_derivatives

contains

  pure subroutine air5_vt_relaxation_source(rho_species, temperature, tv, source, status)
    real(real64), intent(in) :: rho_species(air5_num_species), temperature, tv
    real(real64), intent(out) :: source
    integer, intent(out) :: status
    real(real64) :: dtemperature(6), dtv(6), derivatives(6)

    dtemperature = 0.0_real64
    dtv = 0.0_real64
    call air5_vt_relaxation_source_derivatives(rho_species, temperature, tv, &
      dtemperature, dtv, source, derivatives, status)
  end subroutine air5_vt_relaxation_source

  pure subroutine air5_vt_relaxation_source_derivatives(rho_species, temperature, tv, &
      dtemperature, dtv, source, derivatives, status)
    real(real64), intent(in) :: rho_species(air5_num_species), temperature, tv
    real(real64), intent(in) :: dtemperature(6), dtv(6)
    real(real64), intent(out) :: source, derivatives(6)
    integer, intent(out) :: status
    real(real64) :: concentration(air5_num_species), mole_fraction(air5_num_species)
    real(real64) :: concentration_sum, pressure, gas_density, pressure_derivative
    real(real64) :: pair_time(air5_num_species), pair_time_derivative(air5_num_species)
    real(real64) :: mw_time, park_time, sigma, reduced_mass, mean_speed
    real(real64) :: inverse_time, inverse_time_derivative, dx
    real(real64) :: equilibrium_energy, current_energy, energy_difference
    real(real64) :: energy_derivative, relaxing_number_density
    integer :: molecule, partner, variable, local_status

    source = 0.0_real64
    derivatives = 0.0_real64
    call air5_validate_partial_densities(rho_species, status)
    if (status /= chemistry_status_ok) return
    if (.not. ieee_is_finite(temperature) .or. .not. ieee_is_finite(tv) .or. &
        .not. all(ieee_is_finite(dtemperature)) .or. .not. all(ieee_is_finite(dtv))) then
      status = chemistry_status_nonfinite
      return
    end if
    if (temperature < air5_temperature_min_k .or. &
        temperature > air5_temperature_max_k .or. &
        tv < air5_temperature_min_k .or. tv > air5_temperature_max_k) then
      status = chemistry_status_out_of_domain
      return
    end if
    call air5_pressure(rho_species, temperature, pressure, local_status)
    if (local_status /= chemistry_status_ok) then
      status = local_status
      return
    end if
    if (.not. air5_pressure_is_in_domain(pressure)) then
      status = chemistry_status_out_of_domain
      return
    end if
    if (.not. air5_relaxation_uses_relaxing_species_density) then
      status = chemistry_status_out_of_domain
      return
    end if

    do partner = 1, air5_num_species
      concentration(partner) = rho_species(partner)/air5_molar_mass(partner)
    end do
    concentration_sum = sum(concentration)
    mole_fraction = concentration/concentration_sum
    gas_density = 0.0_real64
    do partner = 1, air5_num_species
      gas_density = gas_density + rho_species(partner)*air5_species_gas_constant(partner)
    end do

    do molecule = 1, air5_num_species
      if (air5_theta_v(molecule) <= 0.0_real64) cycle
      if (rho_species(molecule) <= 0.0_real64) cycle
      relaxing_number_density = concentration(molecule)*avogadro
      do partner = 1, air5_num_species
        mw_time = pressure_atmosphere_pa/pressure*exp(air5_mw_a(molecule,partner)* &
          (temperature**(-1.0_real64/3.0_real64)-air5_mw_b(molecule,partner))-18.42_real64)
        park_time = 0.0_real64
        if (air5_collision_sigma0(molecule,partner) > 0.0_real64) then
          sigma = air5_collision_sigma0(molecule,partner)* &
            temperature**air5_collision_sigma_power(molecule,partner)
          reduced_mass = air5_molar_mass(molecule)*air5_molar_mass(partner)/ &
            (air5_molar_mass(molecule)+air5_molar_mass(partner))
          mean_speed = sqrt(8.0_real64*air5_ru*temperature/(pi*reduced_mass))
          park_time = 1.0_real64/(relaxing_number_density*sigma*mean_speed)
        end if
        pair_time(partner) = mw_time+park_time
      end do
      inverse_time = sum(mole_fraction/pair_time)
      equilibrium_energy = air5_species_vibrational_energy(molecule, temperature)
      current_energy = air5_species_vibrational_energy(molecule, tv)
      energy_difference = equilibrium_energy-current_energy
      source = source + rho_species(molecule)*energy_difference*inverse_time

      do variable = 1, 6
        pressure_derivative = gas_density*dtemperature(variable)
        if (variable <= air5_num_species) then
          pressure_derivative = pressure_derivative + &
            air5_species_gas_constant(variable)*temperature
        end if
        do partner = 1, air5_num_species
          mw_time = pressure_atmosphere_pa/pressure*exp(air5_mw_a(molecule,partner)* &
            (temperature**(-1.0_real64/3.0_real64)-air5_mw_b(molecule,partner))-18.42_real64)
          pair_time_derivative(partner) = mw_time*(-pressure_derivative/pressure - &
            air5_mw_a(molecule,partner)*temperature**(-4.0_real64/3.0_real64)* &
            dtemperature(variable)/3.0_real64)
          if (air5_collision_sigma0(molecule,partner) > 0.0_real64) then
            sigma = air5_collision_sigma0(molecule,partner)* &
              temperature**air5_collision_sigma_power(molecule,partner)
            reduced_mass = air5_molar_mass(molecule)*air5_molar_mass(partner)/ &
              (air5_molar_mass(molecule)+air5_molar_mass(partner))
            mean_speed = sqrt(8.0_real64*air5_ru*temperature/(pi*reduced_mass))
            park_time = 1.0_real64/(relaxing_number_density*sigma*mean_speed)
            pair_time_derivative(partner) = pair_time_derivative(partner) - &
              park_time*(air5_collision_sigma_power(molecule,partner)+0.5_real64)* &
              dtemperature(variable)/temperature
            if (variable == molecule) then
              pair_time_derivative(partner) = pair_time_derivative(partner) - &
                park_time/rho_species(molecule)
            end if
          end if
        end do
        inverse_time_derivative = 0.0_real64
        do partner = 1, air5_num_species
          dx = 0.0_real64
          if (variable <= air5_num_species) then
            dx = -mole_fraction(partner)/(air5_molar_mass(variable)*concentration_sum)
            if (variable == partner) then
              dx = dx + 1.0_real64/(air5_molar_mass(variable)*concentration_sum)
            end if
          end if
          inverse_time_derivative = inverse_time_derivative + dx/pair_time(partner) - &
            mole_fraction(partner)*pair_time_derivative(partner)/ &
            (pair_time(partner)*pair_time(partner))
        end do
        energy_derivative = air5_species_vibrational_cv(molecule,temperature)* &
          dtemperature(variable) - air5_species_vibrational_cv(molecule,tv)*dtv(variable)
        if (variable == molecule) then
          derivatives(variable) = derivatives(variable) + energy_difference*inverse_time
        end if
        derivatives(variable) = derivatives(variable) + rho_species(molecule)*( &
          energy_derivative*inverse_time + energy_difference*inverse_time_derivative)
      end do
    end do
    if (.not. ieee_is_finite(source) .or. .not. all(ieee_is_finite(derivatives))) then
      status = chemistry_status_nonfinite
    else
      status = chemistry_status_ok
    end if
  end subroutine air5_vt_relaxation_source_derivatives

end module chemistry_relaxation
