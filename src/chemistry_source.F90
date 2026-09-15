module chemistry_source
  use iso_fortran_env, only: real64
  use ieee_arithmetic, only: ieee_is_finite
  use chemistry_air5_data, only: air5_num_species, air5_num_reactions, &
    air5_molar_mass, air5_atom_count, air5_nu_reactant, air5_nu_product, &
    air5_third_body_efficiency, air5_has_third_body, air5_rate_uses_tv, &
    air5_arrhenius_a, air5_arrhenius_b, air5_activation_temperature, &
    air5_ln_kc_coefficient, air5_psi_coefficient, air5_psi_bounds
  use chemistry_model, only: chemistry_status_ok, chemistry_status_invalid_composition, &
    chemistry_status_out_of_domain, chemistry_status_nonfinite, &
    chemistry_status_invalid_source_mode, air5_validate_partial_densities, &
    air5_pressure_is_in_domain
  use chemistry_thermo, only: air5_temperature_from_q5, air5_tv_from_ev, &
    air5_temperature_derivatives, air5_tv_derivatives, air5_pressure, &
    air5_species_gas_constant, air5_species_vibrational_energy, &
    air5_species_vibrational_cv
  use chemistry_relaxation, only: air5_vt_relaxation_source_derivatives
  implicit none
  private

  integer, parameter, public :: air5_source_mode_coupled = 0
  integer, parameter, public :: air5_source_mode_chemical = 1
  integer, parameter, public :: air5_source_mode_vt = 2

  public :: air5_instantaneous_source
  public :: air5_instantaneous_source_jacobian
  public :: air5_centered_difference_jacobian
  public :: air5_admissible_difference_jacobian
  public :: air5_source_conservation_residuals

contains

  pure subroutine air5_instantaneous_source(rho, momentum, q5, state, source, status, &
      source_mode, reaction_progress)
    real(real64), intent(in) :: rho, momentum(3), q5, state(6)
    real(real64), intent(out) :: source(6)
    integer, intent(out) :: status
    integer, intent(in), optional :: source_mode
    real(real64), intent(out), optional :: reaction_progress(air5_num_reactions)
    real(real64) :: jacobian(6,6), progress_buffer(air5_num_reactions)
    integer :: active_mode

    active_mode = air5_source_mode_coupled
    if (present(source_mode)) active_mode = source_mode
    call air5_instantaneous_source_jacobian(rho, momentum, q5, state, &
      source, jacobian, status, active_mode, progress_buffer)
    if (present(reaction_progress)) reaction_progress = progress_buffer
  end subroutine air5_instantaneous_source

  pure subroutine air5_instantaneous_source_jacobian(rho, momentum, q5, state, &
      source, jacobian, status, source_mode, reaction_progress)
    real(real64), intent(in) :: rho, momentum(3), q5, state(6)
    real(real64), intent(out) :: source(6), jacobian(6,6)
    integer, intent(out) :: status
    integer, intent(in), optional :: source_mode
    real(real64), intent(out), optional :: reaction_progress(air5_num_reactions)
    real(real64) :: temperature, tv, pressure, concentration(air5_num_species)
    real(real64) :: dtemperature(6), dtv(6), dt_dz(air5_num_species)
    real(real64) :: dtv_dz(air5_num_species), dt_dev, dtv_dev
    real(real64) :: third_body, dthird_body, forward_product, reverse_product
    real(real64) :: forward_coefficient, reverse_coefficient, equilibrium_constant
    real(real64) :: forward_rate, reverse_rate, progress_rate
    real(real64) :: dforward_product, dreverse_product, dforward_coefficient
    real(real64) :: dreverse_coefficient, dforward_rate, dreverse_rate
    real(real64) :: dprogress_rate(6), rate_temperature, drate_temperature
    real(real64) :: z10000, ln_kc, dln_kc_dtemperature, dln_kc
    real(real64) :: energy_per_mass, denergy_per_mass, psi, dpsi_dtemperature
    real(real64) :: mass_factor, vt_source, vt_derivatives(6)
    integer :: reaction, species, variable, local_status, net_stoichiometry
    integer :: active_mode
    logical :: include_chemical, include_vt

    source = 0.0_real64
    jacobian = 0.0_real64
    if (present(reaction_progress)) reaction_progress = 0.0_real64
    active_mode = air5_source_mode_coupled
    if (present(source_mode)) active_mode = source_mode
    if (active_mode < air5_source_mode_coupled .or. active_mode > air5_source_mode_vt) then
      status = chemistry_status_invalid_source_mode
      return
    end if
    include_chemical = active_mode == air5_source_mode_coupled .or. &
      active_mode == air5_source_mode_chemical
    include_vt = active_mode == air5_source_mode_coupled .or. &
      active_mode == air5_source_mode_vt
    call air5_validate_partial_densities(state(1:air5_num_species), status)
    if (status /= chemistry_status_ok) return
    if (.not. all(ieee_is_finite(state))) then
      status = chemistry_status_nonfinite
      return
    end if
    if (state(6) < 0.0_real64) then
      status = chemistry_status_out_of_domain
      return
    end if
    call air5_temperature_from_q5(rho, momentum, state(1:air5_num_species), &
      state(6), q5, temperature, status)
    if (status /= chemistry_status_ok) return
    call air5_tv_from_ev(state(1:air5_num_species), state(6), tv, status)
    if (status /= chemistry_status_ok) return
    call air5_pressure(state(1:air5_num_species), temperature, pressure, status)
    if (status /= chemistry_status_ok) return
    if (.not. air5_pressure_is_in_domain(pressure)) then
      status = chemistry_status_out_of_domain
      return
    end if
    call air5_temperature_derivatives(state(1:air5_num_species), temperature, &
      dt_dz, dt_dev, status)
    if (status /= chemistry_status_ok) return
    call air5_tv_derivatives(state(1:air5_num_species), tv, dtv_dz, dtv_dev, status)
    if (status /= chemistry_status_ok) return
    dtemperature(1:air5_num_species) = dt_dz
    dtemperature(6) = dt_dev
    dtv(1:air5_num_species) = dtv_dz
    dtv(6) = dtv_dev

    if (include_chemical) then
      do species = 1, air5_num_species
        concentration(species) = state(species)/air5_molar_mass(species)
      end do

      do reaction = 1, air5_num_reactions
      third_body = 1.0_real64
      if (air5_has_third_body(reaction)) then
        third_body = dot_product(air5_third_body_efficiency(:,reaction), concentration)
      end if
      forward_product = air5_mass_action(concentration, air5_nu_reactant(:,reaction))
      reverse_product = air5_mass_action(concentration, air5_nu_product(:,reaction))
      if (air5_rate_uses_tv(reaction)) then
        rate_temperature = sqrt(temperature*tv)
      else
        rate_temperature = temperature
      end if
      forward_coefficient = air5_arrhenius_a(reaction)* &
        rate_temperature**air5_arrhenius_b(reaction)* &
        exp(-air5_activation_temperature(reaction)/rate_temperature)
      z10000 = 10000.0_real64/temperature
      ln_kc = air5_ln_kc_coefficient(1,reaction)/z10000 + &
        air5_ln_kc_coefficient(2,reaction) + &
        air5_ln_kc_coefficient(3,reaction)*log(z10000) + &
        air5_ln_kc_coefficient(4,reaction)*z10000 + &
        air5_ln_kc_coefficient(5,reaction)*z10000*z10000
      equilibrium_constant = exp(ln_kc)
      reverse_coefficient = air5_arrhenius_a(reaction)* &
        temperature**air5_arrhenius_b(reaction)* &
        exp(-air5_activation_temperature(reaction)/temperature)/equilibrium_constant
      forward_rate = forward_coefficient*third_body*forward_product
      reverse_rate = reverse_coefficient*third_body*reverse_product
      progress_rate = forward_rate-reverse_rate
      if (present(reaction_progress)) reaction_progress(reaction) = progress_rate

      dln_kc_dtemperature = (-air5_ln_kc_coefficient(1,reaction)/(z10000*z10000) + &
        air5_ln_kc_coefficient(3,reaction)/z10000 + &
        air5_ln_kc_coefficient(4,reaction) + &
        2.0_real64*air5_ln_kc_coefficient(5,reaction)*z10000)* &
        (-z10000/temperature)
      do variable = 1, 6
        if (air5_rate_uses_tv(reaction)) then
          drate_temperature = 0.5_real64*rate_temperature*( &
            dtemperature(variable)/temperature+dtv(variable)/tv)
        else
          drate_temperature = dtemperature(variable)
        end if
        dforward_coefficient = forward_coefficient*( &
          air5_arrhenius_b(reaction)/rate_temperature + &
          air5_activation_temperature(reaction)/(rate_temperature*rate_temperature))* &
          drate_temperature
        dln_kc = dln_kc_dtemperature*dtemperature(variable)
        dreverse_coefficient = reverse_coefficient*( &
          (air5_arrhenius_b(reaction)/temperature + &
           air5_activation_temperature(reaction)/(temperature*temperature))* &
          dtemperature(variable)-dln_kc)
        dthird_body = 0.0_real64
        if (air5_has_third_body(reaction) .and. variable <= air5_num_species) then
          dthird_body = air5_third_body_efficiency(variable,reaction)/ &
            air5_molar_mass(variable)
        end if
        dforward_product = air5_mass_action_derivative(concentration, &
          air5_nu_reactant(:,reaction), variable)
        dreverse_product = air5_mass_action_derivative(concentration, &
          air5_nu_product(:,reaction), variable)
        dforward_rate = dforward_coefficient*third_body*forward_product + &
          forward_coefficient*dthird_body*forward_product + &
          forward_coefficient*third_body*dforward_product
        dreverse_rate = dreverse_coefficient*third_body*reverse_product + &
          reverse_coefficient*dthird_body*reverse_product + &
          reverse_coefficient*third_body*dreverse_product
        dprogress_rate(variable) = dforward_rate-dreverse_rate
      end do

      do species = 1, air5_num_species
        net_stoichiometry = air5_nu_product(species,reaction) - &
          air5_nu_reactant(species,reaction)
        if (net_stoichiometry == 0) cycle
        mass_factor = air5_molar_mass(species)*real(net_stoichiometry,real64)
        source(species) = source(species)+mass_factor*progress_rate
        jacobian(species,:) = jacobian(species,:)+mass_factor*dprogress_rate

        if (air5_rate_uses_tv(reaction)) then
          if (air5_psi_bounds(2,species) <= 0.0_real64) cycle
          call air5_preferential_factor(species, temperature, psi, dpsi_dtemperature)
          energy_per_mass = psi*air5_species_gas_constant(species)* &
            air5_activation_temperature(reaction)
          source(6) = source(6)+mass_factor*progress_rate*energy_per_mass
          do variable = 1, 6
            denergy_per_mass = dpsi_dtemperature*dtemperature(variable)* &
              air5_species_gas_constant(species)*air5_activation_temperature(reaction)
            jacobian(6,variable) = jacobian(6,variable)+mass_factor*( &
              dprogress_rate(variable)*energy_per_mass + &
              progress_rate*denergy_per_mass)
          end do
        else
          if (air5_psi_bounds(2,species) <= 0.0_real64) cycle
          energy_per_mass = air5_species_vibrational_energy(species, tv)
          source(6) = source(6)+mass_factor*progress_rate*energy_per_mass
          do variable = 1, 6
            denergy_per_mass = air5_species_vibrational_cv(species,tv)*dtv(variable)
            jacobian(6,variable) = jacobian(6,variable)+mass_factor*( &
              dprogress_rate(variable)*energy_per_mass + &
              progress_rate*denergy_per_mass)
          end do
        end if
      end do
      end do
    end if

    if (include_vt) then
      call air5_vt_relaxation_source_derivatives(state(1:air5_num_species), &
        temperature, tv, dtemperature, dtv, vt_source, vt_derivatives, local_status)
      if (local_status /= chemistry_status_ok) then
        status = local_status
        return
      end if
      source(6) = source(6)+vt_source
      jacobian(6,:) = jacobian(6,:)+vt_derivatives
    end if
    if (.not. all(ieee_is_finite(source)) .or. .not. all(ieee_is_finite(jacobian))) then
      status = chemistry_status_nonfinite
    else if (present(reaction_progress)) then
      if (.not. all(ieee_is_finite(reaction_progress))) then
        status = chemistry_status_nonfinite
      else
        status = chemistry_status_ok
      end if
    else
      status = chemistry_status_ok
    end if
  end subroutine air5_instantaneous_source_jacobian

  pure function air5_mass_action(concentration, stoichiometry) result(product)
    real(real64), intent(in) :: concentration(air5_num_species)
    integer, intent(in) :: stoichiometry(air5_num_species)
    real(real64) :: product
    integer :: species, exponent

    product = 1.0_real64
    do species = 1, air5_num_species
      do exponent = 1, stoichiometry(species)
        product = product*concentration(species)
      end do
    end do
  end function air5_mass_action

  pure function air5_mass_action_derivative(concentration, stoichiometry, variable) &
      result(derivative)
    real(real64), intent(in) :: concentration(air5_num_species)
    integer, intent(in) :: stoichiometry(air5_num_species), variable
    real(real64) :: derivative, reduced_product
    integer :: species, exponent, reduced_exponent

    derivative = 0.0_real64
    if (variable > air5_num_species) return
    if (stoichiometry(variable) == 0) return
    reduced_product = 1.0_real64
    do species = 1, air5_num_species
      reduced_exponent = stoichiometry(species)
      if (species == variable) reduced_exponent = reduced_exponent-1
      do exponent = 1, reduced_exponent
        reduced_product = reduced_product*concentration(species)
      end do
    end do
    derivative = real(stoichiometry(variable),real64)*reduced_product/ &
      air5_molar_mass(variable)
  end function air5_mass_action_derivative

  pure subroutine air5_preferential_factor(species, temperature, factor, derivative)
    integer, intent(in) :: species
    real(real64), intent(in) :: temperature
    real(real64), intent(out) :: factor, derivative
    real(real64) :: exponent, raw_factor, raw_derivative

    exponent = air5_psi_coefficient(1,species)/temperature + &
      air5_psi_coefficient(2,species) + &
      air5_psi_coefficient(3,species)*log(temperature) + &
      air5_psi_coefficient(4,species)*temperature + &
      air5_psi_coefficient(5,species)*temperature*temperature
    raw_factor = exp(exponent)
    raw_derivative = raw_factor*(-air5_psi_coefficient(1,species)/(temperature*temperature) + &
      air5_psi_coefficient(3,species)/temperature + &
      air5_psi_coefficient(4,species) + &
      2.0_real64*air5_psi_coefficient(5,species)*temperature)
    if (raw_factor <= air5_psi_bounds(1,species)) then
      factor = air5_psi_bounds(1,species)
      derivative = 0.0_real64
    else if (raw_factor >= air5_psi_bounds(2,species)) then
      factor = air5_psi_bounds(2,species)
      derivative = 0.0_real64
    else
      factor = raw_factor
      derivative = raw_derivative
    end if
  end subroutine air5_preferential_factor

  pure subroutine air5_centered_difference_jacobian(rho, momentum, q5, state, &
      jacobian, status)
    real(real64), intent(in) :: rho, momentum(3), q5, state(6)
    real(real64), intent(out) :: jacobian(6,6)
    integer, intent(out) :: status
    real(real64) :: plus_state(6), minus_state(6), plus_source(6), minus_source(6)
    real(real64) :: step, species_scale
    integer :: variable, plus_status, minus_status

    jacobian = 0.0_real64
    species_scale = max(sum(state(1:air5_num_species)), 1.0e-12_real64)
    do variable = 1, 6
      if (variable <= air5_num_species) then
        step = 1.0e-4_real64*max(abs(state(variable)), 1.0e-4_real64*species_scale)
      else
        step = 5.0e-6_real64*max(abs(state(variable)), 1.0_real64)
      end if
      plus_state = state
      minus_state = state
      plus_state(variable) = plus_state(variable)+step
      minus_state(variable) = minus_state(variable)-step
      call air5_instantaneous_source(rho, momentum, q5, plus_state, plus_source, plus_status)
      call air5_instantaneous_source(rho, momentum, q5, minus_state, minus_source, minus_status)
      if (plus_status /= chemistry_status_ok .or. minus_status /= chemistry_status_ok) then
        status = chemistry_status_invalid_composition
        return
      end if
      jacobian(:,variable) = (plus_source-minus_source)/(2.0_real64*step)
    end do
    status = chemistry_status_ok
  end subroutine air5_centered_difference_jacobian

  pure subroutine air5_admissible_difference_jacobian(rho, momentum, q5, state, &
      jacobian, status)
    real(real64), intent(in) :: rho, momentum(3), q5, state(6)
    real(real64), intent(out) :: jacobian(6,6)
    integer, intent(out) :: status
    real(real64) :: base_source(6), plus_half_source(6), plus_source(6)
    real(real64) :: plus2_source(6), minus_source(6)
    real(real64) :: plus_half_state(6), plus_state(6), plus2_state(6), minus_state(6)
    real(real64) :: step, species_scale
    integer :: variable, base_status, plus_half_status, plus_status
    integer :: plus2_status, minus_status

    jacobian = 0.0_real64
    call air5_instantaneous_source(rho, momentum, q5, state, base_source, base_status)
    if (base_status /= chemistry_status_ok) then
      status = base_status
      return
    end if

    species_scale = max(sum(state(1:air5_num_species)), 1.0e-12_real64)
    do variable = 1, 6
      if (variable <= air5_num_species) then
        step = 1.0e-4_real64*max(abs(state(variable)), 1.0e-4_real64*species_scale)
      else
        step = 5.0e-6_real64*max(abs(state(variable)), 1.0_real64)
      end if

      if (state(variable) > 2.0_real64*step) then
        plus_state = state
        plus_state(variable) = plus_state(variable)+step
        call air5_instantaneous_source(rho, momentum, q5, plus_state, &
          plus_source, plus_status)
        if (plus_status /= chemistry_status_ok) then
          status = plus_status
          return
        end if
        minus_state = state
        minus_state(variable) = minus_state(variable)-step
        call air5_instantaneous_source(rho, momentum, q5, minus_state, &
          minus_source, minus_status)
        if (minus_status /= chemistry_status_ok) then
          status = minus_status
          return
        end if
        jacobian(:,variable) = (plus_source-minus_source)/(2.0_real64*step)
      else
        plus_half_state = state
        plus_half_state(variable) = plus_half_state(variable)+0.5_real64*step
        call air5_instantaneous_source(rho, momentum, q5, plus_half_state, &
          plus_half_source, plus_half_status)
        if (plus_half_status /= chemistry_status_ok) then
          status = plus_half_status
          return
        end if
        plus_state = state
        plus_state(variable) = plus_state(variable)+step
        call air5_instantaneous_source(rho, momentum, q5, plus_state, &
          plus_source, plus_status)
        if (plus_status /= chemistry_status_ok) then
          status = plus_status
          return
        end if
        plus2_state = state
        plus2_state(variable) = plus2_state(variable)+2.0_real64*step
        call air5_instantaneous_source(rho, momentum, q5, plus2_state, &
          plus2_source, plus2_status)
        if (plus2_status /= chemistry_status_ok) then
          status = plus2_status
          return
        end if
        jacobian(:,variable) = (4.0_real64*(-3.0_real64*base_source + &
          4.0_real64*plus_half_source-plus_source)/step - &
          (-3.0_real64*base_source+4.0_real64*plus_source-plus2_source)/ &
          (2.0_real64*step))/3.0_real64
      end if
    end do
    status = chemistry_status_ok
  end subroutine air5_admissible_difference_jacobian

  pure subroutine air5_source_conservation_residuals(source, mass_residual, &
      nitrogen_residual, oxygen_residual)
    real(real64), intent(in) :: source(6)
    real(real64), intent(out) :: mass_residual, nitrogen_residual, oxygen_residual
    real(real64) :: mass_scale, element_source, element_scale
    integer :: species, element

    mass_scale = sum(abs(source(1:air5_num_species)))
    if (mass_scale == 0.0_real64) then
      mass_residual = 0.0_real64
    else
      mass_residual = abs(sum(source(1:air5_num_species)))/mass_scale
    end if
    nitrogen_residual = 0.0_real64
    oxygen_residual = 0.0_real64
    do element = 1, 2
      element_source = 0.0_real64
      element_scale = 0.0_real64
      do species = 1, air5_num_species
        element_source = element_source + real(air5_atom_count(element,species),real64)* &
          source(species)/air5_molar_mass(species)
        element_scale = element_scale + abs(real(air5_atom_count(element,species),real64)* &
          source(species)/air5_molar_mass(species))
      end do
      if (element_scale > 0.0_real64) element_source = abs(element_source)/element_scale
      if (element == 1) nitrogen_residual = element_source
      if (element == 2) oxygen_residual = element_source
    end do
  end subroutine air5_source_conservation_residuals

end module chemistry_source
