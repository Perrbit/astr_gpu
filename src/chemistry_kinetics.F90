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

module chemistry_linear6
  use iso_fortran_env, only: real64
  use ieee_arithmetic, only: ieee_is_finite
  use chemistry_model, only: chemistry_status_ok, chemistry_status_linear_failure, &
    chemistry_status_nonfinite
  implicit none
  private

  integer, parameter :: linear_size = 6

  public :: air5_lu_factor_6
  public :: air5_lu_solve_6

contains

  pure subroutine air5_lu_factor_6(matrix, lu, pivots, status)
    real(real64), intent(in) :: matrix(linear_size,linear_size)
    real(real64), intent(out) :: lu(linear_size,linear_size)
    integer, intent(out) :: pivots(linear_size), status
    real(real64) :: row_scale(linear_size), scaled_pivot, best_scaled_pivot
    real(real64) :: row_buffer(linear_size), scale_buffer
    integer :: column, pivot, row

    lu = matrix
    pivots = [(row,row=1,linear_size)]
    if (.not. all(ieee_is_finite(matrix))) then
      status = chemistry_status_nonfinite
      return
    end if
    do row = 1, linear_size
      row_scale(row) = maxval(abs(matrix(row,:)))
    end do
    if (any(row_scale == 0.0_real64)) then
      status = chemistry_status_linear_failure
      return
    end if

    do column = 1, linear_size-1
      pivot = column
      best_scaled_pivot = abs(lu(column,column))/row_scale(column)
      do row = column+1, linear_size
        scaled_pivot = abs(lu(row,column))/row_scale(row)
        if (scaled_pivot > best_scaled_pivot) then
          pivot = row
          best_scaled_pivot = scaled_pivot
        end if
      end do
      if (best_scaled_pivot <= 64.0_real64*epsilon(1.0_real64)) then
        status = chemistry_status_linear_failure
        return
      end if
      pivots(column) = pivot
      if (pivot /= column) then
        row_buffer = lu(column,:)
        lu(column,:) = lu(pivot,:)
        lu(pivot,:) = row_buffer
        scale_buffer = row_scale(column)
        row_scale(column) = row_scale(pivot)
        row_scale(pivot) = scale_buffer
      end if
      do row = column+1, linear_size
        lu(row,column) = lu(row,column)/lu(column,column)
        lu(row,column+1:linear_size) = lu(row,column+1:linear_size) - &
          lu(row,column)*lu(column,column+1:linear_size)
      end do
    end do
    if (abs(lu(linear_size,linear_size))/row_scale(linear_size) <= &
        64.0_real64*epsilon(1.0_real64) .or. &
        .not. all(ieee_is_finite(lu))) then
      status = chemistry_status_linear_failure
      return
    end if
    status = chemistry_status_ok
  end subroutine air5_lu_factor_6

  pure subroutine air5_lu_solve_6(lu, pivots, rhs, solution, status)
    real(real64), intent(in) :: lu(linear_size,linear_size), rhs(linear_size)
    integer, intent(in) :: pivots(linear_size)
    real(real64), intent(out) :: solution(linear_size)
    integer, intent(out) :: status
    real(real64) :: value
    integer :: column, pivot, row

    solution = 0.0_real64
    if (.not. all(ieee_is_finite(lu)) .or. .not. all(ieee_is_finite(rhs))) then
      status = chemistry_status_nonfinite
      return
    end if
    if (any(pivots < 1) .or. any(pivots > linear_size)) then
      status = chemistry_status_linear_failure
      return
    end if

    solution = rhs
    do column = 1, linear_size-1
      pivot = pivots(column)
      if (pivot /= column) then
        value = solution(column)
        solution(column) = solution(pivot)
        solution(pivot) = value
      end if
    end do
    do row = 2, linear_size
      solution(row) = solution(row) - &
        dot_product(lu(row,1:row-1),solution(1:row-1))
    end do
    do row = linear_size, 1, -1
      if (lu(row,row) == 0.0_real64) then
        solution = 0.0_real64
        status = chemistry_status_linear_failure
        return
      end if
      solution(row) = (solution(row) - &
        dot_product(lu(row,row+1:linear_size),solution(row+1:linear_size)))/lu(row,row)
    end do
    if (.not. all(ieee_is_finite(solution))) then
      solution = 0.0_real64
      status = chemistry_status_nonfinite
    else
      status = chemistry_status_ok
    end if
  end subroutine air5_lu_solve_6

end module chemistry_linear6

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
