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
