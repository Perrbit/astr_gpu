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
