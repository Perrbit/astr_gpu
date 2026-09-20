module chemistry_model
  use iso_fortran_env, only: real64
  use ieee_arithmetic, only: ieee_is_finite
  use chemistry_air5_data, only: air5_mechanism_id, air5_num_species, &
    air5_pressure_min_pa, air5_pressure_max_pa
  implicit none
  private

  integer, parameter, public :: chemistry_status_ok = 0
  integer, parameter, public :: chemistry_status_invalid_mechanism = 1
  integer, parameter, public :: chemistry_status_invalid_density = 2
  integer, parameter, public :: chemistry_status_invalid_composition = 3
  integer, parameter, public :: chemistry_status_out_of_domain = 4
  integer, parameter, public :: chemistry_status_no_vibrational_capacity = 5
  integer, parameter, public :: chemistry_status_nonconverged = 6
  integer, parameter, public :: chemistry_status_nonfinite = 7
  integer, parameter, public :: chemistry_status_linear_failure = 8
  integer, parameter, public :: chemistry_status_invalid_timestep = 9
  integer, parameter, public :: chemistry_status_invalid_tolerance = 10
  integer, parameter, public :: chemistry_status_step_limit = 11
  integer, parameter, public :: chemistry_status_invalid_source_mode = 12
  real(real64), parameter, public :: air5_flux_limiter_safety = &
    1.0_real64-1.0e-8_real64

  public :: air5_validate_mechanism_id
  public :: air5_validate_partial_densities
  public :: air5_validate_physical_species_state
  public :: air5_pressure_is_in_domain

contains

  pure subroutine air5_validate_mechanism_id(identifier, status)
    character(len=*), intent(in) :: identifier
    integer, intent(out) :: status

    status = chemistry_status_ok
    if (trim(identifier) /= air5_mechanism_id) status = chemistry_status_invalid_mechanism
  end subroutine air5_validate_mechanism_id

  pure subroutine air5_validate_partial_densities(rho_species, status)
    real(real64), intent(in) :: rho_species(air5_num_species)
    integer, intent(out) :: status

    status = chemistry_status_ok
    if (.not. all(ieee_is_finite(rho_species))) then
      status = chemistry_status_nonfinite
    else if (any(rho_species < 0.0_real64) .or. &
             sum(rho_species) <= tiny(1.0_real64)) then
      status = chemistry_status_invalid_composition
    end if
  end subroutine air5_validate_partial_densities

  pure subroutine air5_validate_physical_species_state(rho, rho_species, status)
    real(real64), intent(in) :: rho
    real(real64), intent(in) :: rho_species(air5_num_species)
    integer, intent(out) :: status
    real(real64) :: tolerance

    status = chemistry_status_ok
    if (.not. ieee_is_finite(rho)) then
      status = chemistry_status_nonfinite
      return
    end if
    if (rho <= 0.0_real64) then
      status = chemistry_status_invalid_density
      return
    end if
    call air5_validate_partial_densities(rho_species, status)
    if (status /= chemistry_status_ok) return
    tolerance = 1.0e-10_real64*max(rho, 1.0_real64)
    if (abs(sum(rho_species)-rho) > tolerance) status = chemistry_status_invalid_composition
  end subroutine air5_validate_physical_species_state

  pure function air5_pressure_is_in_domain(pressure) result(is_in_domain)
    real(real64), intent(in) :: pressure
    logical :: is_in_domain
    real(real64) :: bound_tolerance

    bound_tolerance = 64.0_real64*epsilon(1.0_real64)* &
      max(abs(air5_pressure_max_pa),1.0_real64)
    is_in_domain = ieee_is_finite(pressure) .and. &
      pressure >= air5_pressure_min_pa-bound_tolerance .and. &
      pressure <= air5_pressure_max_pa+bound_tolerance
  end function air5_pressure_is_in_domain

end module chemistry_model

module chemistry_state_layout
  use chemistry_air5_data, only: air5_num_species
  implicit none
  private

  integer, parameter, public :: air5_idx_density = 1
  integer, parameter, public :: air5_idx_momentum_first = 2
  integer, parameter, public :: air5_idx_momentum_last = 4
  integer, parameter, public :: air5_idx_total_energy = 5
  integer, parameter, public :: air5_idx_species_first = 6
  integer, parameter, public :: air5_idx_species_last = &
    air5_idx_species_first + air5_num_species - 1
  integer, parameter, public :: air5_idx_ev = air5_idx_species_last + 1
  integer, parameter, public :: air5_num_mode_equations = 1
  integer, parameter, public :: air5_num_conservative = air5_idx_ev

  integer, parameter, public :: air5_layout_status_ok = 0
  integer, parameter, public :: air5_layout_status_invalid_species = 1
  integer, parameter, public :: air5_layout_status_turbulence = 2
  integer, parameter, public :: air5_layout_status_filter = 3
  integer, parameter, public :: air5_layout_status_nondimensional = 4

  public :: air5_configure_runtime_layout

contains

  pure subroutine air5_configure_runtime_layout(enabled,nondimen,lfilter,turbmode, &
      num_species,num_modequ,numq,status)
    logical, intent(in) :: enabled, nondimen, lfilter
    character(len=*), intent(in) :: turbmode
    integer, intent(in) :: num_species
    integer, intent(inout) :: num_modequ, numq
    integer, intent(out) :: status

    status = air5_layout_status_ok
    if(.not.enabled) return

    if(nondimen) then
      status = air5_layout_status_nondimensional
      return
    endif
    if(num_species /= air5_num_species) then
      status = air5_layout_status_invalid_species
      return
    endif
    if(trim(turbmode) /= 'none') then
      status = air5_layout_status_turbulence
      return
    endif
    if(lfilter) then
      status = air5_layout_status_filter
      return
    endif

    num_modequ = air5_num_mode_equations
    numq = air5_num_conservative
  end subroutine air5_configure_runtime_layout

end module chemistry_state_layout
