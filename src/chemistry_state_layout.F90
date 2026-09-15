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
