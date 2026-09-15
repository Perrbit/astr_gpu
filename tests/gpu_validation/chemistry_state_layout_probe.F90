program chemistry_state_layout_probe
  use chemistry_state_layout, only: air5_idx_density, air5_idx_momentum_first, &
    air5_idx_momentum_last, air5_idx_total_energy, air5_idx_species_first, &
    air5_idx_species_last, air5_idx_ev, air5_num_conservative, &
    air5_layout_status_ok, air5_layout_status_invalid_species, &
    air5_layout_status_turbulence, air5_layout_status_filter, &
    air5_layout_status_nondimensional, &
    air5_configure_runtime_layout
  implicit none

  character(len=32) :: mode

  call get_command_argument(1,mode)
  select case (trim(mode))
  case ('indices')
    write(*,'(8(i0,1x))') air5_idx_density,air5_idx_momentum_first, &
      air5_idx_momentum_last,air5_idx_total_energy,air5_idx_species_first, &
    air5_idx_species_last,air5_idx_ev,air5_num_conservative
  case ('valid')
    call layout_case(.true.,.false.,.false.,'none',5,-7,-9,1,11,air5_layout_status_ok)
  case ('invalid_nondimensional')
    call layout_case(.true.,.true.,.false.,'none',5,4,17,4,17, &
      air5_layout_status_nondimensional)
  case ('invalid_species')
    call layout_case(.true.,.false.,.false.,'none',4,-7,-9,-7,-9, &
      air5_layout_status_invalid_species)
  case ('invalid_turbulence')
    call layout_case(.true.,.false.,.false.,'k-omega',5,2,12,2,12, &
      air5_layout_status_turbulence)
  case ('invalid_filter')
    call layout_case(.true.,.false.,.true.,'none',5,0,10,0,10,air5_layout_status_filter)
  case ('disabled')
    call layout_case(.false.,.true.,.true.,'k-omega',2,2,9,2,9,air5_layout_status_ok)
  case default
    error stop 'unknown chemistry state-layout probe mode'
  end select

contains

  subroutine layout_case(enabled,nondimen,lfilter,turbmode,num_species,initial_modequ,initial_numq, &
      expected_modequ,expected_numq,expected_status)
    logical, intent(in) :: enabled, nondimen, lfilter
    character(len=*), intent(in) :: turbmode
    integer, intent(in) :: num_species, initial_modequ, initial_numq
    integer, intent(in) :: expected_modequ, expected_numq, expected_status
    integer :: num_modequ, numq, status

    num_modequ = initial_modequ
    numq = initial_numq
    call air5_configure_runtime_layout(enabled,nondimen,lfilter,turbmode,num_species, &
      num_modequ,numq,status)
    write(*,'(5(i0,1x))') num_modequ,numq,status,expected_modequ,expected_numq
    write(*,'(2(i0,1x))') status,expected_status
  end subroutine layout_case

end program chemistry_state_layout_probe
