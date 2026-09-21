program chemistry_state_layout_probe
  use iso_fortran_env, only: real64
  use chemistry_model, only: chemistry_status_ok,chemistry_status_invalid_composition, &
    air5_limit_filter_species
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
  case ('valid_filter')
    call layout_case(.true.,.false.,.true.,'none',5,-7,-9,1,11,air5_layout_status_ok)
  case ('disabled')
    call layout_case(.false.,.true.,.true.,'k-omega',2,2,9,2,9,air5_layout_status_ok)
  case ('limit_filter_undershoot')
    call limit_filter_case(.false.)
  case ('limit_filter_roundoff')
    call limit_filter_roundoff_case()
  case ('reject_filter_invalid_base')
    call limit_filter_case(.true.)
  case default
    error stop 'unknown chemistry state-layout probe mode'
  end select

contains

  subroutine limit_filter_case(invalid_base)
    logical, intent(in) :: invalid_base
    real(real64) :: base_species(5),filtered_species(5),theta
    integer :: status
    logical :: limited

    base_species=[0.5999999999999_real64,0.2_real64,0.1_real64,0.1_real64,1.0e-13_real64]
    if(invalid_base) base_species(5)=-1.0e-4_real64
    filtered_species=[0.600000000001_real64,0.2_real64,0.1_real64,0.1_real64,-1.0e-12_real64]
    call air5_limit_filter_species(1.0_real64,base_species,1.0_real64, &
      filtered_species,limited,theta,status)
    write(*,'(2(i0,1x),4(es24.16,1x))') status,merge(1,0,limited),theta, &
      minval(filtered_species),sum(filtered_species)-1.0_real64,filtered_species(1)
  end subroutine limit_filter_case

  subroutine limit_filter_roundoff_case()
    real(real64) :: base_species(5),filtered_species(5),theta
    integer :: status
    logical :: limited

    base_species=[0.5999999999999_real64,0.2_real64,0.1_real64,0.1_real64,1.0e-13_real64]
    filtered_species=[0.60000000000005_real64,0.2_real64,0.1_real64,0.1_real64,-5.0e-14_real64]
    call air5_limit_filter_species(1.0_real64,base_species,1.0_real64, &
      filtered_species,limited,theta,status)
    write(*,'(2(i0,1x),3(es24.16,1x))') status,merge(1,0,limited),theta, &
      minval(filtered_species),sum(filtered_species)-1.0_real64
  end subroutine limit_filter_roundoff_case

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
