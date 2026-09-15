program chemistry_radau_probe
  use iso_fortran_env, only: real64
  use chemistry_air5_data, only: air5_num_reactions
  use chemistry_model, only: chemistry_status_ok
  use chemistry_thermo, only: air5_ev_from_tv, air5_q5_from_state
  use chemistry_source, only: air5_instantaneous_source
  implicit none

  real(real64) :: rho, momentum(3), q5, state(6), source(6)
  real(real64) :: reaction_progress(air5_num_reactions)
  character(len=32) :: mode
  integer :: status

  call get_command_argument(1,mode)
  if (trim(mode) /= 'source') error stop 'unknown chemistry Radau probe mode'
  rho = 0.05_real64
  momentum = rho*[40.0_real64,-5.0_real64,2.0_real64]
  state(1:5) = rho*[0.55_real64,0.15_real64,0.10_real64,0.12_real64,0.08_real64]
  call air5_ev_from_tv(state(1:5),1000.0_real64,state(6),status)
  if (status /= chemistry_status_ok) error stop 'Ev construction failed'
  call air5_q5_from_state(rho,momentum,state(1:5),state(6),6000.0_real64,q5,status)
  if (status /= chemistry_status_ok) error stop 'q5 construction failed'
  call air5_instantaneous_source(rho,momentum,q5,state,source,status, &
    reaction_progress=reaction_progress)
  if (status /= chemistry_status_ok) error stop 'source evaluation failed'
  write(*,'(2(es24.16,1x))') rho,q5
  write(*,'(3(es24.16,1x))') momentum
  write(*,'(6(es24.16,1x))') state
  write(*,'(6(es24.16,1x))') source
  write(*,'(12(es24.16,1x))') reaction_progress
end program chemistry_radau_probe
