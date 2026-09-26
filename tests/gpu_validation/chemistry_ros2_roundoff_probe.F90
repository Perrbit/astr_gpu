! Diagnostic only: fixed-step chemistry, not the production adaptive integrator.
program chemistry_ros2_roundoff_probe
  use iso_fortran_env, only: real64
  use ieee_arithmetic, only: ieee_is_finite
  use chemistry_air5_data, only: air5_atom_count, air5_molar_mass, air5_ru
  use chemistry_model, only: chemistry_status_ok, air5_validate_physical_species_state
  use chemistry_thermo, only: air5_ev_from_tv, air5_q5_from_state, &
    air5_temperature_from_q5, air5_tv_from_ev
  use chemistry_ros2, only: air5_ros2_fixed_step
  implicit none
  ! NVHPC has no extended host REAL; report the actual accumulator precision.
  integer, parameter :: wide = max(real64,selected_real_kind(18))
  real(real64) :: rho, momentum(3), q5, initial(6), state(6,3), candidate(6)
  real(real64) :: terms(6,2), estimate(6), carry(6), trial_carry(6), trial(6)
  real(real64) :: composition(5), temperature, tv, dt, ysum, err, atol(6)
  real(real64) :: max_yerr(3), max_error_norm(3), final_t(3), final_tv(3)
  real(wide) :: initial_elements(2), elements(2), max_mass(3), max_elements(2,3)
  integer :: steps, iteration, path, status, species, atom, term, first_crossing(3)
  character(len=64) :: argument

  steps = 20000
  dt = 5.0e-10_real64
  temperature = 2000.0_real64
  if (command_argument_count() >= 1) then
    call get_command_argument(1,argument)
    read(argument,*) steps
  end if
  if (command_argument_count() >= 2) then
    call get_command_argument(2,argument)
    read(argument,*) dt
  end if
  if (command_argument_count() >= 3) then
    call get_command_argument(3,argument)
    read(argument,*) temperature
  end if
  if (steps <= 0 .or. .not. ieee_is_finite(dt) .or. dt <= 0.0_real64) &
    error stop 'invalid probe controls'
  composition = [0.767_real64,0.233_real64,0.0_real64,0.0_real64,0.0_real64]
  rho = 20000.0_real64/(temperature*sum(composition*air5_ru/air5_molar_mass))
  momentum = rho*[2000.0_real64,0.0_real64,0.0_real64]
  initial(1:5) = rho*composition
  call air5_ev_from_tv(initial(1:5),temperature,initial(6),status)
  if (status /= chemistry_status_ok) error stop 'initial Ev invalid'
  call air5_q5_from_state(rho,momentum,initial(1:5),initial(6),temperature,q5,status)
  if (status /= chemistry_status_ok) error stop 'initial q5 invalid'
  do atom = 1, 2
    initial_elements(atom) = sum(real(initial(1:5),wide)* &
      real(air5_atom_count(atom,:),wide)/real(air5_molar_mass,wide))
  end do
  do path = 1, 3
    state(:,path) = initial
  end do
  carry = 0.0_real64
  max_mass = 0.0_wide
  max_elements = 0.0_wide
  max_yerr = 0.0_real64
  max_error_norm = 0.0_real64
  first_crossing = 0
  atol(1:5) = 1.0e-13_real64*rho
  atol(6) = 1.0e-13_real64*max(abs(initial(6)),1.0_real64)
  do iteration = 1, steps
    do path = 1, 3
      call air5_ros2_fixed_step(rho,momentum,q5,state(:,path),dt,candidate, &
        estimate,status,update_terms=terms)
      if (status /= chemistry_status_ok) then
        write(*,*) 'FAILED_STEP_PATH_STATUS',iteration,path,status
        error stop 'ROS2 failed; no clipping or fallback'
      end if
      trial = candidate
      if (path == 2) trial = state(:,path)+(terms(:,1)+terms(:,2))
      if (path == 3) then
        trial = state(:,path)
        trial_carry = carry
        do term = 1, 2
          call compensated_add(trial,trial_carry,terms(:,term))
        end do
      end if
      call air5_validate_physical_species_state(rho,trial(1:5),status)
      if (status /= chemistry_status_ok) error stop 'trial species invalid'
      call air5_temperature_from_q5(rho,momentum,trial(1:5),trial(6),q5, &
        final_t(path),status)
      if (status /= chemistry_status_ok) error stop 'trial T invalid'
      call air5_tv_from_ev(trial(1:5),trial(6),final_tv(path),status)
      if (status /= chemistry_status_ok) error stop 'trial Tv invalid'
      err = sqrt(sum((estimate/(atol+1.0e-9_real64* &
        max(abs(state(:,path)),abs(trial))))**2)/6.0_real64)
      max_error_norm(path) = max(max_error_norm(path),err)
      if (err > 1.0_real64) error stop 'fixed diagnostic step exceeds adaptive tolerance'
      ! Only commit the carry after all candidate gates succeed.
      state(:,path) = trial
      if (path == 3) carry = trial_carry
      ysum = 0.0_real64
      do species = 1, 5
        ysum = ysum+trial(species)/rho
      end do
      max_yerr(path) = max(max_yerr(path),abs(ysum-1.0_real64))
      if (abs(ysum-1.0_real64) > 128.0_real64*epsilon(rho) .and. &
          first_crossing(path) == 0) first_crossing(path) = iteration
      max_mass(path) = max(max_mass(path), &
        abs(sum(real(trial(1:5),wide))-sum(real(initial(1:5),wide)))/real(rho,wide))
      do atom = 1, 2
        elements(atom) = sum(real(trial(1:5),wide)* &
          real(air5_atom_count(atom,:),wide)/real(air5_molar_mass,wide))
        max_elements(atom,path) = max(max_elements(atom,path), &
          abs(elements(atom)-initial_elements(atom))/initial_elements(atom))
      end do
    end do
  end do
  write(*,*) 'DIAGNOSTIC_FIXED_CHEMISTRY_ONLY',steps,dt,temperature
  write(*,*) 'WIDE_DIGITS',digits(1.0_wide)
  write(*,*) 'PATH: 1=original 2=grouped 3=persistent_compensation'
  do path = 1, 3
    write(*,'(a,i0,6(1x,es24.16),1x,i0)') 'DRIFT ',path, &
      real(max_mass(path),real64),real(max_elements(:,path),real64), &
      max_yerr(path),max_error_norm(path),minval(state(1:5,path)),first_crossing(path)
    write(*,'(a,i0,8(1x,es24.16))') 'STATE ',path,state(:,path),final_t(path),final_tv(path)
  end do
  write(*,'(a,1x,es24.16)') 'FIXED_Q5',q5
contains
  subroutine compensated_add(value,correction,increment)
    real(real64), intent(inout) :: value(6), correction(6)
    real(real64), intent(in) :: increment(6)
    real(real64) :: adjusted(6), updated(6)
    adjusted = increment-correction
    updated = value+adjusted
    correction = (updated-value)-adjusted
    value = updated
  end subroutine compensated_add
end program chemistry_ros2_roundoff_probe
