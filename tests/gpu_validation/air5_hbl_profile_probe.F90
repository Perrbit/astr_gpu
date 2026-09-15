program air5_hbl_profile_probe
  use iso_fortran_env, only: real64
  use chemistry_air5_data, only: air5_num_species
  use chemistry_model, only: chemistry_status_ok
  use chemistry_state_layout, only: air5_num_conservative
  use chemistry_flow_state, only: air5_conservative_to_primitive
  use chemistry_hbl_profile, only: air5_hbl_profile_type, &
    air5_hbl_profile_status_ok,read_air5_hbl_profile, &
    sample_air5_hbl_profile,air5_hbl_profile_bounds, &
    air5_hbl_profile_point_count
  implicit none

  type(air5_hbl_profile_type) :: profile
  character(len=32) :: mode
  character(len=1024) :: path
  real(real64) :: y_min,y_max,y_sample,q(air5_num_conservative)
  real(real64) :: rho,velocity(3),temperature,mass_fraction(air5_num_species)
  real(real64) :: tv,pressure
  integer :: status,state_status

  call get_command_argument(1,mode)
  call get_command_argument(2,path)
  call read_air5_hbl_profile(trim(path),profile,status)
  if(status/=air5_hbl_profile_status_ok) then
    write(*,'(I0)') status
    stop
  endif
  call air5_hbl_profile_bounds(profile,y_min,y_max,status)
  if(status/=air5_hbl_profile_status_ok) then
    write(*,'(I0)') status
    stop
  endif

  select case(trim(mode))
  case('lower')
    y_sample=y_min
  case('upper')
    y_sample=y_max
  case('midpoint')
    y_sample=0.5_real64*(y_min+y_max)
  case default
    error stop 'unknown air5 HBL profile probe mode'
  end select
  call sample_air5_hbl_profile(profile,y_sample,q,status)
  if(status/=air5_hbl_profile_status_ok) then
    write(*,'(I0)') status
    stop
  endif
  call air5_conservative_to_primitive(q,rho,velocity,temperature,mass_fraction, &
    tv,pressure,state_status)
  if(state_status/=chemistry_status_ok) error stop 'sample reconstruction failed'
  write(*,'(2(I0,1X),15(ES25.16E3,1X))') status, &
    air5_hbl_profile_point_count(profile),y_min,y_max,y_sample,rho,velocity, &
    pressure,temperature,tv,mass_fraction,sum(mass_fraction)
end program air5_hbl_profile_probe
