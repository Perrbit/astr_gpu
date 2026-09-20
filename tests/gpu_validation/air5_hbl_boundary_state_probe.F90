program air5_hbl_boundary_state_probe
  use iso_fortran_env, only: real64
  use chemistry_air5_data, only: air5_num_species
  use chemistry_model, only: chemistry_status_ok
  use chemistry_state_layout, only: air5_num_conservative
  use chemistry_flow_state, only: air5_primitive_to_conservative, &
    air5_conservative_to_primitive
  use chemistry_hbl_boundary_state, only: build_air5_hbl_wall_state, &
    build_air5_hbl_outflow_state
  implicit none

  character(len=32) :: mode
  real(real64) :: q1(air5_num_conservative),q2(air5_num_conservative)
  real(real64) :: q_high(air5_num_conservative),limiter_theta
  real(real64) :: result(air5_num_conservative),rho,velocity(3),temperature,tv,pressure
  real(real64) :: y1(air5_num_species),y2(air5_num_species),mass_fraction(air5_num_species)
  integer :: status,state_status

  y1=[0.70_real64,0.20_real64,0.03_real64,0.04_real64,0.03_real64]
  y2=[0.69_real64,0.21_real64,0.035_real64,0.035_real64,0.03_real64]
  call state_from_pressure(101000.0_real64,2000.0_real64,1800.0_real64, &
    [100.0_real64,5.0_real64,0.0_real64],y1,q1,state_status)
  if(state_status/=chemistry_status_ok) error stop 'failed to build inner state one'
  call state_from_pressure(100000.0_real64,1800.0_real64,1600.0_real64, &
    [90.0_real64,3.0_real64,0.0_real64],y2,q2,state_status)
  if(state_status/=chemistry_status_ok) error stop 'failed to build inner state two'

  call get_command_argument(1,mode)
  select case(trim(mode))
  case('wall')
    call build_air5_hbl_wall_state(q1,q2,2925.0_real64,result,status)
  case('outflow')
    call build_air5_hbl_outflow_state(q1,q2,result,status)
  case('outflow-trace')
    y1=[0.76999899899999997_real64,0.23_real64,2.0e-26_real64, &
      1.0e-6_real64,1.0e-9_real64]
    y2=[0.76999899899999997_real64,0.23_real64,1.0e-25_real64, &
      1.0e-6_real64,1.0e-9_real64]
    call air5_primitive_to_conservative(1.0_real64,[2900.0_real64,0.0_real64,0.0_real64], &
      1500.0_real64,y1,1400.0_real64,q1,state_status)
    if(state_status/=chemistry_status_ok) error stop 'failed to build trace inner state one'
    call air5_primitive_to_conservative(1.0_real64,[2800.0_real64,0.0_real64,0.0_real64], &
      1500.0_real64,y2,1400.0_real64,q2,state_status)
    if(state_status/=chemistry_status_ok) error stop 'failed to build trace inner state two'
    q_high=(4.0_real64*q1-q2)/3.0_real64
    call build_air5_hbl_outflow_state(q1,q2,result,status,limiter_theta)
  case default
    error stop 'unknown air5 HBL boundary-state probe mode'
  end select
  if(status/=chemistry_status_ok) then
    write(*,'(I0)') status
    stop
  endif
  call air5_conservative_to_primitive(result,rho,velocity,temperature, &
    mass_fraction,tv,pressure,state_status)
  if(state_status/=chemistry_status_ok) error stop 'boundary state reconstruction failed'
  if(trim(mode)=='outflow-trace') then
    write(*,'(2(I0,1X),17(ES25.16E3,1X))') status,state_status,rho,velocity, &
      pressure,temperature,tv,mass_fraction,sum(mass_fraction),limiter_theta, &
      minval(q_high(6:10)),minval(result(6:10)), &
      max(maxval(abs(result(1:5)-q_high(1:5))),abs(result(11)-q_high(11)))
  else
    write(*,'(2(I0,1X),14(ES25.16E3,1X))') status,state_status,rho,velocity, &
      pressure,temperature,tv,mass_fraction,sum(mass_fraction), &
      maxval(abs(result-(4.0_real64*q1-q2)/3.0_real64))
  endif

contains

  subroutine state_from_pressure(p,t,tv_value,u,y,q,status)
    use chemistry_air5_data, only: air5_molar_mass,air5_ru
    real(real64), intent(in) :: p,t,tv_value,u(3),y(air5_num_species)
    real(real64), intent(out) :: q(air5_num_conservative)
    integer, intent(out) :: status
    real(real64) :: density,mixture_gas_constant

    mixture_gas_constant=sum(y*air5_ru/air5_molar_mass)
    density=p/(mixture_gas_constant*t)
    call air5_primitive_to_conservative(density,u,t,y,tv_value,q,status)
  end subroutine state_from_pressure

end program air5_hbl_boundary_state_probe
