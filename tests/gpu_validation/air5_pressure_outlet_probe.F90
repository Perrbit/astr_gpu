program air5_pressure_outlet_probe
  use iso_fortran_env, only: real64
  use ieee_arithmetic, only: ieee_value,ieee_quiet_nan
  use chemistry_flow_state, only: air5_primitive_to_conservative,air5_conservative_to_primitive
  use chemistry_pressure_outlet_state, only: build_air5_pressure_outlet_state
  use chemistry_thermo, only: air5_species_gas_constant,air5_species_cv_tr
  implicit none
  real(real64) :: inner(11),boundary(11),y(5),yb(5),u(3),ub(3),rho,t,tv,p,c,r,cv
  real(real64) :: rb,tb,tvb,pb,target,dp
  integer :: s,status,n

  y=[0.70_real64,0.20_real64,0.03_real64,0.04_real64,0.03_real64]
  r=0.0_real64; cv=0.0_real64
  do s=1,5
    r=r+y(s)*air5_species_gas_constant(s)
    cv=cv+y(s)*air5_species_cv_tr(s)
  enddo
  rho=0.2_real64; t=4000.0_real64; tv=1700.0_real64
  p=rho*r*t
  c=sqrt((1.0_real64+r/cv)*p/rho)
  u=[0.4_real64*c,12.0_real64,-7.0_real64]
  call air5_primitive_to_conservative(rho,u,t,y,tv,inner,status)
  if(status/=0) error stop 'probe initial state'
  do n=-1,1
    target=p*(1.0_real64+0.05_real64*n)
    call build_air5_pressure_outlet_state(inner,target,boundary,status)
    if(status/=0) error stop 'valid pressure outlet rejected'
    call air5_conservative_to_primitive(boundary,rb,ub,tb,yb,tvb,pb,status)
    if(status/=0) error stop 'pressure outlet reconstruction'
    dp=target-p
    if(abs(pb-target)>2.0e-9_real64) error stop 'pressure target'
    if(abs(rb-rho-dp/(c*c))>2.0e-14_real64) error stop 'outgoing entropy'
    if(abs(ub(1)-u(1)+dp/(rho*c))>2.0e-10_real64) error stop 'outgoing acoustic'
    if(maxval(abs(ub(2:3)-u(2:3)))>2.0e-12_real64) error stop 'tangential velocity'
    if(maxval(abs(yb-y))>2.0e-14_real64) error stop 'composition'
    if(abs(tvb-tv)>2.0e-9_real64) error stop 'vibrational temperature'
    if(abs(sum(boundary(6:10))-boundary(1))>2.0e-14_real64) error stop 'mass closure'
  enddo
  call build_air5_pressure_outlet_state(inner,-p,boundary,status)
  if(status==0) error stop 'negative pressure accepted'
  target=ieee_value(p,ieee_quiet_nan)
  call build_air5_pressure_outlet_state(inner,target,boundary,status)
  if(status==0) error stop 'NaN pressure accepted'
  call build_air5_pressure_outlet_state(inner,2.0_real64*p,boundary,status)
  if(status==0) error stop 'pressure correction reversal accepted'
  u(1)=-1.0_real64
  call air5_primitive_to_conservative(rho,u,t,y,tv,inner,status)
  call build_air5_pressure_outlet_state(inner,p,boundary,status)
  if(status==0) error stop 'inflow accepted'
  u(1)=2.0_real64*c
  call air5_primitive_to_conservative(rho,u,t,y,tv,inner,status)
  call build_air5_pressure_outlet_state(inner,1.05_real64*p,boundary,status)
  if(status/=0 .or. any(boundary/=inner)) error stop 'supersonic state changed'
  print '(A)', 'AIR5_PRESSURE_OUTLET_PROBE_PASS'
end program air5_pressure_outlet_probe
