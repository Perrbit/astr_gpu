program air5_transport_roundoff_probe
  use iso_fortran_env, only: real64
  use chemistry_model, only: air5_transport_species_ratio,air5_transport_roundoff_gamma
  implicit none
  real(real64) :: base,negative,operand_scale,theta,scaled_theta,factor
  integer :: power

  base=1.0e-38_real64
  negative=-base
  operand_scale=16.0_real64*base
  theta=air5_transport_species_ratio(base,negative,operand_scale)
  if(theta<=0.0_real64 .or. theta>=1.0_real64) error stop 'ratio must reserve roundoff'
  if(base+theta*negative<air5_transport_roundoff_gamma*operand_scale) &
    error stop 'insufficient cancellation reserve'
  do power=-100,100,100
    factor=scale(1.0_real64,power)
    scaled_theta=air5_transport_species_ratio(base*factor,negative*factor,operand_scale*factor)
    if(scaled_theta/=theta) error stop 'ratio depends on absolute species magnitude'
  enddo
  if(air5_transport_species_ratio(0.0_real64,0.0_real64,0.0_real64)/=1.0_real64) &
    error stop 'exact zero must remain unmodified'
  if(air5_transport_species_ratio(0.0_real64,-base,base)/=0.0_real64) &
    error stop 'no negative budget allowed from zero'
  if(air5_transport_species_ratio(base,-0.1_real64*base,2.0_real64*base)/=1.0_real64) &
    error stop 'unlimited safe flux must remain unlimited'
  if(air5_transport_species_ratio(base*epsilon(base),-base,base)/=0.0_real64) &
    error stop 'unresolved residual cannot fund a correction'
  write(*,'(A)') 'AIR5_TRANSPORT_ROUNDOFF_PROBE pass'
end program air5_transport_roundoff_probe
