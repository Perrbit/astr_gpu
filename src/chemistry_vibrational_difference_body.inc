    integer, intent(in) :: species
    real(real64), intent(in) :: temperature,tv
    real(real64) :: theta,a,b,delta,em_a,em_b,em_delta

    theta=air5_theta_v(species)
    difference=0.0_real64
    if(theta<=0.0_real64) return
    a=theta/temperature
    b=theta/tv
    delta=theta*((temperature-tv)/(temperature*tv))
    ! expm1(x)=2*exp(x/2)*sinh(x/2), using standard Fortran intrinsics.
    ! Form the small exponent difference from T-Tv, not two rounded reciprocals.
    em_a=2.0_real64*exp(0.5_real64*a)*sinh(0.5_real64*a)
    em_b=2.0_real64*exp(0.5_real64*b)*sinh(0.5_real64*b)
    em_delta=2.0_real64*exp(0.5_real64*delta)*sinh(0.5_real64*delta)
    difference=air5_species_gas_constant(species)*theta*exp(a)*em_delta/(em_a*em_b)
