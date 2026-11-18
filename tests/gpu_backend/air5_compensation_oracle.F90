subroutine reference_compensated_mean(left,right,cl,cr,carry) bind(C,name='air5_reference_mean')
  use iso_c_binding,only:c_double
  use chemistry_compensation,only:compensated_mean
  implicit none
  real(c_double),value :: left,right,cl,cr
  real(c_double),intent(out) :: carry
  call compensated_mean(left,right,cl,cr,carry)
end subroutine
subroutine reference_compensated_normalize(high,carry) bind(C,name='air5_reference_normalize')
  use iso_c_binding,only:c_double
  use chemistry_compensation,only:compensated_normalize
  implicit none
  real(c_double),intent(inout) :: high,carry
  call compensated_normalize(high,carry)
end subroutine
subroutine reference_compensated_primitive(q,preserve,out,status) bind(C,name='air5_reference_primitive')
  use iso_c_binding,only:c_double,c_int
  use chemistry_flow_state,only:air5_conservative_to_primitive,configure_air5_species_cache
  implicit none
  real(c_double),intent(in) :: q(11)
  integer(c_int),value :: preserve
  real(c_double),intent(out) :: out(12)
  integer(c_int),intent(out) :: status
  call configure_air5_species_cache(preserve/=0)
  call air5_conservative_to_primitive(q,out(1),out(2:4),out(5),out(8:12),out(6),out(7),status)
end subroutine
