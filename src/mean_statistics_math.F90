module mean_statistics_math
  use astr_precision, only: real_kind
  implicit none
  private
  public :: mean_point_moments,mean_endpoint_count,mean_moment_count
  integer,parameter :: mean_endpoint_count=14,mean_moment_count=44
contains
#ifdef _CUDA
  attributes(host,device) &
#endif
  pure subroutine mean_point_moments(endpoint,values)
    real(real_kind),intent(in) :: endpoint(mean_endpoint_count)
    real(real_kind),intent(out) :: values(mean_moment_count)
    real(real_kind) :: ro,u,v,w,p,t,s11,s22,s33,s12,s13,s23
    ro=endpoint(1);u=endpoint(2);v=endpoint(3);w=endpoint(4);p=endpoint(5);t=endpoint(6)
    s11=endpoint(7);s22=endpoint(8);s33=endpoint(9)
    s12=endpoint(10);s13=endpoint(11);s23=endpoint(12)
    values(1:33)=[ro,ro*u,ro*v,ro*w,p,ro*t,ro*u*u,ro*v*v,ro*w*w,ro*u*v,ro*u*w,ro*v*w, &
      p*p,ro*t*t,ro*u*t,ro*v*t,ro*w*t,ro*u**3,ro*v**3,ro*w**3, &
      ro*u*u*v,ro*u*u*w,ro*u*v*v,ro*u*w*w,ro*v*v*w,ro*v*w*w,ro*u*v*w, &
      u,v,w,p*u,p*v,p*w]
    values(34:41)=endpoint(7:14)
    values(42:44)=[s11*u+s12*v+s13*w,s12*u+s22*v+s23*w,s13*u+s23*v+s33*w]
  end subroutine
end module mean_statistics_math
