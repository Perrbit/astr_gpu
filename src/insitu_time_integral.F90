module insitu_time_integral
  use astr_precision, only: real_kind
  use ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: clipped_trapezoid
contains
  ! Each component is an integrand (including any precomputed nonlinear moment).
  pure subroutine clipped_trapezoid(t0,t1,g0,g1,window_start,window_end,increment,weight,ok)
    real(real_kind),intent(in) :: t0,t1,g0(:),g1(:),window_start,window_end
    real(real_kind),intent(out) :: increment(size(g0)),weight
    logical,intent(out) :: ok
    real(real_kind) :: dt,a,b,width,alpha,beta,right,mean(size(g0))
    increment=0.e0_real_kind
    weight=0.e0_real_kind
    ok=.false.
    if(size(g0)/=size(g1).or.size(g0)==0) return
    if(.not.all(ieee_is_finite([t0,t1,window_start,window_end]))) return
    if(.not.all(ieee_is_finite(g0)).or..not.all(ieee_is_finite(g1))) return
    if(t1<=t0.or.window_end<=window_start) return
    dt=t1-t0
    if(.not.ieee_is_finite(dt)) return
    a=max(t0,window_start)
    b=min(t1,window_end)
    if(b<=a) then
      ok=.true.
      return
    endif
    width=b-a
    alpha=(a-t0)/dt
    beta=(b-t0)/dt
    right=0.5e0_real_kind*alpha+0.5e0_real_kind*beta
    ! Convex weights avoid overflowing g0+g1 or g1-g0 before scaling.
    mean=(1.e0_real_kind-right)*g0+right*g1
    if(.not.all(ieee_is_finite(mean))) return
    if(width>1.e0_real_kind) then
      if(any(abs(mean)>huge(1.e0_real_kind)/width)) return
    endif
    increment=width*mean
    if(.not.all(ieee_is_finite(increment))) then
      increment=0.e0_real_kind
      return
    endif
    weight=width
    ok=.true.
  end subroutine
end module
