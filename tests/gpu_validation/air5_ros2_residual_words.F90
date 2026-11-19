module air5_ros2_residual_words
  use iso_fortran_env, only: real64
  implicit none
  private
  public :: air5_probe_residual
contains
  attributes(host,device) pure subroutine two_sum(a,b,hi,lo)
    real(real64), intent(in) :: a,b
    real(real64), intent(out) :: hi,lo
    real(real64) :: v
    hi=a+b
    v=hi-a
    lo=(a-(hi-v))+(b-v)
  end subroutine two_sum

  attributes(host,device) pure function product_tail(a,b,hi) result(lo)
    real(real64), intent(in) :: a,b,hi
    real(real64) :: lo,ap,bp,ah,al,bh,bl,e1,e2,e3
    real(real64), parameter :: splitter=134217729.0_real64
    ap=splitter*a; bp=splitter*b
    ah=ap-(ap-a); al=a-ah
    bh=bp-(bp-b); bl=b-bh
    e1=hi-ah*bh
    e2=e1-al*bh
    e3=e2-ah*bl
    lo=al*bl-e3
  end function product_tail

  ! Compile only this source with contraction disabled. Dekker's product tail
  ! matches the explicit FMA tail for the normal bounded probe operands.
  attributes(host,device) pure subroutine air5_probe_residual(a,b,x,r)
    real(real64), intent(in) :: a(6,6),b(6),x(6)
    real(real64), intent(out) :: r(6)
    real(real64) :: hi,lo,coef,p,sh,sl
    integer :: row,col
    do row=1,6
      hi=b(row); lo=0.0_real64
      do col=1,6
        coef=-a(row,col)
        p=coef*x(col)
        call two_sum(hi,p,sh,sl)
        call two_sum(sh,sl+(lo+product_tail(coef,x(col),p)),hi,lo)
      end do
      r(row)=hi
    end do
  end subroutine air5_probe_residual
end module air5_ros2_residual_words
