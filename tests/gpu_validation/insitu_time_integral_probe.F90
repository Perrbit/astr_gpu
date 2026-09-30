program time_integral_probe
  use iso_fortran_env, only: real64
  use ieee_arithmetic, only: ieee_value,ieee_quiet_nan
  use insitu_time_integral
  implicit none
  real(real64) :: inc(3),weight,total(3),duration,ta(4),t0,t1,g0(3),g1(3)
  logical :: ok
  integer :: i
  ! Constant, linear signal and independently sampled second moment.
  call clipped_trapezoid(0.d0,2.d0,[1.d0,0.d0,0.d0],[1.d0,2.d0,4.d0], &
                        0.5d0,1.5d0,inc,weight,ok)
  call require(ok.and.weight==1.d0.and.all(inc==[1.d0,1.d0,2.d0]))
  ! Nonuniform sample times and a window not aligned with either endpoint.
  ta=[0.d0,0.125d0,0.75d0,2.d0]
  total=0.d0
  duration=0.d0
  do i=1,3
    t0=ta(i)
    t1=ta(i+1)
    g0=[2.d0,3.d0*t0+1.d0,-t0]
    g1=[2.d0,3.d0*t1+1.d0,-t1]
    call clipped_trapezoid(t0,t1,g0,g1,0.25d0,1.25d0,inc,weight,ok)
    call require(ok)
    total=total+inc
    duration=duration+weight
  enddo
  call require(duration==1.d0.and.all(total==[2.d0,3.25d0,-0.75d0]))
  ! A window beyond available samples receives no invented coverage.
  call clipped_trapezoid(0.d0,1.d0,[1.d0,2.d0,3.d0],[1.d0,2.d0,3.d0], &
                        2.d0,3.d0,inc,weight,ok)
  call require(ok.and.weight==0.d0.and.all(inc==0.d0))
  call clipped_trapezoid(0.d0,1.d0,[2.d0,2.d0,2.d0],[2.d0,2.d0,2.d0], &
                        -1.d0,2.d0,inc,weight,ok)
  call require(ok.and.weight==1.d0.and.all(inc==2.d0))
  ! Finite large values must not overflow in an intermediate sum or difference.
  call clipped_trapezoid(0.d0,1.d0,[1.d308,1.d308,1.d308],[1.d308,-1.d308,1.d308], &
                        0.d0,1.d0,inc,weight,ok)
  call require(ok.and.inc(1)==1.d308.and.inc(2)==0.d0)
  call clipped_trapezoid(0.d0,2.d0,[1.d308,1.d308,1.d308],[1.d308,1.d308,1.d308], &
                        0.d0,2.d0,inc,weight,ok)
  call require(.not.ok.and.weight==0.d0)
  call clipped_trapezoid(1.d0,1.d0,[1.d0,2.d0,3.d0],[1.d0,2.d0,3.d0], &
                        0.d0,2.d0,inc,weight,ok)
  call require(.not.ok)
  call clipped_trapezoid(1.d0,0.d0,[1.d0,2.d0,3.d0],[1.d0,2.d0,3.d0], &
                        0.d0,2.d0,inc,weight,ok)
  call require(.not.ok)
  call clipped_trapezoid(0.d0,1.d0,[ieee_value(0.d0,ieee_quiet_nan),2.d0,3.d0], &
                        [1.d0,2.d0,3.d0],0.d0,2.d0,inc,weight,ok)
  call require(.not.ok)
  print *, 'PASS: clipped trapezoids, independent moments, no extrapolation, invalid inputs'
contains
  subroutine require(condition)
    logical,intent(in) :: condition
    if(.not.condition) error stop 'time integration assertion failed'
  end subroutine
end program
