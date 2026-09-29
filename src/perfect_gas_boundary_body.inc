#ifndef BOUNDARY_PROC
#define BOUNDARY_PROC pure
#endif
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: pressure_extrapolation_inlet, density_evolved_isothermal_wall
  public :: constant_conservative_boundary, piecewise_conservative_boundary
  public :: admissible
contains
  BOUNDARY_PROC subroutine constant_conservative_boundary(face,halo,hm,target,status)
    integer,intent(in) :: hm
    real(8),intent(inout) :: face(5),halo(5,hm)
    real(8),intent(in) :: target(5)
    integer,intent(out) :: status
    integer :: h
    status=1
    if(hm<1) return
    status=2
    if(.not.admissible(target)) return
    face=target
    do h=1,hm
      halo(:,h)=target
    enddo
    status=0
  end subroutine constant_conservative_boundary

  BOUNDARY_PROC subroutine piecewise_conservative_boundary(face,halo,hm,x,split,left,right,status)
    integer,intent(in) :: hm
    real(8),intent(inout) :: face(5),halo(5,hm)
    real(8),intent(in) :: x,split,left(5),right(5)
    integer,intent(out) :: status
    status=1
    if(.not.all(ieee_is_finite([x,split]))) return
    status=2
    if(.not.admissible(left)) return
    if(.not.admissible(right)) return
    if(x>split) then
      call constant_conservative_boundary(face,halo,hm,right,status)
    else
      call constant_conservative_boundary(face,halo,hm,left,status)
    endif
  end subroutine piecewise_conservative_boundary

  BOUNDARY_PROC logical function admissible(q)
    real(8),intent(in) :: q(5)
    admissible=.false.
    if(.not.all(ieee_is_finite(q))) return
    if(q(1)<=0.d0) return
    admissible=q(5)>0.5d0*sum(q(2:4)**2)/q(1)
  end function admissible

  BOUNDARY_PROC subroutine pressure_extrapolation_inlet(face,halo,hm,gamma,status)
    integer,intent(in) :: hm
    real(8),intent(inout) :: face(5),halo(5,hm)
    real(8),intent(in) :: gamma
    integer,intent(out) :: status
    real(8) :: pressure,sound,trial(5)
    integer :: h
    status=1
    if(.not.ieee_is_finite(gamma)) return
    if(gamma<=1.d0.or.hm<1) return
    status=2
    if(.not.admissible(face)) return
    do h=1,hm
      if(.not.admissible(halo(:,h))) return
    enddo
    pressure=(gamma-1.d0)*(face(5)-0.5d0*sum(face(2:4)**2)/face(1))
    sound=sqrt(gamma*pressure/face(1))
    if(abs(face(2)/face(1))>=sound) then
      face=halo(:,1)
    else
      ! Prescribed halo density/momenta remain unchanged. Only energy extrapolates.
      do h=1,hm
        trial=halo(:,h)
        trial(5)=face(5)
        if(.not.admissible(trial)) return
      enddo
      halo(5,:)=face(5)
    endif
    status=0
  end subroutine pressure_extrapolation_inlet

  BOUNDARY_PROC subroutine density_evolved_isothermal_wall(face,inside,halo,hm,tw,gamma,mach,status)
    integer,intent(in) :: hm
    real(8),intent(inout) :: face(5),halo(5,hm)
    real(8),intent(in) :: inside(5,hm),tw,gamma,mach
    integer,intent(out) :: status
    real(8) :: next_face(5),next_halo(5,hm),pwall,tfirst,tghost,velocity(3),rho
    integer :: h
    status=1
    if(.not.all(ieee_is_finite([tw,gamma,mach]))) return
    if(tw<=0.d0.or.gamma<=1.d0.or.mach<=0.d0.or.hm<1) return
    status=2
    if(.not.admissible(face)) return
    do h=1,hm
      if(.not.admissible(inside(:,h))) return
    enddo
    pwall=face(1)*tw/(gamma*mach**2)
    next_face=[face(1),0.d0,0.d0,0.d0,pwall/(gamma-1.d0)]
    tfirst=(gamma-1.d0)*(inside(5,1)/inside(1,1)- &
                        0.5d0*sum((inside(2:4,1)/inside(1,1))**2))*gamma*mach**2
    ! Validate the whole column before writing, including the furthest ghost cell.
    do h=1,hm
      tghost=(h+1)*tw-h*tfirst
      status=3
      if(.not.ieee_is_finite(tghost)) return
      if(tghost<=0.d0) return
      rho=pwall*gamma*mach**2/tghost
      velocity=-inside(2:4,h)/inside(1,h)
      next_halo(:,h)=[rho,rho*velocity,pwall/(gamma-1.d0)+0.5d0*rho*sum(velocity**2)]
      status=2
      if(.not.admissible(next_halo(:,h))) return
    enddo
    face=next_face
    halo=next_halo
    status=0
  end subroutine density_evolved_isothermal_wall
