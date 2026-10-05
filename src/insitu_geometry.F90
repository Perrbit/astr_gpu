module insitu_geometry
  use iso_fortran_env, only: real64
  use ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: hex_volume,quad_area,geometry_cross,wall_frame
  real(real64),parameter :: gauss(2)=[0.5d0-0.5d0/sqrt(3.d0),0.5d0+0.5d0/sqrt(3.d0)]
contains
  pure subroutine wall_frame(first,second,inward,normal,tangent,ok)
    real(real64),intent(in) :: first(3),second(3),inward(3)
    real(real64),intent(out) :: normal(3),tangent(3)
    logical,intent(out) :: ok
    real(real64) :: length,orientation
    normal=0.d0; tangent=0.d0; ok=.false.
    if(.not.all(ieee_is_finite([first,second,inward]))) return
    normal=geometry_cross(first,second)
    length=sqrt(dot_product(normal,normal))
    if(.not.ieee_is_finite(length).or.length<=0.d0) return
    normal=normal/length
    orientation=dot_product(normal,inward)
    if(.not.ieee_is_finite(orientation).or.orientation==0.d0) return
    if(orientation<0.d0) normal=-normal
    tangent=[1.d0,0.d0,0.d0]-normal(1)*normal
    length=sqrt(dot_product(tangent,tangent))
    if(.not.ieee_is_finite(length).or.length<=0.d0) return
    tangent=tangent/length
    ok=all(ieee_is_finite([normal,tangent]))
  end subroutine

  pure function geometry_cross(a,b) result(c)
    real(real64),intent(in) :: a(3),b(3)
    real(real64) :: c(3)
    c=[a(2)*b(3)-a(3)*b(2),a(3)*b(1)-a(1)*b(3),a(1)*b(2)-a(2)*b(1)]
  end function

  pure subroutine hex_volume(points,volume,ok)
    real(real64),intent(in) :: points(3,0:1,0:1,0:1)
    real(real64),intent(out) :: volume
    logical,intent(out) :: ok
    real(real64) :: shape(0:1,3),derivative(0:1),jac(3,3),det
    integer :: p,q,r,i,j,k
    volume=0.d0; ok=.false.
    if(.not.all(ieee_is_finite(points))) return
    derivative=[-1.d0,1.d0]
    do r=1,2
    do q=1,2
    do p=1,2
      shape(:,1)=[1.d0-gauss(p),gauss(p)]
      shape(:,2)=[1.d0-gauss(q),gauss(q)]
      shape(:,3)=[1.d0-gauss(r),gauss(r)]
      jac=0.d0
      ! Subtract one vertex to avoid derivative cancellation under coordinate translations.
      do k=0,1
      do j=0,1
      do i=0,1
        jac(:,1)=jac(:,1)+(points(:,i,j,k)-points(:,0,0,0))*derivative(i)*shape(j,2)*shape(k,3)
        jac(:,2)=jac(:,2)+(points(:,i,j,k)-points(:,0,0,0))*shape(i,1)*derivative(j)*shape(k,3)
        jac(:,3)=jac(:,3)+(points(:,i,j,k)-points(:,0,0,0))*shape(i,1)*shape(j,2)*derivative(k)
      enddo
      enddo
      enddo
      det=dot_product(geometry_cross(jac(:,1),jac(:,2)),jac(:,3))
      if(.not.ieee_is_finite(det).or.det<=0.d0) then
        volume=0.d0; return
      endif
      volume=volume+det/8.d0
    enddo
    enddo
    enddo
    ok=ieee_is_finite(volume).and.volume>0.d0
    if(.not.ok) volume=0.d0
  end subroutine

  pure subroutine quad_area(points,area,ok)
    real(real64),intent(in) :: points(3,0:1,0:1)
    real(real64),intent(out) :: area
    logical,intent(out) :: ok
    real(real64) :: shape(0:1,2),derivative(0:1),jac(3,2),normal(3),reference(3),measure
    integer :: p,q,i,j
    area=0.d0; ok=.false.
    if(.not.all(ieee_is_finite(points))) return
    derivative=[-1.d0,1.d0]
    reference=0.d0
    do q=1,2
    do p=1,2
      shape(:,1)=[1.d0-gauss(p),gauss(p)]
      shape(:,2)=[1.d0-gauss(q),gauss(q)]
      jac=0.d0
      do j=0,1
      do i=0,1
        jac(:,1)=jac(:,1)+(points(:,i,j)-points(:,0,0))*derivative(i)*shape(j,2)
        jac(:,2)=jac(:,2)+(points(:,i,j)-points(:,0,0))*shape(i,1)*derivative(j)
      enddo
      enddo
      normal=geometry_cross(jac(:,1),jac(:,2))
      measure=sqrt(dot_product(normal,normal))
      if(.not.ieee_is_finite(measure).or.measure<=0.d0) then
        area=0.d0; return
      endif
      normal=normal/measure
      if(p==1.and.q==1) reference=normal
      if(dot_product(reference,normal)<=0.d0) then
        area=0.d0; return
      endif
      area=area+measure/4.d0
    enddo
    enddo
    ok=ieee_is_finite(area).and.area>0.d0
    if(.not.ok) area=0.d0
  end subroutine
end module
