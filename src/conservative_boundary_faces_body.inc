#ifndef FACE_PROC
#define FACE_PROC pure
#endif
  implicit none
  private
  public :: apply_conservative_face_column
#ifdef FACE_CUDA
  public :: apply_conservative_face_kernel
#endif
contains
  FACE_PROC subroutine apply_conservative_face_column(side,s,k,im,jm,km,hm,q,xline, &
                                                      tw,gamma,mach,split,left,right,status)
    integer,intent(in) :: side,s,k,im,jm,km,hm
    real(8),intent(inout) :: q(-hm:im+hm,-hm:jm+hm,-hm:km+hm,5)
    real(8),intent(in) :: xline(-hm:im+hm),tw,gamma,mach,split,left(5),right(5)
    integer,intent(out) :: status
    real(8) :: face(5),halo(5,hm),inside(5,hm),target(5)
    integer :: h
    status=1
    if(hm<1.or.im<hm.or.jm<hm.or.k<0.or.k>km) return
    if(side==1.or.side==2) then
      if(s<0.or.s>jm) return
    elseif(side==3.or.side==4) then
      if(s< -hm.or.s>im+hm) return
    else
      return
    endif
    select case(side)
    case(1)
      face=q(0,s,k,:)
      do h=1,hm
        halo(:,h)=q(-h,s,k,:)
      enddo
      call pressure_extrapolation_inlet(face,halo,hm,gamma,status)
      if(status/=0) return
      q(0,s,k,:)=face
      do h=1,hm
        q(-h,s,k,:)=halo(:,h)
      enddo
    case(2)
      face=q(im,s,k,:)
      halo=0.d0
      target=q(im-1,s,k,:)
      call constant_conservative_boundary(face,halo,hm,target,status)
      if(status/=0) return
      q(im,s,k,:)=face
      do h=1,hm
        q(im+h,s,k,:)=halo(:,h)
      enddo
    case(3)
      face=q(s,0,k,:)
      halo=0.d0
      do h=1,hm
        inside(:,h)=q(s,h,k,:)
      enddo
      call density_evolved_isothermal_wall(face,inside,halo,hm,tw,gamma,mach,status)
      if(status/=0) return
      q(s,0,k,:)=face
      do h=1,hm
        q(s,-h,k,:)=halo(:,h)
      enddo
    case(4)
      face=q(s,jm,k,:)
      halo=0.d0
      call piecewise_conservative_boundary(face,halo,hm,xline(s),split,left,right,status)
      if(status/=0) return
      q(s,jm,k,:)=face
      do h=1,hm
        q(s,jm+h,k,:)=halo(:,h)
      enddo
    end select
  end subroutine apply_conservative_face_column
