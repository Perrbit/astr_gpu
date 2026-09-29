module rectilinear_metric_halo
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: fill_rectilinear_metric_halo
contains
  subroutine fill_rectilinear_metric_halo(im,jm,km,hm,owned,jacob,dxi,status)
    integer,intent(in) :: im,jm,km,hm
    logical,intent(in) :: owned(6)
    real(8),intent(inout) :: jacob(-hm:im+hm,-hm:jm+hm,-hm:km+hm)
    real(8),intent(inout) :: dxi(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3,3)
    integer,intent(out) :: status
    integer :: i,j,k,d,e,h,idx(3),ref(3)
    real(8) :: scale,cofactor,reference
    status=1
    if(min(im,jm,km,hm)<1) return
    ! Validate the physical metric contract before touching any halo.
    do k=0,km
    do j=0,jm
    do i=0,im
      if(.not.ieee_is_finite(jacob(i,j,k))) return
      if(.not.all(ieee_is_finite(dxi(i,j,k,:,:)))) return
      if(jacob(i,j,k)<=0.d0) return
      do d=1,3
        if(dxi(i,j,k,d,d)<=0.d0) return
        scale=dxi(i,j,k,d,d)
        do e=1,3
          if(e/=d.and.abs(dxi(i,j,k,d,e))>1.d-10*scale) return
        enddo
        idx=[i,j,k]; ref=idx; ref(d)=0
        cofactor=jacob(i,j,k)*dxi(i,j,k,d,d)
        reference=jacob(ref(1),ref(2),ref(3))*dxi(ref(1),ref(2),ref(3),d,d)
        if(.not.ieee_is_finite(cofactor).or..not.ieee_is_finite(reference)) return
        if(abs(cofactor-reference)>1.d-10*max(abs(cofactor),abs(reference))) return
      enddo
    enddo
    enddo
    enddo
    ! Constant normal extension preserves the normal cofactor of a separable
    ! rectilinear map. Only face halos with physical transverse indices are used.
    do h=1,hm
      if(owned(1)) then
        jacob(-h,0:jm,0:km)=jacob(0,0:jm,0:km)
        dxi(-h,0:jm,0:km,:,:)=dxi(0,0:jm,0:km,:,:)
      endif
      if(owned(2)) then
        jacob(im+h,0:jm,0:km)=jacob(im,0:jm,0:km)
        dxi(im+h,0:jm,0:km,:,:)=dxi(im,0:jm,0:km,:,:)
      endif
      if(owned(3)) then
        jacob(0:im,-h,0:km)=jacob(0:im,0,0:km)
        dxi(0:im,-h,0:km,:,:)=dxi(0:im,0,0:km,:,:)
      endif
      if(owned(4)) then
        jacob(0:im,jm+h,0:km)=jacob(0:im,jm,0:km)
        dxi(0:im,jm+h,0:km,:,:)=dxi(0:im,jm,0:km,:,:)
      endif
      if(owned(5)) then
        jacob(0:im,0:jm,-h)=jacob(0:im,0:jm,0)
        dxi(0:im,0:jm,-h,:,:)=dxi(0:im,0:jm,0,:,:)
      endif
      if(owned(6)) then
        jacob(0:im,0:jm,km+h)=jacob(0:im,0:jm,km)
        dxi(0:im,0:jm,km+h,:,:)=dxi(0:im,0:jm,km,:,:)
      endif
    enddo
    status=0
  end subroutine fill_rectilinear_metric_halo
end module rectilinear_metric_halo
