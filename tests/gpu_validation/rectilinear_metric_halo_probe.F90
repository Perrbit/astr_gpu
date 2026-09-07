program rectilinear_metric_halo_probe
  use rectilinear_metric_halo
  implicit none
  integer,parameter :: im=8,jm=7,km=6,hm=5
  real(8) :: jac(-hm:im+hm,-hm:jm+hm,-hm:km+hm),original(-hm:im+hm,-hm:jm+hm,-hm:km+hm)
  real(8) :: metric(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3,3),base(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3,3)
  real(8) :: spacing(3)
  logical :: owned(6),filled
  integer :: i,j,k,d,mask,status,ref(3),idx(3),dims(3),axis,side
  dims=[im,jm,km]
  original=-999.d0; base=-999.d0
  do k=0,km
  do j=0,jm
  do i=0,im
    spacing=[1.d0+.01d0*i,1.d0+.02d0*j,1.d0+.03d0*k]
    original(i,j,k)=product(spacing)
    base(i,j,k,:,:)=0.d0
    do d=1,3
      base(i,j,k,d,d)=1.d0/spacing(d)
    enddo
  enddo
  enddo
  enddo
  do mask=0,63
    do d=1,6
      owned(d)=btest(mask,d-1)
    enddo
    jac=original; metric=base
    call fill_rectilinear_metric_halo(im,jm,km,hm,owned,jac,metric,status)
    if(status/=0) error stop 'Valid stretched metric rejected'
    do k=-hm,km+hm
    do j=-hm,jm+hm
    do i=-hm,im+hm
      idx=[i,j,k]; ref=idx; filled=.false.
      if(count(idx<0.or.idx>dims)==1) then
        do axis=1,3
          side=0
          if(idx(axis)<0) side=2*axis-1
          if(idx(axis)>dims(axis)) side=2*axis
          if(side>0) then
            filled=owned(side)
            if(filled) ref(axis)=max(0,min(dims(axis),idx(axis)))
          endif
        enddo
      endif
      if(jac(i,j,k)/=original(ref(1),ref(2),ref(3))) error stop 'Unexpected Jacobian write'
      if(any(metric(i,j,k,:,:)/=base(ref(1),ref(2),ref(3),:,:))) error stop 'Unexpected metric write'
    enddo
    enddo
    enddo
  enddo
  owned=.true.
  jac=original; metric=base; metric(1,1,1,1,2)=0.1d0
  call fill_rectilinear_metric_halo(im,jm,km,hm,owned,jac,metric,status)
  if(status==0.or.any(jac/=original)) error stop 'Skew metric not rejected before writing'
  jac=original; metric=base; jac(1,1,1)=-1.d0
  call fill_rectilinear_metric_halo(im,jm,km,hm,owned,jac,metric,status)
  if(status==0.or.any(metric/=base)) error stop 'Negative Jacobian not rejected'
  jac=original; metric=base; metric(1,1,1,1,1)=2.d0
  call fill_rectilinear_metric_halo(im,jm,km,hm,owned,jac,metric,status)
  if(status==0.or.any(jac/=original)) error stop 'Nonseparable cofactor not rejected'
  print*,'RECTILINEAR_METRIC_HALO_PASS masks=64 hm=5'
end program rectilinear_metric_halo_probe
