program boundary_faces_probe
#ifdef TEST_CUDA
  use cudafor
  use conservative_boundary_faces_gpu
#else
  use conservative_boundary_faces
#endif
  implicit none
  integer,parameter :: im=8,jm=8,km=1,hm=2
  real(8) :: q(-hm:im+hm,-hm:jm+hm,-hm:km+hm,5),xline(-hm:im+hm)
  real(8) :: left(5),right(5),rho,u,v,temp,p
  integer :: i,j,k,ii,jj,side,lo,hi,s,status,mode
  character(len=16) :: arg
#ifdef TEST_CUDA
  real(8),device :: qd(-hm:im+hm,-hm:jm+hm,-hm:km+hm,5),xd(-hm:im+hm),ld(5),rd(5)
  integer,device :: error_d
  integer :: ierr,blocks
#endif
  call get_command_argument(1,arg)
  read(arg,*)mode
  left=[1.d0,0.1d0,0.d0,0.d0,3.d0]
  right=[1.2d0,0.2d0,0.1d0,0.d0,4.d0]
  do i=-hm,im+hm
    xline(i)=10.d0*i
  enddo
  do k=-hm,km+hm
  do j=-hm,jm+hm
  do i=-hm,im+hm
    ii=max(0,min(im,i))
    jj=max(0,min(jm,j))
    rho=1.d0+0.01d0*jj+0.005d0*ii
    u=0.2d0+0.01d0*ii
    v=0.01d0*jj
    temp=1.7d0-0.02d0*jj
    p=rho*temp/5.6d0
    q(i,j,k,:)=[rho,rho*u,rho*v,0.d0,p/0.4d0+0.5d0*rho*(u*u+v*v)]
  enddo
  enddo
  enddo
#ifdef TEST_CUDA
  qd=q
  xd=xline
  ld=left
  rd=right
#endif
  do side=1,4
    if(.not.btest(mode,side-1)) cycle
    lo=0
    hi=jm
    if(side>=3) then
      lo=-hm
      hi=im+hm
    endif
#ifdef TEST_CUDA
    error_d=0
    blocks=((hi-lo+1)*(km+1)+255)/256
    call apply_conservative_face_kernel<<<blocks,256>>>(side,im,jm,km,hm,qd,xd, &
                                                        1.7d0,1.4d0,2.d0,40.d0,ld,rd,error_d)
    ierr=cudaDeviceSynchronize()
    if(ierr/=cudaSuccess) error stop 'Face probe CUDA failure'
    status=error_d
    if(status/=0) error stop 'Face probe invalid boundary'
#else
    do k=0,km
    do s=lo,hi
      call apply_conservative_face_column(side,s,k,im,jm,km,hm,q,xline, &
                                          1.7d0,1.4d0,2.d0,40.d0,left,right,status)
      if(status/=0) error stop 'Face probe invalid boundary'
    enddo
    enddo
#endif
  enddo
#ifdef TEST_CUDA
  q=qd
#endif
  do k=0,km
  do j=-hm,jm+hm
  do i=-hm,im+hm
    print '(5es25.16)',q(i,j,k,:)
  enddo
  enddo
  enddo
end program boundary_faces_probe
