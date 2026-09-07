#ifdef TEST_CUDA
module boundary_contract_launch
  use cudafor
  use perfect_gas_boundary_gpu
contains
  attributes(global) subroutine apply_contract(face,halo,inside,mode,status)
    real(8),device :: face(5),halo(5,6),inside(5,6)
    integer,value :: mode
    integer,device :: status
    if(mode>=10) then
      if(mode==13) then
        call constant_conservative_boundary(face,halo,6,inside(:,1),status)
      else
        call piecewise_conservative_boundary(face,halo,6,real(mode+29,8),40.d0, &
                                             inside(:,1),inside(:,2),status)
      endif
    elseif(mode<=4) then
      call pressure_extrapolation_inlet(face,halo,6,2.d0,status)
    else
      call density_evolved_isothermal_wall(face,inside,halo,6,1.676194d0,1.4d0,2.d0,status)
    endif
  end subroutine apply_contract
end module boundary_contract_launch
#endif
program boundary_contract_probe
#ifdef TEST_CUDA
  use boundary_contract_launch
#else
  use perfect_gas_boundary
#endif
  implicit none
  integer,parameter :: hm=6
  real(8) :: face(5),halo(5,hm),inside(5,hm),u,tw,temp,p,rho
  integer :: mode,h,status
  character(len=32) :: arg
#ifdef TEST_CUDA
  real(8),device :: face_d(5),halo_d(5,hm),inside_d(5,hm)
  integer,device :: status_d
  integer :: ierr
#endif
  call get_command_argument(1,arg)
  read(arg,*)mode
  if(mode>=10) then
    face=-99.d0
    halo=-99.d0
    inside=0.d0
    inside(:,1)=[1.d0,0.1d0,0.d0,0.d0,3.d0]
    inside(:,2)=[1.2d0,0.2d0,0.1d0,0.d0,4.d0]
#ifndef TEST_CUDA
    if(mode==13) then
      call constant_conservative_boundary(face,halo,hm,inside(:,1),status)
    else
      call piecewise_conservative_boundary(face,halo,hm,real(mode+29,8),40.d0, &
                                           inside(:,1),inside(:,2),status)
    endif
#endif
  elseif(mode<=4) then
    ! gamma=2 and p=2 make the sonic speed exactly representable.
    u=0.5d0
    if(mode==2) u=3.d0
    if(mode==3) u=-3.d0
    if(mode==4) u=2.d0
    face=[1.d0,u,0.d0,0.d0,2.d0+0.5d0*u*u]
    do h=1,hm
      halo(:,h)=[1.d0,0.1d0,0.d0,0.d0,3.d0+0.01d0*h]
    enddo
#ifndef TEST_CUDA
    call pressure_extrapolation_inlet(face,halo,hm,2.d0,status)
#endif
  else
    tw=1.676194d0
    face=[1.3d0,0.4d0,0.2d0,0.1d0,2.d0]
    halo=-99.d0
    do h=1,hm
      rho=1.d0+0.01d0*h
      temp=tw-0.02d0*h
      if(mode==6) temp=3.d0*tw
      if(mode==7) temp=1.2d0*tw
      p=rho*temp/(1.4d0*4.d0)
      inside(:,h)=[rho,rho*0.1d0*h,rho*0.02d0*h,rho*0.03d0*h, &
                    p/0.4d0+0.5d0*rho*(0.01d0+0.0004d0+0.0009d0)*h*h]
    enddo
#ifndef TEST_CUDA
    call density_evolved_isothermal_wall(face,inside,halo,hm,tw,1.4d0,2.d0,status)
#endif
  endif
#ifdef TEST_CUDA
  if(mode<=4) inside=0.d0
  face_d=face
  halo_d=halo
  inside_d=inside
  status_d=-1
  call apply_contract<<<1,1>>>(face_d,halo_d,inside_d,mode,status_d)
  ierr=cudaDeviceSynchronize()
  if(ierr/=cudaSuccess) error stop 'Boundary probe CUDA synchronization failed'
  face=face_d
  halo=halo_d
  status=status_d
#endif
  print*,status
  print '(5es25.16)',face
  do h=1,hm
    print '(5es25.16)',halo(:,h)
  enddo
end program boundary_contract_probe
