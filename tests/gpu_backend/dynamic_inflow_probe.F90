program dynamic_inflow_probe
  use iso_c_binding
  use ieee_arithmetic, only: ieee_is_finite
  use device_state
  use fluid_explicit, only: fluid_prepare,fluid_rhs_prepare
  use boundary_state_cpp, only: profile_inflow_prepare,inflow_slices_prepare, &
    inflow_slice_upload,inflow_slice_download,dynamic_inflow_x
  implicit none
  integer,parameter :: im=4,jm=5,km=6,hm=3,nodes=(im+2*hm+1)*(jm+2*hm+1)*(km+2*hm+1)
  integer(c_int),parameter :: slots(4)=[2,0,3,1]
  real(c_double),parameter :: delta=.25d0,time=.37d0,c6=2.5d0,base(5)=[1.2d0,.3d0,.01d0,.02d0,1.4d0]
  type(c_ptr) :: state
  real(c_double) :: q(-hm:im+hm,-hm:jm+hm,-hm:km+hm,5),eq(-hm:im+hm,-hm:jm+hm,-hm:km+hm,5), &
    p(-hm:im+hm,-hm:jm+hm,-hm:km+hm,6),ep(-hm:im+hm,-hm:jm+hm,-hm:km+hm,6), &
    target(0:jm,0:km,9),plane(0:jm,0:km,5),loaded(0:jm,0:km,5),rhs(0:im,0:jm,0:km,5), &
    erhs(0:im,0:jm,0:km,5),values(5),mach,c2,t,amp,pw,density,maximum=0
  integer :: debug,pressure_mode,mach_case,i,j,k,c,n,cases=0
  do debug=0,1
    call check(device_state_create(0_c_int,im,jm,km,hm,int(debug,c_int),state))
    call check(fluid_prepare(state,0_c_int));call check(fluid_rhs_prepare(state,0_c_int))
    call check(profile_inflow_prepare(state));call check(inflow_slices_prepare(state))
    do n=1,4
      t=(n-1)*delta
      do c=1,5
      do k=0,km
      do j=0,jm
        amp=c*1.d-4*(1+.01d0*j+.03d0*k)
        plane(j,k,c)=amp*(1+t+t*t+t*t*t)
      enddo
      enddo
      enddo
      call check(inflow_slice_upload(state,slots(n),plane,bytes(size(plane))))
      call check(inflow_slice_download(state,slots(n),loaded,bytes(size(loaded))))
      if(any(transfer(loaded,[0_c_int64_t],size(loaded))/=transfer(plane,[0_c_int64_t],size(plane)))) &
        error stop 'inflow slot copy bits changed'
    enddo
    do pressure_mode=0,1
    do mach_case=0,1
      mach=.3d0
      if(mach_case==1) mach=8.d0
      c2=1.4d0*mach*mach
      q=17;p=-29;target=0;rhs=7;erhs=7
      do c=1,4
        target(:,:,c)=base(c)
      enddo
      target(:,:,5)=base(1)*base(5)/c2;target(:,:,6)=base(5);target(:,:,7)=1
      p(:,:,:,5)=2
      eq=q;ep=p
      do k=0,km
      do j=0,jm
        do c=1,5
          amp=c*1.d-4*(1+.01d0*j+.03d0*k)
          values(c)=base(c)+amp*(1+time+time*time+time*time*time)
        enddo
        density=values(1);pw=density*values(5)/c2
        if(pressure_mode==1.and.values(2)/(sqrt(values(5))/mach)<1) then
          pw=2;density=pw/values(5)*c2
        endif
        eq(0,j,k,:)=[density,density*values(2:4),pw*c6+.5d0*density*sum(values(2:4)**2)]
        ep(0,j,k,:)=[density,values(2:4),pw,values(5)];erhs(0,j,k,:)=0
      enddo
      enddo
      call check(device_state_upload(state,device_q,q,bytes(size(q))))
      call check(device_state_upload(state,device_rho,p(:,:,:,1),bytes(nodes)))
      call check(device_state_upload(state,device_velocity,p(:,:,:,2:4),bytes(3*nodes)))
      call check(device_state_upload(state,device_pressure,p(:,:,:,5),bytes(nodes)))
      call check(device_state_upload(state,device_temperature,p(:,:,:,6),bytes(nodes)))
      call check(device_state_upload(state,device_rhs,rhs,bytes(size(rhs))))
      call check(device_state_upload(state,device_inflow_target,target,bytes(size(target))))
      call check(dynamic_inflow_x(state,slots,0.d0,delta,time,int(pressure_mode,c_int),mach,c2,c6))
      call check(device_state_download(state,device_q,q,bytes(size(q))))
      call check(device_state_download(state,device_rho,p(:,:,:,1),bytes(nodes)))
      call check(device_state_download(state,device_velocity,p(:,:,:,2:4),bytes(3*nodes)))
      call check(device_state_download(state,device_pressure,p(:,:,:,5),bytes(nodes)))
      call check(device_state_download(state,device_temperature,p(:,:,:,6),bytes(nodes)))
      call check(device_state_download(state,device_rhs,rhs,bytes(size(rhs))))
      if(.not.all(ieee_is_finite(q)).or..not.all(ieee_is_finite(p))) error stop 'nonfinite inlet'
      if(any(rhs/=erhs)) error stop 'dynamic inlet RHS footprint'
      do c=1,5
        maximum=max(maximum,maxval(abs(q(:,:,:,c)-eq(:,:,:,c)))/max(1.d0,maxval(abs(eq(:,:,:,c)))))
      enddo
      do c=1,6
        maximum=max(maximum,maxval(abs(p(:,:,:,c)-ep(:,:,:,c)))/max(1.d0,maxval(abs(ep(:,:,:,c)))))
      enddo
      if(any(q(-hm:-1,:,:,:)/=17).or.any(q(1:im+hm,:,:,:)/=17)) error stop 'off-face inlet q changed'
      cases=cases+1
    enddo
    enddo
    call check(device_state_destroy(state))
  enddo
  if(maximum>2.d-10) error stop 'cubic inlet manufacture differs'
  write(*,'(a,i0,a,es24.16e3)') 'DYNAMIC_INFLOW_CUBIC_PASS cases=',cases,' max_scaled=',maximum
contains
  integer(c_size_t) function bytes(count)
    integer,intent(in) :: count
    bytes=int(count,c_size_t)*c_sizeof(0.d0)
  end function
  subroutine check(status)
    integer(c_int),intent(in) :: status
    if(status/=0) then
      write(*,'(a,i0)') 'dynamic inlet device status=',status
      error stop 'dynamic inlet device operation'
    endif
  end subroutine
end program
