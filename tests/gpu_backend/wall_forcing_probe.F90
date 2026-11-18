program wall_forcing_probe
  use iso_c_binding
  use ieee_arithmetic, only: ieee_is_finite
  use device_state
  use fluid_explicit, only: fluid_prepare
  use boundary_state_cpp, only: wall_forcing_prepare,wall_forcing_upload,wall41_forcing_y
  implicit none
  integer,parameter :: im=7,jm=6,km=5,hm=1,nodes=(im+2*hm+1)*(jm+2*hm+1)*(km+2*hm+1)
  real(c_double),parameter :: tw=1.2d0,c2=1.4d0,c6=2.5d0
  real(c_double) :: q(-hm:im+hm,-hm:jm+hm,-hm:km+hm,5),expected_q(-hm:im+hm,-hm:jm+hm,-hm:km+hm,5), &
    cache(-hm:im+hm,-hm:jm+hm,-hm:km+hm,6),expected_cache(-hm:im+hm,-hm:jm+hm,-hm:km+hm,6), &
    metric(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3,3),speed(0:im,0:km), &
    rhs(0:im,0:jm,0:km,5),normal(3),u(3),pw,density,err,maximum=0
  type(c_ptr) :: state
  integer :: i,j,k,c,geometry,debug,cases=0
  do debug=0,1
    call check(device_state_create(0_c_int,im,jm,km,hm,int(debug,c_int),state))
    call check(fluid_prepare(state,0_c_int))
    call check(wall_forcing_prepare(state))
    do geometry=0,1
      q=17;cache=-29;metric=0;rhs=7
      do k=-hm,km+hm
        do j=-hm,jm+hm
          do i=-hm,im+hm
            cache(i,j,k,5)=2+.01d0*i+.03d0*k+.02d0*j*j
          enddo
        enddo
      enddo
      do k=0,km
        do i=0,im
          normal=[0.d0,1.d0,0.d0]
          if(geometry==1) normal=[.1d0+.01d0*i,1.d0,.05d0*k]
          metric(i,0,k,2,:)=normal
          speed(i,k)=.02d0*sin(real(i-k,c_double))
        enddo
      enddo
      expected_q=q;expected_cache=cache
      do k=0,km
        do i=0,im
          normal=metric(i,0,k,2,:);normal=normal/sqrt(sum(normal**2))
          u=speed(i,k)*normal;pw=2+.01d0*i+.03d0*k;density=pw/tw*c2
          expected_q(i,0,k,:)=[density,density*u,pw*c6+.5d0*density*sum(u**2)]
          expected_cache(i,0,k,:)=[density,u,pw,tw]
        enddo
      enddo
      call check(device_state_upload(state,device_q,q,bytes(size(q))))
      call check(device_state_upload(state,device_rho,cache(:,:,:,1),bytes(nodes)))
      call check(device_state_upload(state,device_velocity,cache(:,:,:,2:4),bytes(3*nodes)))
      call check(device_state_upload(state,device_pressure,cache(:,:,:,5),bytes(nodes)))
      call check(device_state_upload(state,device_temperature,cache(:,:,:,6),bytes(nodes)))
      call check(device_state_upload(state,device_metric,metric,bytes(size(metric))))
      call check(device_state_upload(state,device_rhs,rhs,bytes(size(rhs))))
      call check(wall_forcing_upload(state,speed,bytes(size(speed))))
      call check(wall41_forcing_y(state,tw,c2,c6))
      call check(device_state_download(state,device_q,q,bytes(size(q))))
      call check(device_state_download(state,device_rho,cache(:,:,:,1),bytes(nodes)))
      call check(device_state_download(state,device_velocity,cache(:,:,:,2:4),bytes(3*nodes)))
      call check(device_state_download(state,device_pressure,cache(:,:,:,5),bytes(nodes)))
      call check(device_state_download(state,device_temperature,cache(:,:,:,6),bytes(nodes)))
      call check(device_state_download(state,device_rhs,rhs,bytes(size(rhs))))
      if(.not.all(ieee_is_finite(q)).or..not.all(ieee_is_finite(cache))) error stop 'nonfinite wall state'
      if(any(rhs/=7)) error stop 'wall forcing changed RHS'
      do c=1,5
        err=maxval(abs(q(:,:,:,c)-expected_q(:,:,:,c)))/max(1.d0,maxval(abs(expected_q(:,:,:,c))))
        maximum=max(maximum,err)
        do k=-hm,km+hm
          do j=-hm,jm+hm
            do i=-hm,im+hm
              if(j==0.and.i>=0.and.i<=im.and.k>=0.and.k<=km) cycle
              if(q(i,j,k,c)/=17) error stop 'wall forcing changed an off-face node'
            enddo
          enddo
        enddo
      enddo
      do c=1,6
        maximum=max(maximum,maxval(abs(cache(:,:,:,c)-expected_cache(:,:,:,c)))/ &
          max(1.d0,maxval(abs(expected_cache(:,:,:,c)))))
      enddo
      do k=0,km
        do i=0,im
          normal=metric(i,0,k,2,:);normal=normal/sqrt(sum(normal**2))
          maximum=max(maximum,abs(sum(cache(i,0,k,2:4)*normal)-speed(i,k)))
          maximum=max(maximum,maxval(abs(cache(i,0,k,2:4)-speed(i,k)*normal)))
        enddo
      enddo
      cases=cases+1
    enddo
    call check(device_state_destroy(state))
  enddo
  if(maximum>2.d-10) error stop 'wall forcing analytic mismatch'
  write(*,'(a,i0,a,es24.16e3)') 'WALL_FORCING_PROJECTION_PASS cases=',cases,' max_scaled=',maximum
contains
  integer(c_size_t) function bytes(count)
    integer,intent(in) :: count
    bytes=int(count,c_size_t)*c_sizeof(0.d0)
  end function
  subroutine check(status)
    integer(c_int),intent(in) :: status
    if(status/=0) then
      write(*,'(a,i0)') 'wall forcing device status=',status
      error stop 'wall forcing device operation'
    endif
  end subroutine
end program
