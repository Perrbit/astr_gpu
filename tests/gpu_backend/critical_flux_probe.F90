program critical_flux_probe
  use iso_c_binding
  use ieee_arithmetic, only: ieee_is_finite
  use device_state
  use fluid_explicit, only: fluid_prepare,fluid_rhs_prepare,fluid_rhs_clear
  use upwind_operator_cpp
  use shock_sensor_cpp
  use crashfix_operator_cpp
  implicit none
  type(c_ptr) :: state
  integer :: h(8),im,jm,km,hm,n,unit,frame,debug,axis,c,s,cases=0,left(3),right(3)
  integer(c_int) :: low(3)=1,high(3)=1,first(3)=0,last(3)
  real(c_double) :: constants(3),error,maximum=0
  real(c_double),allocatable :: q(:,:,:,:),pv(:,:,:,:),metric(:,:,:,:,:),jac(:,:,:),expected(:,:), &
    flux(:,:,:,:),observed(:,:,:,:),raw(:,:,:)
  integer(c_int8_t),allocatable :: critical(:,:,:),active(:),fluid(:,:,:),shock(:,:,:)
  integer,allocatable :: points(:,:)
  character(1024) :: root,path
  call get_command_argument(1,root)
  do frame=1,6
    write(path,'(a,a,i1,a)') trim(root),'/reference.flux',frame,'.bin'
    open(newunit=unit,file=trim(path),status='old',access='stream',form='unformatted');read(unit) h,constants
    im=h(1);jm=h(2);km=h(3);hm=h(4);n=h(8);last=int([im,jm,km],c_int)
    if(h(5)/=5.or.n/=30.or.h(6)<0.or.h(6)>2.or.h(7)<0.or.h(7)>1) error stop 'critical flux reference header'
    allocate(critical(-hm:im+hm,-hm:jm+hm,-hm:km+hm),active(n),fluid(0:im,0:jm,0:km), &
      shock(0:im,0:jm,0:km),points(3,n),expected(5,n),q(-hm:im+hm,-hm:jm+hm,-hm:km+hm,5), &
      pv(-hm:im+hm,-hm:jm+hm,-hm:km+hm,6),metric(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3,3), &
      jac(-hm:im+hm,-hm:jm+hm,-hm:km+hm),raw(-hm:im+hm,-hm:jm+hm,-hm:km+hm), &
      flux(-1:im+1,-1:jm+1,-1:km+1,5))
    allocate(observed,mold=q)
    read(unit) critical,points,q,pv,metric,jac,expected,active;close(unit)
    do s=1,n
      axis=(s-1)/10+1;left=points(:,s);right=left;right(axis)=right(axis)+1
      left(axis)=max(0,left(axis));right(axis)=min(last(axis),right(axis))
      if(merge(1,0,critical(left(1),left(2),left(3))/=0.or.critical(right(1),right(2),right(3))/=0)/=active(s)) &
        error stop 'original critical face selection differs from its node mask'
    enddo
    do debug=0,1
      call check(device_state_create(0_c_int,im,jm,km,hm,int(debug,c_int),state))
      call check(fluid_prepare(state,1_c_int));call check(fluid_rhs_prepare(state,0_c_int))
      call check(upwind_prepare_cpp(state));call check(shock_prepare_cpp(state))
      ! With zero physical history, cover both absent and unrelated-halo-only masks.
      if(h(6)>0.or.debug==1) then
        fluid=1
        call check(crashfix_prepare_cpp(state,fluid,int(size(fluid),c_size_t)))
        call check(crashfix_history_upload_cpp(state,critical,int(size(critical),c_size_t)))
      endif
      call check(device_state_upload(state,device_q,q,bytes(size(q))))
      call check(device_state_upload(state,device_rho,pv(:,:,:,1),bytes(size(jac))))
      call check(device_state_upload(state,device_velocity,pv(:,:,:,2:4),bytes(3*size(jac))))
      call check(device_state_upload(state,device_pressure,pv(:,:,:,5),bytes(size(jac))))
      call check(device_state_upload(state,device_temperature,pv(:,:,:,6),bytes(size(jac))))
      call check(device_state_upload(state,device_metric,metric,bytes(size(metric))))
      call check(device_state_upload(state,device_jacobian,jac,bytes(size(jac))))
      raw=real(h(7),c_double)
      call check(device_state_upload(state,device_shock_sensor,raw,bytes(size(raw))))
      call check(shock_expand_cpp(state,low,high,.001d0))
      call check(shock_mask_download_cpp(state,shock,int(size(shock),c_size_t)))
      if(any(shock/=h(7))) error stop 'diagnostic shock mask not selected'
      call check(fluid_rhs_clear(state))
      do axis=1,3
        flux=99;call check(device_state_upload(state,device_interface_flux,flux,bytes(size(flux))))
        call check(upwind_axis_cpp(state,int(axis-1,c_int),5_c_int,1_c_int,constants(1),constants(2),constants(3), &
          0_c_int,low,high,first,last))
        call check(device_state_download(state,device_interface_flux,flux,bytes(size(flux))))
        do c=1,5
        do s=(axis-1)*10+1,axis*10
          if(.not.ieee_is_finite(flux(points(1,s),points(2,s),points(3,s),c))) error stop 'nonfinite critical flux'
          error=abs(flux(points(1,s),points(2,s),points(3,s),c)-expected(c,s))/max(1.d0,maxval(abs(expected(c,:))))
          if(error>2.d-10) then
            write(*,'(a,5(1x,i0),1x,es24.16e3)') 'CRITICAL_FLUX_DIFFERENCE',frame,debug,axis,c,s,error
            error stop 'critical-history flux differs from original GPU'
          endif
          maximum=max(maximum,error)
        enddo
        enddo
      enddo
      call check(device_state_download(state,device_q,observed,bytes(size(q))))
      if(any(transfer(observed,[0_c_int64_t],size(observed))/=transfer(q,[0_c_int64_t],size(q)))) &
        error stop 'critical flux changed q'
      call check(device_state_destroy(state));cases=cases+1
    enddo
    deallocate(critical,active,fluid,shock,points,expected,q,pv,metric,jac,raw,flux,observed)
  enddo
  write(*,'(a,i0,a,es24.16e3)') 'CPP_CRITICAL_FLUX_PASS cases=',cases,' max_scaled=',maximum
contains
  integer(c_size_t) function bytes(count)
    integer,intent(in) :: count
    bytes=int(count,c_size_t)*c_sizeof(0.d0)
  end function
  subroutine check(status)
    integer(c_int),intent(in) :: status
    if(status/=0) then
      write(*,'(a,i0)') 'critical flux device status=',status
      error stop 'critical flux device operation'
    endif
  end subroutine
end program
