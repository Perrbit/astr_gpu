program native_curve_gradient_probe
  use iso_c_binding, only: c_ptr,c_int,c_size_t
  use astr_precision, only: real_kind,real_bytes,real_to_word
  use ieee_arithmetic, only: ieee_is_finite,ieee_value,ieee_quiet_nan
  use device_state
  use fluid_explicit
  implicit none
  type(c_ptr) :: state
  integer :: header(11),im,jm,km,hm,unit,debug,scenario,i,j,k,b
  integer(c_int) :: low(3),high(3)
  real(real_kind),allocatable :: primitive(:,:,:,:),seed(:,:,:,:),observed(:,:,:,:),metric(:,:,:,:,:), &
    gradient(:,:,:,:,:),temperature(:,:,:,:)
  real(real_kind) :: nan,affine_error
  character(1400) :: input,output,path
  call get_command_argument(1,input);call get_command_argument(2,output)
  open(newunit=unit,file=trim(input),access='stream',form='unformatted',status='old')
  read(unit) header
  im=header(1);jm=header(2);km=header(3);hm=header(4)
  low=int(header(5:7),c_int);high=int(header(8:10),c_int)
  if(header(11)/=real_bytes) error stop 'curve gradient packet kind mismatch'
  allocate(primitive(-hm:im+hm,-hm:jm+hm,-hm:km+hm,6),seed(-hm:im+hm,-hm:jm+hm,-hm:km+hm,6), &
    observed(-hm:im+hm,-hm:jm+hm,-hm:km+hm,4),metric(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3,3), &
    gradient(0:im,0:jm,0:km,3,3),temperature(0:im,0:jm,0:km,3))
  read(unit) primitive,metric
  close(unit)
  call check(device_bind(0_c_int))
  nan=ieee_value(0._real_kind,ieee_quiet_nan)
  do debug=0,1
    call check(device_state_create(0_c_int,im,jm,km,hm,int(debug,c_int),state))
    call check(fluid_prepare(state,1_c_int))
    call check(device_state_upload(state,device_metric,metric,bytes(size(metric))))
    do scenario=0,2
      seed=primitive
      if(scenario==1) then
        seed(:,:,:,2)=.25_real_kind;seed(:,:,:,3)=-.5_real_kind;seed(:,:,:,4)=.75_real_kind;seed(:,:,:,6)=4
      elseif(scenario==2) then
        do k=-hm,km+hm
        do j=-hm,jm+hm
        do i=-hm,im+hm
          seed(i,j,k,2:4)=real([i,j,k],real_kind);seed(i,j,k,6)=real(i+j+k,real_kind)
        enddo
        enddo
        enddo
      endif
      if(low(1)==1) seed(:-1,:,:,:)=nan
      if(high(1)==1) seed(im+1:,:,:,:)=nan
      if(low(2)==1) seed(:,:-1,:,:)=nan
      if(high(2)==1) seed(:,jm+1:,:,:)=nan
      if(low(3)==1) seed(:,:,:-1,:)=nan
      if(high(3)==1) seed(:,:,km+1:,:)=nan
      call check(device_state_upload(state,device_velocity,seed(:,:,:,2:4),bytes(3*size(seed(:,:,:,1)))))
      call check(device_state_upload(state,device_temperature,seed(:,:,:,6),bytes(size(seed(:,:,:,1)))))
      call check(fluid_gradient_basic_boundary(state,low,high))
      call check(device_state_download(state,device_dvelocity,gradient,bytes(size(gradient))))
      call check(device_state_download(state,device_dtemperature,temperature,bytes(size(temperature))))
      if(.not.all(ieee_is_finite(gradient)).or..not.all(ieee_is_finite(temperature))) &
        error stop 'curve gradient read outside a physical end'
      call check(device_state_download(state,device_velocity,observed(:,:,:,1:3),bytes(3*size(observed(:,:,:,1)))))
      call check(device_state_download(state,device_temperature,observed(:,:,:,4),bytes(size(observed(:,:,:,1)))))
      if(any(real_to_word(observed(:,:,:,1:3))/=real_to_word(seed(:,:,:,2:4))).or. &
        any(real_to_word(observed(:,:,:,4))/=real_to_word(seed(:,:,:,6)))) error stop 'curve gradient changed primitives'
      if(scenario==0) then
        write(path,'(a,".",i1,".bin")') trim(output),debug
        open(newunit=unit,file=trim(path),access='stream',form='unformatted',status='new')
        write(unit) im,jm,km,real_bytes
        write(unit) gradient,temperature
        close(unit)
      elseif(scenario==1) then
        if(any(gradient/=0).or.any(temperature/=0)) error stop 'curve constant gradient is not zero'
      else
        affine_error=maxval(abs(gradient-metric(0:im,0:jm,0:km,:,:)))
        do b=1,3
          affine_error=max(affine_error,maxval(abs(temperature(:,:,:,b)-sum(metric(0:im,0:jm,0:km,:,b),dim=4))))
        enddo
        write(*,'(a,2(1x,i0),1x,es24.16)') 'CPP_NATIVE_CURVE_AFFINE',real_bytes,debug,affine_error
      endif
    enddo
    call check(device_state_destroy(state))
    write(*,'(a,2(1x,i0))') 'CPP_NATIVE_CURVE_GRADIENT_PASS',real_bytes,debug
  enddo
contains
  integer(c_size_t) function bytes(n)
    integer,intent(in) :: n
    bytes=int(n,c_size_t)*int(real_bytes,c_size_t)
  end function
  subroutine check(status)
    integer,intent(in) :: status
    if(status/=0) then
      print *, 'curve gradient device status=',status
      error stop 'curve gradient device call'
    endif
  end subroutine
end program
