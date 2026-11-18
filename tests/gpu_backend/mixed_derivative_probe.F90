program mixed_derivative_probe
  use iso_c_binding
  use ieee_arithmetic, only: ieee_is_finite
  use mpi
  use device_state
  use fluid_explicit
  implicit none
  type(c_ptr) :: state,address
  type(device_view) :: view
  integer :: rank,ranks,local_rank,status,ierr,unit,h(12),im,jm,km,hm,mode,stage,debug,cases=0
  integer :: topology(3)
  integer(c_int64_t) :: checked=0
  real(c_double) :: constants(8),maximum=0.d0,error,statistics(13),again(13)
  real(c_double),allocatable :: base(:,:,:,:),q(:,:,:,:),primitives(:,:,:,:),metric(:,:,:,:,:),jac(:,:,:), &
    gref(:,:,:,:,:),tref(:,:,:,:),conv(:,:,:,:),full(:,:,:,:),next(:,:,:,:), &
    gradient(:,:,:,:,:),thermal(:,:,:,:),observed(:,:,:,:),observed_gradient(:,:,:,:,:),observed_thermal(:,:,:,:)
  real(c_float),allocatable :: g4(:,:,:,:,:),t4(:,:,:,:)
  character(1024) :: root,path,text
  call get_environment_variable('OMPI_COMM_WORLD_LOCAL_RANK',text,status=status)
  if(status/=0) error stop 'mixed diagnostic probe requires Open MPI local rank'
  read(text,*) local_rank
  call check(device_bind(int(local_rank,c_int)))
  call MPI_Init(ierr);call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr);call MPI_Comm_size(MPI_COMM_WORLD,ranks,ierr)
  call get_command_argument(1,root);call get_command_argument(2,text);read(text,*) topology
  if(product(topology)/=ranks) error stop 'diagnostic packet topology'
  do mode=0,1
  do stage=1,3
    write(path,'(a,a,i1,a,i1,a,i8.8,a)') trim(root),'/mode',mode,'.stage',stage,'.rank',rank,'.bin'
    open(newunit=unit,file=trim(path),access='stream',form='unformatted',status='old')
    read(unit) h,constants
    im=h(4);jm=h(5);km=h(6);hm=h(7)
    allocate(base(-hm:im+hm,-hm:jm+hm,-hm:km+hm,5),q(-hm:im+hm,-hm:jm+hm,-hm:km+hm,5))
    allocate(primitives(-hm:im+hm,-hm:jm+hm,-hm:km+hm,6),metric(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3,3))
    allocate(jac(-hm:im+hm,-hm:jm+hm,-hm:km+hm))
    allocate(gref(0:im,0:jm,0:km,3,3),tref(0:im,0:jm,0:km,3),gradient(0:im,0:jm,0:km,3,3),thermal(0:im,0:jm,0:km,3))
    allocate(conv(0:im,0:jm,0:km,5),full(0:im,0:jm,0:km,5),next(0:im,0:jm,0:km,5))
    allocate(observed,mold=q);allocate(observed_gradient,mold=gradient);allocate(observed_thermal,mold=thermal)
    allocate(g4(0:im,0:jm,0:km,3,3),t4(0:im,0:jm,0:km,3))
    read(unit) base,q,primitives,metric,jac,gref,tref,conv,full,next;close(unit)
    do debug=0,1
      call check(device_state_create(int(local_rank,c_int),int(im,c_int),int(jm,c_int),int(km,c_int), &
        int(hm,c_int),int(debug,c_int),state))
      call check(fluid_prepare(state,1_c_int));call check(fluid_rhs_prepare(state,1_c_int))
      if(device_state_view(state,device_diagnostic_dvelocity,view)/=device_not_allocated) &
        error stop 'unused diagnostic workspace allocated'
      call check(device_state_upload(state,device_q,q,bytes(size(q),8)))
      call check(device_state_upload(state,device_rho,primitives(:,:,:,1),bytes(size(jac),8)))
      call check(device_state_upload(state,device_velocity,primitives(:,:,:,2:4),bytes(3*size(jac),8)))
      call check(device_state_upload(state,device_pressure,primitives(:,:,:,5),bytes(size(jac),8)))
      call check(device_state_upload(state,device_temperature,primitives(:,:,:,6),bytes(size(jac),8)))
      call check(device_state_upload(state,device_metric,metric,bytes(size(metric),8)))
      call check(device_state_upload(state,device_rhs,full,bytes(size(full),8)))
      call check(fluid_gradient(state))
      call check(device_state_download(state,device_dvelocity,gradient,bytes(size(gradient),8)))
      call check(device_state_download(state,device_dtemperature,thermal,bytes(size(thermal),8)))
      call check(fluid_prepare_diagnostic_gradient(state))
      call check(device_state_view(state,device_diagnostic_dvelocity,view));address=view%data
      if(view%element_bytes/=4.or.product(int(view%extent,c_int64_t))/=size(g4)) error stop 'diagnostic storage layout'
      call check(fluid_prepare_diagnostic_gradient(state))
      call check(device_state_view(state,device_diagnostic_dvelocity,view))
      if(.not.c_associated(address,view%data)) error stop 'repeated diagnostic prepare reallocated'
      call check(device_state_view(state,device_diagnostic_dtemperature,view))
      if(view%element_bytes/=4.or.product(int(view%extent,c_int64_t))/=size(t4)) error stop 'thermal diagnostic layout'
      call check(device_state_view(state,device_dvelocity,view))
      if(view%element_bytes/=8) error stop 'solver velocity gradient precision changed'
      call check(device_state_view(state,device_dtemperature,view))
      if(view%element_bytes/=8) error stop 'solver thermal gradient precision changed'
      call check(fluid_diagnostic_gradient(state,[0_c_int,0_c_int,0_c_int],[0_c_int,0_c_int,0_c_int]))
      call check(device_state_download_f32(state,device_diagnostic_dvelocity,g4,bytes(size(g4),4)))
      call check(device_state_download_f32(state,device_diagnostic_dtemperature,t4,bytes(size(t4),4)))
      if(.not.all(ieee_is_finite(g4)).or..not.all(ieee_is_finite(t4))) error stop 'nonfinite diagnostic gradient'
      if(any(transfer(g4,[0_c_int32_t],size(g4))/=transfer(real(gradient,c_float),[0_c_int32_t],size(g4)))) &
        error stop 'velocity diagnostic is not final FP32 conversion'
      if(any(transfer(t4,[0_c_int32_t],size(t4))/=transfer(real(thermal,c_float),[0_c_int32_t],size(t4)))) &
        error stop 'thermal diagnostic is not final FP32 conversion'
      checked=checked+size(g4,kind=c_int64_t)+size(t4,kind=c_int64_t)
      error=max(maxval(abs(real(g4,c_double)-real(real(gref,c_float),c_double))), &
        maxval(abs(real(t4,c_double)-real(real(tref,c_float),c_double))))
      if(.not.(error<=1.d-8)) error stop 'same-state diagnostic differs from original frozen gradient'
      maximum=max(maximum,error)
      call check(fluid_statistics(state,constants(1),constants(4),constants(5),statistics))
      call check(fluid_statistics(state,constants(1),constants(4),constants(5),again))
      if(any(transfer(statistics,[0_c_int64_t],13)/=transfer(again,[0_c_int64_t],13))) error stop 'diagnostic statistics repeat'
      if(.not.all(ieee_is_finite(statistics))) error stop 'nonfinite diagnostic statistics'
      call check(device_state_download(state,device_q,observed,bytes(size(observed),8)))
      if(any(transfer(q,[0_c_int64_t],size(q))/=transfer(observed,[0_c_int64_t],size(observed)))) &
        error stop 'diagnostics changed q or its halo'
      call check(device_state_download(state,device_rhs,observed(0:im,0:jm,0:km,:),bytes(size(full),8)))
      if(any(transfer(full,[0_c_int64_t],size(full))/= &
        transfer(observed(0:im,0:jm,0:km,:),[0_c_int64_t],size(full)))) error stop 'diagnostics changed RHS'
      call check(device_state_download(state,device_dvelocity,observed_gradient,bytes(size(gradient),8)))
      call check(device_state_download(state,device_dtemperature,observed_thermal,bytes(size(thermal),8)))
      if(any(transfer(observed_gradient,[0_c_int64_t],size(gradient))/=transfer(gradient,[0_c_int64_t],size(gradient))).or. &
        any(transfer(observed_thermal,[0_c_int64_t],size(thermal))/=transfer(thermal,[0_c_int64_t],size(thermal)))) &
        error stop 'diagnostics changed solver gradients'
      call check(device_state_destroy(state));cases=cases+1
    enddo
    deallocate(base,q,primitives,metric,jac,gref,tref,gradient,thermal,conv,full,next,observed,g4,t4,observed_gradient,observed_thermal)
  enddo
  enddo
  print *, 'CPP_MIXED_DERIVATIVE_PASS ',rank,cases,maximum,checked
  call MPI_Finalize(ierr)
contains
  integer(c_size_t) function bytes(n,width)
    integer,intent(in) :: n,width
    bytes=int(n,c_size_t)*width
  end function
  subroutine check(code)
    integer,intent(in) :: code
    if(code==0) return
    print *, 'DERIVATIVE_STATUS ',rank,mode,stage,debug,code
    call MPI_Abort(MPI_COMM_WORLD,1,ierr)
  end subroutine
end program
