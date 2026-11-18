program native_explicit_probe
  use iso_c_binding
  use ieee_arithmetic, only: ieee_is_finite
  use mpi
  use astr_precision, only: real_kind,real_bytes,real_to_word
  use device_state
  use fluid_explicit
  use fluid_step
  use halo_transport
  implicit none
  type(c_ptr) :: state
  type(device_view) :: view
  integer :: rank,ranks,local_rank,ierr,status,cart,topology(3),coordinate(3),lo(3),hi(3),h(9), &
    im,jm,km,hm,origin(3),source,filtering,stage,debug,step,unit,workspace,field,phase_cases=0,coupled_cases=0
  integer(c_int) :: transport
  real(real_kind),parameter :: gamma=1.4_real_kind,mach=.1_real_kind,c2=gamma*mach**2,c6=1._real_kind/(gamma-1._real_kind), &
    reynolds=1600._real_kind,prandtl=.72_real_kind,c5=(gamma-1._real_kind)*mach**2, &
    tc=110.3_real_kind/273.15_real_kind,tc1=1._real_kind+tc,dt=1.e-4_real_kind
  real(real_kind) :: error(5),grad_error,thermal_error,maximum=0,coeff_a(3),coeff_b(3),coeff_c(3)
  real(real_kind),allocatable :: base(:,:,:,:),q(:,:,:,:),initial(:,:,:,:),p(:,:,:,:),metric(:,:,:,:,:),jac(:,:,:), &
    gref(:,:,:,:,:),tref(:,:,:,:),conv(:,:,:,:),full(:,:,:,:),following(:,:,:,:),observed(:,:,:,:), &
    gradient(:,:,:,:,:),thermal(:,:,:,:)
  character(1024) :: packets,output,folder,path,text
  character(32) :: layout
  if(real_bytes/=4) error stop 'native explicit32 probe requires the FP32 build'
  call get_environment_variable('OMPI_COMM_WORLD_LOCAL_RANK',text,status=status)
  if(status/=0) error stop 'native explicit32 requires Open MPI local rank'
  read(text,*) local_rank;call check(device_bind(int(local_rank,c_int)))
  call MPI_Init(ierr);call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr);call MPI_Comm_size(MPI_COMM_WORLD,ranks,ierr)
  call get_command_argument(1,packets);call get_command_argument(2,output)
  call get_command_argument(3,text);read(text,*) topology
  call get_command_argument(4,text)
  select case(trim(text))
  case('pinned');transport=halo_pinned
  case('device');transport=halo_device
  case default;error stop 'native explicit32 transport'
  end select
  call get_command_argument(5,text)
  select case(trim(text))
  case('scalar');workspace=1
  case('full');workspace=5
  case default;error stop 'native explicit32 filter storage'
  end select
  call get_command_argument(6,layout)
  if(len_trim(layout)==0) error stop 'native explicit32 packet layout required'
  if(product(topology)/=ranks) error stop 'native explicit32 topology'
  call MPI_Cart_create(MPI_COMM_WORLD,3,topology,[.true.,.true.,.true.],.false.,cart,ierr);call check(ierr)
  call MPI_Cart_coords(cart,rank,3,coordinate,ierr);call check(ierr)
  do field=1,3
    call MPI_Cart_shift(cart,field-1,1,lo(field),hi(field),ierr);call check(ierr)
  enddo
  coeff_a=[1._real_kind,.75_real_kind,1._real_kind/3._real_kind]
  coeff_b=[0._real_kind,.25_real_kind,2._real_kind/3._real_kind]
  coeff_c=[1._real_kind,.25_real_kind,2._real_kind/3._real_kind]
  do source=0,1
  do filtering=0,1
    if(source==0) write(folder,'(a,a,i0,2a)') trim(packets),'/cpu/filter',filtering,'/',trim(layout)
    if(source==1) write(folder,'(a,a,i0,2a)') trim(packets),'/gpu/filter',filtering,'/',trim(layout)
    do stage=1,3
      write(path,'(a,a,i0,a,i8.8,a)') trim(folder),'/stage',stage,'.rank',rank,'.bin'
      open(newunit=unit,file=trim(path),access='stream',form='unformatted',status='old');read(unit) h
      im=h(1);jm=h(2);km=h(3);hm=h(4);origin=h(5:7)
      if(h(8)/=stage.or.h(9)/=real_bytes.or.any(origin/=coordinate*[im,jm,km])) error stop 'native packet identity'
      allocate(base(-hm:im+hm,-hm:jm+hm,-hm:km+hm,5),q(-hm:im+hm,-hm:jm+hm,-hm:km+hm,5))
      allocate(initial,mold=q)
      allocate(p(-hm:im+hm,-hm:jm+hm,-hm:km+hm,6),metric(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3,3), &
        jac(-hm:im+hm,-hm:jm+hm,-hm:km+hm),gref(0:im,0:jm,0:km,3,3),tref(0:im,0:jm,0:km,3))
      allocate(conv(0:im,0:jm,0:km,5),full(0:im,0:jm,0:km,5),following(0:im,0:jm,0:km,5))
      allocate(observed,mold=q);allocate(gradient,mold=gref);allocate(thermal,mold=tref)
      read(unit) base,q,p,metric,jac,gref,tref,conv,full,following;close(unit)
      do debug=0,1
        call prepare()
        call check(device_state_upload(state,device_q,base,bytes(size(base))));call check(fluid_rk_save(state))
        call check(device_state_upload(state,device_q,q,bytes(size(q))))
        call check(device_state_upload(state,device_rho,p(:,:,:,1),bytes(size(jac))))
        call check(device_state_upload(state,device_velocity,p(:,:,:,2:4),bytes(3*size(jac))))
        call check(device_state_upload(state,device_pressure,p(:,:,:,5),bytes(size(jac))))
        call check(device_state_upload(state,device_temperature,p(:,:,:,6),bytes(size(jac))))
        call check(fluid_gradient(state))
        call check(device_state_download(state,device_dvelocity,gradient,bytes(size(gradient))))
        call check(device_state_download(state,device_dtemperature,thermal,bytes(size(thermal))))
        if(.not.all(ieee_is_finite(gradient)).or..not.all(ieee_is_finite(thermal))) error stop 'native gradients nonfinite'
        grad_error=maxval(abs(gradient-gref))/max(1._real_kind,maxval(abs(gref)))
        thermal_error=maxval(abs(thermal-tref))/max(1._real_kind,maxval(abs(tref)))
        call check(fluid_convection(state));call read_rhs();call report('conv',conv)
        call check(fluid_viscous_flux(state,reynolds,prandtl,c5,tc,tc1))
        do field=0,2
          call check(exchange(state,device_stress,int(field,c_int),0_c_int))
          call check(exchange(state,device_heat_flux,int(field,c_int),0_c_int))
        enddo
        call check(fluid_viscous_divergence(state));call read_rhs();call report('full',full)
        call check(fluid_rk_update(state,dt,coeff_a(stage),coeff_b(stage),coeff_c(stage)))
        call check(device_state_download(state,device_q,observed,bytes(size(observed))))
        call admissible();call report('updated',following)
        print *, 'CPP_NATIVE_EXPLICIT32_GRADIENT ',rank,source,filtering,stage,debug,grad_error,thermal_error
        call check(device_state_destroy(state));phase_cases=phase_cases+1
      enddo
      if(stage==1) then
        write(path,'(a,a,i8.8,a)') trim(folder),'/initial.rank',rank,'.bin'
        open(newunit=unit,file=trim(path),access='stream',form='unformatted',status='old');read(unit) initial;close(unit)
        do debug=0,1
          call prepare();call check(device_state_upload(state,device_q,initial,bytes(size(initial))))
          do step=1,4
          do field=1,3
            call check(fluid_advance_stage(state,field,dt,c2,c6,reynolds,prandtl,c5,tc,tc1, &
              filtering==1.and.field==1,.true.,exchange,begin_exchange,end_exchange))
          enddo
          enddo
          call check(device_state_download(state,device_q,observed,bytes(size(observed))));call admissible()
          write(path,'(a,a,i0,a,i0,a,i0,a,i8.8,a)') trim(output),'/source',source,'.filter',filtering,'.debug',debug,'.rank',rank,'.bin'
          open(newunit=unit,file=trim(path),access='stream',form='unformatted',status='new')
          write(unit) im,jm,km,hm,5;write(unit) observed;close(unit)
          if(source==0.and.filtering==0) call affine_and_constant_invariants()
          call check(device_state_destroy(state));coupled_cases=coupled_cases+1
        enddo
      endif
      deallocate(base,q,p,metric,jac,gref,tref,conv,full,following,observed,gradient,thermal,initial)
    enddo
  enddo
  enddo
  print *, 'CPP_NATIVE_EXPLICIT32_PASS ',rank,phase_cases,coupled_cases,maximum
  call MPI_Comm_free(cart,ierr);call check(ierr);call MPI_Finalize(ierr)
contains
  integer(c_size_t) function bytes(n)
    integer,intent(in) :: n
    bytes=int(n,c_size_t)*real_bytes
  end function
  subroutine check(code)
    integer,intent(in) :: code
    if(code==0) return
    print *, 'NATIVE_EXPLICIT32_STATUS ',rank,source,filtering,stage,debug,code
    call MPI_Abort(MPI_COMM_WORLD,1,ierr)
  end subroutine
  subroutine prepare()
    call check(device_state_create(int(local_rank,c_int),int(im,c_int),int(jm,c_int),int(km,c_int),int(hm,c_int),int(debug,c_int),state))
    call check(fluid_prepare(state,1_c_int));call check(fluid_rhs_prepare(state,1_c_int))
    call check(device_transport_prepare(state,int(cart,c_int),int(lo,c_int),int(hi,c_int), &
      int(topology,c_int),6_c_int,transport,0_c_int,[1_c_int,1_c_int,1_c_int]))
    if(workspace==1) then
      call check(fluid_filter_prepare_scalar(state))
    else
      call check(fluid_filter_prepare(state))
    endif
    do field=device_q,device_filter_work
      call check(device_state_view(state,int(field,c_int),view))
      if(view%element_bytes/=real_bytes) error stop 'native explicit field uses a different real type'
    enddo
    if(device_state_view(state,device_diagnostic_dvelocity,view)/=device_not_allocated) &
      error stop 'native precision allocated mixed diagnostic copies'
    call check(device_state_upload(state,device_metric,metric,bytes(size(metric))))
    call check(device_state_upload(state,device_jacobian,jac,bytes(size(jac))))
  end subroutine
  subroutine read_rhs()
    call check(device_state_download(state,device_rhs,observed(0:im,0:jm,0:km,:),bytes(size(conv))))
  end subroutine
  subroutine report(label,reference)
    character(*),intent(in) :: label
    real(real_kind),intent(in) :: reference(:,:,:,:)
    integer :: c
    if(.not.all(ieee_is_finite(observed(0:im,0:jm,0:km,:)))) error stop 'native component nonfinite'
    do c=1,5
      error(c)=maxval(abs(observed(0:im,0:jm,0:km,c)-reference(:,:,:,c)))/max(1._real_kind,maxval(abs(reference(:,:,:,c))))
    enddo
    maximum=max(maximum,maxval(error))
    print *, 'CPP_NATIVE_EXPLICIT32_DIFFERENCE ',rank,source,filtering,stage,debug,label,error
  end subroutine
  subroutine admissible()
    if(.not.all(ieee_is_finite(observed(0:im,0:jm,0:km,:))).or.any(observed(0:im,0:jm,0:km,1)<=0)) &
      error stop 'native state nonfinite or nonpositive density'
    if(any(observed(0:im,0:jm,0:km,5)-.5_real_kind*sum(observed(0:im,0:jm,0:km,2:4)**2,dim=4)/ &
      observed(0:im,0:jm,0:km,1)<=0)) error stop 'native nonpositive internal energy'
  end subroutine
  subroutine affine_and_constant_invariants()
    integer :: c,d,i,j,k
    real(real_kind),allocatable :: constant_q(:,:,:,:),constant_metric(:,:,:,:,:),constant_jac(:,:,:)
    real(real_kind),allocatable :: affine_velocity(:,:,:,:),affine_temperature(:,:,:)
    real(real_kind) :: projection(3,3),slopes(3,3),thermal_slopes(3),expected_gradient(3,3),expected_thermal(3)
    ! Powers of two make the constant preservation checks exact in both formats.
    allocate(constant_q,mold=initial);allocate(constant_metric,mold=metric);allocate(constant_jac,mold=jac)
    allocate(affine_velocity(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3));allocate(affine_temperature,mold=jac)
    constant_q(:,:,:,1)=2;constant_q(:,:,:,2)=.25_real_kind;constant_q(:,:,:,3)=-.5_real_kind
    constant_q(:,:,:,4)=1;constant_q(:,:,:,5)=8.328125_real_kind
    call check(device_state_upload(state,device_q,constant_q,bytes(size(constant_q))))
    call check(fluid_primitives(state,2._real_kind,4._real_kind));call check(fluid_check(state))
    call check(device_state_download(state,device_velocity,affine_velocity,bytes(size(affine_velocity))))
    if(any(affine_velocity(0:im,0:jm,0:km,1)/=.125_real_kind).or. &
       any(affine_velocity(0:im,0:jm,0:km,2)/=-.25_real_kind).or. &
       any(affine_velocity(0:im,0:jm,0:km,3)/=.5_real_kind)) error stop 'dyadic primitive velocities differ'
    do c=3,6
      if(c==4) cycle
      call check(device_state_download(state,int(c,c_int),affine_temperature,bytes(size(affine_temperature))))
      if(any(affine_temperature(0:im,0:jm,0:km)/=2._real_kind)) error stop 'dyadic rho/p/T conversion differs'
    enddo
    call check(fluid_conservative(state,2._real_kind));call check(fluid_check(state))
    call check(device_state_download(state,device_q,observed,bytes(size(observed))))
    if(any(real_to_word(observed)/=real_to_word(constant_q))) error stop 'dyadic conservative roundtrip changed words'
    print *, 'CPP_NATIVE_EXPLICIT32_CONVERSION ',rank,debug,' exact_dyadic_primitive_conservative_roundtrip'
    projection=reshape([1._real_kind,.25_real_kind,.5_real_kind, &
      .5_real_kind,2._real_kind,.25_real_kind,.25_real_kind,.5_real_kind,1._real_kind],[3,3])
    slopes=reshape([1._real_kind,2._real_kind,4._real_kind, &
      2._real_kind,-1._real_kind,.5_real_kind,.5_real_kind,1._real_kind,-2._real_kind],[3,3])
    thermal_slopes=[.125_real_kind,.25_real_kind,.5_real_kind]
    expected_gradient=matmul(transpose(slopes),projection);expected_thermal=matmul(thermal_slopes,projection)
    do k=-hm,km+hm
    do j=-hm,jm+hm
    do i=-hm,im+hm
      affine_velocity(i,j,k,:)=matmul(real([i,j,k],real_kind),slopes)
      affine_temperature(i,j,k)=32._real_kind+dot_product(real([i,j,k],real_kind),thermal_slopes)
      constant_metric(i,j,k,:,:)=projection
    enddo
    enddo
    enddo
    call check(device_state_upload(state,device_velocity,affine_velocity,bytes(size(affine_velocity))))
    call check(device_state_upload(state,device_temperature,affine_temperature,bytes(size(affine_temperature))))
    call check(device_state_upload(state,device_metric,constant_metric,bytes(size(constant_metric))))
    call check(fluid_gradient(state))
    call check(device_state_download(state,device_dvelocity,gradient,bytes(size(gradient))))
    call check(device_state_download(state,device_dtemperature,thermal,bytes(size(thermal))))
    do c=1,3
      do d=1,3
        if(any(gradient(:,:,:,c,d)/=expected_gradient(c,d))) error stop 'affine full-metric gradient differs from exact dyadic result'
      enddo
      if(any(thermal(:,:,:,c)/=expected_thermal(c))) error stop 'affine thermal projection differs from exact dyadic result'
    enddo
    print *, 'CPP_NATIVE_EXPLICIT32_AFFINE ',rank,debug,' exact_full_metric_projection'
    constant_q(:,:,:,1)=2;constant_q(:,:,:,2)=.25_real_kind;constant_q(:,:,:,3)=-.5_real_kind
    constant_q(:,:,:,4)=1;constant_q(:,:,:,5)=512
    constant_metric=0
    do c=1,3;constant_metric(:,:,:,c,c)=1;enddo
    constant_metric(:,:,:,1,1)=.5_real_kind
    constant_jac=2
    call check(device_state_upload(state,device_q,constant_q,bytes(size(constant_q))))
    call check(device_state_upload(state,device_metric,constant_metric,bytes(size(constant_metric))))
    call check(device_state_upload(state,device_jacobian,constant_jac,bytes(size(constant_jac))))
    do d=1,3
      call check(fluid_advance_stage(state,d,dt,c2,c6,reynolds,prandtl,c5,tc,tc1,d==1,.true.,exchange,begin_exchange,end_exchange))
      call read_rhs()
      if(any(observed(0:im,0:jm,0:km,:)/=0)) error stop 'constant state has nonzero RHS'
    enddo
    call check(device_state_download(state,device_q,observed,bytes(size(observed))))
    if(any(real_to_word(observed)/=real_to_word(constant_q))) error stop 'filter/RK changed a constant state'
    print *, 'CPP_NATIVE_EXPLICIT32_CONSTANT ',rank,source,filtering,debug,' exact_zero_rhs_filter_and_rk'
  end subroutine
  integer(c_int) function exchange(handle,id,axis,reconcile) bind(C)
    type(c_ptr),value :: handle
    integer(c_int),value :: id,axis,reconcile
    exchange=device_exchange_axis(handle,id,axis,reconcile)
  end function
  integer(c_int) function begin_exchange(handle,id,axis,reconcile) bind(C)
    type(c_ptr),value :: handle
    integer(c_int),value :: id,axis,reconcile
    begin_exchange=device_exchange_begin(handle,id,axis,reconcile)
  end function
  integer(c_int) function end_exchange(handle,id,axis,reconcile) bind(C)
    type(c_ptr),value :: handle
    integer(c_int),value :: id,axis,reconcile
    end_exchange=device_exchange_end(handle,id,axis,reconcile)
  end function
end program
