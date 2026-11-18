program critical_shared_flux_probe
  use iso_c_binding
  use mpi
  use device_state
  use fluid_explicit
  use halo_transport
  use crashfix_operator_cpp
  use crashfix_runtime_cpp
  use upwind_operator_cpp
  use shock_sensor_cpp
  implicit none
  type(c_ptr) :: state
  type(crashfix_report) :: report
  integer :: rank,ranks,local_rank,devices,status,ierr,unit,h(13),im,jm,km,hm,axis,debug,a,i,j,k, &
    topology(3),coordinate(3),other(3),lower(3),upper(3),periodic(3)=[0,0,1],p(3),neighbor(3),reference_header(3),locations(3,2)
  integer(c_int) :: mode,low(3),high(3),first(3)=0,last(3)
  real(c_double) :: constants(3),stencil(8,11,2),expected(5,2),error,maximum=0,c2,c6
  real(c_double),allocatable :: q(:,:,:,:),pv(:,:,:,:),cpu_q(:,:,:,:),cpu_p(:,:,:,:),seed_q(:,:,:,:),seed_p(:,:,:,:), &
    metric(:,:,:,:,:),jac(:,:,:),flux(:,:,:,:),observed(:,:,:,:),sensor(:,:,:)
  integer(c_int8_t),allocatable :: fluid(:,:,:),initial_mask(:,:,:),cpu_mask(:,:,:),critical(:,:,:),actual_mask(:,:,:)
  character(1024) :: packets,reference,path,text
  character(32) :: seed_mode,transport
  call get_environment_variable('OMPI_COMM_WORLD_LOCAL_RANK',text,status=status)
  if(status/=0) error stop 'critical shared flux needs local MPI rank'
  read(text,*) local_rank
  status=device_count(devices);if(status/=0.or.devices<1) error stop 'critical shared flux needs GPU'
  status=device_bind(int(mod(local_rank,devices),c_int));if(status/=0) error stop 'critical shared flux device binding'
  call MPI_Init(ierr);call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr);call MPI_Comm_size(MPI_COMM_WORLD,ranks,ierr)
  if(ranks/=2) error stop 'critical shared flux uses two ranks'
  call get_command_argument(1,packets);call get_command_argument(2,reference)
  call get_command_argument(3,seed_mode);call get_command_argument(4,transport)
  select case(trim(transport))
  case('pinned');mode=halo_pinned
  case('device');mode=halo_device
  case default;error stop 'critical shared flux transport'
  end select
  open(newunit=unit,file=trim(reference),status='old',access='stream',form='unformatted')
  read(unit) reference_header,locations,constants,stencil,expected;close(unit)
  if(any(reference_header(2:3)/=1)) error stop 'critical flux reference is not the repaired shared contract'
  axis=reference_header(1)
  write(path,'(a,a,i8.8,a)') trim(packets),'/reference.case1.rank',rank,'.bin'
  open(newunit=unit,file=trim(path),status='old',access='stream',form='unformatted');read(unit) h
  im=h(1);jm=h(2);km=h(3);hm=h(4);last=int([im,jm,km],c_int)
  topology=h(9:11)/[im,jm,km];coordinate=h(6:8)/[im,jm,km]
  if(product(topology)/=ranks.or.h(5)/=5.or.h(12)/=1) error stop 'critical shared flux packet layout'
  allocate(q(-hm:im+hm,-hm:jm+hm,-hm:km+hm,5),pv(-hm:im+hm,-hm:jm+hm,-hm:km+hm,6))
  allocate(cpu_q,mold=q);allocate(seed_q,mold=q);allocate(observed,mold=q)
  allocate(cpu_p,mold=pv);allocate(seed_p,mold=pv)
  allocate(initial_mask(-hm:im+hm,-hm:jm+hm,-hm:km+hm),fluid(0:im,0:jm,0:km))
  allocate(cpu_mask,mold=initial_mask);allocate(critical,mold=initial_mask);allocate(actual_mask,mold=initial_mask)
  read(unit) fluid,initial_mask,seed_q,seed_p,cpu_mask,cpu_q,cpu_p,critical,q,pv;close(unit)
  allocate(metric(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3,3),jac(-hm:im+hm,-hm:jm+hm,-hm:km+hm))
  allocate(sensor,mold=jac);allocate(flux(-1:im+1,-1:jm+1,-1:km+1,5))
  metric=0;jac=1;sensor=0
  do a=1,3;metric(:,:,:,a,a)=1;enddo
  lower=MPI_PROC_NULL;upper=MPI_PROC_NULL
  do a=1,3
    other=coordinate;other(a)=other(a)-1
    if(other(a)>=0.or.periodic(a)/=0) then
      other(a)=modulo(other(a),topology(a));lower(a)=rank_number(other)
    endif
    other=coordinate;other(a)=other(a)+1
    if(other(a)<topology(a).or.periodic(a)/=0) then
      other(a)=modulo(other(a),topology(a));upper(a)=rank_number(other)
    endif
  enddo
  low=int(merge(1,0,lower==MPI_PROC_NULL),c_int);high=int(merge(1,0,upper==MPI_PROC_NULL),c_int)
  p=[im/2,jm/2,km/2]
  select case(trim(seed_mode))
  case('shared')
    p(axis)=0
    if(rank==1) critical(p(1),p(2),p(3))=1
  case('halo')
    p(axis)=last(axis)-1
    if(rank==0) critical(p(1),p(2),p(3))=1
  case default;error stop 'critical shared flux seed must be shared/halo'
  end select
  c2=constants(1)*constants(2)**2;c6=1.d0/(constants(1)-1.d0)
  do debug=0,1
    call check(device_state_create(int(mod(local_rank,devices),c_int),im,jm,km,hm,int(debug,c_int),state))
    call check(fluid_prepare(state,1_c_int));call check(fluid_rhs_prepare(state,0_c_int))
    call check(upwind_prepare_cpp(state));call check(shock_prepare_cpp(state))
    call check(crashfix_prepare_cpp(state,fluid,int(size(fluid),c_size_t)))
    call check(device_transport_prepare(state,int(MPI_COMM_WORLD,c_int),int(lower,c_int),int(upper,c_int), &
      int(topology,c_int),6_c_int,mode,0_c_int,int(periodic,c_int)))
    call check(device_state_upload(state,device_q,q,bytes(size(q))))
    call check(device_state_upload(state,device_rho,pv(:,:,:,1),bytes(size(jac))))
    call check(device_state_upload(state,device_velocity,pv(:,:,:,2:4),bytes(3*size(jac))))
    call check(device_state_upload(state,device_pressure,pv(:,:,:,5),bytes(size(jac))))
    call check(device_state_upload(state,device_temperature,pv(:,:,:,6),bytes(size(jac))))
    call check(device_state_upload(state,device_metric,metric,bytes(size(metric))))
    call check(device_state_upload(state,device_jacobian,jac,bytes(size(jac))))
    call check(crashfix_history_upload_cpp(state,critical,int(size(critical),c_size_t)))
    call check(repair_reconciled_cpp(state,int(h(6:8),c_int),int(h(9:11),c_int),int(periodic,c_int), &
      c2,c6,1.d-10/c2,MPI_COMM_WORLD,report))
    call check(crashfix_history_download_cpp(state,actual_mask,int(size(actual_mask),c_size_t)))
    p=locations(:,rank+1)-h(6:8);neighbor=p;neighbor(axis)=neighbor(axis)+1
    if(actual_mask(p(1),p(2),p(3))==0.and.actual_mask(neighbor(1),neighbor(2),neighbor(3))==0) &
      error stop 'canonical critical mark does not reach the shared interface'
    call check(device_state_upload(state,device_shock_sensor,sensor,bytes(size(sensor))))
    call check(shock_expand_cpp(state,low,high,.001d0));call check(fluid_rhs_clear(state))
    call check(upwind_axis_cpp(state,int(axis-1,c_int),5_c_int,1_c_int,constants(1),constants(2),constants(3), &
      0_c_int,low,high,first,last))
    call check(device_state_download(state,device_interface_flux,flux,bytes(size(flux))))
    error=maxval(abs(flux(p(1),p(2),p(3),:)-expected(:,rank+1))/max(1.d0,abs(expected(:,rank+1))))
    if(error>2.d-10) error stop 'critical shared flux differs from the repaired original contract'
    write(*,'(a,2(1x,i0),5(1x,es24.16e3))') 'CPP_CRITICAL_SHARED_FLUX_VALUES',rank,debug,flux(p(1),p(2),p(3),:)
    maximum=max(maximum,error)
    call check(device_state_download(state,device_q,observed,bytes(size(observed))))
    if(any(transfer(q,[0_c_int64_t],size(q))/=transfer(observed,[0_c_int64_t],size(q)))) &
      error stop 'healthy reconciliation or critical flux changed q'
    call check(device_state_destroy(state))
  enddo
  print *, 'CPP_CRITICAL_SHARED_FLUX_PASS ',rank,maximum
  call MPI_Finalize(ierr)
contains
  integer function rank_number(coordinate)
    integer,intent(in) :: coordinate(3)
    rank_number=coordinate(1)+topology(1)*(coordinate(2)+topology(2)*coordinate(3))
  end function
  integer(c_size_t) function bytes(n)
    integer,intent(in) :: n
    bytes=int(n,c_size_t)*8
  end function
  subroutine check(code)
    integer,intent(in) :: code
    if(code/=0) then
      print *, 'CRITICAL_SHARED_FLUX_STATUS ',rank,debug,code
      call MPI_Abort(MPI_COMM_WORLD,1,ierr)
    endif
  end subroutine
end program
