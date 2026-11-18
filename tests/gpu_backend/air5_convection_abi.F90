program air5_convection_abi
  use iso_c_binding
  use mpi
  use device_state
  use halo_transport,only:device_transport_prepare,device_exchange_axis
  use chemistry_device_properties,only:air5_prepare=>air5_device_prepare,air5_check=>air5_device_check, &
    air5_primitives=>air5_device_primitives,air5_primitives_regions=>air5_device_primitives_regions
  use chemistry_device_transport
  use chemistry_device_boundary
  use chemistry_device_coupling,only:air5_device_options,air5_chemistry_prepare
  implicit none
  type(c_ptr) :: handle,buffers(4),top_high_address
  type(device_view) :: view
  type(air5_transport_failure) :: failure
  type(air5_device_options) :: boundary_options
  integer(c_int),parameter :: im=8,jm=7,km=6,h=5
  integer :: n,u,c,d,f,axis,top_mode,index,status,ierr
  integer(c_int) :: code,types(3)
  integer(c_size_t) :: capacity,bytes
  integer,allocatable :: statuses(:,:)
  real(c_double),allocatable :: properties(:,:),data(:)
  real(c_double),allocatable :: boundary_profile(:,:)
  real(c_double),allocatable :: boundary_top_x(:,:)
  real(c_double),allocatable :: boundary_spacing(:,:)
  real(c_double) :: q(11),rho,velocity(3),pressure,y(5),floor(5),constant
  real(c_double) :: filter_tr(5),filter_v(5),node_input(22)
  real(c_double) :: incident_states(11,2)
  real(c_double) :: abi_spacing((im+1)*(km+1))
  integer(c_int) :: filter_global(3),filter_periodic(3),scalar_mode
  integer(c_int) :: region_first(3,2),region_last(3,2)
  integer,parameter :: fields(9)=[device_q,device_rho,device_velocity,device_pressure,device_air5_species, &
    device_metric,device_jacobian,device_rhs,device_save]
  integer,parameter :: flux_fields(5)=[device_stress,device_heat_flux,device_air5_species_flux, &
    device_air5_ev_flux,device_air5_species_energy_flux]
  character(1024) :: input
  call get_command_argument(1,input)
  open(newunit=u,file=trim(input),access='stream',form='unformatted',status='old')
  read(u)n;allocate(statuses(7,n),properties(114,n));read(u)statuses,properties;close(u)
  q=properties(1:11,120);rho=properties(12,120);velocity=properties(13:15,120);pressure=properties(23,120)
  y=properties(17:21,120);floor=properties(110:114,1)
  call check(device_bind(0_c_int))
  call MPI_Init(ierr);call check(int(ierr,c_int))
  call check(device_state_create(0_c_int,im,jm,km,h,1_c_int,handle))
  call check(air5_prepare(handle));call check(air5_geometry_prepare(handle));call check(air5_faces_prepare(handle))
  call check(air5_symmetric_prepare(handle,floor,1_c_int))
  call check(device_state_view(handle,device_air5_top_normal_high,view));top_high_address=view%data
  abi_spacing=0.125_c_double
  call check(air5_characteristic_top_prepare(handle,abi_spacing,int(size(abi_spacing),c_size_t)))
  call check(device_state_view(handle,device_air5_top_normal_high,view))
  if(.not.c_associated(view%data,top_high_address))error stop 'symmetric-first shared high plane moved'
  call check(device_transport_prepare(handle,int(MPI_COMM_SELF,c_int), &
    [int(MPI_PROC_NULL,c_int),int(MPI_PROC_NULL,c_int),int(MPI_PROC_NULL,c_int)], &
    [int(MPI_PROC_NULL,c_int),int(MPI_PROC_NULL,c_int),int(MPI_PROC_NULL,c_int)], &
    [1_c_int,1_c_int,1_c_int],11_c_int,0_c_int,0_c_int,[1_c_int,1_c_int,1_c_int]))
  do f=1,size(fields)
    call check(device_state_view(handle,int(fields(f),c_int),view))
    allocate(data(product(view%extent)));index=0
    do d=1,view%extent(5);do c=1,view%extent(4)
      select case(fields(f))
      case(device_q,device_save);constant=q(c)
      case(device_rho);constant=rho
      case(device_velocity);constant=velocity(c)
      case(device_pressure);constant=pressure
      case(device_air5_species);constant=y(c)
      case(device_jacobian);constant=1.0_c_double
      case(device_metric);constant=merge(1.0_c_double,0.0_c_double,c==d)
      case default;constant=0.0_c_double
      end select
      n=product(view%extent(1:3));data(index+1:index+n)=constant;index=index+n
    enddo;enddo
    bytes=int(size(data),c_size_t)*8_c_size_t;call check(device_state_upload(handle,int(fields(f),c_int),data,bytes));deallocate(data)
  enddo
  call check(device_state_view(handle,device_air5_top_normal_high,view))
  if(any(view%extent/=[im+1,1,km+1,11,1]))error stop 'top plane descriptor mismatch'
  allocate(data(product(view%extent)));data=0.0_c_double
  call check(device_state_upload(handle,device_air5_top_normal_high,data,int(size(data),c_size_t)*8_c_size_t));deallocate(data)
  boundary_options=air5_device_options(1_c_int,0_c_int,0_c_int,1_c_int,1_c_int,32_c_int,0_c_int, &
    1e-9_c_double,1e-13_c_double,0.0_c_double)
  call check(air5_chemistry_prepare(handle,boundary_options))
  ! Explicit component sequence, not the old all-periodic fast controller.
  do axis=1,3
    call check(air5_compensation_axis(handle,int(axis,c_int),0_c_int))
    call check(device_exchange_axis(handle,device_q,int(axis-1,c_int),1_c_int))
    call check(air5_compensation_axis(handle,int(axis,c_int),1_c_int))
  enddo
  call check(device_state_view(handle,device_q,view));allocate(data(product(view%extent)));n=product(view%extent(1:3))
  call check(device_state_download(handle,device_q,data,int(size(data),c_size_t)*8_c_size_t))
  do c=1,11
    if(any(data((c-1)*n+1:c*n)/=q(c)))error stop 'compensation ABI uniform q'
  enddo
  deallocate(data)
  call check(device_state_view(handle,device_air5_carry,view));allocate(data(product(view%extent)))
  call check(device_state_download(handle,device_air5_carry,data,int(size(data),c_size_t)*8_c_size_t))
  if(any(data/=0.0_c_double))error stop 'compensation ABI nonzero carry'
  deallocate(data)
  region_first(:,1)=[0_c_int,0_c_int,0_c_int];region_last(:,1)=[im/2,jm,km]
  region_first(:,2)=[im/2+1,0_c_int,0_c_int];region_last(:,2)=[im,jm,km]
  call check(air5_primitives_regions(handle,2_c_int,region_first,region_last,1_c_int))
  call check(air5_check(handle,status));if(status/=0)error stop 'primitive regions ABI status'
  call check(air5_symmetric_buffers(handle,buffers,capacity))
  if(.not.all([(c_associated(buffers(c)),c=1,4)]))error stop 'missing borrowed plane slots'
  if(capacity/=12_c_size_t*(im+1)*(jm+1))error stop 'plane capacity mismatch'
  if(c_sizeof(failure)/=568_c_size_t)error stop 'failure ABI size mismatch'
  do top_mode=0,1
    types=merge([4_c_int,4_c_int,3_c_int],[3_c_int,3_c_int,3_c_int],top_mode==1)
    call check(air5_symmetric_base(handle,types,1.0_c_double,0.0_c_double,1e-7_c_double,1_c_int,int(top_mode,c_int)))
    call check(air5_check(handle,status));if(status/=0)error stop 'base ABI status'
    if(top_mode==1)then
      call check(air5_symmetric_top_normal(handle,types,1e-7_c_double,1_c_int))
      call check(air5_check(handle,status));if(status/=0)error stop 'top ABI status'
    endif
    do axis=1,3
      if(top_mode==1.and.axis==2)cycle
      call check(air5_symmetric_faces(handle,types,int(axis,c_int),0_c_int,1e-7_c_double,int(top_mode,c_int),1_c_int))
      call check(air5_check(handle,status));if(status/=0)error stop 'face ABI status'
      if(top_mode==0)then
        call check(air5_symmetric_exchange(handle,int(axis,c_int)))
      else
        call check(air5_symmetric_plane(handle,int(axis,c_int),0_c_int,merge(1_c_int,0_c_int,types(axis)==3), &
          merge(1_c_int,0_c_int,types(axis)==3)))
        if(types(axis)==3)call check(air5_symmetric_copy_local_planes(handle,int(axis,c_int)))
        call check(air5_symmetric_plane(handle,int(axis,c_int),1_c_int,merge(1_c_int,0_c_int,types(axis)==3), &
          merge(1_c_int,0_c_int,types(axis)==3)))
      endif
      call check(air5_symmetric_apply(handle,types,int(axis,c_int),1e-7_c_double,int(top_mode,c_int)))
      call check(air5_check(handle,status));if(status/=0)error stop 'apply ABI status'
    enddo
    call check(air5_symmetric_final(handle,types,1.0_c_double,0.0_c_double,1e-7_c_double,1_c_int,int(top_mode,c_int)))
    call check(air5_check(handle,status));if(status/=0)error stop 'final ABI status'
    call check(air5_symmetric_failure(handle,failure))
    if(any(failure%failure/=0).or.any(failure%values/=0).or.failure%cdt/=0)error stop 'failure ABI contents'
  enddo
  call check(air5_transport_prepare(handle,1_c_int))
  do f=1,size(flux_fields)
    call check(device_state_view(handle,int(flux_fields(f),c_int),view))
    allocate(data(product(view%extent)));data=real(f,c_double)
    call check(device_state_upload(handle,int(flux_fields(f),c_int),data,int(size(data),c_size_t)*8_c_size_t))
    deallocate(data)
  enddo
  call check(air5_flux_transport_prepare(handle,air5_flux_packed_overlap))
  call check(air5_flux_exchange(handle))
  do f=1,size(flux_fields)
    call check(device_state_view(handle,int(flux_fields(f),c_int),view))
    allocate(data(product(view%extent)))
    call check(device_state_download(handle,int(flux_fields(f),c_int),data,int(size(data),c_size_t)*8_c_size_t))
    if(any(data/=real(f,c_double)))error stop 'flux halo ABI constant values changed'
    deallocate(data)
  enddo
  call check(device_state_destroy(handle))
  if(command_argument_count()==2)then
    call get_command_argument(2,input)
    open(newunit=u,file=trim(input),access='stream',form='unformatted',status='old')
    read(u)filter_global,filter_periodic,filter_tr,filter_v,node_input;close(u)
    do scalar_mode=0_c_int,1_c_int
      call check(device_state_create(0_c_int,filter_global(1),filter_global(2),filter_global(3),h,1_c_int,handle))
      call check(air5_prepare(handle));call check(air5_geometry_prepare(handle));call check(air5_rk_prepare(handle))
      call check(device_transport_prepare(handle,int(MPI_COMM_SELF,c_int), &
        [int(MPI_PROC_NULL,c_int),int(MPI_PROC_NULL,c_int),int(MPI_PROC_NULL,c_int)], &
        [int(MPI_PROC_NULL,c_int),int(MPI_PROC_NULL,c_int),int(MPI_PROC_NULL,c_int)], &
        [1_c_int,1_c_int,1_c_int],11_c_int,0_c_int,0_c_int,filter_periodic))
      call check(air5_filter_prepare(handle,filter_tr,filter_v,filter_global,[0_c_int,0_c_int,0_c_int], &
        filter_periodic,scalar_mode))
      call check(device_state_view(handle,device_q,view));allocate(data(product(view%extent)))
      n=product(view%extent(1:3))
      do c=1,11
        data((c-1)*n+1:c*n)=node_input(c)
      enddo
      bytes=int(size(data),c_size_t)*8_c_size_t
      call check(device_state_upload(handle,device_q,data,bytes))
      call check(air5_filter_apply(handle,1_c_int,status));if(status/=0)error stop 'filter ABI physics status'
      call check(device_state_download(handle,device_q,data,bytes))
      do c=1,11
        if(any(data((c-1)*n+1:c*n)/=node_input(c)))error stop 'filter ABI constant state changed'
      enddo
      deallocate(data)
      call check(air5_primitives(handle,[0_c_int,0_c_int,0_c_int],filter_global,1_c_int))
      call check(air5_check(handle,status));if(status/=0)error stop 'boundary ABI input status'
      call check(device_state_view(handle,device_pressure,view));allocate(data(product(view%extent)))
      call check(device_state_download(handle,device_pressure,data,int(size(data),c_size_t)*8_c_size_t))
      index=1+int(sum(-int(view%lower(1:3),c_int64_t)*view%stride(1:3)));pressure=data(index)
      deallocate(data)
      call check(air5_pressure_outlet_upper_x(handle,pressure,1_c_int))
      call check(air5_check(handle,status));if(status/=0)error stop 'outlet ABI status'
      call check(air5_wall_lower_y(handle,1000.0_c_double,1_c_int))
      call check(air5_check(handle,status));if(status/=0)error stop 'wall ABI status'
      call check(air5_fixed_boundary_prepare(handle,node_input(1:11),2.0_c_double*node_input(1:11)))
      call check(air5_fixed_boundary_apply(handle,1_c_int,1_c_int,0_c_int,0.0_c_double,1_c_int))
      call check(air5_check(handle,status));if(status/=0)error stop 'fixed boundary ABI status'
      allocate(boundary_profile(0:filter_global(2),11))
      do c=1,11;do axis=0,filter_global(2)
        boundary_profile(axis,c)=node_input(c)*(1.0_c_double+real(axis,c_double)/1024.0_c_double)
      enddo;enddo
      call check(air5_profile_boundary_prepare(handle,boundary_profile,int(size(boundary_profile),c_size_t)))
      call check(air5_profile_inflow_lower_x(handle))
      call check(device_state_view(handle,device_q,view));allocate(data(product(view%extent)))
      call check(device_state_download(handle,device_q,data,int(size(data),c_size_t)*8_c_size_t))
      do c=1,11;do axis=0,filter_global(2)
        index=1+int(sum(-int(view%lower(1:3),c_int64_t)*view%stride(1:3)))+ &
          int(axis*view%stride(2)+(c-1)*view%stride(4))
        if(data(index)/=boundary_profile(axis,c))error stop 'profile ABI layout mismatch'
      enddo;enddo
      deallocate(data,boundary_profile)
      allocate(boundary_top_x(0:filter_global(1),0:filter_global(3)))
      do c=0,filter_global(3);do axis=0,filter_global(1)
        boundary_top_x(axis,c)=real(axis,c_double)*0.125_c_double
      enddo;enddo
      incident_states(:,1)=node_input(1:11);incident_states(:,2)=2.0_c_double*node_input(1:11)
      call check(air5_top_boundary_prepare(handle,node_input(1:11),incident_states,1.0_c_double, &
        0.5_c_double,1_c_int,boundary_top_x,int(size(boundary_top_x),c_size_t)))
      call check(air5_hbl_outflow_upper_x(handle,1_c_int,1_c_int,1_c_int))
      call check(air5_check(handle,status));if(status/=0)error stop 'HBL outflow ABI status'
      call check(air5_farfield_upper_y(handle,1_c_int,1_c_int))
      call check(air5_check(handle,status));if(status/=0)error stop 'HBL dynamic ghost ABI status'
      call check(air5_farfield_upper_y(handle,0_c_int,1_c_int))
      call check(air5_check(handle,status));if(status/=0)error stop 'HBL incident copy ABI status'
      boundary_options=air5_device_options(1_c_int,0_c_int,0_c_int,1_c_int,1_c_int,32_c_int,0_c_int, &
        1e-9_c_double,1e-13_c_double,0.0_c_double)
      call check(air5_chemistry_prepare(handle,boundary_options))
      call check(device_state_view(handle,device_air5_carry,view));allocate(data(product(view%extent)));data=0.125_c_double
      call check(device_state_upload(handle,device_air5_carry,data,int(size(data),c_size_t)*8_c_size_t))
      call check(air5_hbl_boundary_carry_reset(handle,[1_c_int,1_c_int,0_c_int],[1_c_int,1_c_int,0_c_int],1_c_int))
      call check(device_state_download(handle,device_air5_carry,data,int(size(data),c_size_t)*8_c_size_t))
      if(data(1)/=0.0_c_double)error stop 'HBL inlet carry ABI reset'
      index=1+int(filter_global(1)*view%stride(1)+filter_global(2)*view%stride(2))
      if(data(index)/=0.125_c_double)error stop 'HBL dynamic top carry ABI preservation'
      deallocate(data,boundary_top_x)
      allocate(boundary_spacing(0:filter_global(1),0:filter_global(3)));boundary_spacing=0.125_c_double
      incident_states(:,1)=node_input(1:11);incident_states(:,2)=node_input(1:11)
      call check(air5_top_boundary_prepare(handle,node_input(1:11),incident_states,1.0_c_double, &
        0.5_c_double,1_c_int,boundary_spacing,int(size(boundary_spacing),c_size_t)))
      call check(air5_characteristic_top_prepare(handle,boundary_spacing,int(size(boundary_spacing),c_size_t)))
      call check(device_state_view(handle,device_air5_top_normal_high,view));top_high_address=view%data
      call check(air5_faces_prepare(handle));call check(air5_symmetric_prepare(handle,filter_v,1_c_int))
      call check(device_state_view(handle,device_air5_top_normal_high,view))
      if(.not.c_associated(view%data,top_high_address))error stop 'characteristic-first shared high plane moved'
      call check(device_state_view(handle,device_q,view));allocate(data(product(view%extent)));n=product(view%extent(1:3))
      do c=1,11
        data((c-1)*n+1:c*n)=node_input(c)
      enddo
      call check(device_state_upload(handle,device_q,data,int(size(data),c_size_t)*8_c_size_t));deallocate(data)
      call check(device_state_view(handle,device_jacobian,view));allocate(data(product(view%extent)));data=1.0_c_double
      call check(device_state_upload(handle,device_jacobian,data,int(size(data),c_size_t)*8_c_size_t));deallocate(data)
      call check(device_state_view(handle,device_rhs,view));allocate(data(product(view%extent)));data=-91.0_c_double
      call check(device_state_upload(handle,device_rhs,data,int(size(data),c_size_t)*8_c_size_t));deallocate(data)
      call check(device_state_view(handle,device_air5_top_normal_high,view));allocate(data(product(view%extent)));data=-77.0_c_double
      call check(device_state_upload(handle,device_air5_top_normal_high,data,int(size(data),c_size_t)*8_c_size_t));deallocate(data)
      call check(air5_characteristic_top_rhs(handle,1_c_int,1_c_int,0.0_c_double,1500.0_c_double))
      call check(air5_check(handle,status));if(status/=0)error stop 'characteristic ABI uniform-state status'
      call check(device_state_view(handle,device_rhs,view));allocate(data(product(view%extent)))
      call check(device_state_download(handle,device_rhs,data,int(size(data),c_size_t)*8_c_size_t))
      do c=1,11;do f=0,filter_global(3);do axis=0,filter_global(2);do top_mode=0,filter_global(1)
        index=1+int(top_mode*view%stride(1)+axis*view%stride(2)+f*view%stride(3)+(c-1)*view%stride(4))
        constant=merge(0.0_c_double,-91.0_c_double,axis==filter_global(2).and.top_mode>=1)
        if(data(index)/=constant)error stop 'characteristic ABI top/inlet RHS mask'
      enddo;enddo;enddo;enddo
      deallocate(data)
      call check(device_state_view(handle,device_air5_top_normal_high,view));allocate(data(product(view%extent)))
      call check(device_state_download(handle,device_air5_top_normal_high,data,int(size(data),c_size_t)*8_c_size_t))
      do c=1,11;do f=0,filter_global(3);do top_mode=0,filter_global(1)
        index=1+int(top_mode*view%stride(1)+f*view%stride(3)+(c-1)*view%stride(4))
        if(data(index)/=merge(0.0_c_double,-77.0_c_double,top_mode>=1))error stop 'characteristic ABI high plane'
      enddo;enddo;enddo
      deallocate(data,boundary_spacing)
      call check(device_state_destroy(handle))
    enddo
    write(*,*)'AIR5_FILTER_FIELD_ABI_PASS scalar/full'
    write(*,*)'AIR5_BOUNDARY_ABI_PASS outlet/wall'
    write(*,*)'AIR5_BOUNDARY_TARGET_ABI_PASS fixed/profile j-fastest'
    write(*,*)'AIR5_HBL_STATE_ABI_PASS outflow/top/reference/carry'
    write(*,*)'AIR5_CHARACTERISTIC_ABI_PASS uniform zero RHS; inlet-owned column untouched'
  endif
  call MPI_Finalize(ierr);call check(int(ierr,c_int))
  write(*,*)'AIR5_CONVECTION_ABI_PASS',c_sizeof(failure),capacity
  write(*,*)'AIR5_FLUX_EXCHANGE_ABI_PASS'
  write(*,*)'AIR5_COMPENSATION_ABI_PASS explicit axes and normalization; uniform q unchanged'
contains
  subroutine check(code)
    integer(c_int),intent(in) :: code
    if(code/=0)then
      write(*,*)'ABI operation status=',code
      error stop 'ABI operation failed'
    endif
  end subroutine
end program
