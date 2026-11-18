program air5_convection_budget_probe
  use iso_c_binding, only: c_ptr,c_size_t,c_double,c_int,c_int64_t,c_int8_t
  use mpi
  use chemistry_air5_data, only: air5_ru,air5_molar_mass
#ifdef ASTR_OLD_CONVECTION
  use cudafor
  use commvar, only: im,jm,km,hm,lihomo,ljhomo,lkhomo,npdci,npdcj,npdck,lcomb,flowtype,shkcrt
  use parallel, only: mpirank,isize,jsize,ksize,mpileft,mpiright,mpidown,mpiup,mpiback,mpifront
  use commarray_gpu, only: q_d,qsave_d,qrhs_d,jacob_d,dxi_d,rho_d,vel_d,prs_d,spc_d,shock_mask_d, &
    diffusion_ratio_d,convection_fluid_ratio_d,tmp_d,dvel_d,dtmp_d,shock_sensor_d
  use halo_exchange_gpu, only: exchange_field_halo_gpu
  use shock_sensor_gpu, only: compute_shock_sensor_gpu
  use chemistry_flow_solver_gpu, only: configure_air5_vibrational_floor_gpu,air5_flow_status_d,reset_air5_status, &
    reset_air5_diffusion_failure,air5_diffusion_layered_d,air5_convection_consistent_d,air5_convection_symmetric_d, &
    air5_limited_points_d,air5_minimum_ratio_bits_d,air5_constraint_active_points_d,air5_constraint_minimum_ratio_bits_d, &
    air5_species_constraint_active_points_d,air5_species_constraint_minimum_ratio_bits_d,air5_diffusion_failure_location_d, &
    air5_zero_rhs_kernel,air5_full_state_convection_ratio_kernel,air5_limit_full_state_convection_kernel, &
    air5_reset_diffusion_ratio_kernel,air5_convective_rhs_x_kernel,air5_convective_rhs_y_kernel,air5_convective_rhs_z_kernel, &
    air5_selective_convective_rhs_x_kernel,air5_selective_convective_rhs_y_kernel,air5_selective_convective_rhs_z_kernel
#else
  use device_state
  use chemistry_device_properties
  use chemistry_device_transport
  use chemistry_device_step, only: air5_prepare_shock_mask,air5_assemble_raw_convection,air5_limit_convection_rhs
  use halo_transport
  use shock_sensor_cpp, only: shock_prepare_cpp,shock_expand_cpp,shock_mask_download_cpp
#endif
  implicit none
  integer,parameter :: h=5
  real(c_double),parameter :: threshold=1.0e-3_c_double
  integer :: ierr,rank,comm,sizes(3),coords(3),periodic(3),lower(3),upper(3),extent(3),origin(3),whole(3)
  integer :: i,j,k,d,u,p(3),mode,limiter,test,local,stage,ntypes(3),lo(3),hi(3),capturing
  integer :: nx,ny,nz,first_phase,last_phase,phase,species_budget,status,global_status,active(10),location(4)
  integer(c_int64_t) :: bits(10)
  real(c_double) :: floor(5),a,b,cdt,minimum(10)
  real(c_double),allocatable :: input(:,:,:,:),q(:,:,:,:),save(:,:,:,:),rhs(:,:,:,:),raw(:,:,:,:)
  real(c_double),allocatable :: rho(:,:,:),velocity(:,:,:,:),pressure(:,:,:),ys(:,:,:,:),metric(:,:,:,:,:),jac(:,:,:)
  real(c_double),allocatable :: ratios(:,:,:,:)
  real(c_double),allocatable :: temperature(:,:,:),gradient(:,:,:,:,:),sensor(:,:,:)
  integer(c_int8_t),allocatable :: mask(:,:,:)
  character(2048) :: name
  character(32) :: arg
  character(8) :: rank_text
#ifdef ASTR_OLD_CONVECTION
  type(dim3) :: grid,bx,by,bz,gx,gy,gz
  real(c_double),allocatable :: trace_rhs(:,:,:,:,:)
#else
  type(c_ptr) :: context
  type(device_view) :: field_view
  type(air5_convection_report) :: report
#endif
  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)
  call get_environment_variable('OMPI_COMM_WORLD_LOCAL_RANK',arg);read(arg,*)local
  local=mod(local,2)
  do d=1,3
    call get_command_argument(d+2,arg);read(arg,*)sizes(d)
  enddo
  call get_command_argument(6,arg);read(arg,*) limiter
  call get_command_argument(7,arg);read(arg,*) test
  call get_command_argument(8,arg);read(arg,*) mode
  call get_command_argument(1,name)
  open(newunit=u,file=trim(name),access='stream',form='unformatted',status='old')
  read(u) whole,periodic,floor
  allocate(input(0:whole(1),0:whole(2),0:whole(3),20))
  read(u)input;close(u)
  call MPI_Cart_create(MPI_COMM_WORLD,3,sizes,periodic==1,.false.,comm,ierr)
  call MPI_Cart_coords(comm,rank,3,coords,ierr)
  do d=1,3
    call MPI_Cart_shift(comm,d-1,1,lower(d),upper(d),ierr)
  enddo
  extent=whole/sizes;origin=coords*extent
  nx=extent(1);ny=extent(2);nz=extent(3)
  ntypes=3
  where(lower==MPI_PROC_NULL)ntypes=1
  where(upper==MPI_PROC_NULL)ntypes=2
  where(lower==MPI_PROC_NULL.and.upper==MPI_PROC_NULL)ntypes=4
  lo=merge(1,0,lower==MPI_PROC_NULL);hi=extent-merge(1,0,upper==MPI_PROC_NULL)
  allocate(q(-h:nx+h,-h:ny+h,-h:nz+h,11),jac(-h:nx+h,-h:ny+h,-h:nz+h))
  allocate(rho(-h:nx+h,-h:ny+h,-h:nz+h),pressure(-h:nx+h,-h:ny+h,-h:nz+h))
  allocate(velocity(-h:nx+h,-h:ny+h,-h:nz+h,3),ys(-h:nx+h,-h:ny+h,-h:nz+h,5), &
    metric(-h:nx+h,-h:ny+h,-h:nz+h,3,3))
  allocate(save(0:nx,0:ny,0:nz,11),rhs(0:nx,0:ny,0:nz,11),raw(0:nx,0:ny,0:nz,11), &
    ratios(-h:nx+h,-h:ny+h,-h:nz+h,2))
  metric=0.0_c_double;jac=1.0_c_double
  do k=-h,nz+h;do j=-h,ny+h;do i=-h,nx+h
    p=[i,j,k]+origin
    do d=1,3
      if(periodic(d)==1)then
        p(d)=modulo(p(d),whole(d))
      else
        p(d)=max(0,min(p(d),whole(d)))
      endif
      metric(i,j,k,d,d)=1.0_c_double
    enddo
    q(i,j,k,:)=input(p(1),p(2),p(3),1:11)
    rho(i,j,k)=q(i,j,k,1)
    velocity(i,j,k,:)=input(p(1),p(2),p(3),12:14)
    pressure(i,j,k)=input(p(1),p(2),p(3),15)
    ys(i,j,k,:)=input(p(1),p(2),p(3),16:20)
  enddo;enddo;enddo
  stage=merge(1,2,test==0);a=merge(0.0_c_double,0.75_c_double,stage==1);b=1.0_c_double-a
  cdt=5.e-5_c_double
  capturing=merge(1,0,test==1.or.test==3.or.test==4)
  if(test==4)then
    allocate(temperature,source=pressure/(rho*sum(ys*spread(spread(spread(air5_ru/air5_molar_mass, &
      1,size(rho,1)),2,size(rho,2)),3,size(rho,3)),dim=4)))
    allocate(gradient(0:nx,0:ny,0:nz,3,3),sensor(-h:nx+h,-h:ny+h,-h:nz+h),mask(0:nx,0:ny,0:nz))
  endif
  save=0.998_c_double*q(0:nx,0:ny,0:nz,:)
  if(test==3.and.rank==0) save(lo(1),lo(2),lo(3),1)=-2.0_c_double
  ratios=1.0_c_double;active=0;minimum=huge(1.0_c_double);location=0
#ifdef ASTR_OLD_CONVECTION
  call check(cudaSetDevice(local))
  im=nx;jm=ny;km=nz
  isize=sizes(1);jsize=sizes(2);ksize=sizes(3);mpirank=rank
  mpileft=lower(1);mpiright=upper(1);mpidown=lower(2);mpiup=upper(2);mpiback=lower(3);mpifront=upper(3)
  lihomo=periodic(1)==1;ljhomo=periodic(2)==1;lkhomo=periodic(3)==1
  allocate(q_d(-h:nx+h,-h:ny+h,-h:nz+h,11),jacob_d(-h:nx+h,-h:ny+h,-h:nz+h))
  allocate(rho_d(-h:nx+h,-h:ny+h,-h:nz+h),prs_d(-h:nx+h,-h:ny+h,-h:nz+h), &
    vel_d(-h:nx+h,-h:ny+h,-h:nz+h,3),spc_d(-h:nx+h,-h:ny+h,-h:nz+h,5),dxi_d(-h:nx+h,-h:ny+h,-h:nz+h,3,3))
  allocate(shock_mask_d(0:nx,0:ny,0:nz),qsave_d(0:nx,0:ny,0:nz,11),qrhs_d(0:nx,0:ny,0:nz,11))
  allocate(diffusion_ratio_d(-h:nx+h,-h:ny+h,-h:nz+h,1),convection_fluid_ratio_d(-h:nx+h,-h:ny+h,-h:nz+h,1))
  q_d=q;qsave_d=save;jacob_d=jac;dxi_d=metric;rho_d=rho;prs_d=pressure;vel_d=velocity;spc_d=ys
  shock_mask_d=int(capturing,1)
  if(test==4)then
    npdci=ntypes(1);npdcj=ntypes(2);npdck=ntypes(3)
    lcomb=.true.;flowtype='air5hbl';shkcrt=threshold
    allocate(tmp_d(-h:nx+h,-h:ny+h,-h:nz+h),dvel_d(0:nx,0:ny,0:nz,3,3),dtmp_d(0:nx,0:ny,0:nz,3))
    allocate(shock_sensor_d(-h:nx+h,-h:ny+h,-h:nz+h,1))
    tmp_d=temperature;shock_sensor_d=0.0_c_double
    call compute_shock_sensor_gpu(force_active=.true.)
    gradient=dvel_d;sensor=shock_sensor_d(:,:,:,1);mask=shock_mask_d
  endif
  air5_diffusion_layered_d=0
  air5_convection_consistent_d=merge(1,0,limiter==2)
  air5_convection_symmetric_d=0
  call configure_air5_vibrational_floor_gpu()
  bx=dim3(512,1,1);by=dim3(32,16,1);bz=dim3(64,1,8)
  gx=dim3(((nx+1)*(ny+1)*(nz+1)+511)/512,1,1)
  gy=dim3(((nx+1)*(ny+1)+511)/512,1,nz+1);gz=dim3(((nx+1)*(nz+1)+511)/512,ny+1,1)
  grid=dim3(((nx+1)*(ny+1)*(nz+1)+255)/256,1,1)
  call reset_air5_status()
  call air5_zero_rhs_kernel<<<grid,256>>>(nx,ny,nz)
  call check(cudaGetLastError())
  if(capturing/=0)then
    if(command_argument_count()==9)allocate(trace_rhs(0:nx,0:ny,0:nz,11,3))
    call air5_selective_convective_rhs_x_kernel<<<gx,bx>>>(nx,ny,nz,ntypes(1),lo(1),hi(1),lo(2),hi(2),lo(3),hi(3))
    call check(cudaGetLastError())
    if(allocated(trace_rhs))trace_rhs(:,:,:,:,1)=qrhs_d
    call air5_selective_convective_rhs_y_kernel<<<gy,by>>>(nx,ny,nz,ntypes(2),lo(1),hi(1),lo(2),hi(2),lo(3),hi(3))
    call check(cudaGetLastError())
    if(allocated(trace_rhs))trace_rhs(:,:,:,:,2)=qrhs_d
    call air5_selective_convective_rhs_z_kernel<<<gz,bz>>>(nx,ny,nz,ntypes(3),lo(1),hi(1),lo(2),hi(2),lo(3),hi(3))
  else
    call air5_convective_rhs_x_kernel<<<gx,bx>>>(nx,ny,nz,ntypes(1),lo(1),hi(1),lo(2),hi(2),lo(3),hi(3))
    call check(cudaGetLastError())
    call air5_convective_rhs_y_kernel<<<gy,by>>>(nx,ny,nz,ntypes(2),lo(1),hi(1),lo(2),hi(2),lo(3),hi(3))
    call check(cudaGetLastError())
    call air5_convective_rhs_z_kernel<<<gz,bz>>>(nx,ny,nz,ntypes(3),lo(1),hi(1),lo(2),hi(2),lo(3),hi(3))
  endif
  call check(cudaGetLastError());call check(cudaDeviceSynchronize())
  status=air5_flow_status_d
  call check(status)
  raw=qrhs_d
  if(allocated(trace_rhs))then
    trace_rhs(:,:,:,:,3)=raw
    call get_command_argument(2,name)
    write(rank_text,'(I0)')rank
    open(newunit=u,file=trim(name)//'.axes.rank'//trim(rank_text)//'.bin',access='stream',form='unformatted',status='new')
    write(u)extent,trace_rhs
    close(u)
  endif
  call air5_reset_diffusion_ratio_kernel<<<dim3((size(jac)+255)/256,1,1),256>>>(nx,ny,nz,h)
  call check(cudaGetLastError())
  first_phase=merge(1,0,limiter==2);last_phase=merge(2,0,limiter==2)
  do phase=first_phase,last_phase
    species_budget=merge(1,0,limiter==1.or.phase==2)
    call reset_air5_status();call reset_air5_diffusion_failure()
    air5_limited_points_d=0;air5_minimum_ratio_bits_d=transfer(huge(1.0_c_double),bits(1))
    air5_constraint_active_points_d=0;air5_constraint_minimum_ratio_bits_d=transfer(huge(1.0_c_double),bits(1))
    air5_species_constraint_active_points_d=0;air5_species_constraint_minimum_ratio_bits_d=transfer(huge(1.0_c_double),bits(1))
    call air5_full_state_convection_ratio_kernel<<<grid,256>>>(cdt,a,b,1.0_c_double,stage,1,nx,ny,nz, &
      ntypes(1),ntypes(2),ntypes(3),lo(1),hi(1),lo(2),hi(2),lo(3),hi(3),capturing,species_budget,phase)
    call check(cudaGetLastError());call check(cudaDeviceSynchronize())
    status=air5_flow_status_d
    call MPI_Allreduce(status,global_status,1,MPI_INTEGER,MPI_MAX,comm,ierr)
    if(global_status/=0)then
      if(status/=0)location=air5_diffusion_failure_location_d
      exit
    endif
    if(phase==1)then
      call exchange_field_halo_gpu(convection_fluid_ratio_d,1)
    else
      active(1)=air5_limited_points_d;bits(1)=air5_minimum_ratio_bits_d
      active(2:5)=air5_constraint_active_points_d;bits(2:5)=air5_constraint_minimum_ratio_bits_d
      active(6:10)=air5_species_constraint_active_points_d;bits(6:10)=air5_species_constraint_minimum_ratio_bits_d
      minimum=transfer(bits,minimum)
      call exchange_field_halo_gpu(diffusion_ratio_d,1)
      call air5_limit_full_state_convection_kernel<<<grid,256>>>(nx,ny,nz,ntypes(1),ntypes(2),ntypes(3), &
        lo(1),hi(1),lo(2),hi(2),lo(3),hi(3),capturing,phase)
      call check(cudaGetLastError());call check(cudaDeviceSynchronize())
    endif
  enddo
  rhs=qrhs_d;ratios(:,:,:,1)=diffusion_ratio_d(:,:,:,1)
  if(limiter==2)ratios(:,:,:,2)=convection_fluid_ratio_d(:,:,:,1)
#else
  call check(device_bind(local))
  call check(device_state_create(local,nx,ny,nz,h,1_c_int,context))
  call check(air5_device_prepare(context));call check(air5_geometry_prepare(context));call check(air5_rk_prepare(context))
  call check(device_transport_prepare(context,int(comm,c_int),lower,upper,sizes,11_c_int,mode,0_c_int,periodic))
  call check(air5_convection_budget_prepare(context,merge(1_c_int,0_c_int,limiter==2)))
  call check(device_state_upload(context,device_q,q,int(size(q),c_size_t)*8))
  call check(device_state_upload(context,device_save,save,int(size(save),c_size_t)*8))
  call check(device_state_upload(context,device_jacobian,jac,int(size(jac),c_size_t)*8))
  call check(device_state_upload(context,device_metric,metric,int(size(metric),c_size_t)*8))
  call check(device_state_upload(context,device_rho,rho,int(size(rho),c_size_t)*8))
  call check(device_state_upload(context,device_velocity,velocity,int(size(velocity),c_size_t)*8))
  call check(device_state_upload(context,device_pressure,pressure,int(size(pressure),c_size_t)*8))
  call check(device_state_upload(context,device_air5_species,ys,int(size(ys),c_size_t)*8))
  if(capturing/=0)then
    call check(air5_velocity_gradients_prepare(context))
    if(device_state_view(context,device_stress,field_view)/=device_not_allocated)call check(device_invalid_argument)
    if(device_state_view(context,device_air5_dspecies,field_view)/=device_not_allocated)call check(device_invalid_argument)
    call check(shock_prepare_cpp(context))
    if(test==4)then
      call check(device_state_upload(context,device_temperature,temperature,int(size(temperature),c_size_t)*8))
      call air5_prepare_shock_mask(context,ntypes,threshold,int(comm,c_int))
      call check(device_state_download(context,device_dvelocity,gradient,int(size(gradient),c_size_t)*8))
      call check(device_state_download(context,device_shock_sensor,sensor,int(size(sensor),c_size_t)*8))
      call check(shock_mask_download_cpp(context,mask,int(size(mask),c_size_t)))
    else
      jac=1.0_c_double
      call check(device_state_upload(context,device_shock_sensor,jac,int(size(jac),c_size_t)*8))
      call check(shock_expand_cpp(context,merge(1,0,lower==MPI_PROC_NULL),merge(1,0,upper==MPI_PROC_NULL),0.5_c_double))
    endif
  endif
  call check(air5_assemble_raw_convection(context,ntypes,lo,hi,capturing/=0,.false.,int(comm,c_int)))
  call check(device_state_download(context,device_rhs,raw,int(size(raw),c_size_t)*8))
  global_status=air5_limit_convection_rhs(context,ntypes,lo,hi,floor,a,b,cdt,stage==1,capturing/=0,limiter, &
    .true.,int(comm,c_int),active,minimum,report)
  call check(air5_device_check(context,status))
  location=report%location
  if(global_status==0)then
    active=report%active;minimum=transfer(report%minimum_bits,minimum)
  endif
  call check(device_state_download(context,device_rhs,rhs,int(size(rhs),c_size_t)*8))
  call check(device_state_download(context,device_air5_diffusion_ratio,ratios(:,:,:,1),int(size(jac),c_size_t)*8))
  if(limiter==2)call check(device_state_download(context,device_air5_convection_fluid_ratio,ratios(:,:,:,2),int(size(jac),c_size_t)*8))
  call check(device_state_destroy(context))
#endif
  call get_command_argument(2,name)
  write(rank_text,'(I0)')rank
  open(newunit=u,file=trim(name)//'.rank'//trim(rank_text)//'.bin',access='stream',form='unformatted',status='new')
  write(u)extent,status,global_status,location,active,minimum,ratios,raw,rhs
  close(u)
  if(test==4)then
    open(newunit=u,file=trim(name)//'.sensor.rank'//trim(rank_text)//'.bin',access='stream',form='unformatted',status='new')
    write(u)extent,gradient,sensor(0:nx,0:ny,0:nz),mask
    close(u)
  endif
  call MPI_Comm_free(comm,ierr)
  call MPI_Finalize(ierr)
contains
  subroutine check(code)
    integer,intent(in) :: code
    integer :: ignored
    if(code==0)return
    write(*,'(A,I0)')'AIR5 convection probe status ',code
    call MPI_Abort(MPI_COMM_WORLD,abs(code),ignored)
  end subroutine
end program
