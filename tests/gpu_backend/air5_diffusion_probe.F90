program air5_diffusion_probe
  use iso_c_binding, only: c_ptr,c_size_t,c_double,c_int,c_int64_t
  use mpi
#ifdef ASTR_OLD_DIFFUSION
  use cudafor
  use commvar, only: im,jm,km,hm,lihomo,ljhomo,lkhomo
  use parallel, only: mpirank,isize,jsize,ksize,mpileft,mpiright,mpidown,mpiup,mpiback,mpifront
  use commarray_gpu, only: q_d,qsave_d,qrhs_d,jacob_d,dxi_d,sigma_d,qflux_d,species_flux_d, &
    ev_flux_d,species_energy_flux_d,diffusion_ratio_d,diffusion_energy_ratio_d
  use halo_exchange_gpu, only: exchange_field_halo_gpu
  use chemistry_flow_solver_gpu, only: configure_air5_vibrational_floor_gpu,air5_flow_status_d, &
    reset_air5_status,reset_air5_diffusion_failure,air5_diffusion_layered_d,air5_convection_consistent_d, &
    air5_limited_points_d,air5_minimum_ratio_bits_d,air5_diffusion_ratio_kernel,air5_reset_diffusion_ratio_kernel, &
    air5_diffusion_energy_ratio_kernel,air5_diffusion_rhs_x_kernel,air5_diffusion_rhs_y_kernel,air5_diffusion_rhs_z_kernel
#else
  use device_state
  use chemistry_device_properties
  use chemistry_device_transport
  use chemistry_device_step, only: air5_add_viscous_rhs,air5_limit_viscous_rhs
  use halo_transport
#endif
  implicit none
  integer,parameter :: h=5
  integer :: ierr,rank,comm,sz(3),coords(3),periodic(3),lower(3),upper(3),extent(3),origin(3)
  integer :: i,j,k,d,s,u,layered,test,mode,local,stage,regs,part,ntypes(3),lo(3,2),hi(3,2),status,global_status,count
  integer :: nx,ny,nz,global_limited,replay_header(6),trace_axis
  integer(c_int64_t) :: bits
  real(c_double) :: input(11),floor(5),a,b,cdt,r,fac,amp(5),minimum
  real(c_double),allocatable :: q(:,:,:,:),save(:,:,:,:),rhs(:,:,:,:),metric(:,:,:,:,:),jac(:,:,:)
  real(c_double),allocatable :: stress(:,:,:,:),heat(:,:,:,:),js(:,:,:,:,:),evflux(:,:,:,:),carryflux(:,:,:,:),ratio(:,:,:,:)
  character(2048) :: name
  character(32) :: arg
  character(8) :: rank_text
#ifdef ASTR_OLD_DIFFUSION
  type(dim3) :: grid,bx,by,bz,gy,gz,reset_grid
#else
  type(c_ptr) :: context
  type(air5_diffusion_report) :: report
#endif
  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)
  call get_environment_variable('OMPI_COMM_WORLD_LOCAL_RANK',arg);read(arg,*)local
  local=mod(local,2)
  call get_command_argument(1,name)
  open(newunit=u,file=trim(name),access='stream',form='unformatted',status='old')
  read(u) input,floor
  close(u)
  do d=1,3
    call get_command_argument(d+2,arg);read(arg,*)sz(d)
  enddo
  call get_command_argument(6,arg);read(arg,*) layered
  call get_command_argument(7,arg);read(arg,*) test
  call get_command_argument(8,arg);read(arg,*) mode
  trace_axis=0
  if(command_argument_count()==10)then
    call get_command_argument(10,arg);read(arg,*) trace_axis
  endif
  periodic=[1,1,1]
  if(test>=1.and.test<=3)periodic=[0,0,1]
  call MPI_Cart_create(MPI_COMM_WORLD,3,sz,periodic==1,.false.,comm,ierr)
  call MPI_Cart_coords(comm,rank,3,coords,ierr)
  do d=1,3
    call MPI_Cart_shift(comm,d-1,1,lower(d),upper(d),ierr)
  enddo
  extent=[16,16,16]/sz;origin=coords*extent
  nx=extent(1);ny=extent(2);nz=extent(3)
  ntypes=3
  where(lower==MPI_PROC_NULL)ntypes=1
  where(upper==MPI_PROC_NULL)ntypes=2
  where(lower==MPI_PROC_NULL.and.upper==MPI_PROC_NULL)ntypes=4
  lo(:,1)=merge(1,0,lower==MPI_PROC_NULL);hi(:,1)=extent-merge(1,0,upper==MPI_PROC_NULL)
  regs=1
  if(test==2.and.upper(2)==MPI_PROC_NULL)then
    regs=2;lo(:,2)=[lo(1,1),ny,0];hi(:,2)=[nx,ny,nz]
  endif
  allocate(q(-h:nx+h,-h:ny+h,-h:nz+h,11),jac(-h:nx+h,-h:ny+h,-h:nz+h))
  allocate(metric(-h:nx+h,-h:ny+h,-h:nz+h,3,3),stress(-h:nx+h,-h:ny+h,-h:nz+h,6))
  allocate(heat(-h:nx+h,-h:ny+h,-h:nz+h,3),evflux(-h:nx+h,-h:ny+h,-h:nz+h,3))
  allocate(js(-h:nx+h,-h:ny+h,-h:nz+h,5,3),carryflux(-h:nx+h,-h:ny+h,-h:nz+h,6))
  allocate(save(0:nx,0:ny,0:nz,11),rhs(0:nx,0:ny,0:nz,11),ratio(-h:nx+h,-h:ny+h,-h:nz+h,2))
  amp=[-0.6_c_double,0.5_c_double,0.0625_c_double,0.03125_c_double,0.00625_c_double]
  do k=-h,nz+h;do j=-h,ny+h;do i=-h,nx+h
    fac=1.0_c_double+real(sum([i,j,k]+origin),c_double)/1024.0_c_double
    r=sin(real(19*(i+origin(1))+7*(j+origin(2))-3*(k+origin(3)),c_double)/128.0_c_double)
    if(test==4)fac=1.0_c_double
    q(i,j,k,:)=input*fac
    jac(i,j,k)=1.0_c_double+fac/32.0_c_double
    metric(i,j,k,:,:)=0.0_c_double
    do d=1,3
      metric(i,j,k,d,d)=1.0_c_double/fac
      heat(i,j,k,d)=real(d,c_double)*r*100.0_c_double
      evflux(i,j,k,d)=-r*real(d,c_double)*10.0_c_double
      js(i,j,k,:,d)=real(d,c_double)*amp*r*merge(1.0_c_double,1.0e5_c_double,test==0)
      carryflux(i,j,k,d)=-r*500.0_c_double
      carryflux(i,j,k,d+3)=-r*25.0_c_double
    enddo
    stress(i,j,k,:)=real([1,2,3,4,5,6],c_double)*r
    if(test==2)then
      heat(i,j,k,:)=1.0e6_c_double*heat(i,j,k,:)
      evflux(i,j,k,:)=1.0e6_c_double*evflux(i,j,k,:)
    endif
    if(i<0.or.i>nx.or.j<0.or.j>ny.or.k<0.or.k>nz)cycle
    save(i,j,k,:)=q(i,j,k,:)*jac(i,j,k)*0.98_c_double
    rhs(i,j,k,:)=input*0.02_c_double
  enddo;enddo;enddo
  if(test==3.and.rank==0)rhs(lo(1,1),lo(2,1),lo(3,1),1)=-1.e9_c_double
  stage=merge(1,2,test==0.or.test==4)
  a=merge(0.0_c_double,0.75_c_double,stage==1);b=1.0_c_double-a
  cdt=1.e-3_c_double
  if(trace_axis==1)then
    call get_command_argument(2,name)
    write(rank_text,'(I0)')rank
    open(newunit=u,file=trim(name)//'.input.rank'//trim(rank_text)//'.bin',access='stream',form='unformatted',status='new')
    write(u) q,save,rhs,jac,metric,stress,heat,js,evflux,carryflux
    close(u)
  endif
#ifdef ASTR_OLD_DIFFUSION
  call check(cudaSetDevice(local))
  im=nx;jm=ny;km=nz
  isize=sz(1);jsize=sz(2);ksize=sz(3);mpirank=rank
  mpileft=lower(1);mpiright=upper(1);mpidown=lower(2);mpiup=upper(2);mpiback=lower(3);mpifront=upper(3)
  lihomo=periodic(1)==1;ljhomo=periodic(2)==1;lkhomo=periodic(3)==1
  allocate(q_d(-h:nx+h,-h:ny+h,-h:nz+h,11),jacob_d(-h:nx+h,-h:ny+h,-h:nz+h))
  allocate(dxi_d(-h:nx+h,-h:ny+h,-h:nz+h,3,3),sigma_d(-h:nx+h,-h:ny+h,-h:nz+h,6))
  allocate(qflux_d(-h:nx+h,-h:ny+h,-h:nz+h,3),ev_flux_d(-h:nx+h,-h:ny+h,-h:nz+h,3))
  allocate(species_flux_d(-h:nx+h,-h:ny+h,-h:nz+h,5,3),species_energy_flux_d(-h:nx+h,-h:ny+h,-h:nz+h,6))
  allocate(qsave_d(0:nx,0:ny,0:nz,11),qrhs_d(0:nx,0:ny,0:nz,11))
  allocate(diffusion_ratio_d(-h:nx+h,-h:ny+h,-h:nz+h,1),diffusion_energy_ratio_d(-h:nx+h,-h:ny+h,-h:nz+h,1))
  q_d=q;qsave_d=save;qrhs_d=rhs;jacob_d=jac;dxi_d=metric;sigma_d=stress;qflux_d=heat
  ev_flux_d=evflux;species_flux_d=js;species_energy_flux_d=carryflux
  call configure_air5_vibrational_floor_gpu()
  air5_diffusion_layered_d=layered
  air5_convection_consistent_d=0
  air5_limited_points_d=0;air5_minimum_ratio_bits_d=transfer(huge(1.0_c_double),bits)
  call reset_air5_status()
  call reset_air5_diffusion_failure()
  grid=dim3(((nx+1)*(ny+1)*(nz+1)+255)/256,1,1)
  reset_grid=dim3((size(jac)+255)/256,1,1)
  call air5_reset_diffusion_ratio_kernel<<<reset_grid,256>>>(nx,ny,nz,h)
  call check(cudaGetLastError())
  do part=1,regs
    call air5_diffusion_ratio_kernel<<<grid,256>>>(cdt,a,b,1.0_c_double,stage,1,nx,ny,nz,ntypes(1),ntypes(2),ntypes(3), &
      lo(1,part),hi(1,part),lo(2,part),hi(2,part),lo(3,part),hi(3,part))
    call check(cudaGetLastError())
  enddo
  call check(cudaDeviceSynchronize())
  status=air5_flow_status_d
#else
  call check(device_bind(local))
  call check(device_state_create(local,nx,ny,nz,h,1_c_int,context))
  call check(air5_device_prepare(context));call check(air5_geometry_prepare(context));call check(air5_rk_prepare(context))
  call check(air5_transport_prepare(context,layered))
  call check(device_transport_prepare(context,int(comm,c_int),lower,upper,sz,11_c_int,mode,0_c_int,periodic))
  call check(air5_diffusion_prepare(context))
  call check(device_state_upload(context,device_q,q,int(size(q),c_size_t)*8))
  call check(device_state_upload(context,device_save,save,int(size(save),c_size_t)*8))
  call check(device_state_upload(context,device_rhs,rhs,int(size(rhs),c_size_t)*8))
  call check(device_state_upload(context,device_jacobian,jac,int(size(jac),c_size_t)*8))
  call check(device_state_upload(context,device_metric,metric,int(size(metric),c_size_t)*8))
  call check(device_state_upload(context,device_stress,stress,int(size(stress),c_size_t)*8))
  call check(device_state_upload(context,device_heat_flux,heat,int(size(heat),c_size_t)*8))
  call check(device_state_upload(context,device_air5_ev_flux,evflux,int(size(evflux),c_size_t)*8))
  call check(device_state_upload(context,device_air5_species_flux,js,int(size(js),c_size_t)*8))
  if(layered/=0)call check(device_state_upload(context,device_air5_species_energy_flux,carryflux,int(size(carryflux),c_size_t)*8))
  if(test==4)then
    call check(air5_device_primitives(context,[-h,-h,-h],extent+h,0_c_int))
    call check(air5_device_check(context,status));call check(status)
    call check(air5_flux_transport_prepare(context,air5_flux_packed_overlap))
    global_status=air5_add_viscous_rhs(context,ntypes,regs,lo(:,1:regs),hi(:,1:regs),floor,a,b,cdt, &
      .true.,.false.,layered/=0,.true.,int(comm,c_int),global_limited,minimum,report)
    call check(global_status)
    if(global_limited/=0)call check(95)
  else
    global_status=air5_limit_viscous_rhs(context,ntypes,regs,lo(:,1:regs),hi(:,1:regs),floor,a,b,cdt, &
      stage==1,layered/=0,.true.,int(comm,c_int),global_limited,minimum,report)
  endif
  call check(air5_device_check(context,status))
  if(command_argument_count()>=9)then
    ! Diagnostic: take the old budgets as input, without changing the
    ! production limiter, to isolate the divergence on this same state.
    call get_command_argument(9,name)
    write(rank_text,'(I0)')rank
    open(newunit=u,file=trim(name)//'.rank'//trim(rank_text)//'.bin',access='stream',form='unformatted',status='old')
    read(u) replay_header,bits,ratio
    close(u)
    if(any(replay_header(1:3)/=extent).or.replay_header(5)/=0.or.global_status/=0)call check(94)
    call check(device_state_upload(context,device_rhs,rhs,int(size(rhs),c_size_t)*8))
    call check(device_state_upload(context,device_air5_diffusion_ratio,ratio(:,:,:,1),int(size(jac),c_size_t)*8))
    if(layered/=0) &
      call check(device_state_upload(context,device_air5_diffusion_energy_ratio,ratio(:,:,:,2),int(size(jac),c_size_t)*8))
    do part=1,regs;do d=1,3
      if(trace_axis/=0.and.trace_axis/=d)cycle
      call check(air5_diffusion_rhs(context,ntypes,lo(:,part),hi(:,part),d))
    enddo;enddo
  endif
#endif
#ifdef ASTR_OLD_DIFFUSION
  call MPI_Allreduce(status,global_status,1,MPI_INTEGER,MPI_MAX,comm,ierr)
  if(global_status==0)then
    call exchange_field_halo_gpu(diffusion_ratio_d,1)
    if(layered/=0)then
      do part=1,regs
        call air5_diffusion_energy_ratio_kernel<<<grid,256>>>(cdt,a,b,1.0_c_double,stage,nx,ny,nz,ntypes(1),ntypes(2),ntypes(3), &
          lo(1,part),hi(1,part),lo(2,part),hi(2,part),lo(3,part),hi(3,part))
        call check(cudaGetLastError())
      enddo
      call check(cudaDeviceSynchronize())
      call exchange_field_halo_gpu(diffusion_energy_ratio_d,1)
    endif
    bx=dim3(512,1,1);by=dim3(32,16,1);bz=dim3(64,1,8)
    gy=dim3(((nx+1)*(ny+1)+511)/512,1,nz+1);gz=dim3(((nx+1)*(nz+1)+511)/512,ny+1,1)
    grid=dim3(((nx+1)*(ny+1)*(nz+1)+511)/512,1,1)
    do part=1,regs
      if(trace_axis==0.or.trace_axis==1)then
        call air5_diffusion_rhs_x_kernel<<<grid,bx>>>(nx,ny,nz,ntypes(1),lo(1,part),hi(1,part), &
          lo(2,part),hi(2,part),lo(3,part),hi(3,part))
        call check(cudaGetLastError())
      endif
      if(trace_axis==0.or.trace_axis==2)then
        call air5_diffusion_rhs_y_kernel<<<gy,by>>>(nx,ny,nz,ntypes(2),lo(1,part),hi(1,part), &
          lo(2,part),hi(2,part),lo(3,part),hi(3,part))
        call check(cudaGetLastError())
      endif
      if(trace_axis==0.or.trace_axis==3)then
        call air5_diffusion_rhs_z_kernel<<<gz,bz>>>(nx,ny,nz,ntypes(3),lo(1,part),hi(1,part), &
          lo(2,part),hi(2,part),lo(3,part),hi(3,part))
        call check(cudaGetLastError())
      endif
    enddo
    call check(cudaDeviceSynchronize())
  endif
#endif
  ratio=1.0_c_double
#ifdef ASTR_OLD_DIFFUSION
  rhs=qrhs_d;ratio(:,:,:,1)=diffusion_ratio_d(:,:,:,1)
  if(layered/=0)ratio(:,:,:,2)=diffusion_energy_ratio_d(:,:,:,1)
  count=air5_limited_points_d;bits=air5_minimum_ratio_bits_d
#else
  call check(device_state_download(context,device_rhs,rhs,int(size(rhs),c_size_t)*8))
  call check(device_state_download(context,device_air5_diffusion_ratio,ratio(:,:,:,1),int(size(jac),c_size_t)*8))
  if(layered/=0) &
    call check(device_state_download(context,device_air5_diffusion_energy_ratio,ratio(:,:,:,2),int(size(jac),c_size_t)*8))
  count=report%limited;bits=report%minimum_bits
  if(status/=0.and.report%claimed/=1)call check(98)
  if(test==4)then
    do d=1,11
      if(any(rhs(:,:,:,d)/=input(d)*0.02_c_double))call check(96)
    enddo
    if(any(ratio/=1.0_c_double))call check(97)
  endif
  call check(device_state_destroy(context))
#endif
  write(rank_text,'(I0)')rank
  call get_command_argument(2,name)
  open(newunit=u,file=trim(name)//'.rank'//trim(rank_text)//'.bin',access='stream',form='unformatted',status='new')
  write(u) extent,status,global_status,count,bits,ratio,rhs
  close(u)
  call MPI_Comm_free(comm,ierr)
  call MPI_Finalize(ierr)
contains
  subroutine check(code)
    integer,intent(in) :: code
    integer :: ignored
    if(code==0)return
    write(*,'(A,I0)')'AIR5 diffusion probe status ',code
    call MPI_Abort(MPI_COMM_WORLD,abs(code),ignored)
  end subroutine
end program
