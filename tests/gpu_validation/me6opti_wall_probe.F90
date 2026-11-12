! Actual nodal/ME6 production operators on a wall-compatible manufactured field.
program me6opti_wall_probe
  use astr_precision, only: real_kind
  use astr_precision_mpi, only: mpi_real_type
  use commvar
  use commarray, only: rho,vel,tmp,jacob,dxi,qrhs,dvel,dtmp,dspc,vor
  use parallel
  use numerical_forms, only: convection_form,viscous_form
  use bc, only: bctype,twall
  use comsolver, only: solvrinit,gradcal
  use solver, only: diffrsdcal6
  use me6opti_runtime, only: me6_diffusion_cpu,me6_nodal_boundary_point
  use mpi
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_value,ieee_quiet_nan
#ifdef TEST_CUDA
  use cudafor
  use commarray_gpu, only: vel_d,tmp_d,jacob_d,dxi_d,qrhs_d,sigma_d,qflux_d
  use me6opti_runtime_gpu, only: me6_diffusion_gpu
  use halo_exchange_gpu, only: init_halo_transport_gpu
#endif
  implicit none
  real(real_kind),allocatable :: nodal(:,:,:,:),hybrid(:,:,:,:)
  real(real_kind) :: pi,spacing(3),xx,yy,zz,f,u,v,ux,uy,uz,vx,vy,vz,lapu,lapv
  real(real_kind) :: temperature,ty,tyy,mu,muy,kappa,exact(4),errors(6),global_errors(6),weight
  real(real_kind) :: budgets(8),global_budgets(8)
  real(real_kind) :: norm_packet(9),global_norms(9),max_error(4),global_max_error(4),point_error(4)
  integer :: n,axis,parts(3),coords(3),dims(3),first(3),i,j,k,c,dist,category,ierr,rank_count
#ifdef TEST_CUDA
  integer :: devices,cuda_status
#endif
  character(32) :: argument
  call mpi_init(ierr)
  call mpi_comm_rank(MPI_COMM_WORLD,mpirank,ierr)
  call mpi_comm_size(MPI_COMM_WORLD,rank_count,ierr)
  mpisize=rank_count;lio=mpirank==0
  call get_command_argument(1,argument);read(argument,*) n
  call get_command_argument(2,argument);read(argument,*) axis
  parts=1;parts(axis)=mpisize;coords=0;coords(axis)=mpirank
  dims=n/parts;first=coords*dims
  im=dims(1);jm=dims(2);km=dims(3);ia=n;ja=n;ka=n
  isize=parts(1);jsize=parts(2);ksize=parts(3);irk=coords(1);jrk=coords(2);krk=coords(3)
  irkm=isize-1;jrkm=jsize-1;krkm=ksize-1;ig0=first(1);jg0=first(2);kg0=first(3)
  mpileft=MPI_PROC_NULL;mpiright=MPI_PROC_NULL;mpidown=MPI_PROC_NULL;mpiup=MPI_PROC_NULL
  mpiback=MPI_PROC_NULL;mpifront=MPI_PROC_NULL;mpitag=100
  if(isize>1) then
    mpileft=modulo(mpirank-1,mpisize);mpiright=modulo(mpirank+1,mpisize)
  endif
  if(jsize>1) then
    if(jrk>0) mpidown=mpirank-1
    if(jrk<jrkm) mpiup=mpirank+1
  endif
  if(ksize>1) then
    mpiback=modulo(mpirank-1,mpisize);mpifront=modulo(mpirank+1,mpisize)
  endif
  npdci=3;npdck=3;npdcj=3;is=0;ie=im;js=0;je=jm;ks=0;ke=km
  if(jrk==0) then
    npdcj=1;js=1
  endif
  if(jrk==jrkm) then
    npdcj=2;je=jm-1
  endif
  if(jsize==1) npdcj=4
  lihomo=.true.;ljhomo=.false.;lkhomo=.true.;lfftk=.false.;ndims=3
  numq=5;num_species=0;num_modequ=0;lcomb=.false.;nondimen=.true.;lfilter=.false.
  ltimrpt=.false.;lreport=.false.;turbmode='none';flowtype='channel'
  conschm='643e';difschm='643e';convection_form='keep';viscous_form='me6_opti'
  bctype=[1,1,41,41,1,1];twall=1.e0_real_kind
  gamma=1.4e0_real_kind;mach=0.3e0_real_kind;reynolds=3000.e0_real_kind;prandtl=0.72e0_real_kind
  const2=gamma*mach**2;const5=(gamma-1.e0_real_kind)*mach**2
  tempconst=110.4e0_real_kind/273.15e0_real_kind;tempconst1=1.e0_real_kind+tempconst
  pi=acos(-1.e0_real_kind);spacing=[2*pi,pi,2*pi]/real(n,real_kind)
  allocate(rho(-hm:im+hm,-hm:jm+hm,-hm:km+hm),vel(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3), &
    tmp(-hm:im+hm,-hm:jm+hm,-hm:km+hm),jacob(-hm:im+hm,-hm:jm+hm,-hm:km+hm), &
    dxi(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3,3),qrhs(0:im,0:jm,0:km,5), &
    dvel(0:im,0:jm,0:km,3,3),dtmp(0:im,0:jm,0:km,3),dspc(0:im,0:jm,0:km,0,3),vor(0:im,0:jm,0:km,3), &
    nodal(0:im,0:jm,0:km,5),hybrid(0:im,0:jm,0:km,5))
  rho=1.e0_real_kind;tmp=ieee_value(0.e0_real_kind,ieee_quiet_nan);vel=tmp(0,0,0)
  jacob=product(spacing);dxi=0.e0_real_kind
  do c=1,3
    dxi(:,:,:,c,c)=1.e0_real_kind/spacing(c)
  enddo
  do k=-hm,km+hm
  do j=-hm,jm+hm
  do i=-hm,im+hm
    if(j+jg0<0.or.j+jg0>ja) cycle
    xx=(i+ig0)*spacing(1);yy=(j+jg0)*spacing(2);zz=(k+kg0)*spacing(3)
    f=1.e0_real_kind+0.1e0_real_kind*cos(zz)
    vel(i,j,k,:)=[0.1e0_real_kind*sin(xx)*sin(2*yy)*f,-0.1e0_real_kind*cos(xx)*sin(yy)**2*f,0.e0_real_kind]
    tmp(i,j,k)=1.e0_real_kind+0.02e0_real_kind*sin(yy)**2
    if(j+jg0==0.or.j+jg0==ja) vel(i,j,k,:)=0.e0_real_kind
  enddo
  enddo
  enddo
  call solvrinit()
  call gradcal()
  qrhs=0.e0_real_kind
  call diffrsdcal6()
  nodal=qrhs
  qrhs=0.e0_real_kind
  call diffrsdcal6(wall_band_only=.true.)
  call me6_diffusion_cpu()
  hybrid=qrhs;errors=0.e0_real_kind;budgets=0.e0_real_kind;norm_packet=0.e0_real_kind;max_error=0.e0_real_kind
  if(.not.all(ieee_is_finite(hybrid))) error stop 'ME6 wall read physical-exterior NaN'
  if(any(hybrid(:,:,:,1)/=0)) error stop 'ME6 wall changed density RHS'
  do k=0,km
  do j=0,jm
  do i=0,im
    if(me6_nodal_boundary_point(i,j,k)) then
      if(any(hybrid(i,j,k,:)/=nodal(i,j,k,:))) error stop 'ME6 wall did not retain actual nodal RHS'
    endif
    dist=min(j+jg0,ja-j-jg0)
    xx=(i+ig0)*spacing(1);yy=(j+jg0)*spacing(2);zz=(k+kg0)*spacing(3)
    f=1.e0_real_kind+0.1e0_real_kind*cos(zz)
    u=0.1e0_real_kind*sin(xx)*sin(2*yy)*f;v=-0.1e0_real_kind*cos(xx)*sin(yy)**2*f
    ux=0.1e0_real_kind*cos(xx)*sin(2*yy)*f;uy=0.2e0_real_kind*sin(xx)*cos(2*yy)*f
    uz=-0.01e0_real_kind*sin(xx)*sin(2*yy)*sin(zz)
    vx=0.1e0_real_kind*sin(xx)*sin(yy)**2*f;vy=-ux;vz=0.01e0_real_kind*cos(xx)*sin(yy)**2*sin(zz)
    lapu=-5*u-0.01e0_real_kind*sin(xx)*sin(2*yy)*cos(zz)
    lapv=0.1e0_real_kind*cos(xx)*(sin(yy)**2-2*cos(2*yy))*f+0.01e0_real_kind*cos(xx)*sin(yy)**2*cos(zz)
    temperature=tmp(i,j,k);ty=0.02e0_real_kind*sin(2*yy);tyy=0.04e0_real_kind*cos(2*yy)
    mu=temperature*sqrt(temperature)*tempconst1/(temperature+tempconst)/reynolds
    muy=mu*(1.5e0_real_kind/temperature-1.e0_real_kind/(temperature+tempconst))*ty
    kappa=mu/prandtl/const5
    exact(1)=mu*lapu+muy*(uy+vx);exact(2)=mu*lapv+2*muy*vy;exact(3)=muy*vz
    exact(4)=u*exact(1)+v*exact(2)+2*mu*(ux*ux+vy*vy)+mu*((uy+vx)**2+uz*uz+vz*vz) &
      +kappa*tyy+(muy/prandtl/const5)*ty
    if(dist>0) then
      point_error=abs(hybrid(i,j,k,2:5)/jacob(i,j,k)-exact)
      max_error=max(max_error,point_error)
      category=min(dist,5)
      errors(category)=max(errors(category),maxval(point_error))
    endif
    if(i+ig0==ia.or.k+kg0==ka) cycle
    if((i==im.and.irk<irkm).or.(j==jm.and.jrk<jrkm).or.(k==km.and.krk<krkm)) cycle
    weight=1.e0_real_kind
    if(dist==0) weight=0.5e0_real_kind
    budgets(1:4)=budgets(1:4)+weight*nodal(i,j,k,2:5)
    budgets(5:8)=budgets(5:8)+weight*hybrid(i,j,k,2:5)
    if(dist>0) then
      norm_packet(1:4)=norm_packet(1:4)+jacob(i,j,k)*point_error
      norm_packet(5:8)=norm_packet(5:8)+jacob(i,j,k)*point_error**2
      norm_packet(9)=norm_packet(9)+jacob(i,j,k)
    endif
  enddo
  enddo
  enddo
#ifdef TEST_CUDA
  cuda_status=cudaGetDeviceCount(devices)
  if(cuda_status/=cudaSuccess.or.devices<1) error stop 'ME6 wall probe needs a GPU'
  cuda_status=cudaSetDevice(mod(mpirank,devices))
  if(cuda_status/=cudaSuccess) error stop 'ME6 wall probe cannot bind device'
  allocate(vel_d,mold=vel);allocate(tmp_d,mold=tmp);allocate(jacob_d,mold=jacob);allocate(dxi_d,mold=dxi)
  allocate(qrhs_d,mold=qrhs)
  allocate(sigma_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm,6),qflux_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3))
  vel_d=vel;tmp_d=tmp;jacob_d=jacob;dxi_d=dxi;qrhs_d=0.e0_real_kind
  call init_halo_transport_gpu()
  call me6_diffusion_gpu()
  nodal=qrhs_d
  if(.not.all(ieee_is_finite(nodal))) error stop 'ME6 wall GPU read physical-exterior NaN'
  if(any(nodal(:,:,:,1)/=0)) error stop 'ME6 wall GPU changed density RHS'
  errors(6)=maxval(abs(nodal-hybrid)/max(1.e0_real_kind,abs(nodal),abs(hybrid)))
  if(errors(6)>2.e-10_real_kind) error stop 'ME6 wall CPU/GPU differs'
#endif
  call MPI_Allreduce(errors,global_errors,6,mpi_real_type,MPI_MAX,MPI_COMM_WORLD,ierr)
  call MPI_Allreduce(budgets,global_budgets,8,mpi_real_type,MPI_SUM,MPI_COMM_WORLD,ierr)
  call MPI_Allreduce(norm_packet,global_norms,9,mpi_real_type,MPI_SUM,MPI_COMM_WORLD,ierr)
  call MPI_Allreduce(max_error,global_max_error,4,mpi_real_type,MPI_MAX,MPI_COMM_WORLD,ierr)
  if(lio) then
    write(*,'(A,3I5,6ES25.16)') 'ME6_WALL_ERRORS n/np/axis=',n,mpisize,axis,global_errors
    write(*,'(A,8ES25.16)') 'ME6_WALL_NODAL_HYBRID_BUDGET=',global_budgets
    write(*,'(A,4ES25.16)') 'ME6_WALL_GLOBAL_L1=',global_norms(1:4)/global_norms(9)
    write(*,'(A,4ES25.16)') 'ME6_WALL_GLOBAL_L2=',sqrt(global_norms(5:8)/global_norms(9))
    write(*,'(A,4ES25.16)') 'ME6_WALL_GLOBAL_LINF=',global_max_error
  endif
  call MPI_Finalize(ierr)
end program
