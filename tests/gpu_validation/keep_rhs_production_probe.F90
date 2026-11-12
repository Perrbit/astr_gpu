! Link with the root-built solver objects; test RHS ownership, not a flow solution.
program keep_rhs_production_probe
  use astr_precision, only: real_kind
  use astr_precision_mpi, only: mpi_real_type
  use commvar
  use commarray, only: rho,vel,prs,jacob,dxi,qrhs
  use numerical_forms, only: convection_form
  use compact_runtime, only: configure_global_compact,global_keep_convection_cpu,release_global_compact
  use parallel, only: mpirank,mpisize,lio,irk,jrk,krk,isize,jsize,ksize,ig0,jg0,kg0, &
    mpi_igroup,mpi_jgroup,mpi_kgroup
  use mpi
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_value,ieee_quiet_nan
#ifdef TEST_CUDA
  use cudafor
  use commarray_gpu, only: rho_d,vel_d,prs_d,jacob_d,dxi_d,qrhs_d
  use compact_operator_gpu, only: compact_keep_convection_gpu,release_compact_operator_gpu
#endif
  implicit none
  real(real_kind),allocatable :: baseline(:,:,:,:),accumulated(:,:,:,:)
  real(real_kind) :: pi,coord(3),errors(2),global_errors(2)
  integer :: ierr,axis,i,j,k,d,totals(3),offset(3),coordinates(3),ranks(3),types(3)
  logical :: test_repair
#ifdef TEST_CUDA
  integer :: devices,status
#endif
  character(16) :: argument
  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD,mpirank,ierr)
  call MPI_Comm_size(MPI_COMM_WORLD,mpisize,ierr)
  call get_command_argument(1,argument);read(argument,*) axis
  if(axis<1.or.axis>3) error stop 'KEEP RHS probe requires axis 1, 2 or 3'
  call get_command_argument(2,argument);test_repair=trim(argument)=='repair'
  im=8;jm=8;km=8;ndims=3;numq=5;num_species=0;num_modequ=0
  coordinates=0;coordinates(axis)=mpirank;ranks=1;ranks(axis)=mpisize
  irk=coordinates(1);jrk=coordinates(2);krk=coordinates(3)
  isize=ranks(1);jsize=ranks(2);ksize=ranks(3);lio=mpirank==0
  offset=8*coordinates;totals=8*ranks
  ig0=offset(1);jg0=offset(2);kg0=offset(3)
  ia=totals(1);ja=totals(2);ka=totals(3)
  mpi_igroup=MPI_COMM_SELF;mpi_jgroup=MPI_COMM_SELF;mpi_kgroup=MPI_COMM_SELF
  select case(axis)
  case(1);mpi_igroup=MPI_COMM_WORLD
  case(2);mpi_jgroup=MPI_COMM_WORLD
  case(3);mpi_kgroup=MPI_COMM_WORLD
  end select
  types=[3,3,3]
  if(jrk==0) types(2)=1
  if(jrk==jsize-1) types(2)=2
  if(jsize==1) types(2)=4
  npdci=types(1);npdcj=types(2);npdck=types(3)
  is=0;ie=im;js=0;je=jm;ks=0;ke=km
  if(jrk==0) js=1
  if(jrk==jsize-1) je=jm-1
  conschm='643c';difschm='643e';convection_form='keep';gamma=1.4e0_real_kind
  flowtype='channel';turbmode='none';rkscheme='rk3';nondimen=.true.
  lihomo=.true.;ljhomo=.false.;lkhomo=.true.;lcomb=.false.;lcracon=.false.;lfilter=.false.;lrestart=.false.
  lcracon=test_repair
  allocate(rho(-hm:im+hm,-hm:jm+hm,-hm:km+hm),prs(-hm:im+hm,-hm:jm+hm,-hm:km+hm), &
    vel(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3),jacob(-hm:im+hm,-hm:jm+hm,-hm:km+hm), &
    dxi(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3,3),qrhs(0:im,0:jm,0:km,5), &
    baseline(0:im,0:jm,0:km,5),accumulated(0:im,0:jm,0:km,5))
  pi=acos(-1.e0_real_kind);jacob=1.e0_real_kind;dxi=0.e0_real_kind
  do d=1,3
    dxi(:,:,:,d,d)=1.e0_real_kind
  enddo
  rho=ieee_value(0.e0_real_kind,ieee_quiet_nan);vel=rho(0,0,0);prs=rho(0,0,0)
  do k=-hm,km+hm
  do j=-hm,jm+hm
  do i=-hm,im+hm
    if(j+offset(2)<0.or.j+offset(2)>ja) cycle
    coord=pi*real([i,j,k]+offset,real_kind)/real(totals,real_kind)
    rho(i,j,k)=1.e0_real_kind+0.01e0_real_kind*cos(coord(2))
    prs(i,j,k)=1.e0_real_kind+0.02e0_real_kind*cos(2*coord(1))*sin(coord(2))
    vel(i,j,k,:)=[0.1e0_real_kind*sin(coord(2)),0.01e0_real_kind*sin(2*coord(1))*sin(coord(2))**2, &
      0.005e0_real_kind*cos(2*coord(3))*sin(coord(2))]
  enddo
  enddo
  enddo
  call configure_global_compact([1,1,41,41,1,1])
  qrhs=0.e0_real_kind
  call global_keep_convection_cpu()
  baseline=qrhs
  if(.not.all(ieee_is_finite(baseline)).or.maxval(abs(baseline))<1.e-6_real_kind) &
    error stop 'KEEP compact RHS baseline is nonfinite or vacuous'
  qrhs=0.125e0_real_kind
  call global_keep_convection_cpu()
  errors=0.e0_real_kind
  errors(1)=maxval(abs(qrhs-(0.125e0_real_kind+baseline)))
  if(jrk==0.and.any(qrhs(:,0,:,:)/=0.125e0_real_kind)) error stop 'CPU compact erased lower boundary RHS'
  if(jrk==jsize-1.and.any(qrhs(:,jm,:,:)/=0.125e0_real_kind)) error stop 'CPU compact erased upper boundary RHS'
#ifdef TEST_CUDA
  status=cudaGetDeviceCount(devices)
  if(status/=cudaSuccess.or.devices<=0) error stop 'KEEP RHS probe requires a GPU'
  status=cudaSetDevice(mod(mpirank,devices))
  if(status/=cudaSuccess) error stop 'KEEP RHS probe cannot bind device'
  allocate(rho_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm),prs_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm), &
    vel_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3),jacob_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm), &
    dxi_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3,3),qrhs_d(0:im,0:jm,0:km,5))
  rho_d=rho;prs_d=prs;vel_d=vel;jacob_d=jacob;dxi_d=dxi;qrhs_d=0.125e0_real_kind
  call compact_keep_convection_gpu()
  accumulated=qrhs_d
  if(.not.all(ieee_is_finite(accumulated))) error stop 'GPU compact RHS is nonfinite'
  errors(2)=maxval(abs(accumulated-(0.125e0_real_kind-baseline)))
  if(jrk==0.and.any(accumulated(:,0,:,:)/=0.125e0_real_kind)) error stop 'GPU compact erased lower boundary RHS'
  if(jrk==jsize-1.and.any(accumulated(:,jm,:,:)/=0.125e0_real_kind)) error stop 'GPU compact erased upper boundary RHS'
  call release_compact_operator_gpu()
#endif
  call MPI_Allreduce(errors,global_errors,2,mpi_real_type,MPI_MAX,MPI_COMM_WORLD,ierr)
  if(lio) write(*,'(A,2I4,2ES24.16)') 'KEEP_RHS_COMPACT np/axis cpu/gpu=',mpisize,axis,global_errors
  if(maxval(global_errors)>2.e-10_real_kind) error stop 'KEEP compact did not preserve existing RHS'
  if(test_repair) call check_actual_repair()
  call release_global_compact()
  call MPI_Finalize(ierr)
contains
  subroutine check_actual_repair()
    use commarray, only: q,tmp,spc,nodestat,crinod,x
    use mainloop, only: crashfix,fhand_err
#ifdef TEST_CUDA
    use commarray_gpu, only: q_d,tmp_d,crinod_d
    use crashfix_gpu, only: repair_crashfix_state_gpu,crashfix_support_status_gpu,release_crashfix_gpu
#endif
    real(real_kind) :: local_error,global_error,internal
    integer :: repaired_nodes,ii,jj,kk,c
#ifdef TEST_CUDA
    logical,allocatable :: mask(:,:,:)
#endif
    const2=1.e0_real_kind;const6=1.e0_real_kind/(gamma-1.e0_real_kind)
    nstep=0;ctime=0.e0_real_kind;lreport=.false.;ltimrpt=.false.
    allocate(q(-hm:im+hm,-hm:jm+hm,-hm:km+hm,5),tmp(-hm:im+hm,-hm:jm+hm,-hm:km+hm), &
      spc(-hm:im+hm,-hm:jm+hm,-hm:km+hm,0),nodestat(-hm:im+hm,-hm:jm+hm,-hm:km+hm), &
      crinod(-hm:im+hm,-hm:jm+hm,-hm:km+hm),x(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3))
    q(:,:,:,1)=rho
    do d=1,3
      q(:,:,:,d+1)=rho*vel(:,:,:,d)
    enddo
    q(:,:,:,5)=prs*const6+0.5e0_real_kind*rho*sum(vel**2,dim=4)
    tmp=prs/rho*const2;nodestat=0;crinod=.false.;x=0.e0_real_kind
    q(3,3,3,1)=-1.e0_real_kind;q(4,3,3,5)=-1.e0_real_kind
    rho(3,3,3)=-1.e0_real_kind;prs(3:4,3,3)=-1.e0_real_kind;tmp(3:4,3,3)=-1.e0_real_kind
#ifdef TEST_CUDA
    convection_form='standard'
    if(crashfix_support_status_gpu()/=2) error stop 'KEEP repair accidentally admitted standard compact convection'
    convection_form='keep'
    if(crashfix_support_status_gpu()/=0) error stop 'KEEP compact repair is not admitted'
    allocate(q_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm,5),tmp_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm), &
      crinod_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm),mask(-hm:im+hm,-hm:jm+hm,-hm:km+hm))
    q_d=q;rho_d=rho;vel_d=vel;prs_d=prs;tmp_d=tmp;crinod_d=.false.
#endif
    open(newunit=fhand_err,status='scratch',form='formatted',action='write')
    call crashfix(repaired_nodes=repaired_nodes)
    close(fhand_err)
    if(repaired_nodes/=2*mpisize) error stop 'CPU KEEP fixture did not repair both invalid states'
    if(.not.crinod(3,3,3).or..not.crinod(4,3,3)) error stop 'CPU KEEP repair history is missing'
    local_error=0.e0_real_kind
#ifdef TEST_CUDA
    call repair_crashfix_state_gpu(1)
    accumulated=q_d(0:im,0:jm,0:km,:)
    mask=crinod_d
    if(any(mask.neqv.crinod)) error stop 'KEEP CPU/GPU repair histories differ'
    if(.not.all(ieee_is_finite(accumulated))) error stop 'KEEP GPU repair produced nonfinite state'
    do kk=0,km
    do jj=0,jm
    do ii=0,im
      do c=1,5
        local_error=max(local_error,abs(accumulated(ii,jj,kk,c)-q(ii,jj,kk,c))/ &
          max(1.e0_real_kind,abs(accumulated(ii,jj,kk,c)),abs(q(ii,jj,kk,c))))
      enddo
      if(accumulated(ii,jj,kk,1)<=0.e0_real_kind) error stop 'KEEP GPU repair left negative density'
      internal=accumulated(ii,jj,kk,5)-0.5e0_real_kind*sum(accumulated(ii,jj,kk,2:4)**2)/accumulated(ii,jj,kk,1)
      if(internal<=0.e0_real_kind) error stop 'KEEP GPU repair left negative internal energy'
    enddo
    enddo
    enddo
    call release_crashfix_gpu()
#endif
    call MPI_Allreduce(local_error,global_error,1,mpi_real_type,MPI_MAX,MPI_COMM_WORLD,ierr)
    if(global_error>2.e-10_real_kind) error stop 'KEEP CPU/GPU repaired states differ'
    if(lio) write(*,'(A,I0,A,ES24.16)') 'KEEP_CRASHFIX repaired_nodes=',repaired_nodes,' max_scaled=',global_error
  end subroutine
end program
