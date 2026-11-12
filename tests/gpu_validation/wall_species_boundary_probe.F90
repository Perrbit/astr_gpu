! Link against the root-built solver objects to exercise the actual CPU dispatcher.
program wall_species_production_probe
  use astr_precision, only: real_kind
  use commvar
  use commarray, only: rho,vel,prs,tmp,spc,q,qrhs,x
  use parallel, only: mpirank,mpisize,lio,irk,irkm,jrk,jrkm,krk,krkm,isize,jsize,ksize,mpileft,mpiright
  use bc, only: boucon,bctype,twall,wall_blowing_configured,wall_blowing_amplitude
  use boundary_high_order, only: high_order_boundary_enabled,high_order_wall_enabled
  use mpi
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
#ifdef TEST_CUDA
  use cudafor
  use commarray_gpu, only: q_d,qrhs_d,rho_d,prs_d,tmp_d,vel_d
  use boundary_gpu, only: apply_boundary_conditions_gpu
#endif
  implicit none
  integer :: ierr,wall,i,j,k,variant,failed,global_failed
  real(real_kind) :: expected,metrics(7),global_metrics(7),pi
#ifdef TEST_CUDA
  real(real_kind),allocatable :: gpu_q(:,:,:,:)
  integer :: status,devices
#endif
  character(32) :: argument
  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD,mpirank,ierr)
  call MPI_Comm_size(MPI_COMM_WORLD,mpisize,ierr)
  call get_command_argument(1,argument);read(argument,*) wall
  call get_command_argument(2,argument);read(argument,*) num_species
  call get_command_argument(3,argument);read(argument,*) variant
  im=16;jm=8;km=8;ndims=3;numq=5+num_species;num_modequ=0
  irk=mpirank;irkm=mpisize-1;jrk=0;jrkm=0;krk=0;krkm=0;lio=mpirank==0
  isize=mpisize;jsize=1;ksize=1
  mpileft=mpirank-1;mpiright=mpirank+1
  if(irk==0) mpileft=MPI_PROC_NULL
  if(irk==irkm) mpiright=MPI_PROC_NULL
  nondimen=.true.;lcomb=.false.;limmbou=.false.;lfilter=.false.;lrestart=.false.
  lihomo=.false.;ljhomo=.true.;lkhomo=.true.;ltimrpt=.false.
  flowtype='tgv';turbmode='none';conschm='643e';difschm='643e';rkscheme='rk3'
  const2=1.e0_real_kind;const6=2.5e0_real_kind;gamma=1.4e0_real_kind;time=0
  high_order_boundary_enabled=.false.;high_order_wall_enabled=.false.
  wall_blowing_configured=.true.;wall_blowing_amplitude=0
  bctype=[wall,wall,1,1,1,1];twall=1
  allocate(rho(-hm:im+hm,-hm:jm+hm,-hm:km+hm),prs(-hm:im+hm,-hm:jm+hm,-hm:km+hm), &
    tmp(-hm:im+hm,-hm:jm+hm,-hm:km+hm),vel(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3), &
    spc(-hm:im+hm,-hm:jm+hm,-hm:km+hm,num_species), &
    q(-hm:im+hm,-hm:jm+hm,-hm:km+hm,numq),qrhs(0:im,0:jm,0:km,numq), &
    x(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3))
  rho=1;prs=1;tmp=1;vel=0;q=0;qrhs=0;spc=0
  q(:,:,:,1)=1;q(:,:,:,5)=const6
  pi=acos(-1.e0_real_kind)
  do k=-hm,km+hm
  do j=-hm,jm+hm
  do i=-hm,im+hm
    x(i,j,k,:)=real([i+mpirank*im,j,k],real_kind)
    if(variant==3) then
      prs(i,j,k)=1.e0_real_kind+0.001e0_real_kind*(i-im/2)**2+0.02e0_real_kind*sin(2*pi*j/jm)
      tmp(i,j,k)=1.e0_real_kind+0.002e0_real_kind*(i-im/2)**2+0.01e0_real_kind*cos(2*pi*k/km)
      rho(i,j,k)=prs(i,j,k)/tmp(i,j,k)*const2
      q(i,j,k,1)=rho(i,j,k);q(i,j,k,5)=prs(i,j,k)*const6
    endif
    if(num_species>0) then
      expected=species_reference(j,k)
      if(variant==2.and.j<0) expected=expected+0.125e0_real_kind
      spc(i,j,k,1)=expected
      q(i,j,k,6)=expected
    endif
  enddo
  enddo
  enddo
#ifdef TEST_CUDA
  if(num_species/=0) error stop 'GPU five-variable wall probe does not implement species'
  status=cudaGetDeviceCount(devices)
  if(status/=cudaSuccess.or.devices<=0) error stop 'GPU wall probe requires a working device'
  status=cudaSetDevice(mod(mpirank,devices))
  if(status/=cudaSuccess) error stop 'GPU wall probe device binding failed'
  allocate(q_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm,5),qrhs_d(0:im,0:jm,0:km,5), &
    rho_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm),prs_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm), &
    tmp_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm),vel_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3), &
    gpu_q(-hm:im+hm,-hm:jm+hm,-hm:km+hm,5))
  q_d=q;qrhs_d=qrhs;rho_d=rho;prs_d=prs;tmp_d=tmp;vel_d=vel
  call apply_boundary_conditions_gpu()
  status=cudaDeviceSynchronize()
  if(status/=cudaSuccess) error stop 'GPU actual wall dispatcher failed'
  gpu_q=q_d
#endif
  ! Exercise the production dispatcher and its compiled wall routines unchanged.
  call boucon()
  metrics=0;failed=0
  if(.not.all(ieee_is_finite(q(0:im,0:jm,0:km,:)))) failed=1
#ifdef TEST_CUDA
  if(.not.all(ieee_is_finite(gpu_q(0:im,0:jm,0:km,:)))) failed=1
  metrics(7)=maxval(abs(gpu_q(0:im,0:jm,0:km,:)-q(0:im,0:jm,0:km,:))/ &
    max(1.e0_real_kind,abs(gpu_q(0:im,0:jm,0:km,:)),abs(q(0:im,0:jm,0:km,:))))
  deallocate(q_d,qrhs_d,rho_d,prs_d,tmp_d,vel_d,gpu_q)
#endif
  if(num_species>0) then
    do k=0,km
    do j=0,jm
      expected=species_reference(j,k)
      if(irk==0) then
        metrics(1)=max(metrics(1),abs(spc(0,j,k,1)-expected))
        metrics(3)=max(metrics(3),abs(q(0,j,k,6)/q(0,j,k,1)-expected))
      endif
      if(irk==irkm) then
        metrics(2)=max(metrics(2),abs(spc(im,j,k,1)-expected))
        metrics(4)=max(metrics(4),abs(q(im,j,k,6)/q(im,j,k,1)-expected))
        metrics(5)=max(metrics(5),abs((3*spc(im,j,k,1)-4*spc(im-1,j,k,1)+spc(im-2,j,k,1))/2))
      endif
      metrics(6)=max(metrics(6),maxval(abs(spc(1:im-1,j,k,1)-expected)))
    enddo
    enddo
    if(irk==irkm) write(*,'(A,I0,A,3ES24.16)') 'OWNER rank=',mpirank, &
      ' expected/actual/ghost=',species_reference(4,2),spc(im,4,2,1),spc(im,-2,2,1)
  endif
  call MPI_Allreduce(metrics,global_metrics,7,MPI_DOUBLE_PRECISION,MPI_MAX,MPI_COMM_WORLD,ierr)
  if(maxval(global_metrics(1:6))>2.e-12_real_kind.or.global_metrics(7)>2.e-10_real_kind) failed=1
  call MPI_Allreduce(failed,global_failed,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
  if(lio) then
    write(*,'(A,4I5)') 'CONFIG bc/species/variant/np=',wall,num_species,variant,mpisize
    write(*,'(A,7ES24.16)') 'ERROR lowerY upperY lowerQ upperQ normal_derivative interior gpu_scaled=',global_metrics
    if(global_failed/=0) then
      print '(A)', 'WALL_SPECIES_INVARIANCE_FAILED'
    else
      print '(A)', 'WALL_SPECIES_INVARIANCE_PASSED'
    endif
  endif
  call MPI_Finalize(ierr)
  if(global_failed/=0) stop 2
contains
  real(real_kind) function species_reference(j,k) result(value)
    integer,intent(in) :: j,k
    value=0.3e0_real_kind
    if(variant/=0) value=value+0.05e0_real_kind*sin(2*pi*real(j,real_kind)/jm) &
      +0.02e0_real_kind*cos(2*pi*real(k,real_kind)/km)
  end function
end program
