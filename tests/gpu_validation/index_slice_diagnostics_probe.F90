program index_slice_diagnostics_probe
  use mpi
  use iso_fortran_env, only: int64
  use astr_precision, only: real_kind,real_bytes
  use astr_precision_mpi, only: mpi_real_type
  use commvar, only: ia,ja,ka,im,jm,km,hm,numq,use_gpu,lihomo,ljhomo,lkhomo,num_species,num_modequ,lcomb,nondimen, &
    roinf,uinf,vinf,winf,pinf,tinf,ref_den,ref_vel,ref_tem,ref_len,ref_tim,ref_miu,nstep
  use parallel, only: ig0,jg0,kg0
  use commarray, only: rho,vel,prs,tmp,x,dxi
  use insitu_run_config, only: catalog_product
  use index_slice_provider, only: prepare_index_slice
  use index_slice_archive, only: configure_index_slices,observe_index_slices
  use checkpoint_state_io, only: checkpoint_state_identity
  use iso_c_binding, only: c_char,c_int,c_null_char
  use ieee_arithmetic, only: ieee_is_finite
#ifdef TEST_INDEX_FIELDS_GPU
  use cudafor
  use commarray_gpu, only: rho_d,vel_d,prs_d,tmp_d,dxi_d
  use index_slice_gpu, only: allocate_index_slice_gpu,release_index_slice_gpu
#endif
  implicit none
  type(catalog_product) :: definition
  character(32) :: argument,mode
  character(4096) :: directory
  integer :: rank,ranks,err,axis,i,j,k,c,d,origin(3),cells(3),first(3),extent(3),devices,n,intervals,step,topology(3)
  integer(int64) :: workspace,download,exchange
  integer(int64) :: free_bytes,total_bytes,allocation_nodes
  real(real_kind),allocatable :: values(:,:),xyz(:,:),reference_values(:,:),reference_xyz(:,:)
  real(real_kind) :: computational(3),point(3),metric(3,3),gradient(3,3),expected(21),local_error,error,seconds
  real(real_kind) :: allocation_weights(7,0:6)
  logical :: ok
  character(256) :: message
  type(checkpoint_state_identity) :: identity
  interface
    integer(c_int) function rename_file(old,new) bind(C,name='rename')
      import c_char,c_int
      character(c_char),intent(in) :: old(*),new(*)
    end function
  end interface
  call MPI_Init(err)
  call MPI_Comm_rank(MPI_COMM_WORLD,rank,err);call MPI_Comm_size(MPI_COMM_WORLD,ranks,err)
  call get_command_argument(1,argument);read(argument,*) axis
  call get_command_argument(2,mode)
  intervals=16
  if(mode=='convergence') then
    call get_command_argument(3,argument);read(argument,*) intervals
  endif
  ia=intervals;ja=intervals;ka=intervals;cells=[ia,ja,ka];origin=0
  topology=1;topology(axis)=ranks
  if(mode=='archive-ownership') then
    ia=16;ja=12;ka=8
    call get_command_argument(4,argument);read(argument,*) topology
  endif
  cells=[ia,ja,ka]/topology
  origin=cells*[mod(rank,topology(1)),mod(rank/topology(1),topology(2)),rank/(topology(1)*topology(2))]
  im=cells(1);jm=cells(2);km=cells(3);ig0=origin(1);jg0=origin(2);kg0=origin(3)
  numq=5;use_gpu=.false.;lihomo=.false.;ljhomo=.false.;lkhomo=.false.
  num_species=0;num_modequ=0;lcomb=.false.;nondimen=.true.
  roinf=2;uinf=1;vinf=0;winf=0;pinf=2;tinf=3
  ref_den=1;ref_vel=1;ref_tem=1;ref_len=1;ref_tim=1;ref_miu=1;nstep=0
  allocate(rho(-hm:im+hm,-hm:jm+hm,-hm:km+hm),vel(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3), &
    prs(-hm:im+hm,-hm:jm+hm,-hm:km+hm),tmp(-hm:im+hm,-hm:jm+hm,-hm:km+hm), &
    x(0:im,0:jm,0:km,3),dxi(0:im,0:jm,0:km,3,3))
  rho=-999;vel=-999;prs=2;tmp=3
  if(mode=='archive-ownership') prs=2+real(rank,real_kind)/1024
  gradient=reshape([.2_real_kind,.3_real_kind,.4_real_kind,.5_real_kind,.6_real_kind,.7_real_kind, &
    .8_real_kind,.9_real_kind,1._real_kind],[3,3])
  do k=0,km
  do j=0,jm
  do i=0,im
    computational=real(origin+[i,j,k],real_kind)/real(intervals,real_kind)
    point=[computational(1)+computational(2)/8,computational(2), &
      computational(3)+computational(1)*computational(2)/32]
    metric=0
    metric(1,:)=[1._real_kind,-.125_real_kind,0._real_kind]*intervals
    metric(2,2)=intervals
    metric(3,:)=[-computational(2)/32,computational(2)/256-computational(1)/32,1._real_kind]*intervals
    x(i,j,k,:)=point;dxi(i,j,k,:,:)=metric
    rho(i,j,k)=2+sum([.01_real_kind,.02_real_kind,.03_real_kind]*point)
    vel(i,j,k,:)=1+matmul(gradient,point)
    if(mode=='convergence') vel(i,j,k,:)=1+point**7
  enddo
  enddo
  enddo
#ifdef TEST_INDEX_FIELDS_GPU
  use_gpu=.true.
  err=cudaGetDeviceCount(devices)
  if(err/=cudaSuccess.or.devices<1) error stop 'available GPU required'
  err=cudaSetDevice(mod(rank,devices))
  allocate(rho_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm),vel_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3), &
    prs_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm),tmp_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm), &
    dxi_d(0:im,0:jm,0:km,3,3))
  rho_d=rho;vel_d=vel;prs_d=prs;tmp_d=tmp;dxi_d=dxi
  if(mode=='allocation-recovery') then
    allocation_weights=0
    err=cudaMemGetInfo(free_bytes,total_bytes)
    allocation_nodes=total_bytes/(12*real_bytes)+1
    if(err/=cudaSuccess.or.allocation_nodes>huge(n)) error stop 'unrepresentable allocation-recovery probe'
    call allocate_index_slice_gpu(int(allocation_nodes),1,[(i,i=1,12)],[1],allocation_weights,ok)
    if(ok) error stop 'expected oversized device allocation to fail'
    call release_index_slice_gpu()
    err=cudaGetLastError()
    if(err/=cudaSuccess) error stop 'recoverable allocation error was not consumed'
  endif
#endif
  if(mode=='archive'.or.mode=='archive-ownership') then
    call get_command_argument(3,directory)
    call configure_index_slices(trim(directory)//'/output',trim(directory)//'/products.jsonl',2_int64,.false.,.false.)
    do step=0,12
      nstep=step
      if(rank==0.and.any(step==[3,7,9])) then
        write(argument,'("candidate",i0,".jsonl")') step
        err=rename_file(trim(directory)//'/'//trim(argument)//c_null_char,trim(directory)//'/products.jsonl'//c_null_char)
        if(err/=0) call MPI_Abort(MPI_COMM_WORLD,4,err)
      endif
      call MPI_Barrier(MPI_COMM_WORLD,err)
      identity=checkpoint_state_identity(int(step,int64),real(step,real_kind)*.001_real_kind, &
        merge(.001_real_kind,0._real_kind,step>0),.001_real_kind)
      call observe_index_slices(identity)
    enddo
    if(rank==0) print '(a)', 'SLICE_RUNTIME_ARCHIVE_PASS'
    call MPI_Finalize(err)
    stop
  endif
  definition%fields=[1,2,3,4,5,6,(i,i=13,27)]
  definition%axis=2;definition%index=0
  definition%ranges=reshape([0_int64,int(intervals,int64),0_int64,0_int64,0_int64,int(intervals,int64)],[2,3])
  if(mode=='native-reference') then
    use_gpu=.false.
    call prepare_index_slice(definition,0,0_int64,reference_values,reference_xyz,first,extent,ok,message, &
      workspace,download,exchange,seconds)
    if(.not.ok) error stop 'cannot prepare native CPU reference'
#ifdef TEST_INDEX_FIELDS_GPU
    use_gpu=.true.
#endif
  endif
  call prepare_index_slice(definition,0,0_int64,values,xyz,first,extent,ok,message,workspace,download,exchange,seconds)
  if(.not.ok) then
    print *,message;call MPI_Abort(MPI_COMM_WORLD,1,err)
  endif
  expected(7:15)=reshape(gradient,[9]);expected(16)=-.5_real_kind*sum(gradient*transpose(gradient))
  expected(17)=gradient(1,1)+gradient(2,2)+gradient(3,3)
  expected(18:20)=[gradient(3,2)-gradient(2,3),gradient(1,3)-gradient(3,1),gradient(2,1)-gradient(1,2)]
  expected(21)=sqrt(.01_real_kind**2+.02_real_kind**2+.03_real_kind**2)
  local_error=0
  do n=1,size(values,1)
    expected(1)=2+sum([.01_real_kind,.02_real_kind,.03_real_kind]*xyz(:,n))
    expected(2:4)=1+matmul(gradient,xyz(:,n));expected(5:6)=[2._real_kind,3._real_kind]
    if(mode=='convergence') then
      gradient=0
      do d=1,3
        gradient(d,d)=7*xyz(d,n)**6
      enddo
      local_error=max(local_error,maxval(abs(values(n,7:15)-reshape(gradient,[9]))))
    else
      local_error=max(local_error,maxval(abs(values(n,:)-expected)))
    endif
  enddo
  call MPI_Allreduce(local_error,error,1,mpi_real_type,MPI_MAX,MPI_COMM_WORLD,err)
  if(rank==0) print '(a,es24.16)','SLICE_DIAGNOSTIC_ERROR=',error
  if(mode=='native-reference') then
    if(.not.all(ieee_is_finite(values))) error stop 'nonfinite native diagnostic'
    local_error=maxval(abs(values-reference_values))
    call MPI_Allreduce(local_error,error,1,mpi_real_type,MPI_MAX,MPI_COMM_WORLD,err)
    if(rank==0) print '(a,i0,a,es24.16)','SLICE_NATIVE_BYTES=',real_bytes,' BACKEND_DIFFERENCE=',error
  endif
  if(mode/='convergence'.and.real_bytes==8.and.error>2e-10_real_kind) call MPI_Abort(MPI_COMM_WORLD,2,err)
  if(use_gpu.and.download/=int(size(values),int64)*real_bytes) call MPI_Abort(MPI_COMM_WORLD,3,err)
  call MPI_Finalize(err)
end program
