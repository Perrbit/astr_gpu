program index_slice_diagnostics_probe
  use mpi
  use iso_fortran_env, only: int64
  use astr_precision, only: real_kind,real_bytes
  use astr_precision_mpi, only: mpi_real_type
  use commvar, only: ia,ja,ka,im,jm,km,hm,numq,use_gpu,lihomo,ljhomo,lkhomo,num_species,num_modequ,lcomb,nondimen, &
    roinf,uinf,vinf,winf,pinf,tinf,ref_den,ref_vel,ref_tem,ref_len,ref_tim,ref_miu,nstep
  use parallel, only: ig0,jg0,kg0,isize,jsize,ksize,mpileft,mpiright,mpidown,mpiup,mpiback,mpifront,mpirank,mpisize
  use commarray, only: rho,vel,prs,tmp,x,dxi
  use insitu_run_config, only: catalog_product
  use index_slice_provider, only: prepare_index_slice
  use index_slice_archive, only: configure_index_slices,observe_index_slices
  use checkpoint_state_io, only: checkpoint_state_identity
  use iso_c_binding, only: c_char,c_int,c_null_char,c_f_pointer
  use ieee_arithmetic, only: ieee_is_finite
#ifdef TEST_INDEX_FIELDS_GPU
  use cudafor
  use commarray_gpu, only: rho_d,vel_d,prs_d,tmp_d,dxi_d,x_d
  use index_slice_gpu, only: allocate_index_slice_gpu,release_index_slice_gpu,index_slice_device_view,download_index_slice_gpu, &
    index_slice_exchange_usage
  use insitu_products_gpu, only: begin_catalog_streamlines_gpu,end_catalog_streamlines_gpu,catalog_streamline_pointers_gpu
#endif
  implicit none
  type(catalog_product) :: definition
  character(32) :: argument,mode,transport
  character(4096) :: directory
  integer :: rank,ranks,err,axis,i,j,k,c,d,origin(3),cells(3),first(3),extent(3),devices,n,intervals,step,topology(3)
  integer :: plane_axis,plane_index,cell_count,total_cells
  integer(int64) :: workspace,download,exchange
  integer(int64) :: private_download,private_upload
  integer(int64) :: free_bytes,total_bytes,allocation_nodes
  real(real_kind),allocatable :: values(:,:),xyz(:,:),reference_values(:,:),reference_xyz(:,:)
  real(real_kind) :: computational(3),point(3),metric(3,3),gradient(3,3),expected(21),local_error,error,seconds
  real(real_kind) :: allocation_weights(7,0:6)
  logical :: ok
#ifdef TEST_INDEX_FIELDS_GPU
  type(c_devptr) :: retained_values,retained_coordinates
  real(real_kind),pointer,contiguous,device :: oracle_coordinates(:,:)=>null()
  real(real_kind),pointer,contiguous,device :: trace_halo(:,:,:,:)=>null(),trace_coordinates(:,:,:,:)=>null()
  real(real_kind),allocatable :: trace_values(:,:,:,:),trace_points(:,:,:,:)
  type(c_devptr) :: trace_pointer,trace_coordinate_pointer
  integer :: neighbors(2,3),stride,coordinate
#endif
  character(256) :: message
  type(checkpoint_state_identity) :: identity
  interface
    integer(c_int) function rename_file(old,new) bind(C,name='rename')
      import c_char,c_int
      character(c_char),intent(in) :: old(*),new(*)
    end function
  end interface
#ifdef TEST_INDEX_FIELDS_GPU
  ! Bind this diagnostic before CUDA-aware MPI probes the primary context.
  call get_environment_variable('OMPI_COMM_WORLD_LOCAL_RANK',argument,status=err)
  if(err/=0) error stop 'MPI local rank is required by the GPU diagnostic'
  read(argument,*) rank
  err=cudaGetDeviceCount(devices)
  if(err/=cudaSuccess.or.devices<1) error stop 'available GPU required'
  err=cudaSetDevice(mod(rank,devices))
  if(err/=cudaSuccess) error stop 'cannot bind diagnostic GPU'
  err=cudaMemGetInfo(free_bytes,total_bytes)
  if(err/=cudaSuccess) error stop 'cannot initialize diagnostic CUDA context'
#endif
  call MPI_Init(err)
  call MPI_Comm_rank(MPI_COMM_WORLD,rank,err);call MPI_Comm_size(MPI_COMM_WORLD,ranks,err)
  mpirank=rank;mpisize=ranks
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
    dxi_d(0:im,0:jm,0:km,3,3),x_d(0:im,0:jm,0:km,3))
  rho_d=rho;vel_d=vel;prs_d=prs;tmp_d=tmp;dxi_d=dxi;x_d=x
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
  if(mode=='trace-input') then
#ifdef TEST_INDEX_FIELDS_GPU
    definition%fields=[2,3,4];definition%axis=0
    definition%ranges(1,:)=0;definition%ranges(2,:)=intervals
    call prepare_index_slice(definition,0,0_int64,values,xyz,first,extent,ok,message,workspace,download,exchange,seconds, &
      include_cell_endpoints=.true.,retain_device=.true.)
    if(.not.ok) error stop 'cannot prepare frozen trace source'
    isize=topology(1);jsize=topology(2);ksize=topology(3);stride=1;neighbors=MPI_PROC_NULL
    do d=1,3
      coordinate=mod(rank/stride,topology(d))
      if(coordinate>0) neighbors(1,d)=rank-stride
      if(coordinate<topology(d)-1) neighbors(2,d)=rank+stride
      stride=stride*topology(d)
    enddo
    mpileft=neighbors(1,1);mpiright=neighbors(2,1);mpidown=neighbors(1,2);mpiup=neighbors(2,2)
    mpiback=neighbors(1,3);mpifront=neighbors(2,3)
    call index_slice_device_view(retained_values,retained_coordinates)
    call begin_catalog_streamlines_gpu('pinned',retained_values,retained_coordinates,definition%fields,.true., &
      4_int64*1024**3,2_int64*1024**3,1024_int64**3,workspace+6_int64*product(extent)*real_bytes,ok,message)
    if(.not.ok) then
      print *,message;error stop 'cannot prepare private trace halos'
    endif
    call catalog_streamline_pointers_gpu(trace_pointer,trace_coordinate_pointer)
    call c_f_pointer(trace_pointer,trace_halo,[im+7,jm+7,km+7,3])
    call c_f_pointer(trace_coordinate_pointer,trace_coordinates,[im+7,jm+7,km+7,3])
    allocate(trace_values(im+7,jm+7,km+7,3),trace_points(im+7,jm+7,km+7,3))
    trace_values=trace_halo;trace_points=trace_coordinates;local_error=0
    do k=1,km+7
    do j=1,jm+7
    do i=1,im+7
      if(any(origin+[i-4,j-4,k-4]<0).or.any(origin+[i-4,j-4,k-4]>intervals)) cycle
      computational=real(origin+[i-4,j-4,k-4],real_kind)/real(intervals,real_kind)
      point=[computational(1)+computational(2)/8,computational(2), &
        computational(3)+computational(1)*computational(2)/32]
      local_error=max(local_error,maxval(abs(trace_points(i,j,k,:)-point)), &
        maxval(abs(trace_values(i,j,k,:)-1-matmul(gradient,point))))
    enddo
    enddo
    enddo
    nullify(trace_halo,trace_coordinates)
    call end_catalog_streamlines_gpu(0);call release_index_slice_gpu()
    call MPI_Allreduce(local_error,error,1,mpi_real_type,MPI_MAX,MPI_COMM_WORLD,err)
    if(error>2e-10_real_kind.or..not.ieee_is_finite(error)) error stop 'private streamline input differs'
    if(rank==0) print '(a,es24.16)','TRACE_INPUT_MAXABS=',error
    call MPI_Finalize(err);stop
#else
    error stop 'trace input probe requires a GPU build'
#endif
  endif
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
  if(mode=='mesh'.or.mode=='retained-mesh') then
    call get_command_argument(3,argument);read(argument,*) plane_axis
    call get_command_argument(4,argument);read(argument,*) plane_index
    transport='host';call get_command_argument(5,argument)
    if(len_trim(argument)>0) transport=argument
    definition%axis=plane_axis;definition%index=plane_index
    definition%ranges(1,:)=0;definition%ranges(2,:)=intervals
    definition%ranges(:,plane_axis)=plane_index
    call prepare_index_slice(definition,0,0_int64,values,xyz,first,extent,ok,message,workspace,download,exchange,seconds, &
      include_cell_endpoints=.true.,retain_device=mode=='retained-mesh',transport=trim(transport))
    if(.not.ok) then
      print *,message;call MPI_Abort(MPI_COMM_WORLD,1,err)
    endif
    cell_count=0
    if(product(extent)>0) then
      cell_count=1
      do d=1,3
        if(d/=plane_axis) cell_count=cell_count*(extent(d)-1)
      enddo
    endif
    call MPI_Allreduce(cell_count,total_cells,1,MPI_INTEGER,MPI_SUM,MPI_COMM_WORLD,err)
    if(total_cells/=intervals**2) error stop 'missing or duplicate image cells at an MPI interface'
    if(mode=='retained-mesh') then
#ifdef TEST_INDEX_FIELDS_GPU
      call index_slice_exchange_usage(private_download,private_upload)
      write(*,'(a,i0,2a,a,i0,a,i0)') 'PRIVATE_MPI rank=',rank,' mode=',trim(transport), &
        ' download=',private_download,' upload=',private_upload
      if(size(values)/=0.or.size(xyz)/=0.or.download/=0) error stop 'retained mesh acquired host field buffers'
      call index_slice_device_view(retained_values,retained_coordinates)
      allocate(reference_values(product(extent),size(definition%fields)),reference_xyz(3,product(extent)))
      call download_index_slice_gpu(reference_values,ok)
      call c_f_pointer(retained_coordinates,oracle_coordinates,[3,max(1,product(extent))])
      if(size(reference_xyz)>0) reference_xyz=oracle_coordinates(:,:product(extent))
      nullify(oracle_coordinates)
      if(.not.ok) error stop 'cannot read diagnostic-only retained mesh oracle'
      call move_alloc(reference_values,values);call move_alloc(reference_xyz,xyz)
      call release_index_slice_gpu()
      if(rank==0) print '(a)','RETAINED_MESH field_d2h_bytes=0 geometry_host_bytes=0 oracle_readback_separate=1'
#else
      error stop 'retained mesh probe requires the GPU build'
#endif
    endif
    if(rank==0) print '(a,i0)','IMAGE_MESH_CELLS=',total_cells
  else
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
  endif
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
  if(use_gpu.and.mode/='retained-mesh'.and.download/=int(size(values),int64)*real_bytes) call MPI_Abort(MPI_COMM_WORLD,3,err)
  call MPI_Finalize(err)
end program
