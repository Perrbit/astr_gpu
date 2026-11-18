program critical_history_probe
  use iso_c_binding
  use astr_precision, only: real_kind,real_bytes,real_to_word
  use mpi
#if defined(REFERENCE_CPU) || defined(REFERENCE_CUDA)
  use commvar, only: im,jm,km,hm,ia,ja,ka,lihomo,ljhomo,lkhomo
  use parallel, only: mpirank,mpisize,lio,isize,jsize,ksize,mpileft,mpiright,mpidown,mpiup,mpiback,mpifront, &
    mpitag,reconcile_crashfix_history
#ifdef REFERENCE_CUDA
  use cudafor
  use commarray_gpu, only: crinod_d
  use crashfix_gpu, only: reconcile_crashfix_history_gpu,release_crashfix_gpu
#endif
#else
  use device_state
  use fluid_explicit, only: fluid_prepare
  use halo_transport
  use crashfix_operator_cpp
#endif
  implicit none
  integer,parameter :: cells(3)=[9,7,6],halo=5
  integer :: rank,ranks,local_rank,devices,status,ierr,topology(3),coordinate(3),origin(3),global(3),periods(3), &
    lower(3),upper(3),axis,i,j,k,c,debug,peer,peer_coordinate(3),p(3),g(3),offsets(3),candidate(3), &
    sx,sy,sz,outsides,pass,physical_nodes,halo_nodes
  integer(c_int) :: mode
  logical,allocatable :: history(:,:,:)
  integer(c_int8_t),allocatable :: seed(:,:,:),actual(:,:,:),expected(:,:,:),first(:,:,:),fluid(:,:,:)
  real(real_kind),allocatable :: q(:,:,:,:),observed_q(:,:,:,:)
  real(c_double) :: metrics(6),sent_bytes
  logical :: allowed,value
  character(128) :: text,transport
#if !defined(REFERENCE_CPU) && !defined(REFERENCE_CUDA)
  type(c_ptr) :: state
#endif
  call get_environment_variable('OMPI_COMM_WORLD_LOCAL_RANK',text,status=status)
  if(status/=0) error stop 'critical history test requires Open MPI local rank'
  read(text,*) local_rank
#ifdef REFERENCE_CUDA
  status=cudaGetDeviceCount(devices);if(status/=cudaSuccess) error stop 'critical GPU count'
  if(devices<1) error stop 'critical history test requires a visible GPU'
  status=cudaSetDevice(mod(local_rank,devices));if(status/=cudaSuccess) error stop 'critical GPU binding'
#elif !defined(REFERENCE_CPU)
  status=device_count(devices);if(status/=0) error stop 'critical C++ device count'
  if(devices<1) error stop 'critical history test requires a visible GPU'
  status=device_bind(int(mod(local_rank,devices),c_int))
  if(status/=0) error stop 'critical history pre-MPI device binding failed'
#endif
  call MPI_Init(ierr);call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr);call MPI_Comm_size(MPI_COMM_WORLD,ranks,ierr)
  call get_command_argument(1,text);read(text,*) topology
  call get_command_argument(2,text);read(text,*) periods
  call get_command_argument(3,transport)
  select case(trim(transport))
  case('pinned');mode=0
  case('device');mode=1
  case default;error stop 'critical history test transport must be pinned/device'
  end select
  if(product(topology)/=ranks) error stop 'critical history topology mismatch'
  coordinate=rank_coordinate(rank);origin=coordinate*cells;global=topology*cells
  lower=MPI_PROC_NULL;upper=MPI_PROC_NULL
  do axis=1,3
    candidate=coordinate;candidate(axis)=candidate(axis)-1
    if(candidate(axis)>=0.or.periods(axis)/=0) then
      candidate(axis)=modulo(candidate(axis),topology(axis));lower(axis)=rank_number(candidate)
    endif
    candidate=coordinate;candidate(axis)=candidate(axis)+1
    if(candidate(axis)<topology(axis).or.periods(axis)/=0) then
      candidate(axis)=modulo(candidate(axis),topology(axis));upper(axis)=rank_number(candidate)
    endif
  enddo
  allocate(seed(-halo:cells(1)+halo,-halo:cells(2)+halo,-halo:cells(3)+halo))
  allocate(actual,mold=seed);allocate(expected,mold=seed);allocate(first,mold=seed)
  allocate(history(-halo:cells(1)+halo,-halo:cells(2)+halo,-halo:cells(3)+halo))
  seed=1
  do k=0,cells(3)
  do j=0,cells(2)
  do i=0,cells(1)
    seed(i,j,k)=merge(1_c_int8_t,0_c_int8_t,seeded(rank,[i,j,k]))
  enddo
  enddo
  enddo
  expected=seed;physical_nodes=0;halo_nodes=0
  do k=-halo,cells(3)+halo
  do j=-halo,cells(2)+halo
  do i=-halo,cells(1)+halo
    p=[i,j,k];g=origin+p;outsides=count(p<0.or.p>cells)
    if(outsides>1) cycle
    allowed=.true.
    do axis=1,3
      if(periods(axis)/=0) then
        g(axis)=modulo(g(axis),global(axis))
      elseif(g(axis)<0.or.g(axis)>global(axis)) then
        allowed=.false.
      endif
    enddo
    if(.not.allowed) cycle
    value=.false.
    do peer=0,ranks-1
      peer_coordinate=rank_coordinate(peer)
      do sz=0,periods(3)
      do sy=0,periods(2)
      do sx=0,periods(1)
        candidate=g+[sx,sy,sz]*global-peer_coordinate*cells
        if(all(candidate>=0.and.candidate<=cells)) value=value.or.seeded(peer,candidate)
      enddo
      enddo
      enddo
    enddo
    expected(i,j,k)=merge(1_c_int8_t,0_c_int8_t,value)
    if(outsides==0) physical_nodes=physical_nodes+1
    if(outsides==1) halo_nodes=halo_nodes+1
  enddo
  enddo
  enddo
#if defined(REFERENCE_CPU) || defined(REFERENCE_CUDA)
  if(hm/=halo) error stop 'critical history reference halo width mismatch'
  im=cells(1);jm=cells(2);km=cells(3);ia=global(1);ja=global(2);ka=global(3)
  lihomo=periods(1)/=0;ljhomo=periods(2)/=0;lkhomo=periods(3)/=0
  mpirank=rank;mpisize=ranks;lio=rank==0;mpitag=100
  isize=topology(1);jsize=topology(2);ksize=topology(3)
  mpileft=lower(1);mpiright=upper(1);mpidown=lower(2);mpiup=upper(2);mpiback=lower(3);mpifront=upper(3)
#ifdef REFERENCE_CUDA
  allocate(crinod_d,mold=history)
#endif
#else
  allocate(fluid(0:cells(1),0:cells(2),0:cells(3)));fluid=1
  allocate(q(-halo:cells(1)+halo,-halo:cells(2)+halo,-halo:cells(3)+halo,5));allocate(observed_q,mold=q)
  do c=1,5
  do k=-halo,cells(3)+halo
  do j=-halo,cells(2)+halo
  do i=-halo,cells(1)+halo
    q(i,j,k,c)=real(10000*c+100*rank+3*i+5*j+7*k,real_kind)
    if(c==5) q(i,j,k,c)=q(i,j,k,c)+1.e6_real_kind
  enddo
  enddo
  enddo
  enddo
#endif
  do debug=0,1
#if defined(REFERENCE_CPU) || defined(REFERENCE_CUDA)
    history=seed/=0
#ifdef REFERENCE_CUDA
    crinod_d=history
#endif
#else
    call check(device_state_create(int(mod(local_rank,devices),c_int),int(cells(1),c_int),int(cells(2),c_int), &
      int(cells(3),c_int),halo,int(debug,c_int),state))
    call check(fluid_prepare(state,0_c_int))
    call check(device_transport_prepare(state,int(MPI_COMM_WORLD,c_int),int(lower,c_int),int(upper,c_int), &
      int(topology,c_int),6_c_int,mode,1_c_int,int(periods,c_int)))
    call check(crashfix_prepare_cpp(state,fluid,int(size(fluid),c_size_t)))
    call check(device_state_upload(state,device_q,q,int(size(q),c_size_t)*real_bytes))
    call check(crashfix_history_upload_cpp(state,seed,int(size(seed),c_size_t)))
#endif
    do pass=1,2
#ifdef REFERENCE_CPU
      call reconcile_crashfix_history(history);actual=merge(1_c_int8_t,0_c_int8_t,history)
#elif defined(REFERENCE_CUDA)
      call reconcile_crashfix_history_gpu();history=crinod_d;actual=merge(1_c_int8_t,0_c_int8_t,history)
#else
      call check(crashfix_history_exchange_cpp(state))
      call check(crashfix_history_download_cpp(state,actual,int(size(actual),c_size_t)))
#endif
      if(any(actual/=expected)) then
        write(*,'(a,3(1x,i0))') 'CRITICAL_HISTORY_DIFFERENCE rank/debug/pass',rank,debug,pass
        error stop 'shared union or hm halo differs from enumerated owners'
      endif
      if(pass==1) first=actual
      if(pass==2.and.any(actual/=first)) error stop 'history exchange is not idempotent'
    enddo
#if !defined(REFERENCE_CPU) && !defined(REFERENCE_CUDA)
    call check(device_state_download(state,device_q,observed_q,int(size(q),c_size_t)*real_bytes))
    if(any(real_to_word(q)/=real_to_word(observed_q))) &
      error stop 'history exchange changed q'
    call check(device_transport_metrics(state,metrics,0_c_int))
    sent_bytes=0
    do axis=1,3
      if(topology(axis)==1) cycle
      sent_bytes=sent_bytes+2.d0*(1+halo)*product(cells(pack([1,2,3],[1,2,3]/=axis))+1)* &
        (merge(1,0,lower(axis)/=MPI_PROC_NULL)+merge(1,0,upper(axis)/=MPI_PROC_NULL))
    enddo
    if(metrics(5)/=sent_bytes.or.metrics(6)/=merge(2.d0*sent_bytes,0.d0,mode==0)) &
      error stop 'critical history byte transport/staging totals differ'
    call check(device_state_destroy(state))
#endif
    print *, 'CRITICAL_HISTORY_PASS ',rank,debug,physical_nodes,halo_nodes
  enddo
#ifdef REFERENCE_CUDA
  call release_crashfix_gpu();deallocate(crinod_d)
#endif
  call MPI_Finalize(ierr)
contains
  logical function seeded(owner,point)
    integer,intent(in) :: owner,point(3)
    seeded=modulo(point(1)+3*point(2)+5*point(3)+7*owner,19)==0
  end function
  function rank_coordinate(owner) result(coordinate)
    integer,intent(in) :: owner
    integer :: coordinate(3)
    coordinate=[modulo(owner,topology(1)),modulo(owner/topology(1),topology(2)),owner/(topology(1)*topology(2))]
  end function
  integer function rank_number(coordinate)
    integer,intent(in) :: coordinate(3)
    rank_number=coordinate(1)+topology(1)*(coordinate(2)+topology(2)*coordinate(3))
  end function
  subroutine check(code)
    integer,intent(in) :: code
    if(code/=0) then
      print *, 'CRITICAL_HISTORY_STATUS ',rank,code
      call MPI_Abort(MPI_COMM_WORLD,1,ierr)
    endif
  end subroutine
end program
