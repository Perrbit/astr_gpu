program me6opti_halo_probe
  use astr_precision, only: real_kind
  use me6opti_halo, only: extend_me6_edges_cpu,release_me6_edges_cpu
  use mpi
#ifdef TEST_CUDA
  use cudafor
  use me6opti_halo_gpu, only: extend_me6_edges_gpu,release_me6_edges_gpu
  use halo_transport_gpu, only: begin_transport_setup,end_transport_setup
#endif
  implicit none
  integer,parameter :: hm=5,global_last(3)=[17,19,13]
  integer :: parts(3),coords(3),last(3),offset(3),peers(2,3),cart,rank,nranks,ierr
  integer :: i,j,k,c,axis,sample,ghosts,nfail,allfail
  real(real_kind),allocatable :: v(:,:,:,:),t(:,:,:),initial_v(:,:,:,:),initial_t(:,:,:)
  real(real_kind) :: expected
  logical :: periodic(3)
  character(32) :: arg
#ifdef TEST_CUDA
  real(real_kind),allocatable,device :: vd(:,:,:,:),td(:,:,:)
  integer :: devices,local_rank,code
  logical :: pinned
  call get_environment_variable('OMPI_COMM_WORLD_LOCAL_RANK',arg)
  read(arg,*) local_rank
  code=cudaGetDeviceCount(devices)
  if(code/=cudaSuccess.or.devices<1) error stop 'ME6 halo probe requires a GPU'
  code=cudaSetDevice(mod(local_rank,devices))
  if(code/=cudaSuccess) error stop 'ME6 halo device binding failed'
#endif
  call mpi_init(ierr)
  call mpi_comm_rank(MPI_COMM_WORLD,rank,ierr)
  call mpi_comm_size(MPI_COMM_WORLD,nranks,ierr)
  do axis=1,3
    call get_command_argument(axis,arg)
    read(arg,*) parts(axis)
  enddo
  if(product(parts)/=nranks) error stop 'ME6 halo probe rank count mismatch'
  periodic=.true.
  call get_command_argument(4,arg)
  if(trim(arg)=='wall') periodic(2)=.false.
  call mpi_cart_create(MPI_COMM_WORLD,3,parts,periodic,.false.,cart,ierr)
  call mpi_cart_coords(cart,rank,3,coords,ierr)
  do axis=1,3
    call mpi_cart_shift(cart,axis-1,1,peers(1,axis),peers(2,axis),ierr)
    offset(axis)=coords(axis)*global_last(axis)/parts(axis)
    last(axis)=(coords(axis)+1)*global_last(axis)/parts(axis)-offset(axis)
  enddo
  if(any(last<hm)) error stop 'ME6 halo probe needs hm interior nodes'
  allocate(v(-hm:last(1)+hm,-hm:last(2)+hm,-hm:last(3)+hm,3), &
    t(-hm:last(1)+hm,-hm:last(2)+hm,-hm:last(3)+hm))
  allocate(initial_v,mold=v);allocate(initial_t,mold=t)
#ifdef TEST_CUDA
  allocate(vd(-hm:last(1)+hm,-hm:last(2)+hm,-hm:last(3)+hm,3), &
    td(-hm:last(1)+hm,-hm:last(2)+hm,-hm:last(3)+hm))
  call begin_transport_setup(pinned)
  call end_transport_setup(pinned)
#endif
  do sample=1,2
    v=-huge(1.e0_real_kind);t=-huge(1.e0_real_kind)
    ! Only physical nodes and the six faces are initially valid, not edges/corners.
    do k=-hm,last(3)+hm
    do j=-hm,last(2)+hm
    do i=-hm,last(1)+hm
      ghosts=count([i<0.or.i>last(1),j<0.or.j>last(2),k<0.or.k>last(3)])
      if(ghosts>1) cycle
      if(.not.periodic(2).and.(offset(2)+j<0.or.offset(2)+j>global_last(2))) cycle
      do c=1,3
        v(i,j,k,c)=node_value(i,j,k,c,sample)
      enddo
      t(i,j,k)=node_value(i,j,k,4,sample)
    enddo
    enddo
    enddo
    initial_v=v;initial_t=t
    call extend_me6_edges_cpu(last(1),last(2),last(3),hm,v,t,parts,peers,periodic)
    call check_values('CPU')
#ifdef TEST_CUDA
    vd=initial_v;td=initial_t
    call extend_me6_edges_gpu(last(1),last(2),last(3),hm,vd,td,parts,peers,periodic)
    v=vd;t=td
    call check_values('GPU')
#endif
  enddo
  call release_me6_edges_cpu()
#ifdef TEST_CUDA
  call release_me6_edges_gpu()
  deallocate(vd,td)
#endif
  if(rank==0) print '(A,3I3)', 'ME6_EXTENDED_HALO_PASS physical/faces/edges/corners topology=',parts
  call mpi_comm_free(cart,ierr)
  call mpi_finalize(ierr)
contains
  function node_value(ii,jj,kk,component,epoch) result(value)
    integer,intent(in) :: ii,jj,kk,component,epoch
    real(real_kind) :: value
    integer :: g(3)
    g=modulo(offset+[ii,jj,kk],global_last)
    if(.not.periodic(2)) g(2)=offset(2)+jj
    value=real(g(1)+100*g(2)+10000*g(3)+1000000*component+10000000*epoch,real_kind)
  end function
  subroutine check_values(label)
    character(*),intent(in) :: label
    nfail=0
    do k=-hm,last(3)+hm
    do j=-hm,last(2)+hm
    do i=-hm,last(1)+hm
      do c=1,4
        expected=node_value(i,j,k,c,sample)
        if(.not.periodic(2).and.(offset(2)+j<0.or.offset(2)+j>global_last(2))) expected=-huge(1.e0_real_kind)
        if(c<=3) then
          if(v(i,j,k,c)/=expected) nfail=nfail+1
        else
          if(t(i,j,k)/=expected) nfail=nfail+1
        endif
      enddo
    enddo
    enddo
    enddo
    if(any(v(0:last(1),0:last(2),0:last(3),:)/=initial_v(0:last(1),0:last(2),0:last(3),:))) nfail=nfail+1
    if(any(t(0:last(1),0:last(2),0:last(3))/=initial_t(0:last(1),0:last(2),0:last(3)))) nfail=nfail+1
    call mpi_allreduce(nfail,allfail,1,MPI_INTEGER,MPI_SUM,MPI_COMM_WORLD,ierr)
    if(allfail/=0) then
      print *, 'ME6 halo mismatch backend,rank,sample,count=',label,rank,sample,nfail
      call mpi_abort(MPI_COMM_WORLD,1,ierr)
    endif
  end subroutine
end program
