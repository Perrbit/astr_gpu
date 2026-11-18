program owned_vector_probe
  use iso_c_binding
  use mpi
  use device_state
  use fluid_explicit, only: fluid_prepare
  use insitu_moments_cpp
  use insitu_sampling_cpp
  implicit none
  integer,parameter :: global(3)=[16,12,8],hm=5
  type(c_ptr) :: parent,moments
  type(sample_plan) :: sample
  type(device_view) :: source
  integer :: rank,ranks,local_rank,ierr,status,cart,dims(3),coords(3),extent(3),origin(3),neighbors(2,3), &
    i,j,k,a,kind,debug,g(3),checks=0
  integer(c_int) :: mode
  real(c_double),allocatable :: packed(:,:,:,:),after(:,:,:,:),halo(:,:,:,:),q(:,:,:,:),saved_q(:,:,:,:)
  real(c_double) :: error=0,expected(3)
  integer(c_int64_t) :: usage(9)
  character(64) :: argument,transport
  call get_environment_variable('OMPI_COMM_WORLD_LOCAL_RANK',argument,status=status)
  if(status/=0) error stop 'Open MPI local rank required'
  read(argument,*) local_rank
  call check(device_bind(int(local_rank,c_int)),'bind before MPI')
  call MPI_Init(ierr);call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr);call MPI_Comm_size(MPI_COMM_WORLD,ranks,ierr)
  call get_command_argument(1,argument);read(argument,*) dims
  if(any(dims<1).or.product(dims)/=ranks.or.any(mod(global,dims)/=0)) error stop 'owned vector topology'
  call get_command_argument(2,transport)
  select case(trim(transport))
  case('pinned');mode=0
  case('device-aware');mode=1
  case default;error stop 'owned vector transport'
  end select
  call MPI_Cart_create(MPI_COMM_WORLD,3,dims,[.true.,.true.,.true.],.false.,cart,ierr)
  call MPI_Cart_coords(cart,rank,3,coords,ierr)
  do a=1,3
    call MPI_Cart_shift(cart,a-1,1,neighbors(1,a),neighbors(2,a),ierr)
    if(dims(a)==1) neighbors(:,a)=MPI_PROC_NULL
  enddo
  extent=global/dims;origin=coords*extent
  allocate(packed(extent(1),extent(2),extent(3),34),after(extent(1),extent(2),extent(3),34), &
    halo(-3:extent(1)+3,-3:extent(2)+3,-3:extent(3)+3,3), &
    q(-hm:extent(1)+hm,-hm:extent(2)+hm,-hm:extent(3)+hm,5), &
    saved_q(-hm:extent(1)+hm,-hm:extent(2)+hm,-hm:extent(3)+hm,5))
  packed=0;packed(:,:,:,2)=1;packed(:,:,:,3)=.5d0;packed(:,:,:,4)=2
  packed(:,:,:,8)=.5d0;packed(:,:,:,9)=1;packed(:,:,:,34)=1
  do k=1,extent(3)
  do j=1,extent(2)
  do i=1,extent(1)
    g=origin+[i,j,k]-1
    packed(i,j,k,10:12)=real(g,c_double)/1024
    packed(i,j,k,13:15)=2+real(g,c_double)/512
  enddo
  enddo
  enddo
  do debug=0,1
    call check(device_state_create(int(local_rank,c_int),int(extent(1),c_int),int(extent(2),c_int), &
      int(extent(3),c_int),hm,int(debug,c_int),parent),'create parent')
    call check(fluid_prepare(parent,0_c_int),'prepare independent flow buffers')
    q=77
    call check(device_state_upload(parent,device_q,q,bytes(size(q))),'independent q')
    call check(moments_create(parent,int(extent,c_int),[0.d0,1.d0],moments),'create saved moments')
    call check(moments_restore(moments,packed,bytes(size(packed)),.5d0),'restore manufactured saved moments')
    call check(moments_view(moments,source),'borrow unique-node statistics')
    call check(sample_create(parent,cart,int(neighbors(1,:),c_int),int(neighbors(2,:),c_int), &
      int(dims,c_int),[1_c_int,1_c_int,1_c_int],mode,.false.,.true.,0_c_int,.false.,sample),'create mean halo')
    do kind=0,1
      call check(sample_capture_owned_vector(sample%handle,source,int(extent,c_int), &
        int([6,7,8]+3*kind,c_int)),'complete saved-mean owners and halo')
      call check(sample_metrics(sample%handle,usage),'capture transfer receipts')
      if(usage(8)/=int(kind,c_int64_t)*bytes(size(halo))) error stop 'mean capture downloaded source array'
      if(mode==1.and.any(usage(5:6)/=0)) error stop 'device mean capture staged halo on host'
      call check(sample_download(sample%handle,sample_halo,halo,bytes(size(halo))),'observe completed mean halo')
      do k=-3,extent(3)+3
      do j=-3,extent(2)+3
      do i=-3,extent(1)+3
        g=modulo(origin+[i,j,k],global)
        expected=2*kind+real(g,c_double)/real(1024/(kind+1),c_double)
        error=max(error,maxval(abs(halo(i,j,k,:)-expected)))
      enddo
      enddo
      enddo
      if(error/=0) error stop 'unique owner or periodic mean halo differs'
      checks=checks+1
    enddo
    call check(moments_pack(moments,after,bytes(size(after))),'saved statistics after capture')
    if(any(transfer(packed,[0_c_int64_t],size(packed))/=transfer(after,[0_c_int64_t],size(after)))) &
      error stop 'capture changed saved statistics'
    call check(device_state_download(parent,device_q,saved_q,bytes(size(q))),'parent q isolation')
    if(any(saved_q/=q)) error stop 'mean capture changed parent q'
    call check(sample_destroy(sample),'release mean halo')
    call check(moments_destroy(moments),'release saved moments')
    call check(device_state_destroy(parent),'release parent')
  enddo
  write(*,'(a,2(1x,i0),1x,es24.16e3)') 'CPP_OWNED_VECTOR_PASS',rank,checks,error
  call MPI_Comm_free(cart,ierr);call MPI_Finalize(ierr)
contains
  integer(c_size_t) function bytes(n)
    integer,intent(in) :: n
    bytes=8_c_size_t*n
  end function
  subroutine check(code,label)
    integer(c_int),intent(in) :: code
    character(*),intent(in) :: label
    if(code==0) return
    print *,trim(label),code
    call MPI_Abort(MPI_COMM_WORLD,1,ierr)
  end subroutine
end program
