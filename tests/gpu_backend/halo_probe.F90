program halo_probe
  use iso_c_binding
  use mpi
  use device_state
  use halo_transport
  implicit none
  integer,parameter :: dims_local(3)=[9,7,6]
  integer :: ierr,rank,local_rank,status,axis,hm,debug,a,cart,nranks,iteration,side
  integer :: topology(3),coordinates(3),low(3),high(3)
  logical :: periodic(3)
  integer(c_int) :: transport_mode
  character(40) :: value
  type(c_ptr) :: state
  type(device_view) :: before,after
  real(c_double),allocatable :: initial(:,:,:,:),expected(:,:,:,:),actual(:,:,:,:)
  integer :: i,j,k,c
  call get_environment_variable('OMPI_COMM_WORLD_LOCAL_RANK',value,status=status)
  if(status/=0) error stop 'halo probe requires Open MPI local rank'
  read(value,*) local_rank
  status=device_bind(int(local_rank,c_int))
  if(status/=0) error stop 'halo probe pre-MPI device binding'
  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)
  call MPI_Comm_size(MPI_COMM_WORLD,nranks,ierr)
  call get_command_argument(1,value)
  read(value,*) axis
  if(axis<0.or.axis>2) call fail('axis must be 0, 1 or 2')
  call get_command_argument(2,value)
  select case(trim(value))
  case('','pinned')
    transport_mode=halo_pinned
  case('device')
    transport_mode=halo_device
  case default
    call fail('transport must be pinned or device')
  end select
  topology=1;topology(axis+1)=nranks;periodic=.true.
  call MPI_Cart_create(MPI_COMM_WORLD,3,topology,periodic,.false.,cart,ierr)
  call check(ierr,'Cartesian communicator')
  call MPI_Cart_coords(cart,rank,3,coordinates,ierr)
  call check(ierr,'Cartesian coordinates')
  do a=1,3
    call MPI_Cart_shift(cart,a-1,1,low(a),high(a),ierr)
    call check(ierr,'Cartesian neighbors')
  enddo
  do hm=3,5,2
  do debug=0,1
    allocate(initial(-hm:9+hm,-hm:7+hm,-hm:6+hm,5), &
      expected(-hm:9+hm,-hm:7+hm,-hm:6+hm,5),actual(-hm:9+hm,-hm:7+hm,-hm:6+hm,5))
    do c=1,5
    do k=-hm,6+hm
    do j=-hm,7+hm
    do i=-hm,9+hm
      initial(i,j,k,c)=8192.d0*c+1024.d0*rank+8.d0*i+.5d0*j+.03125d0*k
    enddo
    enddo
    enddo
    enddo
    expected=initial
    call check(device_state_create(int(local_rank,c_int),9_c_int,7_c_int,6_c_int,int(hm,c_int), &
      int(debug,c_int),state),'create state')
    call check(device_transport_prepare(state,int(cart,c_int),int(low,c_int),int(high,c_int), &
      int(topology,c_int),5_c_int,transport_mode,0_c_int),'prepare fixed halo transport')
    call check(device_state_view(state,device_q,before),'initial q view')
    call check(device_state_upload(state,device_q,initial,int(size(initial),c_size_t)*8_c_size_t),'upload')
    do iteration=1,2
      do a=0,2
        if(iteration==1) then
          call check(device_exchange_begin(state,device_q,int(a,c_int),1_c_int),'begin halo exchange')
          if(device_exchange_begin(state,device_q,int(a,c_int),1_c_int)==0) call fail('nested begin accepted')
          if(device_exchange_end(state,device_q,int(a,c_int),0_c_int)==0) call fail('mismatched end accepted')
          if(device_state_destroy(state)==0) call fail('in-flight transport destroyed')
          call check(device_exchange_end(state,device_q,int(a,c_int),1_c_int),'end halo exchange')
          if(device_exchange_end(state,device_q,int(a,c_int),1_c_int)==0) call fail('duplicate end accepted')
        else
          call check(device_exchange_axis(state,device_q,int(a,c_int),0_c_int),'halo-only exchange')
        endif
        call reference_exchange(a+1,iteration==1)
        call check(device_state_download(state,device_q,actual,int(size(actual),c_size_t)*8_c_size_t),'download')
        if(any(transfer(actual,[0_c_int64_t],size(actual))/=transfer(expected,[0_c_int64_t],size(expected)))) &
          call fail('physical/shared/halo/corner bits differ')
      enddo
    enddo
    call check(device_state_view(state,device_q,after),'final q view')
    if(.not.c_associated(before%data,after%data)) call fail('q buffer replaced during exchanges')
    call check(device_state_destroy(state),'destroy before MPI finalize')
    write(*,'(a,4(i0,1x),a)') 'HALO PASS rank/axis/hm/debug=',rank,axis,hm,debug, &
      ' all phase bits exact; q pointer stable; begin/end lifecycle checked'
    deallocate(initial,expected,actual)
  enddo
  enddo
  call MPI_Comm_free(cart,ierr)
  call check(ierr,'release Cartesian communicator')
  call MPI_Finalize(ierr)
contains
  subroutine fail(message)
    character(*),intent(in) :: message
    write(*,'(a,i0,2a)') 'HALO FAIL rank=',rank,': ',message
    call MPI_Abort(MPI_COMM_WORLD,1,ierr)
    error stop 1
  end subroutine
  subroutine check(code,message)
    integer,intent(in) :: code
    character(*),intent(in) :: message
    if(code/=0) then
      write(*,'(a,i0)') 'status=',code
      call fail(message)
    endif
  end subroutine
  subroutine reference_exchange(direction,reconcile)
    integer,intent(in) :: direction
    logical,intent(in) :: reconcile
    integer :: shape(3),count,mpi_status(MPI_STATUS_SIZE)
    real(c_double),allocatable :: sl(:,:,:,:),sr(:,:,:,:),rl(:,:,:,:),rr(:,:,:,:)
    shape=dims_local;shape(direction)=hm
    allocate(sl(0:shape(1),0:shape(2),0:shape(3),5),sr(0:shape(1),0:shape(2),0:shape(3),5), &
      rl(0:shape(1),0:shape(2),0:shape(3),5),rr(0:shape(1),0:shape(2),0:shape(3),5))
    select case(direction)
    case(1)
      sl=expected(0:hm,0:7,0:6,:);sr=expected(9-hm:9,0:7,0:6,:)
    case(2)
      sl=expected(0:9,0:hm,0:6,:);sr=expected(0:9,7-hm:7,0:6,:)
    case(3)
      sl=expected(0:9,0:7,0:hm,:);sr=expected(0:9,0:7,6-hm:6,:)
    end select
    count=size(sl)
    call MPI_Sendrecv(sl,count,MPI_DOUBLE_PRECISION,low(direction),901, &
      rr,count,MPI_DOUBLE_PRECISION,high(direction),901,cart,mpi_status,ierr)
    call check(ierr,'Fortran reference low packet')
    call MPI_Sendrecv(sr,count,MPI_DOUBLE_PRECISION,high(direction),902, &
      rl,count,MPI_DOUBLE_PRECISION,low(direction),902,cart,mpi_status,ierr)
    call check(ierr,'Fortran reference high packet')
    select case(direction)
    case(1)
      expected(-hm:-1,0:7,0:6,:)=rl(0:hm-1,:,:,:)
      expected(10:9+hm,0:7,0:6,:)=rr(1:hm,:,:,:)
      if(reconcile) then
        expected(0,0:7,0:6,:)=.5d0*(expected(0,0:7,0:6,:)+rl(hm,:,:,:))
        expected(9,0:7,0:6,:)=.5d0*(expected(9,0:7,0:6,:)+rr(0,:,:,:))
      endif
    case(2)
      expected(0:9,-hm:-1,0:6,:)=rl(:,0:hm-1,:,:)
      expected(0:9,8:7+hm,0:6,:)=rr(:,1:hm,:,:)
      if(reconcile) then
        expected(0:9,0,0:6,:)=.5d0*(expected(0:9,0,0:6,:)+rl(:,hm,:,:))
        expected(0:9,7,0:6,:)=.5d0*(expected(0:9,7,0:6,:)+rr(:,0,:,:))
      endif
    case(3)
      expected(0:9,0:7,-hm:-1,:)=rl(:,:,0:hm-1,:)
      expected(0:9,0:7,7:6+hm,:)=rr(:,:,1:hm,:)
      if(reconcile) then
        expected(0:9,0:7,0,:)=.5d0*(expected(0:9,0:7,0,:)+rl(:,:,hm,:))
        expected(0:9,0:7,6,:)=.5d0*(expected(0:9,0:7,6,:)+rr(:,:,0,:))
      endif
    end select
    deallocate(sl,sr,rl,rr)
  end subroutine
end program
