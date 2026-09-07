program halo_transport_test
  use mpi
  use cudafor
  use iso_c_binding, only: c_loc,c_size_t
  use halo_transport_gpu, only: exchange_host_pair,begin_transport_setup,end_transport_setup
  implicit none
  real(8),allocatable,target :: send_low(:,:,:,:),send_high(:,:,:,:)
  real(8),allocatable,target :: recv_low(:,:,:,:),recv_high(:,:,:,:)
  integer :: rank,nranks,ierr,low,high,width,nvar,n,mode,v,j
  integer,parameter :: variables(3)=[1,3,6]
  integer :: devices,work_calls
  integer(c_size_t) :: bytes
  character(32) :: backend,work_mode
  logical :: pinned,selected_pinned

  call mpi_init(ierr)
  call mpi_comm_rank(MPI_COMM_WORLD,rank,ierr)
  call mpi_comm_size(MPI_COMM_WORLD,nranks,ierr)
  mode=0
  width=0
  nvar=0
  call get_command_argument(1,backend)
  call get_command_argument(2,work_mode)
  call require(len_trim(work_mode)==0 .or. trim(work_mode)=='work')
  work_calls=0
  pinned=trim(backend)=='pinned'
  call require(len_trim(backend)==0 .or. trim(backend)=='pageable' .or. pinned)
  if(pinned) then
    call require(cudaGetDeviceCount(devices)==cudaSuccess)
    call require(devices>0)
    call require(cudaSetDevice(mod(rank,devices))==cudaSuccess)
  endif
  call begin_transport_setup(selected_pinned)
  call require(.not.selected_pinned .or. pinned)
  call end_transport_setup(selected_pinned)
  do mode=1,2
    low=mod(rank+nranks-1,nranks)
    high=mod(rank+1,nranks)
    if(mode==2) then
      if(rank==0) low=MPI_PROC_NULL
      if(rank==nranks-1) high=MPI_PROC_NULL
    endif
    do width=5,6
      do v=1,3
        nvar=variables(v)
        allocate(send_low(width,3,2,nvar),send_high(width,3,2,nvar), &
                 recv_low(width,3,2,nvar),recv_high(width,3,2,nvar))
        n=size(send_low)
        if(pinned) then
          bytes=int(n,c_size_t)*8_c_size_t
          call require(cudaHostRegister(c_loc(send_low),bytes,0)==cudaSuccess)
          call require(cudaHostRegister(c_loc(send_high),bytes,0)==cudaSuccess)
          call require(cudaHostRegister(c_loc(recv_low),bytes,0)==cudaSuccess)
          call require(cudaHostRegister(c_loc(recv_high),bytes,0)==cudaSuccess)
        endif
        send_low=reshape([(real(10000*rank+1000+j,8),j=1,n)],shape(send_low))
        send_high=reshape([(real(10000*rank+2000+j,8),j=1,n)],shape(send_high))
        recv_low=-999.d0
        recv_high=-999.d0
        if(trim(work_mode)=='work') then
          call exchange_host_pair(send_low,send_high,recv_high,recv_low, &
                                  n,low,high,21001,21002,independent_work)
        else
          call exchange_host_pair(send_low,send_high,recv_high,recv_low, &
                                  n,low,high,21001,21002)
        endif
        if(low==MPI_PROC_NULL) then
          call require(all(recv_low==-999.d0))
        else
          call require(all(recv_low== &
            reshape([(real(10000*low+2000+j,8),j=1,n)],shape(recv_low))))
        endif
        if(high==MPI_PROC_NULL) then
          call require(all(recv_high==-999.d0))
        else
          call require(all(recv_high== &
            reshape([(real(10000*high+1000+j,8),j=1,n)],shape(recv_high))))
        endif
        if(pinned) then
          call require(cudaHostUnregister(c_loc(send_low))==cudaSuccess)
          call require(cudaHostUnregister(c_loc(send_high))==cudaSuccess)
          call require(cudaHostUnregister(c_loc(recv_low))==cudaSuccess)
          call require(cudaHostUnregister(c_loc(recv_high))==cudaSuccess)
        endif
        deallocate(send_low,send_high,recv_low,recv_high)
      enddo
    enddo
  enddo
  if(trim(work_mode)=='work') call require(work_calls==12)
  if(rank==0) print *, 'halo transport exact-value tests passed; ranks=',nranks,' pinned=',pinned, &
                      ' work_calls=',work_calls
  call mpi_finalize(ierr)
contains
  subroutine independent_work(requests)
    integer,intent(inout) :: requests(4)
    integer :: status(MPI_STATUS_SIZE,4),code
    logical :: done
    work_calls=work_calls+1
    call mpi_testall(4,requests,done,status,code)
    call require(code==MPI_SUCCESS)
  end subroutine independent_work

  subroutine require(condition)
    logical,intent(in) :: condition
    if(condition) return
    print *, 'halo transport mismatch: rank, mode, width, nvar=',rank,mode,width,nvar
    call mpi_abort(MPI_COMM_WORLD,1,ierr)
  end subroutine require
end program halo_transport_test
