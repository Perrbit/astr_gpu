module benchmark_runtime
  use mpi
  implicit none
  private
  public :: configure_benchmark_runtime,benchmark_field_io_disabled
  logical,save :: configured=.false.,field_io_disabled=.false.
contains
  integer function switch_choice(name)
    character(*),intent(in) :: name
    character(32) :: value
    integer :: length,status
    value=''
    call get_environment_variable(name,value,length=length,status=status)
    switch_choice=0
    if(status==1 .or. length==0) return
    if(status/=0) then
      switch_choice=-1
      return
    endif
    select case(trim(adjustl(value)))
    case('1','t','T','true','TRUE','on','ON')
      switch_choice=1
    case('0','f','F','false','FALSE','off','OFF')
      switch_choice=0
    case default
      switch_choice=-1
    end select
  end function switch_choice

  subroutine configure_benchmark_runtime(use_gpu,flowtype,ndims,lihomo,ljhomo,lkhomo)
    logical,intent(in) :: use_gpu,lihomo,ljhomo,lkhomo
    character(*),intent(in) :: flowtype
    integer,intent(in) :: ndims
    integer :: requested,rk_timing,lowest,highest,rk_lowest,rk_highest
    integer :: ierr,rank,ignored
#ifdef _CUDA
    logical,parameter :: cuda_build=.true.
#else
    logical,parameter :: cuda_build=.false.
#endif
    if(configured) return
    requested=switch_choice('ASTR_GPU_BENCHMARK_NO_FIELD_IO')
    rk_timing=switch_choice('ASTR_GPU_RK_TIMING')
    call mpi_allreduce(requested,lowest,1,MPI_INTEGER,MPI_MIN,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call mpi_abort(MPI_COMM_WORLD,ierr,ignored)
    call mpi_allreduce(requested,highest,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call mpi_abort(MPI_COMM_WORLD,ierr,ignored)
    call mpi_allreduce(rk_timing,rk_lowest,1,MPI_INTEGER,MPI_MIN,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call mpi_abort(MPI_COMM_WORLD,ierr,ignored)
    call mpi_allreduce(rk_timing,rk_highest,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call mpi_abort(MPI_COMM_WORLD,ierr,ignored)
    if(lowest<0 .or. lowest/=highest) then
      print *, 'Invalid or inconsistent GPU benchmark environment'
      call mpi_abort(MPI_COMM_WORLD,1,ignored)
    endif
    if(lowest==1 .and. (rk_lowest<0 .or. rk_lowest/=rk_highest)) then
      print *, 'Invalid or inconsistent ASTR_GPU_RK_TIMING for no-field-I/O benchmark'
      call mpi_abort(MPI_COMM_WORLD,1,ignored)
    endif
    field_io_disabled=requested==1
    if(field_io_disabled .and. (.not.cuda_build .or. .not.use_gpu .or. &
       ndims/=3 .or. trim(flowtype)/='tgv' .or. &
       .not.(lihomo.and.ljhomo.and.lkhomo) .or. &
       rk_timing/=1)) then
      print *, 'ASTR_GPU_BENCHMARK_NO_FIELD_IO requires CUDA 3-D periodic GPU TGV with RK timing'
      call mpi_abort(MPI_COMM_WORLD,1,ignored)
    endif
    configured=.true.
    if(field_io_disabled) then
      call mpi_comm_rank(MPI_COMM_WORLD,rank,ierr)
      if(ierr/=MPI_SUCCESS) call mpi_abort(MPI_COMM_WORLD,ierr,ignored)
      if(rank==0) write(*,'(A)') 'ASTR_GPU_BENCHMARK_NO_FIELD_IO enabled'
    endif
  end subroutine configure_benchmark_runtime

  logical function benchmark_field_io_disabled()
    benchmark_field_io_disabled=configured.and.field_io_disabled
  end function benchmark_field_io_disabled
end module benchmark_runtime
