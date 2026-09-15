program benchmark_runtime_probe
  use mpi
  use benchmark_runtime, only: configure_benchmark_runtime,benchmark_field_io_disabled
  implicit none
  character(32) :: gpu_arg,flow_arg,periodic_arg,expected_arg
  logical :: use_gpu,periodic,expected
  integer :: ierr,rank

  call mpi_init(ierr)
  call mpi_comm_rank(MPI_COMM_WORLD,rank,ierr)
  call get_command_argument(1,gpu_arg)
  call get_command_argument(2,flow_arg)
  call get_command_argument(3,periodic_arg)
  call get_command_argument(4,expected_arg)
  use_gpu=trim(gpu_arg)=='gpu'
  periodic=trim(periodic_arg)=='periodic'
  expected=trim(expected_arg)=='enabled'
  call configure_benchmark_runtime(use_gpu,trim(flow_arg),3,periodic,periodic,periodic)
  if(benchmark_field_io_disabled().neqv.expected) then
    write(*,'(A,I0)') 'benchmark runtime probe mismatch on rank ',rank
    call mpi_abort(MPI_COMM_WORLD,1,ierr)
  endif
  if(rank==0) write(*,'(A,L1)') 'benchmark runtime probe passed enabled=',expected
  call mpi_finalize(ierr)
end program benchmark_runtime_probe
