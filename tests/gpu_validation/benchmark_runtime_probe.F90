program benchmark_runtime_probe
  use mpi
  use benchmark_runtime, only: configure_benchmark_runtime,benchmark_field_io_disabled
  implicit none
  character(32) :: gpu_arg,flow_arg,homogeneous_arg,boundary_arg,expected_arg
  logical :: use_gpu,homogeneous,periodic_boundary,expected
  integer :: ierr,rank

  call mpi_init(ierr)
  call mpi_comm_rank(MPI_COMM_WORLD,rank,ierr)
  call get_command_argument(1,gpu_arg)
  call get_command_argument(2,flow_arg)
  call get_command_argument(3,homogeneous_arg)
  call get_command_argument(4,boundary_arg)
  call get_command_argument(5,expected_arg)
  use_gpu=trim(gpu_arg)=='gpu'
  homogeneous=trim(homogeneous_arg)=='periodic'
  periodic_boundary=trim(boundary_arg)=='periodic'
  expected=trim(expected_arg)=='enabled'
  call configure_benchmark_runtime(use_gpu,trim(flow_arg),3,homogeneous,homogeneous,homogeneous, &
                                   periodic_boundary)
  if(benchmark_field_io_disabled().neqv.expected) then
    write(*,'(A,I0)') 'benchmark runtime probe mismatch on rank ',rank
    call mpi_abort(MPI_COMM_WORLD,1,ierr)
  endif
  if(rank==0) write(*,'(A,L1)') 'benchmark runtime probe passed enabled=',expected
  call mpi_finalize(ierr)
end program benchmark_runtime_probe
