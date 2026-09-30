program insitu_runtime_probe
  use mpi
  use insitu_runtime, only: insitu_check
  implicit none
  integer :: ierr
  call MPI_Init(ierr)
  call insitu_check()
  call MPI_Finalize(ierr)
end program
