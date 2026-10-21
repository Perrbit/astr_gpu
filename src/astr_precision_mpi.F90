module astr_precision_mpi
  use mpi, only: MPI_REAL4,MPI_REAL8
  use astr_precision, only: real_bytes
  implicit none
  private
  public :: mpi_real_type
  integer,parameter :: mpi_real_type=merge(MPI_REAL4,MPI_REAL8,real_bytes==4)
end module astr_precision_mpi
