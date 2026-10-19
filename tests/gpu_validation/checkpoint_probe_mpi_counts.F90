module checkpoint_probe_mpi_counts
  implicit none
  integer,save :: allreduce_calls=0
end module

subroutine mpi_allreduce(send,receive,count,datatype,operation,communicator,status)
  use mpi, only: PMPI_Allreduce
  use checkpoint_probe_mpi_counts, only: allreduce_calls
  implicit none
  integer :: send(*),receive(*),count,datatype,operation,communicator,status
  allreduce_calls=allreduce_calls+1
  call PMPI_Allreduce(send,receive,count,datatype,operation,communicator,status)
end subroutine
