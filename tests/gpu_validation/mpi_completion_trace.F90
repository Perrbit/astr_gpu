subroutine mpi_testall(count,requests,flag,statuses,ierr)
  use mpi, only: MPI_STATUS_SIZE,MPI_REQUEST_NULL,MPI_SUCCESS,PMPI_Testall
  use iso_c_binding, only: c_int
  implicit none
  integer,intent(in) :: count
  integer,intent(inout) :: requests(count)
  logical,intent(out) :: flag
  integer,intent(out) :: statuses(MPI_STATUS_SIZE,count),ierr
  logical :: active
  interface
    subroutine trace_completion(completed) bind(C,name='astr_trace_mpi_completion')
      import c_int
      integer(c_int),value :: completed
    end subroutine trace_completion
  end interface

  active=count==4 .and. any(requests/=MPI_REQUEST_NULL)
  call PMPI_Testall(count,requests,flag,statuses,ierr)
  if(active .and. ierr==MPI_SUCCESS) call trace_completion(merge(1_c_int,0_c_int,flag))
end subroutine mpi_testall
