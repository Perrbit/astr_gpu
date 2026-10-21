program native_precision_mpi_probe
  use mpi
  use astr_precision, only: real_kind,real_bytes
  use astr_precision_mpi, only: mpi_real_type
  implicit none
  real(real_kind) :: send(4),receive(4),total
  integer :: rank,ranks,left,right,bytes,ierr,failed,any_failed

  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)
  call MPI_Comm_size(MPI_COMM_WORLD,ranks,ierr)
  call MPI_Type_size(mpi_real_type,bytes,ierr)
  failed=merge(0,1,bytes==real_bytes)
  left=modulo(rank-1,ranks)
  right=modulo(rank+1,ranks)
  send=real([rank,rank+1,rank+2,rank+3],kind=real_kind)
  call MPI_Sendrecv(send,4,mpi_real_type,right,0,receive,4,mpi_real_type,left,0, &
    MPI_COMM_WORLD,MPI_STATUS_IGNORE,ierr)
  if(any(receive/=real([left,left+1,left+2,left+3],kind=real_kind))) failed=1
  call MPI_Allreduce(send(1),total,1,mpi_real_type,MPI_SUM,MPI_COMM_WORLD,ierr)
  if(total/=real(ranks*(ranks-1)/2,kind=real_kind)) failed=1
  call MPI_Allreduce(failed,any_failed,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
  if(any_failed/=0) call MPI_Abort(MPI_COMM_WORLD,1,ierr)
  if(rank==0) write(*,'(A,I0,A,I0)') 'PASS: numerical MPI bytes=',bytes,' ranks=',ranks
  call MPI_Finalize(ierr)
end program native_precision_mpi_probe
