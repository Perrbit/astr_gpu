program mpi_completion_probe
  use mpi
  implicit none
  integer :: rank,nranks,ierr,requests(4),statuses(MPI_STATUS_SIZE,4),status(MPI_STATUS_SIZE)
  integer,asynchronous :: payload
  integer :: ack
  logical :: done
  call mpi_init(ierr)
  call mpi_comm_rank(MPI_COMM_WORLD,rank,ierr)
  call mpi_comm_size(MPI_COMM_WORLD,nranks,ierr)
  if(nranks/=2) call mpi_abort(MPI_COMM_WORLD,1,ierr)
  requests=MPI_REQUEST_NULL
  if(rank==0) then
    call mpi_recv(ack,1,MPI_INTEGER,1,21002,MPI_COMM_WORLD,status,ierr)
    payload=42
    call mpi_send(payload,1,MPI_INTEGER,1,21001,MPI_COMM_WORLD,ierr)
  else
    payload=-1
    call mpi_irecv(payload,1,MPI_INTEGER,0,21001,MPI_COMM_WORLD,requests(1),ierr)
    call mpi_testall(4,requests,done,statuses,ierr)
    if(done) call mpi_abort(MPI_COMM_WORLD,2,ierr)
    ! The sender cannot post its payload until this deliberately pending test.
    ack=1
    call mpi_send(ack,1,MPI_INTEGER,0,21002,MPI_COMM_WORLD,ierr)
    do
      call mpi_testall(4,requests,done,statuses,ierr)
      if(done) exit
    enddo
    if(payload/=42) call mpi_abort(MPI_COMM_WORLD,3,ierr)
    call mpi_testall(4,requests,done,statuses,ierr)
    if(.not.done) call mpi_abort(MPI_COMM_WORLD,4,ierr)
    print *, 'MPI completion probe passed: pending, completed, null-request test'
  endif
  call mpi_finalize(ierr)
end program mpi_completion_probe
