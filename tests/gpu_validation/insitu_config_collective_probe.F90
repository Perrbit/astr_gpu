program insitu_config_collective_probe
  use mpi
  use insitu_run_config
  use insitu_config_collective
  implicit none
  type(insitu_options) :: options
  character(1024) :: prefix,filename,message
  integer :: rank,ierr
  logical :: ok
  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)
  call get_command_argument(1,prefix)
  write(filename,'(A,I0,A)') trim(prefix),rank,'.nml'
  options%output_directory='unchanged-on-failure'
  call read_insitu_options_collective(trim(filename),MPI_COMM_WORLD,options,ok,message)
  if(.not.ok) then
    if(options%output_directory/='unchanged-on-failure') call MPI_Abort(MPI_COMM_WORLD,2,ierr)
    write(*,'(A,I0,2A)') 'REJECT rank ',rank,': ',trim(message)
  else
    write(*,'(A,I0)') 'PASS collective rank ',rank
    if(options%slice_definition=='plane') write(*,'(A,I0,6(1X,ES25.17))') &
      'PLANE rank ',rank,options%slice_origin,options%slice_normal
  endif
  call MPI_Finalize(ierr)
  if(.not.ok) stop 1
end program
