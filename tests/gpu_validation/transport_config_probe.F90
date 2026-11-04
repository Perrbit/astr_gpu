program transport_config_probe
  use mpi
  use astr_precision, only: real_kind
  use case_model_options, only: reset_model_options,read_model_group,publish_model_options
  use perfect_gas_transport, only: read_transport_environment
  implicit none
  real(real_kind) :: pr,suth
  integer :: status,ierr,unit
  logical :: changed
  character(1024) :: path
  character(512) :: message
  call MPI_Init(ierr)
  write(*,'(A)') 'TRANSPORT_CONFIG_BEGIN'
  call reset_model_options()
  call get_command_argument(1,path)
  if(len_trim(path)>0) then
    open(newunit=unit,file=trim(path),status='old')
    call read_model_group(unit,'transport_options',status,message)
    close(unit)
    if(status/=0) call MPI_Abort(MPI_COMM_WORLD,1,ierr)
  endif
  call publish_model_options(MPI_COMM_WORLD)
  call read_transport_environment(pr,suth,changed,status)
  if(status/=0) call MPI_Abort(MPI_COMM_WORLD,1,ierr)
  write(*,*) 'TRANSPORT_CONFIG ',pr,suth,changed
  call MPI_Finalize(ierr)
end program transport_config_probe
