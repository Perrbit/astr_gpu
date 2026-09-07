program transport_config_probe
  use perfect_gas_transport, only: read_transport_environment
  implicit none
  real(8) :: pr,suth
  integer :: status
  logical :: changed
  call read_transport_environment(pr,suth,changed,status)
  if(status/=0) stop 1
  write(*,*)pr,suth,changed
end program transport_config_probe
