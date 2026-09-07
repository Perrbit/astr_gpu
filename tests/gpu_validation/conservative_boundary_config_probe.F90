program conservative_boundary_config_probe
  use conservative_boundary_config
  implicit none
  type(conservative_boundary_settings) :: settings
  character(len=1024) :: filename
  integer :: status
  call get_command_argument(1,filename)
  call read_conservative_boundary_config(trim(filename),settings,status)
  print*,status,settings%enabled
  if(status/=0) stop 1
  print '(es25.16)',settings%split_x
  print '(5es25.16)',settings%q_left
  print '(5es25.16)',settings%q_right
end program conservative_boundary_config_probe
