program insitu_run_config_probe
  use insitu_run_config
  implicit none
  type(insitu_options) :: options
  character(1024) :: filename,message
  logical :: ok
  call get_command_argument(1,filename)
  call read_insitu_options(trim(filename),options,ok,message)
  if(.not.ok) then
    print *, trim(message)
    stop 1
  endif
  print *, 'PASS: parsed in-situ run configuration, enabled=',options%enabled
  print '(A)', 'rendering_pipeline='//trim(options%rendering_pipeline)
  print '(A,L1)', 'mean_streamline_render=',options%mean_streamline_render
end program
