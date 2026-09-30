program output_config_probe
  use iso_fortran_env, only: int64
  use output_config
  implicit none
  type(output_options) :: options
  character(1024) :: filename,message
  logical :: ok
  call get_command_argument(1,filename)
  options%format_version=77
  call read_output_options(trim(filename),options,ok,message)
  if(.not.ok) then
    if(options%format_version/=77) error stop 'failed parse modified live options'
    print '(A)', 'REJECT unchanged: '//trim(message)
    stop 1
  endif
  call validate_output_grid(options,[16_int64,24_int64,32_int64],ok,message)
  if(.not.ok) then
    print '(A)', 'REJECT grid: '//trim(message)
    stop 1
  endif
  print '(A,I0)', 'format_version=',options%format_version
  print '(A,I0)', 'keep=',options%keep
  print '(A,L1)', 'checkpoint_final=',options%checkpoint%final_frame
  print '(A,3(I0,1X))', 'slice_counts=',count(options%i_indices>=0), &
    count(options%j_indices>=0),count(options%k_indices>=0)
  print '(A,I0)', 'checkpoint_interval_steps=',options%checkpoint%interval_steps
  print '(A)', 'PASS: output configuration'
end program
