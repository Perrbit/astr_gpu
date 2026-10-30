program insitu_run_config_probe
  use insitu_run_config
  implicit none
  type(insitu_options) :: options
  character(1024) :: filename,message,mode
  logical :: ok
  integer :: i
  call get_command_argument(1,filename)
  call get_command_argument(2,mode)
  call read_insitu_options(trim(filename),options,ok,message,native_products=trim(mode)=='catalog')
  if(.not.ok) then
    print *, trim(message)
    stop 1
  endif
  do i=options%view_count+1,insitu_max_views
    if(any(options%views(i)%camera/=0.d0).or.any(options%views(i)%focal/=0.d0).or. &
      any(options%views(i)%up/=0.d0).or.options%views(i)%scale/=0.d0.or. &
      any(options%views(i)%bounds/=0.d0).or.any(options%views(i)%range/=0.d0).or. &
      options%views(i)%width/=0.d0) then
      print *, 'FAIL: unused candidate values are not canonical zero',i
      stop 2
    endif
  enddo
  print *, 'PASS: parsed in-situ run configuration, enabled=',options%enabled
  print '(A)', 'rendering_pipeline='//trim(options%rendering_pipeline)
  print '(A)', 'streamline_seeds='//trim(options%streamline_seeds)
  print '(A,L1)', 'mean_streamline_render=',options%mean_streamline_render
  if(trim(mode)=='catalog') then
    print '(A,I0)', 'legacy_product_count=',options%product_count
    print '(A,I0,A,L1,A,L1)', 'legacy_clock=',options%step_interval, &
      ' initial=',options%initial_frame,' final=',options%final_frame
  endif
  if(options%view_count>0) then
    print '(A,I0)', 'preview_candidates=',options%view_count
    do i=1,options%view_count
      print '(A)', 'trajectory='//trim(streamline_trajectory_key(options%views(i)))
      print '(A)', 'display='//trim(options%views(i)%id)//':'//trim(streamline_display_key(options%views(i)))
    enddo
  endif
end program
