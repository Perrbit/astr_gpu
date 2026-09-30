module insitu_run_config
  use iso_fortran_env, only: int64,real64
  use ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: insitu_options,read_insitu_options
  type :: insitu_options
    logical :: enabled=.false.,statistics=.false.,render=.false.
    logical :: initial_frame=.false.,final_frame=.false.
    character(16) :: schedule_mode='steps'
    integer(int64) :: step_interval=0,host_budget_bytes=0,device_budget_bytes=0,device_reserve_bytes=0
    real(real64) :: time_interval=0,statistics_window(2)=0
    character(1024) :: implementation_path='',pipeline_file='',output_directory=''
    character(1024) :: batch_prefix='',restore_batch=''
  end type
contains
  subroutine read_insitu_options(filename,options,ok,message)
    character(*),intent(in) :: filename
    type(insitu_options),intent(inout) :: options
    logical,intent(out) :: ok
    character(*),intent(out) :: message
    logical :: enabled,statistics,render,initial_frame,final_frame
    character(16) :: schedule_mode
    integer(int64) :: step_interval,host_budget_bytes,device_budget_bytes,device_reserve_bytes
    real(real64) :: time_interval,statistics_window(2)
    character(1024) :: implementation_path,pipeline_file,output_directory,batch_prefix,restore_batch
    integer :: unit,status,closed
    namelist /insitu_run/ enabled,statistics,render,initial_frame,final_frame, &
      schedule_mode,step_interval,time_interval,statistics_window,host_budget_bytes, &
      device_budget_bytes,device_reserve_bytes,implementation_path,pipeline_file,output_directory,batch_prefix,restore_batch
    ok=.false.
    message=''
    enabled=.false.; statistics=.false.; render=.false.
    initial_frame=.false.; final_frame=.false.
    schedule_mode='steps'
    step_interval=0; time_interval=0; statistics_window=0
    host_budget_bytes=0; device_budget_bytes=0; device_reserve_bytes=0
    implementation_path=''; pipeline_file=''; output_directory=''
    batch_prefix=''; restore_batch=''
    open(newunit=unit,file=filename,status='old',action='read',iostat=status,iomsg=message)
    if(status/=0) return
    read(unit,nml=insitu_run,iostat=status,iomsg=message)
    close(unit,iostat=closed)
    if(status/=0) return
    message='cannot close in-situ configuration'
    if(closed/=0) return
    message='nonfinite statistics window or render interval'
    if(.not.all(ieee_is_finite([time_interval,statistics_window]))) return
    message='resource budgets and device reserve must not be negative'
    if(min(host_budget_bytes,device_budget_bytes,device_reserve_bytes)<0) return
    message='in-situ path or mode exceeds supported character capacity'
    if(len_trim(implementation_path)>=len(implementation_path).or. &
       len_trim(pipeline_file)>=len(pipeline_file).or. &
       len_trim(output_directory)>=len(output_directory).or. &
       len_trim(batch_prefix)>1000.or.len_trim(restore_batch)>1000.or. &
       len_trim(schedule_mode)>=len(schedule_mode)) return
    if(len_trim(batch_prefix)>0.or.len_trim(restore_batch)>0) then
      message='paired batches require enabled statistics'
      if(.not.enabled.or..not.statistics) return
    endif
    if(enabled) then
      message='enabled in-situ configuration requires statistics or rendering'
      if(.not.statistics.and..not.render) return
      message='enabled in-situ configuration requires output_directory and positive host budget'
      if(len_trim(output_directory)==0.or.host_budget_bytes<=0) return
    endif
    if(statistics) then
      message='statistics_window must have strictly increasing endpoints'
      if(statistics_window(2)<=statistics_window(1)) return
    endif
    if(render) then
      message='rendering requires implementation_path and pipeline_file'
      if(len_trim(implementation_path)==0.or.len_trim(pipeline_file)==0) return
      message='render schedule requires exactly one positive steps/time interval'
      select case(trim(schedule_mode))
      case('steps')
        if(step_interval<=0.or.time_interval/=0) return
      case('time')
        if(time_interval<=0.or.step_interval/=0) return
      case default
        return
      end select
    endif
    options%enabled=enabled; options%statistics=statistics; options%render=render
    options%initial_frame=initial_frame; options%final_frame=final_frame
    options%schedule_mode=schedule_mode; options%step_interval=step_interval
    options%time_interval=time_interval; options%statistics_window=statistics_window
    options%host_budget_bytes=host_budget_bytes; options%device_budget_bytes=device_budget_bytes
    options%device_reserve_bytes=device_reserve_bytes
    options%implementation_path=implementation_path; options%pipeline_file=pipeline_file
    options%output_directory=output_directory
    options%batch_prefix=batch_prefix; options%restore_batch=restore_batch
    ok=.true.
    message=''
  end subroutine
end module
