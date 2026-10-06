module insitu_run_config
  use iso_fortran_env, only: int64,real64
  use ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: insitu_options,read_insitu_options
  type :: insitu_options
    logical :: enabled=.false.,statistics=.false.,render=.false.
    logical :: initial_frame=.false.,final_frame=.false.
    logical :: air5_volume_statistics=.false.
    logical :: air5_volume_reduction=.false.
    logical :: wall_mean_render=.false.
    logical :: wall_separation=.false.
    character(16) :: schedule_mode='steps'
    character(16) :: derivative_backend='cpu'
    character(16) :: processing_backend='host'
    character(16) :: rendering_pipeline='compatible'
    character(16) :: postprocess_transport=''
    character(16) :: products='all'
    character(16) :: slice_axis='z'
    integer :: slice_index=4
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
    logical :: enabled,statistics,render,initial_frame,final_frame,air5_volume_statistics,air5_volume_reduction,wall_mean_render
    logical :: wall_separation
    character(16) :: schedule_mode,derivative_backend,products,slice_axis,processing_backend,postprocess_transport
    character(16) :: rendering_pipeline
    integer(int64) :: step_interval,host_budget_bytes,device_budget_bytes,device_reserve_bytes
    real(real64) :: time_interval,statistics_window(2)
    character(1024) :: implementation_path,pipeline_file,output_directory,batch_prefix,restore_batch
    integer :: unit,status,closed,slice_index
    namelist /insitu_run/ enabled,statistics,render,initial_frame,final_frame, &
      schedule_mode,step_interval,time_interval,statistics_window,host_budget_bytes, &
      device_budget_bytes,device_reserve_bytes,implementation_path,pipeline_file,output_directory,batch_prefix,restore_batch, &
      derivative_backend,products,slice_axis,slice_index,air5_volume_statistics,air5_volume_reduction,wall_mean_render,wall_separation, &
      processing_backend,postprocess_transport,rendering_pipeline
    ok=.false.
    message=''
    enabled=.false.; statistics=.false.; render=.false.
    initial_frame=.false.; final_frame=.false.
    air5_volume_statistics=.false.
    air5_volume_reduction=.false.
    wall_mean_render=.false.
    wall_separation=.false.
    schedule_mode='steps'
    derivative_backend='cpu'
    processing_backend='host'; postprocess_transport=''
    rendering_pipeline=''
    products='all'
    slice_axis='z'; slice_index=4
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
    message='derivative_backend must be cpu or gpu'
    if(derivative_backend/='cpu'.and.derivative_backend/='gpu') return
    message='processing_backend must be host or device'
    if(processing_backend/='host'.and.processing_backend/='device') return
    if(rendering_pipeline=='') then
      rendering_pipeline='compatible'
      if(enabled.and.render.and.processing_backend=='device') rendering_pipeline='standard-device'
    endif
    message='rendering_pipeline must be compatible, standard-device or direct-device'
    if(rendering_pipeline/='compatible'.and.rendering_pipeline/='standard-device'.and. &
      rendering_pipeline/='direct-device') return
    message='device rendering pipelines require enabled device processing and GPU derivatives'
    if(rendering_pipeline/='compatible'.and.(.not.enabled.or..not.render.or. &
      processing_backend/='device'.or.derivative_backend/='gpu')) return
    if(processing_backend=='device') then
      message='device processing requires explicit postprocess_transport=device-aware or pinned'
      if(postprocess_transport/='device-aware'.and.postprocess_transport/='pinned') return
      message='device processing requires enabled GPU rendering and positive device budget'
      if(.not.enabled.or..not.render.or.derivative_backend/='gpu'.or.device_budget_bytes<=0) return
      message='device processing does not support wall products'
      if(products=='channel_walls'.or.products=='air5_walls') return
    else
      message='postprocess_transport is only valid with processing_backend=device'
      if(postprocess_transport/='') return
    endif
    message='invalid products selection'
    if(products/='all'.and.products/='q_surface'.and.products/='streamlines'.and. &
      products/='q_streamlines'.and.products/='velocity_slice'.and.products/='channel_walls'.and. &
      products/='air5_walls'.and.products/='tgv256_demo') return
    message='tgv256_demo requires device rendering without statistics'
    if(products=='tgv256_demo'.and.(processing_backend/='device'.or.statistics)) return
    message='air5_volume_statistics requires enabled statistics and products=air5_walls'
    if(air5_volume_statistics.and.(.not.enabled.or..not.statistics.or.products/='air5_walls')) return
    message='air5_volume_reduction requires air5_volume_statistics'
    if(air5_volume_reduction.and..not.air5_volume_statistics) return
    message='wall_mean_render requires enabled statistics, rendering and a wall product'
    if(wall_mean_render.and.(.not.enabled.or..not.statistics.or..not.render.or. &
      (products/='channel_walls'.and.products/='air5_walls'))) return
    message='wall_separation requires enabled AIR5 wall statistics'
    if(wall_separation.and.(.not.enabled.or..not.statistics.or.products/='air5_walls')) return
    message='slice_axis must be x, y or z and slice_index must not be negative'
    if((slice_axis/='x'.and.slice_axis/='y'.and.slice_axis/='z').or.slice_index<0) return
    message='nondefault slice settings require products=velocity_slice'
    if(products/='velocity_slice'.and.(slice_axis/='z'.or.slice_index/=4)) return
    message='selected products require derivative_backend=gpu'
    if(enabled.and.render.and.products/='all'.and.products/='channel_walls'.and. &
      products/='air5_walls'.and.derivative_backend/='gpu') return
    message='wall products require CPU wall diagnostics'
    if(enabled.and.(products=='channel_walls'.or.products=='air5_walls').and.derivative_backend/='cpu') return
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
    options%air5_volume_statistics=air5_volume_statistics
    options%air5_volume_reduction=air5_volume_reduction
    options%wall_mean_render=wall_mean_render
    options%wall_separation=wall_separation
    options%schedule_mode=schedule_mode; options%step_interval=step_interval
    options%derivative_backend=derivative_backend
    options%processing_backend=processing_backend
    options%rendering_pipeline=rendering_pipeline
    options%postprocess_transport=postprocess_transport
    options%products=products
    options%slice_axis=slice_axis; options%slice_index=slice_index
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
