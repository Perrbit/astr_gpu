module insitu_run_config
  use iso_fortran_env, only: int64,real64
  use ieee_arithmetic, only: ieee_is_finite
  use adaptive_output, only: adaptive_binding,adaptive_capacity,validate_adaptive_binding,canonical_adaptive_binding
  implicit none
  private
  public :: insitu_options,read_insitu_options,insitu_max_products,product_scene_allowed
  integer,parameter :: insitu_max_products=64
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
    integer :: product_count=0
    character(64) :: product_ids(insitu_max_products)=''
    character(16) :: product_modes(insitu_max_products)=''
    integer(int64) :: product_steps(insitu_max_products)=0
    real(real64) :: product_times(insitu_max_products)=0.d0
    type(adaptive_binding) :: product_adaptive(insitu_max_products)
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
    character(64) :: product_ids(insitu_max_products),temporary_id
    character(16) :: product_modes(insitu_max_products),temporary_mode
    integer(int64) :: product_steps(insitu_max_products),temporary_steps
    real(real64) :: product_times(insitu_max_products),temporary_time
    logical :: product_adaptive(insitu_max_products)
    integer(int64) :: product_dense_steps(insitu_max_products)
    real(real64) :: product_dense_times(insitu_max_products)
    character(64) :: product_events(adaptive_capacity,insitu_max_products), &
      product_windows(adaptive_capacity,insitu_max_products),temporary_associations(adaptive_capacity)
    type(adaptive_binding) :: binding
    logical :: temporary_adaptive,valid_binding
    integer(int64) :: step_interval,host_budget_bytes,device_budget_bytes,device_reserve_bytes
    real(real64) :: time_interval,statistics_window(2)
    character(1024) :: implementation_path,pipeline_file,output_directory,batch_prefix,restore_batch
    integer :: unit,status,closed,slice_index,n,i,j,dot
    namelist /insitu_run/ enabled,statistics,render,initial_frame,final_frame, &
      schedule_mode,step_interval,time_interval,statistics_window,host_budget_bytes, &
      device_budget_bytes,device_reserve_bytes,implementation_path,pipeline_file,output_directory,batch_prefix,restore_batch, &
      derivative_backend,products,slice_axis,slice_index,air5_volume_statistics,air5_volume_reduction,wall_mean_render,wall_separation, &
      processing_backend,postprocess_transport,rendering_pipeline,product_ids,product_modes,product_steps,product_times, &
      product_adaptive,product_dense_steps,product_dense_times,product_events,product_windows
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
    product_ids=''; product_modes=''; product_steps=0; product_times=0.d0
    product_adaptive=.false.; product_dense_steps=0; product_dense_times=0; product_events=''; product_windows=''
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
    n=count(product_ids/='')
    message='product entries must be contiguous enabled render products, without legacy pairs'
    if(n>0) then
      if(.not.enabled.or..not.render.or.any(product_ids(1:n)=='').or. &
        len_trim(batch_prefix)>0.or.len_trim(restore_batch)>0) return
    endif
    if(any(product_modes(n+1:)/='').or.any(product_steps(n+1:)/=0).or.any(product_times(n+1:)/=0.d0)) return
    if(any(product_adaptive(n+1:)).or.any(product_dense_steps(n+1:)/=0).or.any(product_dense_times(n+1:)/=0).or. &
      any(product_events(:,n+1:)/='').or.any(product_windows(:,n+1:)/='')) return
    do i=1,n
      dot=index(trim(product_ids(i)),'.',back=.true.)
      message='product id must be an admitted scene followed by .image or .geometry'
      if(dot<2) return
      if(.not.product_scene_allowed(product_ids(i)(:dot-1),products,statistics)) return
      select case(trim(product_ids(i)(dot+1:)))
      case('image')
      case('geometry')
        message='geometry products require compatible rendering and are disabled for tgv256_demo'
        if(rendering_pipeline/='compatible'.or.products=='tgv256_demo') return
      case default
        return
      end select
      binding%enabled=product_adaptive(i); binding%dense_steps=product_dense_steps(i)
      binding%dense_time=product_dense_times(i); binding%events=product_events(:,i); binding%windows=product_windows(:,i)
      call validate_adaptive_binding(binding,product_modes(i),product_steps(i),product_times(i),ok=valid_binding)
      message='invalid adaptive product binding or dense interval'
      if(.not.valid_binding) return
      message='product clock requires steps with positive steps only, or time with finite positive time only'
      if(.not.ieee_is_finite(product_times(i))) return
      select case(product_modes(i))
      case('steps')
        if(product_steps(i)<=0.or.product_times(i)/=0.d0) return
      case('time')
        if(product_steps(i)/=0.or.product_times(i)<=0.d0) return
      case default
        return
      end select
    enddo
    ! Canonical identity is independent of namelist entry order.
    do i=2,n
      j=i
      do while(j>1)
        if(product_ids(j)>=product_ids(j-1)) exit
        temporary_id=product_ids(j); product_ids(j)=product_ids(j-1); product_ids(j-1)=temporary_id
        temporary_mode=product_modes(j); product_modes(j)=product_modes(j-1); product_modes(j-1)=temporary_mode
        temporary_steps=product_steps(j); product_steps(j)=product_steps(j-1); product_steps(j-1)=temporary_steps
        temporary_time=product_times(j); product_times(j)=product_times(j-1); product_times(j-1)=temporary_time
        temporary_adaptive=product_adaptive(j)
        product_adaptive(j)=product_adaptive(j-1); product_adaptive(j-1)=temporary_adaptive
        temporary_steps=product_dense_steps(j)
        product_dense_steps(j)=product_dense_steps(j-1); product_dense_steps(j-1)=temporary_steps
        temporary_time=product_dense_times(j)
        product_dense_times(j)=product_dense_times(j-1); product_dense_times(j-1)=temporary_time
        temporary_associations=product_events(:,j)
        product_events(:,j)=product_events(:,j-1); product_events(:,j-1)=temporary_associations
        temporary_associations=product_windows(:,j)
        product_windows(:,j)=product_windows(:,j-1); product_windows(:,j-1)=temporary_associations
        j=j-1
      enddo
    enddo
    message='duplicate in-situ product id'
    do i=2,n
      if(product_ids(i)==product_ids(i-1)) return
    enddo
    if(render) then
      message='rendering requires implementation_path and pipeline_file'
      if(len_trim(implementation_path)==0.or.len_trim(pipeline_file)==0) return
      message='render schedule requires exactly one positive steps/time interval'
      if(n==0) then
      select case(trim(schedule_mode))
      case('steps')
        if(step_interval<=0.or.time_interval/=0) return
      case('time')
        if(time_interval<=0.or.step_interval/=0) return
      case default
        return
      end select
      else
        message='independent product clocks cannot be combined with a common interval'
        if(step_interval/=0.or.time_interval/=0.d0) return
      endif
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
    options%product_count=n; options%product_ids=product_ids; options%product_modes=product_modes
    options%product_steps=product_steps; options%product_times=product_times
    do i=1,insitu_max_products
      options%product_adaptive(i)%enabled=product_adaptive(i)
      options%product_adaptive(i)%dense_steps=product_dense_steps(i)
      options%product_adaptive(i)%dense_time=product_dense_times(i)
      options%product_adaptive(i)%events=product_events(:,i); options%product_adaptive(i)%windows=product_windows(:,i)
      call canonical_adaptive_binding(options%product_adaptive(i))
    enddo
    ok=.true.
    message=''
  end subroutine
  logical function product_scene_allowed(scene,profile,statistics) result(allowed)
    character(*),intent(in) :: scene,profile
    logical,intent(in) :: statistics
    allowed=.false.
    ! PF keeps the approved periodic TGV product boundary. Wall presets retain their common clock.
    select case(scene)
    case('q_surface')
      allowed=profile=='all'.or.profile=='q_surface'.or.profile=='q_streamlines'.or.profile=='tgv256_demo'
    case('velocity_slice')
      allowed=profile=='all'.or.profile=='velocity_slice'
    case('instantaneous_streamlines')
      allowed=profile=='all'.or.profile=='streamlines'.or.profile=='q_streamlines'.or.profile=='tgv256_demo'
    case('crossing_streamlines')
      allowed=profile=='all'.or.profile=='streamlines'.or.profile=='q_streamlines'
    case('mean_reynolds_streamlines','mean_favre_streamlines')
      allowed=profile=='all'.and.statistics
    end select
  end function
end module
