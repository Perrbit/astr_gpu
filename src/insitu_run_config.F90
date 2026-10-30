module insitu_run_config
  use iso_fortran_env, only: int64,real64
  use astr_precision, only: real_kind
  use ieee_arithmetic, only: ieee_is_finite
  use adaptive_output, only: adaptive_binding,adaptive_capacity,validate_adaptive_binding,canonical_adaptive_binding
  implicit none
  private
  public :: insitu_options,read_insitu_options,insitu_max_products,product_scene_allowed,canonical_slice_plane
  public :: streamline_view,insitu_max_views,read_streamline_views,streamline_trajectory_key,streamline_display_key
  public :: catalog_product,catalog_field_names,catalog_field_id,catalog_definition_equal
  public :: catalog_render,pack_catalog_render,unpack_catalog_render
  character(32),parameter :: catalog_field_names(27)=[character(32) :: &
    'density','velocity_x','velocity_y','velocity_z','pressure','temperature', &
    'vibrational_temperature','mass_fraction_N2','mass_fraction_O2','mass_fraction_N', &
    'mass_fraction_O','mass_fraction_NO', &
    'velocity_gradient_xx','velocity_gradient_yx','velocity_gradient_zx', &
    'velocity_gradient_xy','velocity_gradient_yy','velocity_gradient_zy', &
    'velocity_gradient_xz','velocity_gradient_yz','velocity_gradient_zz', &
    'Q_rs','velocity_divergence','vorticity_x','vorticity_y','vorticity_z','density_gradient_magnitude']
  type :: catalog_render
    character(32) :: kind='slice',channel='data',format='hdf5-xdmf',sampling='grid_index',field=''
    character(32) :: projection='',color_map='',seed_layout='',integrator='',direction='',side=''
    integer :: image_size(2)=0,jpeg_quality=0,dpi=0,formats=0,maximum_steps=0
    logical :: lighting=.false.
    real(real64) :: origin(3)=0,normal(3)=0,bounds(6)=0,camera(3)=0,focal(3)=0,up(3)=0
    ! The selected-quantity range is a color range for geometry, a y range for wall charts.
    real(real64) :: scale=0,color_range(2)=0,line_width=0,isovalue=0,integration(5)=0
  end type
  type :: catalog_product
    character(:),allocatable :: id,source,statistics_source,record,error
    character(:),allocatable :: event_ids(:),window_ids(:)
    integer,allocatable :: fields(:)
    integer :: line=0,axis=0
    integer(int64) :: index=0,ranges(2,3)=0,enable_request=0,interval_steps=0,dense_interval_steps=0
    real(real64) :: interval_time=0,dense_interval_time=0
    logical :: valid=.false.
    type(catalog_render) :: view
  end type
  integer,parameter :: insitu_max_products=64
  integer,parameter :: insitu_max_views=6
  ! An explicit -1 quota resolves to a nonallocating, overflow-safe sentinel.
  integer(int64),parameter,public :: unlimited_budget_bytes=huge(0_int64)/4096_int64
  type :: streamline_view
    character(64) :: id=''
    character(16) :: velocity='',seeds='',color=''
    real(real_kind) :: camera(3)=0.e0_real_kind,focal(3)=0.e0_real_kind,up(3)=0.e0_real_kind,scale=0.e0_real_kind
    real(real_kind) :: bounds(6)=0.e0_real_kind,range(2)=0.e0_real_kind,width=0.e0_real_kind
  end type
  type :: insitu_options
    logical :: enabled=.false.,statistics=.false.,render=.false.
    logical :: initial_frame=.false.,final_frame=.false.
    logical :: air5_volume_statistics=.false.
    logical :: air5_volume_reduction=.false.
    logical :: wall_mean_render=.false.
    logical :: mean_streamline_render=.false.
    logical :: wall_separation=.false.
    character(16) :: schedule_mode='steps'
    character(16) :: derivative_backend='cpu'
    character(16) :: processing_backend='host'
    character(16) :: rendering_pipeline='compatible'
    character(16) :: postprocess_transport=''
    character(16) :: products='all'
    character(16) :: streamline_seeds='line16'
    character(16) :: slice_axis='z'
    character(16) :: slice_definition='index'
    real(real_kind) :: slice_origin(3)=0.e0_real_kind,slice_normal(3)=0.e0_real_kind
    integer :: slice_index=4
    integer(int64) :: step_interval=0,host_budget_bytes=0,device_budget_bytes=0,device_reserve_bytes=0
    real(real_kind) :: time_interval=0,statistics_window(2)=0
    character(1024) :: implementation_path='',pipeline_file='',output_directory=''
    character(1024) :: batch_prefix='',restore_batch=''
    character(1024) :: streamline_views_file=''
    integer :: view_count=0
    type(streamline_view) :: views(insitu_max_views)
    integer :: product_count=0
    character(64) :: product_ids(insitu_max_products)=''
    character(16) :: product_modes(insitu_max_products)=''
    integer(int64) :: product_steps(insitu_max_products)=0
    real(real_kind) :: product_times(insitu_max_products)=0.e0_real_kind
    type(adaptive_binding) :: product_adaptive(insitu_max_products)
  end type
contains
  subroutine pack_catalog_render(view,labels,integers,reals)
    type(catalog_render),intent(in) :: view
    character(32),intent(out) :: labels(11)
    integer,intent(out) :: integers(7)
    real(real64),intent(out) :: reals(31)
    labels=[view%kind,view%channel,view%format,view%sampling,view%field,view%projection,view%color_map, &
      view%seed_layout,view%integrator,view%direction,view%side]
    integers=[view%image_size,view%jpeg_quality,view%dpi,view%formats,merge(1,0,view%lighting),view%maximum_steps]
    reals=[view%origin,view%normal,view%bounds,view%camera,view%focal,view%up,view%scale, &
      view%color_range,view%line_width,view%isovalue,view%integration]
  end subroutine

  subroutine unpack_catalog_render(view,labels,integers,reals)
    type(catalog_render),intent(out) :: view
    character(32),intent(in) :: labels(11)
    integer,intent(in) :: integers(7)
    real(real64),intent(in) :: reals(31)
    view%kind=labels(1);view%channel=labels(2);view%format=labels(3);view%sampling=labels(4);view%field=labels(5)
    view%projection=labels(6);view%color_map=labels(7);view%seed_layout=labels(8)
    view%integrator=labels(9);view%direction=labels(10);view%side=labels(11)
    view%image_size=integers(1:2);view%jpeg_quality=integers(3);view%dpi=integers(4)
    view%formats=integers(5);view%lighting=integers(6)==1;view%maximum_steps=integers(7)
    view%origin=reals(1:3);view%normal=reals(4:6);view%bounds=reals(7:12)
    view%camera=reals(13:15);view%focal=reals(16:18);view%up=reals(19:21);view%scale=reals(22)
    view%color_range=reals(23:24);view%line_width=reals(25);view%isovalue=reals(26);view%integration=reals(27:31)
  end subroutine

  logical function catalog_definition_equal(a,b) result(same)
    type(catalog_product),intent(in) :: a,b
    character(32) :: labels_a(11),labels_b(11)
    integer :: integers_a(7),integers_b(7)
    real(real64) :: reals_a(31),reals_b(31)
    same=.false.
    if(a%id/=b%id.or.a%source/=b%source.or.a%statistics_source/=b%statistics_source.or. &
      a%axis/=b%axis.or.a%index/=b%index.or.any(a%ranges/=b%ranges)) return
    if(size(a%fields)/=size(b%fields)) return
    if(any(a%fields/=b%fields)) return
    call pack_catalog_render(a%view,labels_a,integers_a,reals_a)
    call pack_catalog_render(b%view,labels_b,integers_b,reals_b)
    same=all(labels_a==labels_b).and.all(integers_a==integers_b).and.all(reals_a==reals_b)
  end function

  integer function catalog_field_id(name) result(id)
    character(*),intent(in) :: name
    integer :: i
    id=0
    do i=1,size(catalog_field_names)
      if(name/=trim(catalog_field_names(i))) cycle
      id=i
      return
    enddo
  end function

  subroutine read_insitu_options(filename,options,ok,message,native_products)
    character(*),intent(in) :: filename
    type(insitu_options),intent(inout) :: options
    logical,intent(out) :: ok
    character(*),intent(out) :: message
    logical,optional,intent(in) :: native_products
    logical :: catalog
    logical :: enabled,statistics,render,initial_frame,final_frame,air5_volume_statistics,air5_volume_reduction,wall_mean_render
    logical :: wall_separation,mean_streamline_render
    character(16) :: schedule_mode,derivative_backend,products,slice_axis,processing_backend,postprocess_transport
    character(16) :: rendering_pipeline,slice_definition,streamline_seeds
    real(real_kind) :: slice_origin(3),slice_normal(3)
    character(64) :: product_ids(insitu_max_products),temporary_id
    character(16) :: product_modes(insitu_max_products),temporary_mode
    integer(int64) :: product_steps(insitu_max_products),temporary_steps
    real(real_kind) :: product_times(insitu_max_products),temporary_time
    logical :: product_adaptive(insitu_max_products)
    integer(int64) :: product_dense_steps(insitu_max_products)
    real(real_kind) :: product_dense_times(insitu_max_products)
    character(64) :: product_events(adaptive_capacity,insitu_max_products), &
      product_windows(adaptive_capacity,insitu_max_products),temporary_associations(adaptive_capacity)
    type(adaptive_binding) :: binding
    logical :: temporary_adaptive,valid_binding
    integer(int64) :: step_interval,host_budget_bytes,device_budget_bytes,device_reserve_bytes
    real(real_kind) :: time_interval,statistics_window(2)
    character(1024) :: implementation_path,pipeline_file,output_directory,batch_prefix,restore_batch,streamline_views_file
    type(streamline_view) :: views(insitu_max_views)
    integer :: view_count
    logical :: views_ok,views_have_means
    integer :: unit,status,closed,slice_index,n,i,j,dot
    namelist /insitu_run/ enabled,statistics,render,initial_frame,final_frame, &
      schedule_mode,step_interval,time_interval,statistics_window,host_budget_bytes, &
      device_budget_bytes,device_reserve_bytes,implementation_path,pipeline_file,output_directory,batch_prefix,restore_batch, &
      derivative_backend,products,slice_axis,slice_index,air5_volume_statistics,air5_volume_reduction,wall_mean_render,wall_separation, &
      mean_streamline_render,streamline_seeds,streamline_views_file, &
      processing_backend,postprocess_transport,rendering_pipeline,slice_definition,slice_origin,slice_normal, &
      product_ids,product_modes,product_steps,product_times, &
      product_adaptive,product_dense_steps,product_dense_times,product_events,product_windows
    ok=.false.
    message=''
    enabled=.false.; statistics=.false.; render=.false.
    catalog=.false.
    if(present(native_products)) catalog=native_products
    initial_frame=.false.; final_frame=.false.
    air5_volume_statistics=.false.
    air5_volume_reduction=.false.
    wall_mean_render=.false.
    mean_streamline_render=.false.
    wall_separation=.false.
    schedule_mode='steps'
    derivative_backend='cpu'
    processing_backend='host'; postprocess_transport=''
    rendering_pipeline=''
    products='all'
    streamline_seeds='line16'
    slice_axis='z'; slice_index=4
    slice_definition='index'; slice_origin=0.e0_real_kind; slice_normal=0.e0_real_kind
    step_interval=0; time_interval=0; statistics_window=0
    host_budget_bytes=0; device_budget_bytes=0; device_reserve_bytes=0
    implementation_path=''; pipeline_file=''; output_directory=''
    batch_prefix=''; restore_batch=''
    streamline_views_file=''; view_count=0
    call clear_streamline_views(views)
    product_ids=''; product_modes=''; product_steps=0; product_times=0.e0_real_kind
    product_adaptive=.false.; product_dense_steps=0; product_dense_times=0; product_events=''; product_windows=''
    open(newunit=unit,file=filename,status='old',action='read',iostat=status,iomsg=message)
    if(status/=0) return
    read(unit,nml=insitu_run,iostat=status,iomsg=message)
    close(unit,iostat=closed)
    if(status/=0) return
    message='cannot close in-situ configuration'
    if(closed/=0) return
    if(catalog) then
      ! Inert legacy control-envelope values; only the runtime catalog schedules products.
      schedule_mode='steps';step_interval=1;time_interval=0.e0_real_kind
      initial_frame=.false.;final_frame=.false.
      product_ids='';product_modes='';product_steps=0;product_times=0.e0_real_kind
      product_adaptive=.false.;product_dense_steps=0;product_dense_times=0.e0_real_kind
      product_events='';product_windows=''
      slice_axis='z';slice_index=4
      slice_definition='index';slice_origin=0.e0_real_kind;slice_normal=0.e0_real_kind
      streamline_seeds='line16';streamline_views_file=''
      wall_mean_render=.false.;mean_streamline_render=.false.
      ! These fields only identify the legacy provider envelope, never product geometry.
      select case(trim(products))
      case('boundary_layer')
        streamline_seeds='bl-layered64';slice_definition='plane'
        slice_origin=[0.e0_real_kind,0.e0_real_kind,45.e0_real_kind]
        slice_normal=[0.e0_real_kind,0.e0_real_kind,1.e0_real_kind]
      case('m12_diagnostics');streamline_seeds='bl-layered256'
      case('sbli_diagnostics');streamline_seeds='sbli-layered16'
      end select
    endif
    if(products=='sbli_2d') then
      if(host_budget_bytes==-1_int64) host_budget_bytes=unlimited_budget_bytes
      if(device_budget_bytes==-1_int64) device_budget_bytes=unlimited_budget_bytes
    endif
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
      message='device walls require standard-device or direct-device, without a compatible fallback'
      if((products=='channel_walls'.or.products=='air5_walls').and.rendering_pipeline=='compatible'.and..not.catalog) return
      message='resident wall separation reduction is not yet admitted'
      if(products=='channel_walls'.and.wall_separation) return
      message='resident AIR5 volume statistics are not yet admitted'
      if(products=='air5_walls'.and.air5_volume_statistics) return
    else
      message='postprocess_transport is only valid with processing_backend=device'
      if(postprocess_transport/='') return
    endif
    message='invalid products selection'
    if(products/='all'.and.products/='q_surface'.and.products/='streamlines'.and. &
      products/='q_streamlines'.and.products/='velocity_slice'.and.products/='channel_walls'.and. &
      products/='air5_walls'.and.products/='tgv256_demo'.and.products/='curve_demo'.and.products/='boundary_layer'.and. &
      products/='m12_diagnostics'.and.products/='sbli_diagnostics'.and.products/='sbli_2d') return
    message='unsupported streamline seed layout'
    if(streamline_seeds/='line16'.and.streamline_seeds/='tgv-stratified'.and.streamline_seeds/='bl-layered64'.and. &
      streamline_seeds/='bl-layered256'.and.streamline_seeds/='sbli-layered16') return
    if(streamline_seeds=='tgv-stratified') then
      message='tgv-stratified seeds require enabled strict device streamline rendering'
      if(.not.enabled.or..not.render.or.processing_backend/='device'.or.rendering_pipeline=='compatible'.or. &
        (products/='all'.and.products/='streamlines'.and.products/='q_streamlines'.and. &
         products/='tgv256_demo'.and.products/='curve_demo')) return
    endif
    message='tgv256_demo requires device rendering without statistics'
    if(products=='tgv256_demo'.and.(processing_backend/='device'.or.statistics)) return
    message='curve_demo requires strict device rendering without statistics or independent clocks'
    if(products=='curve_demo'.and.(processing_backend/='device'.or.statistics.or..not.render.or. &
      rendering_pipeline=='compatible'.or.any(product_ids/=''))) return
    message='boundary_layer requires the approved strict render-only preset and common clock'
    if(products=='boundary_layer') then
      if(.not.enabled.or..not.render.or.statistics.or.processing_backend/='device'.or. &
        rendering_pipeline/='standard-device'.or.derivative_backend/='gpu'.or. &
        postprocess_transport/='pinned'.or.streamline_seeds/='bl-layered64'.or. &
        slice_definition/='plane'.or.any(slice_origin/=[0.e0_real_kind,0.e0_real_kind,45.e0_real_kind]).or. &
        any(slice_normal/=[0.e0_real_kind,0.e0_real_kind,1.e0_real_kind]).or.any(product_ids/='').or. &
        schedule_mode/='steps'.or.(.not.catalog.and.step_interval/=5).or.initial_frame.or.final_frame) return
    endif
    message='bl-layered64 seeds require the bounded boundary_layer profile'
    if(streamline_seeds=='bl-layered64'.and.products/='boundary_layer') return
    message='m12_diagnostics requires strict device rendering, independent clocks and bl-layered256 seeds'
    if(products=='m12_diagnostics') then
      if(.not.enabled.or..not.render.or.statistics.or.processing_backend/='device'.or. &
        rendering_pipeline=='compatible'.or.(.not.catalog.and.rendering_pipeline/='standard-device').or. &
        derivative_backend/='gpu'.or.(.not.catalog.and.postprocess_transport/='pinned').or. &
        streamline_seeds/='bl-layered256'.or.slice_definition/='index'.or.initial_frame.or.final_frame.or. &
        (.not.catalog.and.count(product_ids/='')==0)) return
    endif
    message='bl-layered256 seeds require m12_diagnostics'
    if(streamline_seeds=='bl-layered256'.and.products/='m12_diagnostics') return
    message='sbli_diagnostics requires strict device rendering without cumulative statistics'
    if(products=='sbli_diagnostics'.or.products=='sbli_2d') then
      if(.not.enabled.or..not.render.or.statistics.or.processing_backend/='device'.or. &
        rendering_pipeline=='compatible'.or.(products=='sbli_diagnostics'.and.rendering_pipeline/='standard-device').or. &
        derivative_backend/='gpu'.or. &
        (products=='sbli_diagnostics'.and.streamline_seeds/='sbli-layered16').or.slice_definition/='index') return
    endif
    message='sbli-layered16 seeds require sbli_diagnostics'
    if(streamline_seeds=='sbli-layered16'.and.products/='sbli_diagnostics') return
    message='air5_volume_statistics requires enabled statistics and products=air5_walls'
    if(air5_volume_statistics.and.(.not.enabled.or..not.statistics.or.products/='air5_walls')) return
    message='air5_volume_reduction requires air5_volume_statistics'
    if(air5_volume_reduction.and..not.air5_volume_statistics) return
    message='wall_mean_render requires enabled statistics, rendering and a wall product'
    if(wall_mean_render.and.(.not.enabled.or..not.statistics.or..not.render.or. &
      (products/='channel_walls'.and.products/='air5_walls'))) return
    message='mean_streamline_render requires enabled statistics and strict device streamlines'
    if(mean_streamline_render.and.(.not.enabled.or..not.statistics.or..not.render.or. &
      products/='streamlines'.or.processing_backend/='device'.or.rendering_pipeline=='compatible')) return
    message='wall_separation requires enabled AIR5 wall statistics'
    if(wall_separation.and.(.not.enabled.or..not.statistics.or.products/='air5_walls')) return
    message='slice_axis must be x, y or z and slice_index must not be negative'
    if((slice_axis/='x'.and.slice_axis/='y'.and.slice_axis/='z').or.slice_index<0) return
    message='nondefault slice settings require products=velocity_slice'
    if(products/='velocity_slice'.and.(slice_axis/='z'.or.slice_index/=4)) return
    message='slice_definition must be index or plane'
    if(slice_definition/='index'.and.slice_definition/='plane') return
    message='physical plane values must be finite'
    if(.not.all(ieee_is_finite(slice_origin)).or..not.all(ieee_is_finite(slice_normal))) return
    if(slice_definition=='plane') then
      message='physical planes require an admitted strict device slice profile without index settings or product clocks'
      if((products/='velocity_slice'.and.products/='boundary_layer').or. &
        processing_backend/='device'.or.rendering_pipeline=='compatible'.or. &
        .not.enabled.or..not.render.or.slice_axis/='z'.or.slice_index/=4.or.statistics.or. &
        any(product_ids/='').or.len_trim(batch_prefix)>0.or.len_trim(restore_batch)>0) return
      message='physical plane normal must be nonzero'
      if(maxval(abs(slice_normal))==0.e0_real_kind) return
    else
      message='physical plane values require slice_definition=plane'
      if(any(slice_origin/=0.e0_real_kind).or.any(slice_normal/=0.e0_real_kind)) return
    endif
    message='selected products require derivative_backend=gpu'
    if(enabled.and.render.and.products/='all'.and.products/='channel_walls'.and. &
      products/='air5_walls'.and.derivative_backend/='gpu') return
    message='wall products require CPU diagnostics or an admitted device wall provider'
    if(enabled.and.(products=='channel_walls'.or.products=='air5_walls').and.derivative_backend/='cpu') then
      if(processing_backend/='device') return
    endif
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
      ! Dynamic data products need neither a statistical accumulator nor a renderer.
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
    if(any(product_modes(n+1:)/='').or.any(product_steps(n+1:)/=0).or.any(product_times(n+1:)/=0.e0_real_kind)) return
    if(any(product_adaptive(n+1:)).or.any(product_dense_steps(n+1:)/=0).or.any(product_dense_times(n+1:)/=0).or. &
      any(product_events(:,n+1:)/='').or.any(product_windows(:,n+1:)/='')) return
    do i=1,n
      dot=index(trim(product_ids(i)),'.',back=.true.)
      message='product id must be an admitted scene followed by .image or .geometry'
      if(dot<2) return
      if(.not.product_scene_allowed(product_ids(i)(:dot-1),products,statistics,mean_streamline_render)) return
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
        if(product_steps(i)<=0.or.product_times(i)/=0.e0_real_kind) return
      case('time')
        if(product_steps(i)/=0.or.product_times(i)<=0.e0_real_kind) return
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
        if(step_interval/=0.or.time_interval/=0.e0_real_kind) return
      endif
    endif
    if(len_trim(streamline_views_file)>0) then
      message='streamline views require enabled strict device streamlines without legacy pairs'
      if(.not.enabled.or..not.render.or.products/='streamlines'.or.processing_backend/='device'.or. &
        rendering_pipeline=='compatible'.or.len_trim(batch_prefix)>0.or.len_trim(restore_batch)>0.or. &
        len_trim(streamline_views_file)>=len(streamline_views_file)) return
      message='first streamline views use the common render clock, not independent product clocks'
      if(n/=0) return
      call read_streamline_views(trim(streamline_views_file),views,view_count,views_ok,message)
      if(.not.views_ok) return
      views_have_means=any(views(1:view_count)%velocity/='instantaneous')
      message='mean view candidates require statistics and mean_streamline_render'
      if(views_have_means.and.(.not.statistics.or..not.mean_streamline_render)) return
    endif
    options%enabled=enabled; options%statistics=statistics; options%render=render
    options%initial_frame=initial_frame; options%final_frame=final_frame
    options%air5_volume_statistics=air5_volume_statistics
    options%air5_volume_reduction=air5_volume_reduction
    options%wall_mean_render=wall_mean_render
    options%mean_streamline_render=mean_streamline_render
    options%wall_separation=wall_separation
    options%schedule_mode=schedule_mode; options%step_interval=step_interval
    options%derivative_backend=derivative_backend
    options%processing_backend=processing_backend
    options%rendering_pipeline=rendering_pipeline
    options%postprocess_transport=postprocess_transport
    options%products=products
    options%streamline_seeds=streamline_seeds
    options%streamline_views_file=streamline_views_file
    options%view_count=view_count; options%views=views
    options%slice_axis=slice_axis; options%slice_index=slice_index
    options%slice_definition=slice_definition
    options%slice_origin=slice_origin; options%slice_normal=slice_normal
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

  subroutine read_streamline_views(filename,views,count,ok,message)
    character(*),intent(in) :: filename
    type(streamline_view),intent(out) :: views(insitu_max_views)
    integer,intent(out) :: count
    logical,intent(out) :: ok
    character(*),intent(out) :: message
    character(64) :: candidate_ids(insitu_max_views)
    character(16) :: velocity(insitu_max_views),seeds(insitu_max_views),color_field(insitu_max_views)
    real(real_kind) :: camera_position(3,insitu_max_views),focal_point(3,insitu_max_views),view_up(3,insitu_max_views)
    real(real_kind) :: parallel_scale(insitu_max_views),display_bounds(6,insitu_max_views)
    real(real_kind) :: color_range(2,insitu_max_views),line_width(insitu_max_views),direction(3),up(3),cross(3),delta
    integer :: image_size(2),unit,status,closed,i,j
    namelist /streamline_views/ candidate_ids,velocity,seeds,camera_position,focal_point,view_up, &
      parallel_scale,display_bounds,color_field,color_range,line_width,image_size
    call clear_streamline_views(views)
    count=0; ok=.false.
    candidate_ids='';velocity='';seeds='';color_field=''
    camera_position=0.e0_real_kind;focal_point=0.e0_real_kind;view_up=0.e0_real_kind;parallel_scale=0.e0_real_kind
    display_bounds=0.e0_real_kind;color_range=0.e0_real_kind;line_width=0.e0_real_kind;image_size=[800,600]
    open(newunit=unit,file=filename,status='old',action='read',iostat=status,iomsg=message)
    if(status/=0) return
    read(unit,nml=streamline_views,iostat=status,iomsg=message)
    close(unit,iostat=closed)
    if(status/=0) return
    message='cannot close streamline view configuration'
    if(closed/=0) return
    count=count_nonempty(candidate_ids)
    message='streamline views require one to six contiguous explicit candidates at 800x600'
    if(count<1.or.any(candidate_ids(1:count)=='').or.any(image_size/=[800,600])) return
    message='unexpected values beyond the declared streamline candidates'
    if(any(velocity(count+1:)/='').or.any(seeds(count+1:)/='').or.any(color_field(count+1:)/='').or. &
      any(camera_position(:,count+1:)/=0.e0_real_kind).or.any(focal_point(:,count+1:)/=0.e0_real_kind).or. &
      any(view_up(:,count+1:)/=0.e0_real_kind).or.any(parallel_scale(count+1:)/=0.e0_real_kind).or. &
      any(display_bounds(:,count+1:)/=0.e0_real_kind).or.any(color_range(:,count+1:)/=0.e0_real_kind).or. &
      any(line_width(count+1:)/=0.e0_real_kind)) return
    do i=1,count
      message='candidate ids must be unique ASCII names, not paths'
      if(len_trim(candidate_ids(i))>=len(candidate_ids(i)).or. &
        verify(trim(candidate_ids(i)),'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-')/=0) return
      do j=1,i-1
        if(candidate_ids(i)==candidate_ids(j)) return
      enddo
      message='unsupported streamline velocity, seed layout or color field'
      if(velocity(i)/='instantaneous'.and.velocity(i)/='reynolds'.and.velocity(i)/='favre') return
      if(seeds(i)/='line16'.and.seeds(i)/='tgv-stratified'.and. &
        seeds(i)/='bl-layered64'.and.seeds(i)/='bl-layered256') return
      if(color_field(i)/='u'.and.color_field(i)/='v'.and.color_field(i)/='w'.and.color_field(i)/='speed') return
      message='nonfinite streamline view values'
      if(.not.all(ieee_is_finite([camera_position(:,i),focal_point(:,i),view_up(:,i),parallel_scale(i), &
        display_bounds(:,i),color_range(:,i),line_width(i)]))) return
      direction=camera_position(:,i)-focal_point(:,i);up=view_up(:,i)
      message='streamline camera must have a finite nonzero direction and nonparallel up vector'
      if(.not.all(ieee_is_finite(direction))) return
      if(maxval(abs(direction))==0.e0_real_kind.or.maxval(abs(up))==0.e0_real_kind) return
      direction=direction/maxval(abs(direction));up=up/maxval(abs(up))
      cross=[direction(2)*up(3)-direction(3)*up(2),direction(3)*up(1)-direction(1)*up(3), &
        direction(1)*up(2)-direction(2)*up(1)]
      if(maxval(abs(cross))==0.e0_real_kind) return
      message='streamline scale, line width, bounds and color range must be positive and ordered'
      if(parallel_scale(i)<=0.e0_real_kind.or.line_width(i)<=0.e0_real_kind) return
      do j=1,3
        if(display_bounds(2*j,i)<=display_bounds(2*j-1,i)) return
      enddo
      if(color_range(2,i)<=color_range(1,i)) return
      delta=color_range(2,i)-color_range(1,i)
      if(.not.ieee_is_finite(delta)) return
      if(delta<tiny(delta)) return
      views(i)%id=candidate_ids(i);views(i)%velocity=velocity(i);views(i)%seeds=seeds(i);views(i)%color=color_field(i)
      views(i)%camera=camera_position(:,i);views(i)%focal=focal_point(:,i);views(i)%up=view_up(:,i)
      views(i)%scale=parallel_scale(i);views(i)%bounds=display_bounds(:,i);views(i)%range=color_range(:,i)
      views(i)%width=line_width(i)
    enddo
    ok=.true.;message=''
  end subroutine

  subroutine clear_streamline_views(views)
    type(streamline_view),intent(out) :: views(:)
    integer :: i
    ! Explicit assignments avoid indeterminate unused fields with NVHPC derived constructors.
    do i=1,size(views)
      views(i)%id='';views(i)%velocity='';views(i)%seeds='';views(i)%color=''
      views(i)%camera=0.e0_real_kind;views(i)%focal=0.e0_real_kind;views(i)%up=0.e0_real_kind;views(i)%scale=0.e0_real_kind
      views(i)%bounds=0.e0_real_kind;views(i)%range=0.e0_real_kind;views(i)%width=0.e0_real_kind
    enddo
  end subroutine

  integer function count_nonempty(values) result(n)
    character(*),intent(in) :: values(:)
    n=count(values/='')
  end function

  function streamline_trajectory_key(view) result(key)
    type(streamline_view),intent(in) :: view
    character(128) :: key
    ! The state, topology, statistics window and fixed integrator scope the cache externally.
    key=trim(view%velocity)//':'//trim(view%seeds)
  end function

  function streamline_display_key(view) result(key)
    type(streamline_view),intent(in) :: view
    character(1024) :: key
    write(key,'(a,19(1x,es26.17e3))') trim(view%color), &
      view%camera,view%focal,view%up,view%scale,view%bounds,view%range,view%width
  end function

  subroutine canonical_slice_plane(normal,ok)
    real(real_kind),intent(inout) :: normal(3)
    logical,intent(out) :: ok
    real(real_kind) :: original(3),scaled(3),scale,length,direction
    integer :: axis,d
    ok=.false.
    original=normal
    if(.not.all(ieee_is_finite(original))) return
    axis=maxloc(abs(original),dim=1); scale=abs(original(axis))
    if(scale==0.e0_real_kind) return
    scaled=original/scale
    if(any(original/=0.e0_real_kind.and.scaled==0.e0_real_kind)) return
    length=sqrt((scaled(1)*scaled(1)+scaled(2)*scaled(2))+scaled(3)*scaled(3))
    direction=1.e0_real_kind
    if(original(axis)<0.e0_real_kind) direction=-1.e0_real_kind
    do d=1,3
      scaled(d)=(scaled(d)/length)*direction
      if(original(d)/=0.e0_real_kind.and.scaled(d)==0.e0_real_kind) return
      if(scaled(d)==0.e0_real_kind) scaled(d)=0.e0_real_kind
    enddo
    normal=scaled; ok=.true.
  end subroutine
  logical function product_scene_allowed(scene,profile,statistics,mean_streamline_render) result(allowed)
    character(*),intent(in) :: scene,profile
    logical,intent(in) :: statistics
    logical,intent(in),optional :: mean_streamline_render
    logical :: mean_requested
    mean_requested=.false.
    if(present(mean_streamline_render)) mean_requested=mean_streamline_render
    allowed=.false.
    if(profile=='sbli_2d') then
      select case(scene)
      case('velocity_slice','density_gradient_slice','streamwise_velocity_slice','pressure_ratio_slice', &
           'temperature_slice','wall_pressure_curve','skin_friction_curve','wall_heat_flux_curve')
        allowed=.true.
      end select
      return
    endif
    ! PF keeps the approved periodic TGV product boundary. Wall presets retain their common clock.
    select case(scene)
    case('q_surface')
      allowed=profile=='all'.or.profile=='q_surface'.or.profile=='q_streamlines'.or. &
        profile=='tgv256_demo'.or.profile=='curve_demo'.or.profile=='boundary_layer'.or.profile=='m12_diagnostics'.or. &
        profile=='sbli_diagnostics'
    case('velocity_slice')
      allowed=profile=='all'.or.profile=='velocity_slice'.or.profile=='boundary_layer'.or.profile=='m12_diagnostics'.or. &
        profile=='sbli_diagnostics'
    case('instantaneous_streamlines')
      allowed=profile=='all'.or.profile=='streamlines'.or.profile=='q_streamlines'.or. &
        profile=='tgv256_demo'.or.profile=='curve_demo'.or.profile=='boundary_layer'.or.profile=='m12_diagnostics'.or. &
        profile=='sbli_diagnostics'
    case('crossing_streamlines')
      allowed=profile=='all'.or.profile=='streamlines'.or.profile=='q_streamlines'
    case('mean_reynolds_streamlines','mean_favre_streamlines')
      allowed=statistics.and.(profile=='all'.or.(profile=='streamlines'.and.mean_requested))
    end select
  end function
end module
