module insitu_catalog_render
  use mpi
  use iso_fortran_env, only: int64,real64
  use iso_c_binding, only: c_int,c_int64_t,c_char,c_double,c_null_char
  use astr_precision, only: real_kind,c_real_kind,real_bytes
  use commvar, only: use_gpu,uinf,pinf,tinf,ia,ja,ka,lreadgrid,flowtype
  use insitu_run_config, only: insitu_options,pack_catalog_render,catalog_product
  use insitu_session, only: catalog_render_options,begin_catalog_renderer
  use insitu_product_schedule, only: runtime_product_state,runtime_catalog_count,runtime_catalog_state, &
    runtime_catalog_due,runtime_catalog_success,runtime_catalog_wait,runtime_catalog_disable,runtime_catalog_reject, &
    runtime_active,runtime_waiting,runtime_catalog_image_work
  use benchmark_runtime, only: benchmark_field_io_disabled
  use index_slice_source, only: slice_source_info,slice_source_on_gpu
  use index_slice_provider, only: prepare_index_slice
  use index_slice_archive, only: runtime_product_segment,refresh_runtime_product_status,runtime_catalog_changed
  use checkpoint_state_io, only: checkpoint_state_identity
#if defined(_CUDA) && defined(ASTR_INSITU_DEVICE_RENDERING)
  use cudafor, only: c_devptr
  use index_slice_gpu, only: index_slice_device_view,release_index_slice_gpu,index_slice_exchange_usage
  use insitu_products_gpu, only: begin_catalog_streamlines_gpu,end_catalog_streamlines_gpu,catalog_streamline_pointers_gpu
#endif
  implicit none
  private
  public :: observe_catalog_images
  type :: frame_image
    type(runtime_product_state) :: state
    integer :: source=0
    real(real_kind) :: window(2)=0,measure=0
    integer(int64) :: samples=0
    logical :: selected=.false.
  end type
contains
  subroutine observe_catalog_images(identity)
    type(checkpoint_state_identity),intent(in) :: identity
    type(runtime_product_state) :: state
    type(insitu_options) :: options
    type(frame_image),allocatable :: frames(:)
    type(catalog_product) :: sample_definition
    real(real_kind),allocatable :: values(:,:),xyz(:,:)
    real(real_kind) :: window(2),measure,seconds
    integer(int64) :: count,workspace,download,exchange
    integer(int64) :: retained
    integer(int64) :: private_download,private_upload
    integer :: p,q,source,first(3),extent(3),axes(2),d,n,status,rank,err,old_status
    character(:),allocatable :: directory
    character(16) :: input_transport
    character(512) :: message,input_message
    logical :: available,ready,ok,dirty,input_ok,requested(27),physical,retire_needed,lines,curve,host
    logical,allocatable :: group(:)
#if defined(_CUDA) && defined(ASTR_INSITU_DEVICE_RENDERING)
    character(32) :: labels(11)
    integer(c_int) :: words(7)
    real(real64) :: settings(31)
    real(real_kind) :: references(3),statistics(3)
    character(c_char) :: error_buffer(512)
    type(c_devptr) :: device_values,device_coordinates,line_velocity,line_coordinates
    interface
      integer(c_int) function render_host(backend,script,comm,step,t,id,source,definition,directory,version, &
          labels,words,settings,field_ids,nfields,extent,axis,coordinates,values,references,provider,statistics,samples, &
          selected_field,host_budget,download) bind(C,name='astr_insitu_render_host_product')
        import c_int,c_char,c_int64_t,c_double,c_real_kind
        character(c_char),intent(in) :: backend(*),script(*),id(*),source(*),definition(*),directory(*),labels(*)
        integer(c_int),value :: comm,step,nfields,axis,provider,selected_field
        integer(c_int64_t),value :: version,samples,host_budget,download
        integer(c_int),intent(in) :: words(7),field_ids(*),extent(3)
        real(c_double),intent(in) :: settings(31)
        real(c_real_kind),value :: t
        real(c_real_kind),intent(in) :: coordinates(*),values(*),references(3),statistics(3)
      end function
      subroutine host_group(begin) bind(C,name='astr_insitu_runtime_host_group')
        import c_int
        integer(c_int),value :: begin
      end subroutine
      integer(c_int) function render(pipeline,backend,script,comm,step,t,id,source,definition,directory,version, &
          labels,words,settings,field_ids,nfields,width,height,coordinates,values,references,provider,statistics,samples, &
          selected_field,host_budget) &
          bind(C,name='astr_insitu_render_index_product')
        import c_int,c_char,c_int64_t,c_double,c_real_kind,c_devptr
        character(c_char),intent(in) :: pipeline(*),backend(*),script(*),id(*),source(*),definition(*),directory(*),labels(*)
        integer(c_int),value :: comm,step,nfields,width,height,provider,selected_field
        integer(c_int64_t),value :: version,samples,host_budget
        integer(c_int),intent(in) :: words(7),field_ids(*)
        real(c_double),intent(in) :: settings(31)
        real(c_real_kind),value :: t
        real(c_real_kind),intent(in) :: references(3),statistics(3)
        type(c_devptr),value :: coordinates,values
      end function
      integer(c_int) function render_lines(pipeline,backend,script,comm,step,t,id,source,definition,directory,version, &
          labels,words,settings,global,extent,offset,physical,velocity,coordinate_halo,coordinates,references, &
          provider,statistics,samples,host_budget) bind(C,name='astr_insitu_render_line_product')
        import c_int,c_char,c_int64_t,c_double,c_real_kind,c_devptr
        character(c_char),intent(in) :: pipeline(*),backend(*),script(*),id(*),source(*),definition(*),directory(*),labels(*)
        integer(c_int),value :: comm,step,provider,physical
        integer(c_int64_t),value :: version,samples,host_budget
        integer(c_int),intent(in) :: words(7),global(3),extent(3),offset(3)
        real(c_double),intent(in) :: settings(31)
        real(c_real_kind),value :: t
        real(c_real_kind),intent(in) :: references(3),statistics(3)
        type(c_devptr),value :: velocity,coordinate_halo,coordinates
      end function
      integer(c_int) function render_physical(pipeline,backend,script,comm,step,t,id,source,definition,directory,version, &
          labels,words,settings,field_ids,nfields,global,extent,offset,coordinates,values,references, &
          provider,statistics,samples,selected_field,budget,retained,host_budget) bind(C,name='astr_insitu_render_physical_product')
        import c_int,c_char,c_int64_t,c_double,c_real_kind,c_devptr
        character(c_char),intent(in) :: pipeline(*),backend(*),script(*),id(*),source(*),definition(*),directory(*),labels(*)
        integer(c_int),value :: comm,step,nfields,provider,selected_field
        integer(c_int64_t),value :: version,samples,budget,retained,host_budget
        integer(c_int),intent(in) :: words(7),field_ids(*),global(3),extent(3),offset(3)
        real(c_double),intent(in) :: settings(31)
        real(c_real_kind),value :: t
        real(c_real_kind),intent(in) :: references(3),statistics(3)
        type(c_devptr),value :: coordinates,values
      end function
      subroutine index_group(begin) bind(C,name='astr_insitu_runtime_index_group')
        import c_int
        integer(c_int),value :: begin
      end subroutine
      subroutine physical_group(begin) bind(C,name='astr_insitu_runtime_physical_group')
        import c_int
        integer(c_int),value :: begin
      end subroutine
      subroutine line_group(begin) bind(C,name='astr_insitu_runtime_line_group')
        import c_int
        integer(c_int),value :: begin
      end subroutine
      integer(c_int) function line_metadata(source,labels,words,settings,comm) bind(C,name='astr_insitu_runtime_line_metadata')
        import c_int,c_char,c_double
        character(c_char),intent(in) :: source(*),labels(*)
        integer(c_int),intent(in) :: words(7)
        real(c_double),intent(in) :: settings(31)
        integer(c_int),value :: comm
      end function
      integer(c_int) function cleanup(comm,step,t) bind(C,name='astr_insitu_runtime_catalog_cleanup')
        import c_int,c_real_kind
        integer(c_int),value :: comm,step
        real(c_real_kind),value :: t
      end function
      subroutine get_error(buffer,capacity) bind(C,name='astr_insitu_runtime_error')
        import c_char,c_int
        character(c_char),intent(out) :: buffer(*)
        integer(c_int),value :: capacity
      end subroutine
    end interface
#endif
    if(benchmark_field_io_disabled()) return
    if(.not.runtime_catalog_image_work().and..not.runtime_catalog_changed()) return
    call MPI_Comm_rank(MPI_COMM_WORLD,rank,err);call require_mpi(err)
    call catalog_render_options(options,available)
    host=options%processing_backend=='host'
    input_transport='host'
    if(len_trim(options%postprocess_transport)>0) input_transport=options%postprocess_transport
    dirty=.false.;retire_needed=.false.
#if defined(_CUDA) && defined(ASTR_INSITU_DEVICE_RENDERING)
    if(available.and.runtime_catalog_changed()) then
      status=cleanup(int(MPI_COMM_WORLD,c_int),int(identity%step,c_int),identity%time)
      if(status/=0) then
        print *, 'runtime renderer control cleanup failed'
        call MPI_Abort(MPI_COMM_WORLD,status,err)
      endif
    endif
#endif
    allocate(frames(runtime_catalog_count()),group(runtime_catalog_count()))
    do p=1,runtime_catalog_count()
      call runtime_catalog_state(p,state)
      if(state%definition%view%channel/='image'.and.state%definition%view%channel/='geometry') cycle
      if(state%definition%view%kind=='wall_curve') cycle
      if(state%status/=runtime_active.and.state%status/=runtime_waiting) cycle
      message=''
      if(.not.available) then
        message='image rendering was not enabled in the startup backend configuration'
      elseif(.not.use_gpu) then
        message='runtime image backend is not yet implemented for this source/processing pipeline'
      elseif(host.and.options%rendering_pipeline/='compatible') then
        message='host processing requires the explicitly selected compatible renderer'
      elseif(state%definition%view%channel=='geometry'.and.options%rendering_pipeline/='compatible') then
        message='geometry export requires explicit compatible processing; resident pipelines do not download geometry'
      elseif(.not.((state%definition%view%kind=='slice'.and. &
          (state%definition%view%sampling=='grid_index'.or.state%definition%view%sampling=='physical_plane')).or. &
          ((state%definition%view%kind=='q_surface'.or.state%definition%view%kind=='streamlines').and. &
           state%definition%view%sampling=='volume'))) then
        message='runtime image producer is not yet implemented for this kind/sampling mode'
      elseif(state%definition%view%kind=='streamlines'.and. &
          any(state%definition%fields<2.or.state%definition%fields>4)) then
        message='existing streamlines support velocity components or magnitude only'
      endif
#if !defined(_CUDA) || !defined(ASTR_INSITU_DEVICE_RENDERING)
      message='runtime device images require the CUDA/Catalyst strict-device build'
#endif
      if(len_trim(message)>0) then
        call runtime_catalog_reject(p,trim(message));dirty=.true.;retire_needed=.true.;cycle
      endif
      call slice_source_info(state%definition,source,ready,ok,window,measure,count,message)
      if(.not.ok) then
        call runtime_catalog_reject(p,trim(message));dirty=.true.;retire_needed=.true.;cycle
      endif
      if(host.and.options%derivative_backend=='gpu'.and. &
          (state%definition%view%kind=='q_surface'.or.any(state%definition%fields>=13)).and. &
          .not.slice_source_on_gpu(source)) then
        message='GPU diagnostics are not implemented for this host-resident statistics source'
        call runtime_catalog_reject(p,trim(message));dirty=.true.;retire_needed=.true.;cycle
      endif
      old_status=state%status
      call runtime_catalog_wait(p,.not.ready)
      if((old_status==runtime_waiting).eqv.ready) dirty=.true.
      if(.not.ready) cycle
      if(.not.runtime_catalog_due(p,identity%step,real(identity%time,real64))) cycle
      frames(p)%state=state;frames(p)%source=source
      frames(p)%window=window;frames(p)%measure=measure;frames(p)%samples=count;frames(p)%selected=.true.
    enddo
    do p=1,size(frames)
      if(.not.frames(p)%selected) cycle
      sample_definition=frames(p)%state%definition
      requested=.false.;group=.false.
      do q=p,size(frames)
        if(.not.frames(q)%selected) cycle
        if(.not.same_sample(sample_definition,frames(q)%state%definition)) cycle
        group(q)=.true.;requested(frames(q)%state%definition%fields)=.true.
      enddo
      if(sample_definition%view%kind=='q_surface') requested(22)=.true.
      lines=sample_definition%view%kind=='streamlines'
      if(lines) requested(2:4)=.true.
      sample_definition%fields=pack([(d,d=1,27)],requested)
      physical=sample_definition%view%sampling=='physical_plane'.or.sample_definition%view%kind=='q_surface'.or.lines
      curve=lreadgrid.or.trim(flowtype)/='tgv'.or.any([ia,ja,ka]/=ia)
      if(physical) then
        sample_definition%axis=0
        sample_definition%ranges(1,:)=0;sample_definition%ranges(2,:)=[ia,ja,ka]
      endif
      call prepare_index_slice(sample_definition,frames(p)%source,frames(p)%samples,values,xyz,first,extent,input_ok,input_message, &
        workspace,download,exchange,seconds,include_cell_endpoints=.true.,retain_device=.not.host, &
        transport=trim(input_transport),host_diagnostics=host.and.options%derivative_backend=='cpu', &
        staging_download=private_download,staging_upload=private_upload)
      if(input_ok) write(*,'(a,i0,a,a,a,i0,a,i0,a,i0,a,i0)') 'ASTR_RUNTIME_INPUT rank=',rank,' id=',sample_definition%id, &
        ' step=',identity%step,' selected_field_d2h_bytes=',download,' private_exchange_bytes=',exchange, &
        ' fields=',size(sample_definition%fields)
      if(input_ok) call begin_catalog_renderer(input_ok,input_message)
#if defined(_CUDA) && defined(ASTR_INSITU_DEVICE_RENDERING)
      if(input_ok) then
        if(host) then
          write(*,'(a,i0,a,i0,a,i0,a,i0)') 'ASTR_RUNTIME_TRANSFER rank=',rank,' step=',identity%step, &
            ' transport=host private_d2h_bytes=',private_download,' private_h2d_bytes=',private_upload
          call host_group(1_c_int)
        else
        call index_slice_exchange_usage(private_download,private_upload)
        write(*,'(a,i0,a,i0,2a,a,i0,a,i0)') 'ASTR_RUNTIME_TRANSFER rank=',rank,' step=',identity%step, &
          ' transport=',trim(options%postprocess_transport),' private_d2h_bytes=',private_download,' private_h2d_bytes=',private_upload
        call index_slice_device_view(device_values,device_coordinates)
        if(lines) then
          retained=workspace+product(int(extent,int64))*(3+size(sample_definition%fields))*real_bytes
          call begin_catalog_streamlines_gpu(trim(options%postprocess_transport),device_values,device_coordinates, &
            sample_definition%fields,curve,options%host_budget_bytes,options%device_budget_bytes,options%device_reserve_bytes, &
            retained,input_ok,input_message)
          if(input_ok) then
            call catalog_streamline_pointers_gpu(line_velocity,line_coordinates)
            call line_group(1_c_int)
            do q=p,size(frames)
              if(.not.group(q).or.frames(q)%state%definition%view%channel/='geometry') cycle
              call pack_catalog_render(frames(q)%state%definition%view,labels,words,settings)
              status=line_metadata(frames(q)%state%definition%source//c_null_char,labels,words,settings, &
                int(MPI_COMM_WORLD,c_int))
              if(status==0) cycle
              call get_error(error_buffer,int(size(error_buffer),c_int));message=''
              do d=1,size(error_buffer)
                if(error_buffer(d)==c_null_char) exit
                message(d:d)=error_buffer(d)
              enddo
              call runtime_catalog_disable(q,identity%step,real(identity%time,real64),trim(message))
              group(q)=.false.;frames(q)%selected=.false.;dirty=.true.;retire_needed=.true.
            enddo
          endif
        elseif(physical) then
          call physical_group(1_c_int)
        else
          call index_group(1_c_int)
        endif
        endif
      endif
#endif
      do q=p,size(frames)
        if(.not.group(q)) cycle
        frames(q)%selected=.false.
        state=frames(q)%state;ok=input_ok;message=input_message
        if(ok) call runtime_product_segment(q,state,directory,ok,message)
#if defined(_CUDA) && defined(ASTR_INSITU_DEVICE_RENDERING)
        if(ok) then
        call pack_catalog_render(state%definition%view,labels,words,settings)
        references=[uinf,pinf,tinf]
        statistics=[frames(q)%window,frames(q)%measure]
        if(host) then
          status=render_host(trim(options%implementation_path)//c_null_char,trim(options%pipeline_file)//c_null_char, &
            int(MPI_COMM_WORLD,c_int),int(identity%step,c_int),identity%time,state%definition%id//c_null_char, &
            state%definition%source//c_null_char,state%definition%record//c_null_char,directory//c_null_char, &
            int(state%version,c_int64_t),labels,words,settings,int(sample_definition%fields,c_int), &
            int(size(sample_definition%fields),c_int),int(extent,c_int),int(sample_definition%axis,c_int),xyz,values, &
            references,int(frames(q)%source,c_int),statistics,int(frames(q)%samples,c_int64_t), &
            int(state%definition%fields(1),c_int),int(options%host_budget_bytes,c_int64_t),int(download,c_int64_t))
        elseif(lines) then
          status=render_lines(trim(options%rendering_pipeline)//c_null_char,trim(options%implementation_path)//c_null_char, &
            trim(options%pipeline_file)//c_null_char,int(MPI_COMM_WORLD,c_int),int(identity%step,c_int),identity%time, &
            state%definition%id//c_null_char,state%definition%source//c_null_char,state%definition%record//c_null_char, &
            directory//c_null_char,int(state%version,c_int64_t),labels,words,settings,int([ia,ja,ka],c_int), &
            int(extent-1,c_int),int(first,c_int),merge(1_c_int,0_c_int,curve),line_velocity,line_coordinates,device_coordinates, &
            references,int(frames(q)%source,c_int),statistics,int(frames(q)%samples,c_int64_t),int(options%host_budget_bytes,c_int64_t))
        elseif(physical) then
          retained=workspace+product(int(extent,int64))*(3+size(sample_definition%fields))*real_bytes
          status=render_physical(trim(options%rendering_pipeline)//c_null_char,trim(options%implementation_path)//c_null_char, &
            trim(options%pipeline_file)//c_null_char,int(MPI_COMM_WORLD,c_int),int(identity%step,c_int),identity%time, &
            state%definition%id//c_null_char,state%definition%source//c_null_char,state%definition%record//c_null_char, &
            directory//c_null_char,int(state%version,c_int64_t),labels,words,settings,int(sample_definition%fields,c_int), &
            int(size(sample_definition%fields),c_int),int([ia,ja,ka],c_int),int(extent,c_int),int(first,c_int), &
            device_coordinates,device_values,references,int(frames(q)%source,c_int),statistics,int(frames(q)%samples,c_int64_t), &
            int(state%definition%fields(1),c_int),int(options%device_budget_bytes,c_int64_t),int(retained,c_int64_t), &
            int(options%host_budget_bytes,c_int64_t))
        else
        n=0
        do d=1,3
          if(d==state%definition%axis) cycle
          n=n+1;axes(n)=d
        enddo
        status=render(trim(options%rendering_pipeline)//c_null_char,trim(options%implementation_path)//c_null_char, &
          trim(options%pipeline_file)//c_null_char,int(MPI_COMM_WORLD,c_int),int(identity%step,c_int),identity%time, &
          state%definition%id//c_null_char,state%definition%source//c_null_char,state%definition%record//c_null_char, &
          directory//c_null_char,int(state%version,c_int64_t),labels,words,settings,int(sample_definition%fields,c_int), &
          int(size(sample_definition%fields),c_int),int(extent(axes(1)),c_int),int(extent(axes(2)),c_int), &
          device_coordinates,device_values,references,int(frames(q)%source,c_int),statistics,int(frames(q)%samples,c_int64_t), &
          int(state%definition%fields(1),c_int),int(options%host_budget_bytes,c_int64_t))
        endif
        ok=status==0
        if(.not.ok) then
          call get_error(error_buffer,int(size(error_buffer),c_int))
          message=''
          do d=1,size(error_buffer)
            if(error_buffer(d)==c_null_char) exit
            message(d:d)=error_buffer(d)
          enddo
        endif
        endif
#endif
        if(ok) then
          call runtime_catalog_success(q,identity%step,real(identity%time,real64))
        else
          call runtime_catalog_disable(q,identity%step,real(identity%time,real64),trim(message))
          retire_needed=.true.
          if(rank==0) then
            if(state%definition%view%channel=='geometry') then
              print '(a)','ASTR_RUNTIME_GEOMETRY_DISABLED '//state%definition%id//': '//trim(message)
            else
              print '(a)','ASTR_RUNTIME_IMAGE_DISABLED '//state%definition%id//': '//trim(message)
            endif
          endif
        endif
        dirty=.true.
      enddo
#if defined(_CUDA) && defined(ASTR_INSITU_DEVICE_RENDERING)
      if(input_ok) then
        if(host) then
          call host_group(0_c_int)
        elseif(lines) then
          call line_group(0_c_int)
        elseif(physical) then
          call physical_group(0_c_int)
        else
          call index_group(0_c_int)
        endif
      endif
      if(lines.and..not.host) call end_catalog_streamlines_gpu(int(identity%step))
      call release_index_slice_gpu()
#endif
    enddo
#if defined(_CUDA) && defined(ASTR_INSITU_DEVICE_RENDERING)
    if(available.and.retire_needed) then
      status=cleanup(int(MPI_COMM_WORLD,c_int),int(identity%step,c_int),identity%time)
      if(status/=0) call MPI_Abort(MPI_COMM_WORLD,status,err)
    endif
#endif
    if(dirty) call refresh_runtime_product_status(identity)
  end subroutine

  logical function same_sample(a,b) result(same)
    type(catalog_product),intent(in) :: a,b
    same=a%source==b%source.and.a%statistics_source==b%statistics_source.and. &
      a%view%kind==b%view%kind.and.a%view%sampling==b%view%sampling.and. &
      a%axis==b%axis.and.a%index==b%index.and.all(a%ranges==b%ranges)
    if(same.and.a%view%sampling=='physical_plane') same=all(a%view%origin==b%view%origin).and. &
      all(a%view%normal==b%view%normal).and.all(a%view%bounds==b%view%bounds)
  end function

  subroutine require_mpi(status)
    integer,intent(in) :: status
    integer :: err
    if(status/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,status,err)
  end subroutine
end module
