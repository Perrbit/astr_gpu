module index_slice_archive
  use mpi
  use iso_fortran_env, only: int64,real64
  use iso_c_binding, only: c_int,c_char,c_int64_t,c_double,c_size_t,c_null_char
  use astr_precision, only: real_kind,real_bytes
  use astr_precision_mpi, only: mpi_real_type
  use commvar, only: ia,ja,ka,numq,nondimen,roinf,uinf,vinf,winf,pinf,tinf,ref_den,ref_vel,ref_tem,ref_len,ref_tim,ref_miu
  use insitu_catalog, only: configure_catalog_watch,poll_catalog_watch,write_catalog_status
  use insitu_product_schedule, only: runtime_product_state,runtime_catalog_count,runtime_catalog_state, &
    runtime_catalog_due,runtime_catalog_success,runtime_catalog_wait,runtime_catalog_disable,runtime_catalog_reject, &
    runtime_active,runtime_waiting
  use index_slice_source, only: slice_source_info
  use insitu_selected_runtime, only: observe_selected_runtime,selected_runtime_sampling_info, &
    selected_runtime_profile,selected_profile_result,refresh_selected_runtime,selected_runtime_workspace, &
    selected_runtime_wall_references,selected_runtime_wall_grid,selected_runtime_wall_scale,selected_runtime_cartesian
  use insitu_selected_statistics, only: selected_statistics_group,node_key
  use insitu_spatial_statistics, only: canonical_node_intervals
  use insitu_sample_moments, only: push_sample_moments
  use index_slice_stencil, only: slice_stencil_start
  use insitu_run_config, only: catalog_product,catalog_wall_mean
  use index_slice_provider, only: prepare_index_slice
  use output_fields, only: write_grid_xml,catalog_quantity_units
  use checkpoint_state_io, only: checkpoint_state_identity
  use benchmark_runtime, only: insitu_clock,report_insitu_timing
  implicit none
  private
  public :: configure_index_slices,observe_index_slices
  public :: runtime_product_segment,refresh_runtime_product_status,runtime_catalog_changed
  type :: saved_frame
    character(40) :: file=''
    type(checkpoint_state_identity) :: identity
  end type
  type :: archive_segment
    integer(int64) :: version=-1,request=-1
    character(:),allocatable :: path
    type(saved_frame),allocatable :: frames(:)
    logical :: geometry=.false.,csv_started=.false.
  end type
  type(archive_segment),allocatable,save :: segments(:)
  character(:),allocatable,save :: archive_root,status_path,status_root
  logical,save :: configured=.false.,suppressed=.false.,status_root_ready=.false.
  logical,save :: control_changed=.false.
  interface
    integer(c_int) function ensure_directory(path) bind(C,name='astr_index_slice_directory')
      import c_char,c_int
      character(c_char),intent(in) :: path(*)
    end function
    integer(c_int) function new_segment(root,path,capacity) bind(C,name='astr_index_slice_segment')
      import c_char,c_int,c_size_t
      character(c_char),intent(in) :: root(*)
      character(c_char),intent(out) :: path(*)
      integer(c_size_t),value :: capacity
    end function
    integer(c_int) function write_slice(path,geometry,only_geometry,comm,bytes,dimensions,offset,extent, &
      nf,names,units,field_ids,values,coordinates,step,clock,references,ranges,definition,provenance) &
      bind(C,name='astr_index_slice_write')
      import c_char,c_int,c_int64_t,c_double,real_kind
      character(c_char),intent(in) :: path(*),geometry(*),names(*),units(*),definition(*),provenance(*)
      integer(c_int),value :: only_geometry,comm,bytes,nf
      integer(c_int),intent(in) :: dimensions(2),offset(2),extent(2)
      integer(c_int64_t),intent(in) :: field_ids(*),ranges(*)
      integer(c_int64_t),value :: step
      real(c_double),intent(in) :: clock(3),references(12)
      real(real_kind),intent(in) :: values(*),coordinates(*)
    end function
    integer(c_int) function rename_file(old,new) bind(C,name='rename')
      import c_char,c_int
      character(c_char),intent(in) :: old(*),new(*)
    end function
    integer(c_int) function publish_text(path,text,n,error,capacity) bind(C,name='astr_catalog_publish_text')
      import c_char,c_size_t,c_int
      character(c_char),intent(in) :: path(*),text(*)
      integer(c_size_t),value :: n,capacity
      character(c_char),intent(out) :: error(*)
    end function
  end interface
contains
  subroutine geometry_source_c(ids,count,references,system,units) bind(C,name='astr_insitu_runtime_geometry_source')
    integer(c_int),intent(in) :: ids(*)
    integer(c_int),value :: count
    real(c_double),intent(out) :: references(12)
    character(c_char),intent(out) :: system(*),units(*)
    character(16) :: unit_system
    character(32) :: unit
    integer :: i,j
    unit_system='dimensionless'
    if(.not.nondimen) unit_system='si'
    references=real([roinf,uinf,vinf,winf,pinf,tinf,ref_den,ref_vel,ref_tem,ref_len,ref_tim,ref_miu],real64)
    do j=1,len(unit_system)
      system(j)=unit_system(j:j)
    enddo
    do i=1,count
      unit=catalog_quantity_units(ids(i),unit_system)
      do j=1,len(unit)
        units((i-1)*len(unit)+j)=unit(j:j)
      enddo
    enddo
  end subroutine
  subroutine configure_index_slices(root,catalog,interval,restoring,no_field_io)
    character(*),intent(in) :: root,catalog
    integer(int64),intent(in) :: interval
    logical,intent(in) :: restoring,no_field_io
    archive_root=root//'/slice_products';status_path=root//'/products.status.json'
    status_root=root;status_root_ready=.false.
    suppressed=no_field_io;configured=.true.
    if(allocated(segments)) deallocate(segments)
    allocate(segments(0))
    call configure_catalog_watch(catalog,interval,0_int64,MPI_COMM_WORLD,restoring)
  end subroutine

  subroutine observe_index_slices(identity)
    type(checkpoint_state_identity),intent(in) :: identity
    type(runtime_product_state) :: state
    type(catalog_product) :: cached_definition
    type(catalog_product) :: selection
    type(archive_segment),allocatable :: expanded(:)
    real(real_kind),allocatable :: values(:,:),xyz(:,:)
    real(real_kind) :: window(2),measure,compute_seconds
    integer(int64) :: count,workspace,download,exchange
    integer :: p,n,source,rank,err,first(3),extent(3),old_status,status,cached_source
    logical :: changed,dirty,ready,ok,cached,reused
    real(8) :: phase_started
    character(:),allocatable :: update_message
    character(256) :: message
    call MPI_Comm_rank(MPI_COMM_WORLD,rank,err);call mpi_ok(err)
    if(.not.configured.or.suppressed) return
    cached=.false.;cached_source=-1
    control_changed=.false.
    phase_started=insitu_clock()
    call poll_catalog_watch(identity%step,real(identity%time,real64),[ia,ja,ka],numq,MPI_COMM_WORLD,changed,update_message)
#ifdef ASTR_GPU_CPP
    block
      use commvar, only: use_gpu
      if(use_gpu.and.runtime_catalog_count()>0) then
        do p=1,runtime_catalog_count()
          call runtime_catalog_state(p,state)
          if(state%status/=runtime_active.and.state%status/=runtime_waiting) cycle
          if(state%definition%source/='instantaneous'.or.state%definition%view%channel/='data'.or. &
             state%definition%view%kind/='slice') then
            if(rank==0) write(*,'(a)') 'C++ mean/render product field providers are not implemented'
            call MPI_Abort(MPI_COMM_WORLD,1,err)
          endif
        enddo
      endif
    end block
#endif
    call report_insitu_timing('index_slice_catalog',phase_started,identity%step)
    dirty=changed.or.len(update_message)>0
    control_changed=changed
    if(rank==0.and.len(update_message)>0) print '(a)','ASTR_CATALOG '//update_message
    call observe_selected_runtime(identity%step,identity%time,changed,sample_selected_wall_group)
    n=runtime_catalog_count()
    if(n>size(segments)) then
      allocate(expanded(n));expanded(:size(segments))=segments
      call move_alloc(expanded,segments)
    endif
    do p=1,n
      call runtime_catalog_state(p,state)
      if(state%status/=runtime_active.and.state%status/=runtime_waiting) cycle
      if(state%definition%view%channel/='data') cycle
      if(state%definition%view%kind/='slice'.and.state%definition%view%kind/='probe'.and. &
         state%definition%view%kind/='station_profile'.and.state%definition%view%kind/='wall_profile') cycle
      call slice_source_info(state%definition,source,ready,ok,window,measure,count,message)
      if(.not.ok) then
        call runtime_catalog_reject(p,trim(message));dirty=.true.;cycle
      endif
      old_status=state%status
      call runtime_catalog_wait(p,.not.ready)
      if((old_status==runtime_waiting).eqv.ready) dirty=.true.
      if(.not.ready) cycle
      if(.not.runtime_catalog_due(p,identity%step,real(identity%time,real64))) cycle
      if(state%definition%view%kind=='station_profile'.or.state%definition%view%kind=='wall_profile') then
        if(state%definition%view%kind=='wall_profile') then
          call write_wall_profile(p,state,identity,ok,message)
        else
          call write_station_profile(p,state,identity,ok,message)
        endif
        if(ok) then
          call runtime_catalog_success(p,identity%step,real(identity%time,real64))
        else
          call runtime_catalog_disable(p,identity%step,real(identity%time,real64),trim(message))
          if(rank==0) print '(a)','ASTR_STATION_PROFILE_DISABLED '//state%definition%id//': '//trim(message)
        endif
        dirty=.true.;cycle
      endif
      reused=.false.
      if(cached.and.source==cached_source) then
        reused=state%definition%axis==cached_definition%axis.and.state%definition%index==cached_definition%index.and. &
          all(state%definition%ranges==cached_definition%ranges)
        if(reused) reused=size(state%definition%fields)==size(cached_definition%fields)
        if(reused) reused=all(state%definition%fields==cached_definition%fields)
      endif
      if(reused) then
        workspace=0;download=0;exchange=0;compute_seconds=0;ok=.true.
      else
        if(allocated(values)) deallocate(values)
        if(allocated(xyz)) deallocate(xyz)
        selection=state%definition
        if(selection%view%kind=='probe') then
          selection%axis=1;selection%index=selection%ranges(1,1)
        endif
        call prepare_index_slice(selection,source,count,values,xyz,first,extent,ok,message, &
          workspace,download,exchange,compute_seconds)
        cached=ok
        if(cached) then
          cached_definition=state%definition;cached_source=source
        endif
      endif
      if(ok) then
        if(state%definition%view%kind=='probe') then
          call write_probe(p,state,identity,values,xyz,window,measure,count,ok,message)
        else
          call write_product(p,state,identity,values,xyz,first,extent,window,measure,count,ok,message)
        endif
      endif
      if(ok) then
        call runtime_catalog_success(p,identity%step,real(identity%time,real64))
        write(*,'(a,i0,a,a,a,i0,a,i0,a,i0,a,i0,a,es16.8,a,l1)') &
          'ASTR_SLICE rank=',rank,' id=',state%definition%id,' step=',identity%step,' neighborhood_bytes=',workspace, &
          ' field_d2h_bytes=',download,' private_exchange_bytes=',exchange,' diagnostic_seconds=',compute_seconds, &
          ' reused_2d=',reused
      else
        call runtime_catalog_disable(p,identity%step,real(identity%time,real64),trim(message))
        if(rank==0) print '(a)','ASTR_SLICE_DISABLED '//state%definition%id//': '//trim(message)
      endif
      dirty=.true.
    enddo
    call refresh_selected_runtime()
    if(dirty) then
      if(.not.status_root_ready) then
        status=0
        if(rank==0) status=ensure_directory(status_root//c_null_char)
        call MPI_Bcast(status,1,MPI_INTEGER,0,MPI_COMM_WORLD,err);call mpi_ok(err)
        status_root_ready=status==0
      endif
      call write_catalog_status(status_path,identity%step,real(identity%time,real64),MPI_COMM_WORLD,update_message,ok)
    endif
  end subroutine

  logical function runtime_catalog_changed() result(changed)
    changed=control_changed
  end function

  subroutine refresh_runtime_product_status(identity)
    type(checkpoint_state_identity),intent(in) :: identity
    logical :: ok
    call write_catalog_status(status_path,identity%step,real(identity%time,real64),MPI_COMM_WORLD,'',ok)
  end subroutine

  subroutine runtime_product_segment(index,state,path,ok,message)
    integer,intent(in) :: index
    type(runtime_product_state),intent(in) :: state
    character(:),allocatable,intent(out) :: path
    logical,intent(out) :: ok
    character(*),intent(out) :: message
    path=''
    call begin_product_segment(index,state,ok,message)
    if(ok) path=segments(index)%path
  end subroutine

  subroutine begin_product_segment(index,state,ok,message)
    integer,intent(in) :: index
    type(runtime_product_state),intent(in) :: state
    logical,intent(out) :: ok
    character(*),intent(out) :: message
    character(c_char) :: path_buffer(4096),errors(256)
    character(:),allocatable :: root
    character(64) :: version
    integer :: status,rank,err,i,n
    call MPI_Comm_rank(MPI_COMM_WORLD,rank,err);call mpi_ok(err)
    ok=.false.;message=''
    write(version,'("version",i12.12)') state%version
    root=archive_root//'/'//state%definition%id//'/'//trim(version)
    if(state%definition%view%kind=='probe'.or.state%definition%view%kind=='m12_probes') &
      root=status_root//'/monitor_products/'//state%definition%id//'/'//trim(version)
    if(state%definition%view%kind=='time_space'.or.state%definition%view%kind=='probe_spectrum') &
      root=status_root//'/temporal_products/'//state%definition%id//'/'//trim(version)
    if(state%definition%view%kind=='station_profile') &
      root=status_root//'/profile_products/'//state%definition%id//'/'//trim(version)
    if(state%definition%view%kind=='wall_profile') &
      root=status_root//'/wall_profile_products/'//state%definition%id//'/'//trim(version)
    if(state%definition%view%kind=='wall_curve'.or.state%definition%view%kind=='m12_wall') &
      root=status_root//'/wall_products/'//state%definition%id//'/'//trim(version)
    if(state%definition%view%channel=='image') root=status_root//'/render_products/'//state%definition%id//'/'//trim(version)
    if(state%definition%view%channel=='geometry') root=status_root//'/geometry_products/'//state%definition%id//'/'//trim(version)
    if(segments(index)%version/=state%version.or.segments(index)%request/=state%consumed_request) then
      path_buffer=c_null_char;status=0
      if(rank==0) status=new_segment(root//c_null_char,path_buffer,size(path_buffer,kind=c_size_t))
      call MPI_Bcast(status,1,MPI_INTEGER,0,MPI_COMM_WORLD,err);call mpi_ok(err)
      if(status/=0) then
        message='cannot create a self-contained slice segment';return
      endif
      call MPI_Bcast(path_buffer,size(path_buffer),MPI_CHARACTER,0,MPI_COMM_WORLD,err);call mpi_ok(err)
      n=0
      do i=1,size(path_buffer)
        if(path_buffer(i)==c_null_char) exit
        n=n+1
      enddo
      segments(index)%path=repeat(' ',n)
      do i=1,n
        segments(index)%path(i:i)=path_buffer(i)
      enddo
      segments(index)%version=state%version;segments(index)%geometry=.false.
      segments(index)%csv_started=.false.
      segments(index)%request=state%consumed_request
      if(allocated(segments(index)%frames)) deallocate(segments(index)%frames)
      allocate(segments(index)%frames(0))
      status=0
      if(rank==0) status=publish_text(segments(index)%path//'/definition.json'//c_null_char, &
        state%definition%record,len(state%definition%record,kind=c_size_t),errors,size(errors,kind=c_size_t))
      call MPI_Bcast(status,1,MPI_INTEGER,0,MPI_COMM_WORLD,err);call mpi_ok(err)
      if(status/=0) then
        message='cannot save the effective product definition';return
      endif
    endif
    ok=.true.
  end subroutine

  subroutine sample_selected_wall_group(group,weights,previous,ok)
    use commvar, only: use_gpu,lihomo,ljhomo,lkhomo,im,jm,km
    use parallel, only: ig0,jg0,kg0
#ifdef _CUDA
    use cudafor, only: c_devptr
    use index_slice_gpu, only: index_slice_device_view,release_index_slice_gpu
    use insitu_sample_moments_gpu, only: push_scattered_moments_gpu
#endif
    type(selected_statistics_group),intent(inout) :: group
    real(real_kind),intent(in) :: weights(2)
    logical,intent(in) :: previous
    logical,intent(out) :: ok
    type(catalog_product) :: selection
    real(real_kind),allocatable :: values(:,:),xyz(:,:)
    real(real_kind) :: seconds,window(2),measure,fields(5)
    integer(int64) :: nn,capacity,extra_host,extra_device,workspace,download,exchange,count,key,peak_workspace,total_down,total_exchange
    integer,allocatable :: positions(:)
    integer :: n,nf,nraw,d,lo(3),hi(3),first(3),extent(3),origin(3),intervals(3),rank,ranks,err,p,source, &
      cells(3),owned_upper(3),xranges(2,2),zranges(2,2),nx,nz,xi,zi,m,maximum_rows,g,node(3),sampled
    logical :: ready,periodic(3),node_ok,gradient_needed
    character(512) :: message
    character(16) :: transport
#ifdef _CUDA
    type(c_devptr) :: data_pointer
#endif
    n=size(group%keys);nf=group%nfields
    selection%source='instantaneous';selection%statistics_source=''
    selection%axis=2;selection%index=0;selection%ranges=int(group%bounds,int64)
    allocate(selection%fields(nf+1));selection%fields=[1,group%field_ids(:nf)]
    call slice_source_info(selection,source,ready,ok,window,measure,count,message)
    if(.not.ok) return
    call MPI_Comm_rank(MPI_COMM_WORLD,rank,err);call mpi_ok(err)
    call MPI_Comm_size(MPI_COMM_WORLD,ranks,err);call mpi_ok(err)
    periodic=[lihomo,ljhomo,lkhomo];origin=[ig0,jg0,kg0];intervals=[ia,ja,ka];cells=[im,jm,km]
    owned_upper=origin+cells-1
    where(origin+cells==intervals.and..not.periodic) owned_upper=owned_upper+1
    ! Split periodic aliases without sampling the unrequested gaps between them.
    call canonical_node_intervals(group%bounds(1,1),group%bounds(2,1),ia,periodic(1),xranges,nx)
    call canonical_node_intervals(group%bounds(1,3),group%bounds(2,3),ka,periodic(3),zranges,nz)
    nraw=1;nn=0;maximum_rows=0
    gradient_needed=any(group%field_ids>=31.and.group%field_ids<=33)
    if(any(group%field_ids==31.or.group%field_ids==32)) nraw=nraw+3
    do zi=1,nz
    do xi=1,nx
      lo=max(origin,[xranges(1,xi),0,zranges(1,zi)])
      hi=min(owned_upper,[xranges(2,xi),0,zranges(2,zi)])
      if(any(hi<lo)) cycle
      maximum_rows=max(maximum_rows,product(hi-lo+1))
      if(gradient_needed) then
        do d=1,3
          lo(d)=slice_stencil_start(lo(d),intervals(d),periodic(d))
          hi(d)=slice_stencil_start(hi(d),intervals(d),periodic(d))+6
        enddo
      endif
      nn=max(nn,product(int(hi-lo+1,int64)))
    enddo
    enddo
    call MPI_Allreduce(nn,capacity,1,MPI_INTEGER8,MPI_MAX,MPI_COMM_WORLD,err);call mpi_ok(err)
    extra_host=(8*capacity+12_int64*ranks+maximum_rows)*storage_size(0)/8
    extra_device=0;transport='host'
    if(use_gpu) then
      transport='device-aware'
      extra_device=((nn+2*capacity)*nraw+int(maximum_rows,int64)*(nf+1)+49)*real_bytes+ &
        (8*capacity+68)*storage_size(0)/8
#ifdef _CUDA
      if(.not.associated(group%device%external_positions)) extra_device=extra_device+int(max(1,n),int64)*storage_size(0)/8
#endif
    else
      extra_host=extra_host+((nn+2*capacity)*nraw+int(maximum_rows,int64)*(nf+4))*real_bytes
    endif
    call selected_runtime_workspace(extra_host,extra_device,ok)
    if(.not.ok) return
    sampled=0;peak_workspace=0;total_down=0;total_exchange=0
    do zi=1,nz
    do xi=1,nx
      selection%ranges(:,1)=xranges(:,xi);selection%ranges(:,3)=zranges(:,zi)
      call prepare_index_slice(selection,0,0_int64,values,xyz,first,extent,ok,message,workspace,download,exchange,seconds, &
        retain_device=use_gpu,transport=transport,unique_periodic_nodes=.true.,retain_device_geometry=.false.)
      peak_workspace=max(peak_workspace,workspace);total_down=total_down+download;total_exchange=total_exchange+exchange
      if(.not.ok) exit
      m=product(extent);allocate(positions(m));g=1
      do p=1,m
        node=first-origin+[mod(p-1,extent(1)),mod((p-1)/extent(1),extent(2)),(p-1)/(extent(1)*extent(2))]
        key=node_key(node,cells)
        do while(g<=n)
          if(group%keys(g)>=key) exit
          g=g+1
        enddo
        if(g>n) then
          ok=.false.;exit
        endif
        if(group%keys(g)/=key) then
          ok=.false.;exit
        endif
        positions(p)=g
      enddo
      call agree_wall(ok)
      if(ok) then
#ifdef _CUDA
        if(use_gpu) then
          call index_slice_device_view(data_pointer)
          call push_scattered_moments_gpu(group%device,data_pointer,positions,nf,weights,previous,ok)
          if(m>0) total_down=total_down+storage_size(0)/8
        else
#endif
          do p=1,m
            fields(:nf)=values(p,2:)
            call push_sample_moments(group%states(positions(p),:),fields(:nf),values(p,1),weights,previous,[0,0,0],node_ok)
            ok=ok.and.node_ok
          enddo
#ifdef _CUDA
        endif
#endif
      endif
      deallocate(positions)
      call agree_wall(ok)
      if(.not.ok) exit
      sampled=sampled+m
#ifdef _CUDA
      if(use_gpu) call release_index_slice_gpu()
#endif
    enddo
      if(.not.ok) exit
    enddo
#ifdef _CUDA
    if(use_gpu) call release_index_slice_gpu()
#endif
    ok=ok.and.sampled==n;call agree_wall(ok)
    write(*,'(a,i0,a,i0,a,i0,a,i0,a,i0)') 'ASTR_WALL_SAMPLE rank=',rank,' source_nodes=',n, &
      ' private_workspace_bytes=',peak_workspace,' scalar_status_d2h_bytes=',total_down,' private_exchange_bytes=',total_exchange
    if(.not.ok) print '(a)','ASTR_WALL_SAMPLE_FAILED '//trim(message)
  contains
    subroutine agree_wall(valid)
      logical,intent(inout) :: valid
      integer :: local_bad,global_bad
      local_bad=merge(0,1,valid)
      call MPI_Allreduce(local_bad,global_bad,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,err);call mpi_ok(err)
      valid=global_bad==0
    end subroutine
  end subroutine

  subroutine write_wall_profile(index,state,identity,ok,message)
    use insitu_run_config, only: catalog_field_names
    use insitu_boundary_layer, only: wall_reference,wall_reference_result,resolve_recovery_temperature,wall_distance_plus
    integer,intent(in) :: index
    type(runtime_product_state),intent(in) :: state
    type(checkpoint_state_identity),intent(in) :: identity
    logical,intent(out) :: ok
    character(*),intent(out) :: message
    type(selected_profile_result) :: profile
    type(wall_reference_result),allocatable :: reference_values(:)
    real(real_kind),allocatable :: grid_distances(:,:)
    logical,allocatable :: grid_available(:,:)
    character(:),allocatable :: header,target,temporary,system,unit
    character(64) :: frame
    character(32) :: columns(22),time_unit,cp_unit,reference_mode
    integer :: rank,err,status,closed,unit_number,i,c,nf,slots(5),mode,normalization_status,ncolumns
    logical :: normalized,recovery_ready,row_ok,grid_ok
    real(real_kind) :: fixed(6),recovery,shear(3),distance_plus
    integer(int64) :: retained_bytes
    real(8) :: started,reference_started,product_started
    product_started=insitu_clock()
    call selected_runtime_profile(state%definition,profile,ok,message)
    if(.not.ok) return
    call MPI_Comm_rank(MPI_COMM_WORLD,rank,err);call mpi_ok(err)
    normalized=state%definition%wall_reference%flags(1)==1
    mode=state%definition%wall_reference%flags(2);recovery_ready=.false.;recovery=0
    nf=profile%nfields;normalization_status=0
    if(normalized) then
      reference_started=insitu_clock()
      slots=[findloc(profile%fields(:nf),6,dim=1),findloc(profile%fields(:nf),31,dim=1), &
        findloc(profile%fields(:nf),32,dim=1),findloc(profile%fields(:nf),33,dim=1), &
        findloc(profile%fields(:nf),34,dim=1)]
      if(any(slots==0)) then
        ok=.false.;message='wall reference dependencies were not accumulated together';return
      endif
      retained_bytes=int(size(profile%coordinate),int64)* &
        ((27+2*nf)*real_bytes+storage_size(wall_reference_result())/8)
      call selected_runtime_wall_grid(state%definition,retained_bytes,grid_distances,grid_available,ok,message)
      if(.not.ok) return
      call selected_runtime_wall_references(fixed)
      if(mode>0) call resolve_recovery_temperature(mode,real(state%definition%wall_reference%value,real_kind), &
        fixed(3),fixed(5),fixed(6),recovery,recovery_ready)
      if(rank==0) then
        allocate(reference_values(size(profile%coordinate)),stat=normalization_status)
        if(normalization_status==0) then
          do i=1,size(profile%coordinate)
            shear=[profile%mean_r(slots(2),i),0.e0_real_kind,profile%mean_r(slots(3),i)]
            if(recovery_ready) then
              call wall_reference(shear,profile%density(i),profile%mean_r(slots(5),i), &
                profile%mean_r(slots(1),i),profile%mean_r(slots(4),i),fixed(1),fixed(2),reference_values(i),row_ok, &
                cp_inf=fixed(4),recovery_temperature=recovery)
            else
              call wall_reference(shear,profile%density(i),profile%mean_r(slots(5),i), &
                profile%mean_r(slots(1),i),profile%mean_r(slots(4),i),fixed(1),fixed(2),reference_values(i),row_ok)
            endif
            if(.not.row_ok) then
              normalization_status=1;exit
            endif
          enddo
        endif
      endif
      call MPI_Bcast(normalization_status,1,MPI_INTEGER,0,MPI_COMM_WORLD,err);call mpi_ok(err)
      if(normalization_status/=0) then
        ok=.false.;message='cannot evaluate the one-dimensional wall reference table';return
      endif
      call report_insitu_timing('wall_reference_evaluate',reference_started,identity%step)
    endif
    call begin_product_segment(index,state,ok,message)
    if(.not.ok) return
    call MPI_Comm_rank(MPI_COMM_WORLD,rank,err);call mpi_ok(err)
    header='global_i,x,rho_time_mean'
    do c=1,nf
      header=header//','//trim(catalog_field_names(profile%fields(c)))
    enddo
    if(normalized) then
      header=header//',wall_scale_available,cf_available,cq_available,u_tau,nu_wall,viscous_length,Cf,Cq_inf'
      if(mode>0) header=header//',stanton_available,St_stat'
      header=header//',y_first_plus,dx_minus_plus,dx_plus_plus,dz_min_plus,dz_max_plus'
    endif
    write(frame,'("frame.step",i0.12,".csv")') identity%step
    target=segments(index)%path//'/'//trim(frame);temporary=target//'.tmp'
    started=insitu_clock();status=0;closed=0
    if(rank==0) then
      open(newunit=unit_number,file=temporary,status='new',action='write',iostat=status)
      if(status==0) then
        system='dimensionless';unit='1'
        if(.not.nondimen) then
          system='si';unit='m'
        endif
        write(unit_number,'(a)',iostat=status) &
          '# phase=completed_step; source=reynolds; statistics_source=selected-time; spatial_sampling='// &
          trim(state%definition%view%sampling)
        if(status==0) write(unit_number,'(a,i0,a,es24.16)',iostat=status) '# step=',identity%step,'; time=',identity%time
        if(status==0) write(unit_number,'(a,es24.16,a,a)',iostat=status) '# wall_y=',profile%fixed_coordinate, &
          '; coordinate_unit=',unit
        if(status==0) write(unit_number,'(a,2(i0,1x),a,2(es24.16,1x))',iostat=status) &
          '# global_k=',state%definition%ranges(:,3),'; native_span=',profile%span
        if(status==0) write(unit_number,'(a,es24.16)',iostat=status) '# spatial_measure=',profile%measure(1)
        if(status==0) write(unit_number,'(a,2(es24.16,1x),a,2(es24.16,1x),a,es24.16,a,i0)',iostat=status) &
          '# requested_window=',profile%window,'; actual_sample_range=',profile%actual_window, &
          '; covered_duration=',profile%duration,'; sampling_segments=',profile%segments
        if(status==0) write(unit_number,'(a)',iostat=status) &
          '# stress is the complete molecular tangential stress; heat is Fourier conduction positive into solid'
        if(status==0) write(unit_number,'(a)',iostat=status) &
          '# instantaneous material times gradient is accumulated before averaging; no product of primitive means'
        if(normalized) then
          time_unit='solver-dimensionless';cp_unit='solver-dimensionless'
          if(.not.nondimen) then
            time_unit='s';cp_unit='J kg^-1 K^-1'
          endif
          if(status==0) write(unit_number,'(a)',iostat=status) '# time_unit='//trim(time_unit)//'; cp_inf_unit='//trim(cp_unit)
          if(nondimen) then
            if(status==0) write(unit_number,'(a)',iostat=status) &
              '# u_tau_unit=solver-dimensionless; nu_wall_unit=solver-dimensionless; viscous_length_unit=solver-dimensionless'
          else
            if(status==0) write(unit_number,'(a)',iostat=status) &
              '# u_tau_unit=m s^-1; nu_wall_unit=m^2 s^-1; viscous_length_unit=m'
          endif
          if(status==0) write(unit_number,'(a)',iostat=status) '# Cf_unit=1; Cq_inf_unit=1; St_stat_unit=1'
          if(status==0) write(unit_number,'(a)',iostat=status) &
            '# grid columns are physical lengths divided by the same statistical viscous_length; all have unit 1'
          if(status==0) write(unit_number,'(a)',iostat=status) &
            '# y_first_plus uses the first off-wall node; dx_minus/plus use adjacent cells in decreasing/increasing index'
          if(status==0) write(unit_number,'(a)',iostat=status) &
            '# dz_min/max use cells contained in the selected span; a wall_line uses cells incident on its selected node'
          if(status==0) write(unit_number,'(a)',iostat=status) &
            '# periodic seams use real terminal cell lengths; absent nonperiodic neighbors and unavailable scales leave blank cells'
          if(status==0) write(unit_number,'(a,6(a,es25.16e3))',iostat=status) '# fixed_primary_free_stream', &
            '; rho_inf=',fixed(1),'; U_inf=',fixed(2),'; T_inf=',fixed(3),'; cp_inf=',fixed(4), &
            '; gamma=',fixed(5),'; Mach_inf=',fixed(6)
          if(status==0) write(unit_number,'(a)',iostat=status) &
            '# U_inf is fixed speed; Cf is signed global-x shear; wall scales use the magnitude of mean shear vector'
          if(status==0) write(unit_number,'(a)',iostat=status) &
            '# normalized values unavailable in floating point are empty CSV fields, with the corresponding flag equal to zero'
          if(mode==0) then
            if(status==0) write(unit_number,'(a)',iostat=status) '# Stanton number not requested: no recovery reference supplied'
          else
            reference_mode='temperature'
            if(mode==2) reference_mode='factor'
            if(status==0) write(unit_number,'(a,a,a,es25.16e3)',iostat=status) '# recovery_mode=',trim(reference_mode), &
              '; recovery_input=',state%definition%wall_reference%value
            if(recovery_ready) then
              if(status==0) write(unit_number,'(a,es25.16e3)',iostat=status) '# recovery_temperature=',recovery
            else
              if(status==0) write(unit_number,'(a)',iostat=status) '# recovery_reference_available=0; St_stat unavailable'
            endif
            if(status==0) write(unit_number,'(a)',iostat=status) &
              '# St_stat=mean(q_into_wall)/(rho_inf*U_inf*cp_inf*(Tr-mean(Tw))); not mean(St_instantaneous)'
          endif
          if(nondimen.and.status==0) write(unit_number,'(a)',iostat=status) &
            '# effective_mu=mu_phys/(rho_ref*U_ref*L_ref), including 1/Re; cp_inf=1/((gamma-1)*Mach_inf^2)'
        endif
        if(status==0) write(unit_number,'(a)',iostat=status) &
          '# rho_time_mean_unit='//catalog_quantity_units(1,system)
        do c=1,nf
          if(status==0) write(unit_number,'(a)',iostat=status) '# '//trim(catalog_field_names(profile%fields(c)))// &
            '_unit='//catalog_quantity_units(profile%fields(c),system)
        enddo
        if(status==0) write(unit_number,'(a)',iostat=status) header
        do i=1,size(profile%coordinate)
          if(normalized) then
            call number(profile%coordinate(i),.true.,columns(1));call number(profile%density(i),.true.,columns(2))
            do c=1,nf
              call number(profile%mean_r(c,i),.true.,columns(2+c))
            enddo
            ncolumns=2+nf
            columns(ncolumns+1)=merge('1','0',reference_values(i)%scale_available)
            columns(ncolumns+2)=merge('1','0',reference_values(i)%cf_available)
            columns(ncolumns+3)=merge('1','0',reference_values(i)%cq_available)
            call number(reference_values(i)%u_tau,reference_values(i)%scale_available,columns(ncolumns+4))
            call number(reference_values(i)%nu,reference_values(i)%scale_available,columns(ncolumns+5))
            call number(reference_values(i)%length,reference_values(i)%scale_available,columns(ncolumns+6))
            call number(reference_values(i)%cf,reference_values(i)%cf_available,columns(ncolumns+7))
            call number(reference_values(i)%cq,reference_values(i)%cq_available,columns(ncolumns+8))
            ncolumns=ncolumns+8
            if(mode>0) then
              columns(ncolumns+1)=merge('1','0',reference_values(i)%stanton_available)
              call number(reference_values(i)%stanton,reference_values(i)%stanton_available,columns(ncolumns+2))
              ncolumns=ncolumns+2
            endif
            do c=1,5
              distance_plus=0;grid_ok=.false.
              if(reference_values(i)%scale_available.and.grid_available(c,i)) &
                call wall_distance_plus(grid_distances(c,i),reference_values(i)%length,distance_plus,grid_ok)
              call number(distance_plus,grid_ok,columns(ncolumns+c))
            enddo
            ncolumns=ncolumns+5
            if(status==0) write(unit_number,'(i0,*(a))',iostat=status) int(state%definition%ranges(1,1))+i-1, &
              (','//trim(adjustl(columns(c))),c=1,ncolumns)
          else
            if(status==0) write(unit_number,'(i0,*(a,es25.16e3))',iostat=status) &
              int(state%definition%ranges(1,1))+i-1,',',profile%coordinate(i),',',profile%density(i), &
              (',',profile%mean_r(c,i),c=1,nf)
          endif
        enddo
        close(unit_number,iostat=closed)
        if(status==0.and.closed==0) status=rename_file(temporary//c_null_char,target//c_null_char)
      endif
    endif
    call MPI_Bcast(status,1,MPI_INTEGER,0,MPI_COMM_WORLD,err);call mpi_ok(err)
    call MPI_Bcast(closed,1,MPI_INTEGER,0,MPI_COMM_WORLD,err);call mpi_ok(err)
    ok=status==0.and.closed==0;message='wall profile CSV write/close/publication failed'
    if(ok) message=''
    call report_insitu_timing('wall_profile_csv',started,identity%step)
    call report_insitu_timing('wall_profile_publish_total',product_started,identity%step)
  contains
    subroutine number(value,available,text)
      real(real_kind),intent(in) :: value
      logical,intent(in) :: available
      character(*),intent(out) :: text
      text=''
      if(available) write(text,'(es25.16e3)') value
    end subroutine
  end subroutine

  subroutine write_station_profile(index,state,identity,ok,message)
    use insitu_run_config, only: catalog_field_names
    use insitu_boundary_layer, only: wall_reference_result,wall_scaled_quantity,boundary_layer_edge, &
      boundary_layer_thickness,thickness_result,nonreacting_wall_material,thickness_reynolds
    use commvar, only: tempconst,tempconst1,reynolds,prandtl,const5,cp
    use bc, only: bctype
    integer,intent(in) :: index
    type(runtime_product_state),intent(in) :: state
    type(checkpoint_state_identity),intent(in) :: identity
    logical,intent(out) :: ok
    character(*),intent(out) :: message
    type(selected_profile_result) :: profile
    type(wall_reference_result) :: wall
    type(thickness_result) :: layer
    character(:),allocatable :: header,target,temporary,system,unit,means_unit
    character(512) :: units_message
    character(512) :: layer_message
    character(64) :: frame
    character(2),parameter :: pairs(6)=['xx','xy','yy','xz','yz','zz']
    real(real_kind) :: row(22)
    real(real_kind) :: rho_wall,wall_y
    real(real_kind) :: edge(4),edge_mu,unused_conductivity,layer_re(2)
    real(real_kind),allocatable :: cumulative_d(:),cumulative_t(:)
    character(32) :: raw_columns(22),unit_columns(20)
    character(32) :: layer_columns(3)
    integer :: rank,err,status,closed,unit_number,j,m,n,nf,c,unit_count
    integer(int64) :: retained
    logical :: units_requested,units_ready,reference_ok
    logical :: layer_requested,layer_ready,edge_ready,mu_ready,layer_re_ready(2)
    real(8) :: started
    call selected_runtime_profile(state%definition,profile,ok,message)
    if(.not.ok) return
    units_requested=.false.;units_ready=.false.;units_message='';rho_wall=0;wall_y=0
    if(allocated(state%definition%wall_units)) units_requested=len(state%definition%wall_units)>0
    if(units_requested) then
      retained=int(size(profile%coordinate),int64)*(27+2*profile%nfields)*real_bytes
      call selected_runtime_wall_scale(state%definition,retained,wall,rho_wall,wall_y,reference_ok,units_message)
      units_ready=reference_ok.and.wall%scale_available
    endif
    layer_requested=state%definition%thickness%flags(1)==1
    if(layer_requested) then
      started=insitu_clock()
      call prepare_thickness()
      call report_insitu_timing('station_thickness_diagnostic',started,identity%step)
      if(.not.ok) return
    endif
    call begin_product_segment(index,state,ok,message)
    if(.not.ok) return
    call MPI_Comm_rank(MPI_COMM_WORLD,rank,err);call mpi_ok(err)
    nf=profile%nfields
    header='global_j,y,rho_time_mean'
    do m=1,nf
      header=header//','//trim(catalog_field_names(profile%fields(m)))
    enddo
    if(profile%covariance) then
      header=header//',rms_velocity_x,rms_velocity_y,rms_velocity_z'
      do m=1,6
        header=header//',covariance_'//pairs(m)
      enddo
      do m=1,6
        header=header//',rho_covariance_f_'//pairs(m)
      enddo
    endif
    if(units_requested) then
      header=header//',wall_scale_available,y_plus'
      do m=1,nf
        if(profile%fields(m)>=2.and.profile%fields(m)<=4) &
          header=header//','//trim(catalog_field_names(profile%fields(m)))//'_plus'
      enddo
      if(profile%covariance) then
        header=header//',rms_velocity_x_plus,rms_velocity_y_plus,rms_velocity_z_plus'
        do m=1,6
          header=header//',covariance_'//pairs(m)//'_plus'
        enddo
        do m=1,6
          header=header//',rho_covariance_f_'//pairs(m)//'_plus'
        enddo
      endif
    endif
    if(layer_requested) header=header//',thickness_integrals_available,cumulative_displacement,cumulative_momentum'
    write(frame,'("frame.step",i0.12,".csv")') identity%step
    target=segments(index)%path//'/'//trim(frame);temporary=target//'.tmp'
    started=insitu_clock();status=0;closed=0
    if(rank==0) then
      open(newunit=unit_number,file=temporary,status='new',action='write',iostat=status)
      if(status==0) then
        system='dimensionless';unit='1'
        if(.not.nondimen) then
          system='si';unit='m'
        endif
        write(unit_number,'(a)',iostat=status) '# phase=completed_step; source='//state%definition%source// &
          '; statistics_source=selected-time; spatial_sampling='//trim(state%definition%view%sampling)
        if(status==0) write(unit_number,'(a,i0,a,es24.16)',iostat=status) '# step=',identity%step,'; time=',identity%time
        if(status==0) write(unit_number,'(a,i0,a,es24.16,a,a)',iostat=status) '# global_i=',state%definition%index, &
          '; x=',profile%fixed_coordinate,'; coordinate_unit=',unit
        if(status==0) write(unit_number,'(a,2(i0,1x),a,2(es24.16,1x))',iostat=status) &
          '# global_k=',state%definition%ranges(:,3),'; native_span=',profile%span
        if(status==0) write(unit_number,'(a,es24.16)',iostat=status) '# spatial_measure=',profile%measure(1)
        if(status==0) write(unit_number,'(a,2(es24.16,1x),a,2(es24.16,1x),a,es24.16,a,i0)',iostat=status) &
          '# requested_window=',profile%window,'; actual_sample_range=',profile%actual_window, &
          '; covered_duration=',profile%duration,'; sampling_segments=',profile%segments
        if(status==0) write(unit_number,'(a)',iostat=status) '# rho_time_mean is the ordinary time mean of density, not Favre density'
        if(status==0) write(unit_number,'(a)',iostat=status) &
          '# covariance is the spatial average of local temporal covariance; spatial variation of local means is not added'
        if(status==0) write(unit_number,'(a)',iostat=status) &
          '# rho_covariance_f is the spatial average of mean_time(rho*u_double_prime_i*u_double_prime_j); no minus sign'
        if(layer_requested) call write_thickness_metadata()
        if(units_requested) then
          if(status==0) write(unit_number,'(a)',iostat=status) &
            '# wall_units_reference='//state%definition%wall_units//'; matching positive-weight sampling identity required'
          if(status==0) write(unit_number,'(a)',iostat=status) &
            '# wall units use statistical mean wall vector/rho/mu, not an average of instantaneous normalizations'
          if(status==0) write(unit_number,'(a)',iostat=status) &
            '# y_plus=(y-y_wall)/viscous_length; velocity and RMS plus divide by u_tau; covariance plus divides by u_tau^2'
          if(status==0) write(unit_number,'(a)',iostat=status) &
            '# rho_covariance_f_plus divides by rho_wall*u_tau^2; all plus columns have unit 1 and retain the covariance sign'
          if(units_ready) then
            if(status==0) write(unit_number,'(a,4(a,es25.16e3))',iostat=status) '# statistical_wall_reference', &
              '; rho_wall=',rho_wall,'; y_wall=',wall_y,'; u_tau=',wall%u_tau,'; viscous_length=',wall%length
          else
            if(status==0) write(unit_number,'(a)',iostat=status) '# wall_units_unavailable='//trim(units_message)
          endif
          if(status==0) write(unit_number,'(a)',iostat=status) &
            '# unavailable normalized quantities are empty cells; raw statistics are retained without replacement'
        endif
        means_unit=catalog_quantity_units(1,system)
        if(status==0) write(unit_number,'(a)',iostat=status) '# rho_time_mean_unit='//trim(means_unit)
        do m=1,nf
          means_unit=catalog_quantity_units(profile%fields(m),system)
          if(status==0) write(unit_number,'(a)',iostat=status) '# '//trim(catalog_field_names(profile%fields(m)))// &
            '_unit='//trim(means_unit)
        enddo
        if(profile%covariance) then
          if(nondimen) then
            if(status==0) write(unit_number,'(a)',iostat=status) '# rms_unit=1; covariance_unit=1; rho_covariance_f_unit=1'
          else
            if(status==0) write(unit_number,'(a)',iostat=status) '# rms_unit=m/s; covariance_unit=m^2/s^2; rho_covariance_f_unit=Pa'
          endif
        endif
        if(status==0) write(unit_number,'(a)',iostat=status) header
        do j=1,size(profile%coordinate)
          row=0;row(1)=profile%coordinate(j);row(2)=profile%density(j);n=2+nf
          if(state%definition%source=='favre') then
            row(3:n)=profile%mean_f(:,j)
          else
            row(3:n)=profile%mean_r(:,j)
          endif
          if(profile%covariance) then
            if(state%definition%source=='favre') then
              row(n+1:n+3)=profile%rms_f(:,j);row(n+4:n+9)=profile%covariance_f(:,j)
            else
              row(n+1:n+3)=profile%rms_r(:,j);row(n+4:n+9)=profile%covariance_r(:,j)
            endif
            row(n+10:n+15)=profile%density_stress(:,j);n=n+15
          endif
          if(units_requested.or.layer_requested) then
            do c=1,n
              write(raw_columns(c),'(es25.16e3)') row(c)
            enddo
            unit_columns='';unit_count=0
            if(units_requested) then
            unit_count=1;unit_columns(1)=merge('1','0',units_ready)
            call append_unit(profile%coordinate(j)-wall_y,[wall%length])
            do m=1,nf
              if(profile%fields(m)<2.or.profile%fields(m)>4) cycle
              call append_unit(row(2+m),[wall%u_tau])
            enddo
            if(profile%covariance) then
              do m=1,3
                call append_unit(row(2+nf+m),[wall%u_tau])
              enddo
              do m=1,6
                call append_unit(row(5+nf+m),[wall%u_tau,wall%u_tau])
              enddo
              do m=1,6
                call append_unit(row(11+nf+m),[rho_wall,wall%u_tau,wall%u_tau])
              enddo
            endif
            endif
            layer_columns=''
            if(layer_requested) then
              layer_columns(1)=merge('1','0',layer_ready)
              if(layer_ready) then
                write(layer_columns(2),'(es25.16e3)') cumulative_d(j)
                write(layer_columns(3),'(es25.16e3)') cumulative_t(j)
              endif
            endif
            if(status==0) write(unit_number,'(i0,*(a))',iostat=status) int(state%definition%ranges(1,2))+j-1, &
              (','//trim(adjustl(raw_columns(c))),c=1,n),(','//trim(adjustl(unit_columns(c))),c=1,unit_count), &
              (','//trim(adjustl(layer_columns(c))),c=1,merge(3,0,layer_requested))
          else
            if(status==0) write(unit_number,'(i0,*(a,es25.16e3))',iostat=status) &
              int(state%definition%ranges(1,2))+j-1,(',',row(c),c=1,n)
          endif
        enddo
        close(unit_number,iostat=closed)
        if(status==0.and.closed==0) status=rename_file(temporary//c_null_char,target//c_null_char)
      endif
    endif
    call MPI_Bcast(status,1,MPI_INTEGER,0,MPI_COMM_WORLD,err);call mpi_ok(err)
    call MPI_Bcast(closed,1,MPI_INTEGER,0,MPI_COMM_WORLD,err);call mpi_ok(err)
    ok=status==0.and.closed==0;message='station profile CSV write/close/publication failed'
    if(ok) message=''
    call report_insitu_timing('station_profile_csv',started,identity%step)
  contains
    subroutine prepare_thickness()
      integer :: band(2),slots(3),ny,allocation_status
      layer=thickness_result();edge=0;edge_mu=0;layer_re=0
      layer_ready=.false.;edge_ready=.false.;mu_ready=.false.;layer_re_ready=.false.
      layer_message='thickness needs an axis-aligned bottom wall with inward +y'
      ny=size(profile%coordinate)
      ! Retained profile plus cumulative, quadrature and edge-weight workspaces.
      call selected_runtime_workspace(int(35+2*profile%nfields,int64)*ny*real_bytes,0_int64,ok)
      message='thickness profile workspace exceeds configured resources'
      if(.not.ok) return
      allocate(cumulative_d(ny),cumulative_t(ny),stat=allocation_status)
      ok=allocation_status==0
      call MPI_Allreduce(MPI_IN_PLACE,ok,1,MPI_LOGICAL,MPI_LAND,MPI_COMM_WORLD,err);call mpi_ok(err)
      message='thickness cumulative profile allocation failed'
      if(.not.ok) return
      cumulative_d=0;cumulative_t=0
      if(.not.any(bctype(3)==[41,42,411,421]).or..not.selected_runtime_cartesian(.true.)) return
      slots=[findloc(profile%fields(:profile%nfields),2,dim=1),findloc(profile%fields(:profile%nfields),6,dim=1), &
        findloc(profile%fields(:profile%nfields),5,dim=1)]
      if(any(slots==0)) then
        layer_message='thickness history lacks velocity_x, temperature or pressure';return
      endif
      band=int(state%definition%thickness%edge_band-state%definition%ranges(1,2))+1
      call boundary_layer_edge(profile%coordinate,profile%density,profile%mean_f(slots(1),:), &
        profile%mean_r(slots(2),:),profile%mean_r(slots(3),:),band,edge,edge_ready)
      layer_message='external reference needs finite positive rho/U/T/p on an increasing explicit band'
      if(.not.edge_ready) return
      call nonreacting_wall_material(edge(3),nondimen,tempconst,tempconst1,reynolds,prandtl,const5,cp, &
        .false.,edge_mu,unused_conductivity,mu_ready)
      if(state%definition%thickness%flags(2)==1) then
        call boundary_layer_thickness(profile%coordinate,profile%density,profile%mean_f(slots(1),:),edge(1),edge(2), &
          band,layer,cumulative_d,cumulative_t,layer_ready, &
          real(state%definition%thickness%explicit_upper,real_kind))
      else
        call boundary_layer_thickness(profile%coordinate,profile%density,profile%mean_f(slots(1),:),edge(1),edge(2), &
          band,layer,cumulative_d,cumulative_t,layer_ready)
      endif
      layer_message='thickness integral requires finite positive density and strictly increasing physical y'
      if(layer_ready) layer_message=''
      if(mu_ready.and.layer%delta_available) call thickness_reynolds(edge(1),edge(2),layer%momentum99,edge_mu, &
        layer_re(1),layer_re_ready(1))
      if(mu_ready.and.layer%explicit_available) call thickness_reynolds(edge(1),edge(2),layer%momentum_explicit,edge_mu, &
        layer_re(2),layer_re_ready(2))
    end subroutine

    subroutine write_thickness_metadata()
      if(status==0) write(unit_number,'(a,2(i0,1x))',iostat=status) '# thickness_edge_global_j=', &
        state%definition%thickness%edge_band
      if(status==0) write(unit_number,'(a)',iostat=status) &
        '# thickness_edge_estimator=physical-y trapezoid length means rho/p/T; density-length weighted Favre U'
      if(status==0) write(unit_number,'(a)',iostat=status) &
        '# thickness_velocity=local temporal Favre mean with declared span processing; edge p/T are Reynolds means'
      if(status==0) write(unit_number,'(a)',iostat=status) &
        '# thickness_integration=piecewise-linear integrands, trapezoid and partial final interval; no extrapolation'
      if(status==0) write(unit_number,'(a)',iostat=status) &
        '# thickness_lengths_unit='//unit//'; H and Re_theta are dimensionless; cumulative origins are the wall'
      if(status==0) write(unit_number,'(a,6(i0,1x))',iostat=status) '# thickness_available=', &
        merge(1,0,edge_ready),merge(1,0,layer_ready),merge(1,0,layer%delta_available), &
        merge(1,0,layer%explicit_available),merge(1,0,layer_re_ready(1)),merge(1,0,layer_re_ready(2))
      if(status==0) write(unit_number,'(a)',iostat=status) &
        '# thickness_available_order=edge,cumulative,delta99,explicit_upper,Re_theta99,Re_theta_explicit'
      if(len_trim(layer_message)>0.and.status==0) write(unit_number,'(a)',iostat=status) &
        '# thickness_unavailable='//trim(layer_message)
      if(edge_ready) then
        call layer_scalar('rho_edge',edge(1));call layer_scalar('U_edge',edge(2))
        call layer_scalar('T_edge',edge(3));call layer_scalar('p_edge',edge(4))
      endif
      if(mu_ready) then
        call layer_scalar('mu_edge',edge_mu)
        if(status==0) write(unit_number,'(a)',iostat=status) &
          '# thickness_mu_edge_definition=case viscosity at Reynolds mean edge temperature; not mean instantaneous viscosity'
      endif
      if(status==0) write(unit_number,'(a,i0)',iostat=status) '# thickness_threshold_components=',layer%threshold_components
      if(layer%delta_available) then
        call layer_scalar('delta99',layer%delta99);call layer_scalar('displacement99',layer%displacement99)
        call layer_scalar('momentum99',layer%momentum99)
        if(layer%shape99_available) call layer_scalar('H99',layer%shape99)
      endif
      if(layer%explicit_available) then
        call layer_scalar('explicit_upper',layer%upper);call layer_scalar('displacement_explicit',layer%displacement_explicit)
        call layer_scalar('momentum_explicit',layer%momentum_explicit)
        if(layer%shape_explicit_available) call layer_scalar('H_explicit',layer%shape_explicit)
      endif
      if(state%definition%thickness%flags(2)==1.and..not.layer%explicit_available.and.status==0) &
        write(unit_number,'(a)',iostat=status) '# thickness_explicit_unavailable=upper outside measured y range or invalid integrand'
      if(.not.layer%delta_available.and.status==0) write(unit_number,'(a)',iostat=status) &
        '# thickness_delta99_unavailable=no crossing connected to the entire declared edge band'
      if(layer_re_ready(1)) call layer_scalar('Re_theta99',layer_re(1))
      if(layer_re_ready(2)) call layer_scalar('Re_theta_explicit',layer_re(2))
    end subroutine

    subroutine layer_scalar(name,value)
      character(*),intent(in) :: name
      real(real_kind),intent(in) :: value
      if(status==0) write(unit_number,'(a,es25.16e3)',iostat=status) '# thickness_'//name//'=',value
    end subroutine

    subroutine append_unit(quantity,denominators)
      real(real_kind),intent(in) :: quantity,denominators(:)
      real(real_kind) :: value
      logical :: valid
      unit_count=unit_count+1
      if(.not.units_ready) return
      call wall_scaled_quantity(quantity,denominators,value,valid)
      if(valid) write(unit_columns(unit_count),'(es25.16e3)') value
    end subroutine
  end subroutine

  subroutine write_probe(index,state,identity,values,xyz,window,measure,count,ok,message)
    integer,intent(in) :: index
    type(runtime_product_state),intent(in) :: state
    type(checkpoint_state_identity),intent(in) :: identity
    real(real_kind),intent(in) :: values(:,:),xyz(:,:),window(2),measure
    integer(int64),intent(in) :: count
    logical,intent(out) :: ok
    character(*),intent(out) :: message
    real(real_kind) :: local(size(values,2)+3),global(size(values,2)+3)
    real(real64) :: recorded(6)
    character(:),allocatable :: header,line,filename,system,provider
    character(32) :: field_unit
    character(32) :: number
    real(real64) :: references(12)
    integer :: rank,err,status,closed,unit,i,owned,total
    real(8) :: started
    call MPI_Comm_rank(MPI_COMM_WORLD,rank,err);call mpi_ok(err)
    ok=.false.;message='';local=0;owned=size(values,1)
    call MPI_Allreduce(owned,total,1,MPI_INTEGER,MPI_SUM,MPI_COMM_WORLD,err);call mpi_ok(err)
    if(total/=1) then
      message='a grid-node probe requires exactly one output owner';return
    endif
    if(owned==1) local=[xyz(:,1),values(1,:)]
    call MPI_Reduce(local,global,size(local),mpi_real_type,MPI_SUM,0,MPI_COMM_WORLD,err);call mpi_ok(err)
    call begin_product_segment(index,state,ok,message)
    if(.not.ok) return
    started=insitu_clock();status=0
    if(rank==0) then
      filename=segments(index)%path//'/samples.csv'
      if(segments(index)%csv_started) then
        open(newunit=unit,file=filename,status='old',position='append',action='write',iostat=status)
      else
        open(newunit=unit,file=filename,status='new',action='write',iostat=status)
      endif
      if(status==0) then
        if(.not.segments(index)%csv_started) then
          write(number,'(i0)') real_bytes
          line='# phase=completed_step; source='//state%definition%source//'; source_bytes='//trim(number)
          if(len(state%definition%statistics_source)>0) &
            line=line//'; statistics_source='//state%definition%statistics_source
          write(unit,'(a)',iostat=status) line
          system='dimensionless'
          if(.not.nondimen) system='si'
          line='# unit_system='//system
          do i=1,size(state%definition%fields)
            block
              use insitu_run_config, only: catalog_field_names
              field_unit=catalog_quantity_units(state%definition%fields(i),system)
              line=line//'; '//trim(catalog_field_names(state%definition%fields(i)))//'='//trim(field_unit)
            end block
          enddo
          if(status==0) write(unit,'(a)',iostat=status) line
          provider='not_applicable'
          if(state%definition%source/='instantaneous') then
            provider='velocity-time'
            if(count>0) provider='solver-samples'
            if(state%definition%statistics_source=='selected-time') provider='selected-time'
          endif
          if(status==0) write(unit,'(a)',iostat=status) '# statistics_provider='//provider
          references=real([roinf,uinf,vinf,winf,pinf,tinf,ref_den,ref_vel,ref_tem,ref_len,ref_tim,ref_miu],real64)
          line='# reference_state[rhoinf,uinf,vinf,winf,pinf,tinf,ref_den,ref_vel,ref_tem,ref_len,ref_tim,ref_miu]='
          do i=1,size(references)
            write(number,'(es26.17e3)') references(i)
            if(i>1) line=line//','
            line=line//trim(adjustl(number))
          enddo
          if(status==0) write(unit,'(a)',iostat=status) line
          header='step,time,x,y,z,window_start,window_end,covered_duration,sample_count'
          do i=1,size(state%definition%fields)
            block
              use insitu_run_config, only: catalog_field_names
              header=header//','//trim(catalog_field_names(state%definition%fields(i)))
            end block
          enddo
          if(status==0) write(unit,'(a)',iostat=status) header
        endif
        write(number,'(i0)') identity%step;line=trim(number)
        recorded=[real(identity%time,real64),real(global(1:3),real64),real(window,real64)]
        do i=1,size(recorded)
          write(number,'(es26.17e3)') recorded(i)
          line=line//','//trim(adjustl(number))
        enddo
        write(number,'(es26.17e3)') real(measure,real64);line=line//','//trim(adjustl(number))
        write(number,'(i0)') count;line=line//','//trim(number)
        do i=4,size(global)
          write(number,'(es26.17e3)') real(global(i),real64);line=line//','//trim(adjustl(number))
        enddo
        if(status==0) write(unit,'(a)',iostat=status) line
        close(unit,iostat=closed)
        if(closed/=0) status=closed
      endif
    endif
    call MPI_Bcast(status,1,MPI_INTEGER,0,MPI_COMM_WORLD,err);call mpi_ok(err)
    ok=status==0
    if(ok) then
      segments(index)%csv_started=.true.
    else
      message='probe CSV write/close failed'
    endif
    call report_insitu_timing('runtime_probe_csv',started,identity%step)
  end subroutine

  subroutine write_product(index,state,identity,values,xyz,first,extent,window,measure,count,ok,message)
    integer,intent(in) :: index,first(3),extent(3)
    type(runtime_product_state),intent(in) :: state
    type(checkpoint_state_identity),intent(in) :: identity
    real(real_kind),intent(in) :: values(:,:),xyz(:,:),window(2),measure
    integer(int64),intent(in) :: count
    logical,intent(out) :: ok
    character(*),intent(out) :: message
    type(saved_frame),allocatable :: frames(:)
    character(:),allocatable :: names,units,temporary,target,provenance,system,provider
    character(64) :: frame
    character(512) :: description
    integer :: status,i,n,axes(2),dimensions(2),offset(2),local_extent(2),bounds(2,3),d
    real(real64) :: clock(3),references(12)
    real(real_kind) :: requested(2)
    integer(int64) :: sampling_segments
    real(8) :: phase_started
    call begin_product_segment(index,state,ok,message)
    if(.not.ok) return
    ok=.false.
    bounds=int(state%definition%ranges);bounds(:,state%definition%axis)=int(state%definition%index)
    n=0
    do d=1,3
      if(d==state%definition%axis) cycle
      n=n+1;axes(n)=d
    enddo
    dimensions=bounds(2,axes)-bounds(1,axes)+1;offset=first(axes)-bounds(1,axes);local_extent=extent(axes)
    names='';units='';system='dimensionless'
    if(.not.nondimen) system='si'
    do i=1,size(state%definition%fields)
      block
        use insitu_run_config, only: catalog_field_names
        character(32) :: unit
        names=names//catalog_field_names(state%definition%fields(i))
        unit=catalog_quantity_units(state%definition%fields(i),system);units=units//unit
      end block
    enddo
    provider=state%definition%statistics_source
    if(state%definition%source=='instantaneous') then
      description='phase=completed_step; weight=not_applicable'
    elseif(count>0) then
      provider='solver-samples'
      write(description,'(a,2(es24.16,1x),a,i0,a)') 'sample_time_range=',window,'; samples=',count, &
        '; weight=equal_sample_count; sampling_phase=existing_pre_step_RK1'
    else
      if(provider=='selected-time') then
        call selected_runtime_sampling_info(state%definition%id,requested,sampling_segments)
        write(description,'(a,2(es24.16,1x),a,2(es24.16,1x),a,es24.16,a,i0)') &
          'requested_window=',requested,'; actual_sample_range=',window,'; covered_duration=',measure, &
          '; sampling_segments=',sampling_segments
        description=trim(description)//'; phase=completed_step; gaps_excluded=true'
      else
        provider='velocity-time'
        write(description,'(a,2(es24.16,1x),a,es24.16)') 'requested_window=',window,'; covered_duration=',measure
      endif
      if(state%definition%source=='favre') then
        description=trim(description)//'; weight=density_time_integral'
      else
        description=trim(description)//'; weight=time_integral'
      endif
    endif
    provenance='source='//state%definition%source//'; statistics_source='//provider//'; '//trim(description)// &
      '; values='//system//'; diagnostics=explicit6-seven-node; periodic-stencil-endpoint=node0'
    if(state%definition%fluctuation) provenance=provenance// &
      '; quantity=instantaneous_minus_local_accumulated_mean; current_phase=completed_step'
    if(catalog_wall_mean(state%definition)) provenance=provenance// &
      '; wall_transport=mean_of_instantaneous_complete_viscous_tensor_and_Fourier_flux; spatial_reduction=none'
    clock=real([identity%time,identity%dt_used,identity%dt_next],real64)
    references=real([roinf,uinf,vinf,winf,pinf,tinf,ref_den,ref_vel,ref_tem,ref_len,ref_tim,ref_miu],real64)
    phase_started=insitu_clock()
    if(.not.segments(index)%geometry) then
      temporary=segments(index)%path//'/geometry.h5.tmp';target=segments(index)%path//'/geometry.h5'
      status=write_slice(temporary//c_null_char,'geometry.h5'//c_null_char,1_c_int,int(MPI_COMM_WORLD,c_int), &
        real_bytes,int(dimensions,c_int),int(offset,c_int),int(local_extent,c_int), &
        size(state%definition%fields),names,units,int(state%definition%fields,int64),values,xyz, &
        identity%step,clock,references,int(reshape(bounds,[6]),int64), &
        state%definition%record//c_null_char,provenance//c_null_char)
      if(status==0) call publish_hdf(temporary,target,status)
      if(status/=0) then
        message='slice geometry HDF5 write/close/publication failed';return
      endif
      segments(index)%geometry=.true.
    endif
    write(frame,'("step",i0.12,".h5")') identity%step
    temporary=segments(index)%path//'/'//trim(frame)//'.tmp';target=segments(index)%path//'/'//trim(frame)
    status=write_slice(temporary//c_null_char,'geometry.h5'//c_null_char,0_c_int,int(MPI_COMM_WORLD,c_int), &
      real_bytes,int(dimensions,c_int),int(offset,c_int),int(local_extent,c_int), &
      size(state%definition%fields),names,units,int(state%definition%fields,int64),values,xyz, &
      identity%step,clock,references,int(reshape(bounds,[6]),int64), &
      state%definition%record//c_null_char,provenance//c_null_char)
    if(status==0) call publish_hdf(temporary,target,status)
    if(status/=0) then
      message='slice field HDF5 write/close/publication failed';return
    endif
    call report_insitu_timing('index_slice_hdf5',phase_started,identity%step)
    n=size(segments(index)%frames)
    allocate(frames(n+1));frames(:n)=segments(index)%frames
    frames(n+1)%file=frame;frames(n+1)%identity=identity
    phase_started=insitu_clock()
    call write_series(index,state,frames,dimensions,system,status)
    call report_insitu_timing('index_slice_index',phase_started,identity%step)
    if(status/=0) then
      message='cannot publish slice time index; the failed frame was not indexed';return
    endif
    call move_alloc(frames,segments(index)%frames)
    ok=.true.
  end subroutine

  subroutine write_series(index,state,frames,dimensions,system,status)
    integer,intent(in) :: index,dimensions(2)
    type(runtime_product_state),intent(in) :: state
    type(saved_frame),intent(in) :: frames(:)
    character(*),intent(in) :: system
    integer,intent(out) :: status
    character(:),allocatable :: temporary,target
    integer :: rank,err,unit,closed,i
    temporary=segments(index)%path//'/series.xdmf.tmp';target=segments(index)%path//'/series.xdmf'
    call MPI_Comm_rank(MPI_COMM_WORLD,rank,err);call mpi_ok(err)
    status=0
    if(rank==0) then
      open(newunit=unit,file=temporary,status='replace',action='write',iostat=status)
      if(status==0) then
        write(unit,'(a)',iostat=status) '<?xml version="1.0"?>', &
          '<Xdmf Version="3.0"><Domain><Grid Name="series" GridType="Collection" CollectionType="Temporal">'
        do i=1,size(frames)
          if(status/=0) exit
          call write_grid_xml(unit,dimensions,6,frames(i)%identity,system,'slice','',status, &
            geometry_file='geometry.h5',data_file=trim(frames(i)%file),field_selection=state%definition%fields)
        enddo
        if(status==0) write(unit,'(a)',iostat=status) '</Grid></Domain></Xdmf>'
        close(unit,iostat=closed)
        if(closed/=0) status=closed
        if(status==0) status=rename_file(temporary//c_null_char,target//c_null_char)
      endif
    endif
    call MPI_Bcast(status,1,MPI_INTEGER,0,MPI_COMM_WORLD,err);call mpi_ok(err)
  end subroutine

  subroutine publish_hdf(temporary,target,status)
    character(*),intent(in) :: temporary,target
    integer,intent(out) :: status
    integer :: rank,err
    call MPI_Comm_rank(MPI_COMM_WORLD,rank,err);call mpi_ok(err)
    status=0
    if(rank==0) status=rename_file(temporary//c_null_char,target//c_null_char)
    call MPI_Bcast(status,1,MPI_INTEGER,0,MPI_COMM_WORLD,err);call mpi_ok(err)
  end subroutine

  subroutine mpi_ok(code)
    integer,intent(in) :: code
    integer :: err
    if(code/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,code,err)
  end subroutine
end module
