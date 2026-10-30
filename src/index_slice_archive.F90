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
  use insitu_run_config, only: catalog_product
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
    call report_insitu_timing('index_slice_catalog',phase_started,int(identity%step))
    dirty=changed.or.len(update_message)>0
    control_changed=changed
    if(rank==0.and.len(update_message)>0) print '(a)','ASTR_CATALOG '//update_message
    n=runtime_catalog_count()
    if(n>size(segments)) then
      allocate(expanded(n));expanded(:size(segments))=segments
      call move_alloc(expanded,segments)
    endif
    do p=1,n
      call runtime_catalog_state(p,state)
      if(state%status/=runtime_active.and.state%status/=runtime_waiting) cycle
      if(state%definition%view%channel/='data') cycle
      if(state%definition%view%kind/='slice'.and.state%definition%view%kind/='probe') cycle
      call slice_source_info(state%definition,source,ready,ok,window,measure,count,message)
      if(.not.ok) then
        call runtime_catalog_reject(p,trim(message));dirty=.true.;cycle
      endif
      old_status=state%status
      call runtime_catalog_wait(p,.not.ready)
      if((old_status==runtime_waiting).eqv.ready) dirty=.true.
      if(.not.ready) cycle
      if(.not.runtime_catalog_due(p,identity%step,real(identity%time,real64))) cycle
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
    if(state%definition%view%kind=='probe') root=status_root//'/monitor_products/'//state%definition%id//'/'//trim(version)
    if(state%definition%view%kind=='wall_curve') root=status_root//'/wall_products/'//state%definition%id//'/'//trim(version)
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
    call report_insitu_timing('runtime_probe_csv',started,int(identity%step))
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
    character(256) :: description
    integer :: status,i,n,axes(2),dimensions(2),offset(2),local_extent(2),bounds(2,3),d
    real(real64) :: clock(3),references(12)
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
      provider='velocity-time'
      write(description,'(a,2(es24.16,1x),a,es24.16)') 'requested_window=',window,'; covered_duration=',measure
      if(state%definition%source=='favre') then
        description=trim(description)//'; weight=density_time_integral'
      else
        description=trim(description)//'; weight=time_integral'
      endif
    endif
    provenance='source='//state%definition%source//'; statistics_source='//provider//'; '//trim(description)// &
      '; values='//system//'; diagnostics=explicit6-seven-node; periodic-stencil-endpoint=node0'
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
    write(frame,'("step",i12.12,".h5")') identity%step
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
    call report_insitu_timing('index_slice_hdf5',phase_started,int(identity%step))
    n=size(segments(index)%frames)
    allocate(frames(n+1));frames(:n)=segments(index)%frames
    frames(n+1)%file=frame;frames(n+1)%identity=identity
    phase_started=insitu_clock()
    call write_series(index,state,frames,dimensions,system,status)
    call report_insitu_timing('index_slice_index',phase_started,int(identity%step))
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
