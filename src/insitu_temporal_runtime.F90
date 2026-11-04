module insitu_temporal_runtime
  use mpi
  use iso_fortran_env, only: int64,real64
  use iso_c_binding, only: c_int,c_int64_t,c_double,c_char,c_null_char
  use astr_precision, only: real_kind,c_real_kind,real_bytes
  use astr_precision_mpi, only: mpi_real_type
  use commvar, only: numq,lcomb,nstep,time,nondimen,roinf,uinf,vinf,winf,pinf,tinf, &
    ref_den,ref_vel,ref_tem,ref_len,ref_tim,ref_miu
  use run_control_runtime, only: active_run
  use timestep_control, only: timestep_fixed
  use checkpoint_state_io, only: checkpoint_state_identity
  use insitu_run_config, only: catalog_product,insitu_options,catalog_field_names,pack_catalog_render
  use insitu_catalog, only: quote_catalog_text
  use insitu_product_schedule, only: runtime_product_state,runtime_catalog_count,runtime_catalog_state, &
    runtime_active,runtime_waiting,runtime_catalog_due,runtime_catalog_success,runtime_catalog_wait, &
    runtime_catalog_disable,runtime_catalog_reject
  use insitu_session, only: catalog_product_options,begin_catalog_renderer
  use insitu_temporal_history
  use insitu_temporal_analysis
  use insitu_selected_runtime, only: selected_runtime_workspace
  use index_slice_source, only: slice_source_info
  use index_slice_provider, only: prepare_index_slice
  use index_slice_archive, only: runtime_product_segment,refresh_runtime_product_status
  use output_fields, only: catalog_quantity_units
  use benchmark_runtime, only: insitu_clock,report_insitu_timing,insitu_timing_active
  implicit none
  private
  public :: observe_temporal_products,finish_temporal_products
contains
  subroutine mpi_ok(status)
    integer,intent(in) :: status
    integer :: ignored
    if(status/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,status,ignored)
  end subroutine

  logical function retained(index) result(ready)
    integer,intent(in) :: index
    ready=.false.
    if(.not.allocated(histories)) return
    if(index>size(histories)) return
    ready=histories(index)%configured
  end function

  subroutine report_product_time(stage,product,step,started)
    character(*),intent(in) :: stage,product
    integer(int64),intent(in) :: step
    real(8),intent(in) :: started
    character(:),allocatable :: encoded
    logical :: ok
    integer :: rank,err
    if(.not.insitu_timing_active()) return
    call quote_catalog_text(product,encoded,ok)
    if(.not.ok) then
      print '(a)','ASTR_TEMPORAL_TIMING_OMITTED cannot encode product name';return
    endif
    call MPI_Comm_rank(MPI_COMM_WORLD,rank,err);call mpi_ok(err)
    write(*,'(a,a,a,a,a,i0,a,i0,a,es24.16e3,a)') 'ASTR_TEMPORAL_TIMING {"product":',encoded, &
      ',"stage":"',stage,'","step":',step,',"rank":',rank,',"seconds":',insitu_clock()-started,'}'
  end subroutine

  subroutine observe_temporal_products(identity)
    type(checkpoint_state_identity),intent(in) :: identity
    type(insitu_options) :: options
    type(runtime_product_state) :: state
    type(catalog_product) :: selection
    real(real_kind),allocatable :: values(:,:),coordinates(:,:),line(:,:),xyz(:,:)
    real(real_kind) :: window(2),measure,seconds,dt
    integer(int64) :: samples,workspace,download,exchange
    integer :: p,rank,err,source,first(3),extent(3),flags(2),status,capacity
    logical :: enabled,ok,ready,dirty,fixed,available,spectral_sample
    character(512) :: message
    character(:),allocatable :: directory
    real(8) :: started
    if(.not.temporal_catalog_present()) return
    call MPI_Comm_rank(MPI_COMM_WORLD,rank,err);call mpi_ok(err)
    call catalog_product_options(options,enabled)
    spectral_sample=identity%time>=options%statistics_window(1).and.identity%time<=options%statistics_window(2)
    dirty=.false.;fixed=active_run%timestep%mode==timestep_fixed
    dt=identity%dt_used
    if(identity%step==0) dt=identity%dt_next
    if(rank==0) call observe_temporal_clock(identity%step,dt,fixed)
    call suspend_temporal_products(identity)
    do p=1,runtime_catalog_count()
      call runtime_catalog_state(p,state)
      if(state%definition%view%kind/='time_space'.and.state%definition%view%kind/='probe_spectrum') cycle
      if(state%status/=runtime_active.and.state%status/=runtime_waiting) cycle
      message='';ok=enabled.and.options%statistics.and.options%products=='catalog_stats'
      if(.not.ok) then
        call runtime_catalog_reject(p,'temporal histories require the enabled catalog_stats statistics session')
        dirty=.true.;cycle
      endif
      call slice_source_info(state%definition,source,ready,ok,window,measure,samples,message)
      if(.not.ok) then
        call runtime_catalog_reject(p,trim(message));dirty=.true.;cycle
      endif
      call runtime_catalog_wait(p,.not.ready)
      if(.not.ready) cycle
      if(rank==0.and.retained(p)) histories(p)%was_active=.true.
      call runtime_product_segment(p,state,directory,ok,message)
      if(.not.ok) then
        call disable_product(p,identity,message);dirty=.true.;cycle
      endif
      flags=[1,1]
      if(rank==0.and.retained(p)) then
        flags(1)=merge(0,1,temporal_sampling_same(histories(p),state%definition))
        if(flags(1)==0) then
          if(identity%step<=histories(p)%last_step) flags(2)=0
          if(identity%step>histories(p)%last_step) &
            flags(2)=merge(1,0,identity%step-histories(p)%last_step>=state%definition%temporal%sample_steps)
        endif
        if(flags(1)==1.and.histories(p)%block%filled>0) then
          call publish_block(directory,histories(p),.true.,ok,message)
          flags(1)=merge(1,-1,ok)
        endif
      endif
      call MPI_Bcast(flags,2,MPI_INTEGER,0,MPI_COMM_WORLD,err);call mpi_ok(err)
      call MPI_Bcast(message,len(message),MPI_CHARACTER,0,MPI_COMM_WORLD,err);call mpi_ok(err)
      if(flags(1)<0) then
        call disable_product(p,identity,message);dirty=.true.;cycle
      endif
      if(flags(2)==1) then
        started=insitu_clock();selection=state%definition
        if(selection%view%kind=='probe_spectrum') then
          selection%axis=1;selection%index=selection%ranges(1,1)
        endif
        call prepare_index_slice(selection,source,samples,values,coordinates,first,extent,ok,message, &
          workspace,download,exchange,seconds)
        if(ok) call gather_line(state%definition,values,coordinates,first,line,xyz,ok,message)
        if(ok) then
          status=0
          if(rank==0) then
            if(flags(1)==1) call initialize_temporal_history(p,state%definition,xyz,identity%step,ok,[window,measure])
            if(.not.ok) then
              status=1;message='cannot allocate this temporal line/probe history'
              if(retained(p)) call discard_temporal_history(p)
            else
              capacity=int(state%definition%temporal%block_samples)
              if(capacity==0) capacity=int(state%definition%temporal%segment_length)
              if(capacity/=histories(p)%block%capacity) then
                call publish_block(directory,histories(p),.true.,ok,message)
                if(ok) then
                  call clear_time_block(histories(p)%block)
                  call resize_temporal_block(histories(p),capacity,ok)
                endif
              endif
              if(ok.and.histories(p)%block%filled==histories(p)%block%capacity) then
                call publish_block(directory,histories(p),.false.,ok,message)
                if(ok) call clear_time_block(histories(p)%block)
              endif
              if(ok) then
                histories(p)%definition=state%definition
                if(histories(p)%last_step<identity%step) &
                  call push_temporal_history(histories(p),identity%step,identity%time,dt,fixed,line, &
                    spectral_sample,ok,[window,measure])
                if(.not.ok) then
                  message='temporal sample failed; unusable private history discarded'
                  call discard_temporal_history(p)
                else
                  call mirror_temporal_packet(p,identity,dt,fixed,line,spectral_sample)
                endif
              endif
              if(.not.ok) status=1
            endif
          endif
          call MPI_Bcast(status,1,MPI_INTEGER,0,MPI_COMM_WORLD,err);call mpi_ok(err)
          call MPI_Bcast(message,len(message),MPI_CHARACTER,0,MPI_COMM_WORLD,err);call mpi_ok(err)
          ok=status==0
        endif
        if(.not.ok) then
          call disable_product(p,identity,message);dirty=.true.;cycle
        endif
        if(insitu_timing_active()) write(*,'(a,i0,a,a,a,i0,a,i0,a,i0,a,i0)') &
          'ASTR_TEMPORAL_SAMPLE rank=',rank,' id=',state%definition%id, &
          ' step=',identity%step,' one_dimensional_d2h_bytes=',download,' private_exchange_bytes=',exchange, &
          ' neighborhood_bytes=',workspace
        call report_insitu_timing('temporal_sample',started,identity%step)
        call report_product_time('sample',state%definition%id,identity%step,started)
        if(rank==0.and.insitu_timing_active()) write(*,'(a,a,a,i0,a,i0,a,i0)') &
          'ASTR_TEMPORAL_STORAGE id=',state%definition%id,' step=',identity%step, &
          ' retained_host_array_bytes=',temporal_array_bytes(p),' retained_device_array_bytes=',0
      endif
      if(runtime_catalog_due(p,identity%step,real(identity%time,real64))) then
        started=insitu_clock()
        call publish_output(p,state,identity,directory,options,available,ok,message)
        call report_product_time('publication',state%definition%id,identity%step,started)
        if(ok) then
          call runtime_catalog_success(p,identity%step,real(identity%time,real64))
          call runtime_catalog_wait(p,.not.available)
        else
          call disable_product(p,identity,message)
        endif
        dirty=.true.
      endif
      status=0
      if(rank==0.and.retained(p)) then
        if(histories(p)%block%filled==histories(p)%block%capacity) then
          call publish_block(directory,histories(p),.false.,ok,message)
          if(ok) then
            call clear_time_block(histories(p)%block)
          else
            status=1
          endif
        endif
      endif
      call MPI_Bcast(status,1,MPI_INTEGER,0,MPI_COMM_WORLD,err);call mpi_ok(err)
      call MPI_Bcast(message,len(message),MPI_CHARACTER,0,MPI_COMM_WORLD,err);call mpi_ok(err)
      if(status/=0) then
        call disable_product(p,identity,message);dirty=.true.
      endif
    enddo
    if(dirty) call refresh_runtime_product_status(identity)
  end subroutine

  subroutine disable_product(index,identity,message)
    integer,intent(in) :: index
    type(checkpoint_state_identity),intent(in) :: identity
    character(*),intent(in) :: message
    integer :: rank,err
    call runtime_catalog_disable(index,identity%step,real(identity%time,real64),trim(message))
    call MPI_Comm_rank(MPI_COMM_WORLD,rank,err);call mpi_ok(err)
    if(rank==0.and.retained(index)) histories(index)%was_active=.false.
    if(rank==0) print '(a)','ASTR_TEMPORAL_DISABLED '//trim(message)
  end subroutine

  subroutine suspend_temporal_products(identity)
    type(checkpoint_state_identity),intent(in) :: identity
    type(runtime_product_state) :: state
    integer :: p,rank,err,flag,status
    logical :: ok
    character(:),allocatable :: directory
    character(512) :: message
    call MPI_Comm_rank(MPI_COMM_WORLD,rank,err);call mpi_ok(err)
    do p=1,runtime_catalog_count()
      call runtime_catalog_state(p,state)
      if(state%status==runtime_active.or.state%status==runtime_waiting) cycle
      flag=0
      if(rank==0.and.retained(p)) flag=merge(1,0,histories(p)%was_active)
      call MPI_Bcast(flag,1,MPI_INTEGER,0,MPI_COMM_WORLD,err);call mpi_ok(err)
      if(flag==0) cycle
      call runtime_product_segment(p,state,directory,ok,message)
      status=merge(0,1,ok)
      if(rank==0.and.ok) then
        call publish_block(directory,histories(p),.true.,ok,message)
        status=merge(0,1,ok)
        if(ok) then
          call clear_time_block(histories(p)%block)
          histories(p)%was_active=.false.
        else
          call discard_temporal_history(p)
        endif
      endif
      call MPI_Bcast(status,1,MPI_INTEGER,0,MPI_COMM_WORLD,err);call mpi_ok(err)
      call MPI_Bcast(message,len(message),MPI_CHARACTER,0,MPI_COMM_WORLD,err);call mpi_ok(err)
      if(status/=0) call disable_product(p,identity,message)
    enddo
  end subroutine

  subroutine mirror_temporal_packet(source_index,identity,dt,fixed,values,spectral_sample)
    integer,intent(in) :: source_index
    type(checkpoint_state_identity),intent(in) :: identity
    real(real_kind),intent(in) :: dt,values(:,:)
    logical,intent(in) :: fixed,spectral_sample
    type(runtime_product_state) :: state
    integer :: p
    logical :: ok
    ! Inactive means do not inherit another accumulator's evolving coverage.
    if(histories(source_index)%definition%source/='instantaneous') return
    do p=1,size(histories)
      if(p==source_index) cycle
      if(.not.temporal_sampling_same(histories(p),histories(source_index)%definition)) cycle
      call runtime_catalog_state(p,state)
      if(state%status==runtime_active.or.state%status==runtime_waiting) cycle
      if(identity%step<=histories(p)%last_step) cycle
      if(modulo(identity%step-histories(p)%last_step,histories(p)%definition%temporal%sample_steps)/=0) cycle
      if(histories(p)%block%filled==histories(p)%block%capacity) call clear_time_block(histories(p)%block)
      call push_temporal_history(histories(p),identity%step,identity%time,dt,fixed,values,spectral_sample,ok)
      if(.not.ok) then
        print '(a,i0)','ASTR_TEMPORAL_SHARED_HISTORY_DISCARDED index=',p
        call discard_temporal_history(p)
      endif
    enddo
  end subroutine

  subroutine gather_line(definition,values,coordinates,first,line,xyz,ok,message)
    type(catalog_product),intent(in) :: definition
    real(real_kind),intent(in) :: values(:,:),coordinates(:,:)
    integer,intent(in) :: first(3)
    real(real_kind),allocatable,intent(out) :: line(:,:),xyz(:,:)
    logical,intent(out) :: ok
    character(*),intent(out) :: message
    real(real_kind),allocatable :: local(:,:),global(:,:)
    integer,allocatable :: owners(:),total(:)
    integer :: n,axis,c,i,slot,rank,err,status,bad,all_bad
    integer(int64) :: extra
    axis=maxloc(definition%ranges(2,:)-definition%ranges(1,:),dim=1)
    n=int(definition%ranges(2,axis)-definition%ranges(1,axis))+1;c=size(values,2)
    extra=int(n,int64)*(2_int64*(c+3)*real_bytes+2_int64*storage_size(0)/8+(c+3)*real_bytes)
    call selected_runtime_workspace(extra,0_int64,ok)
    message='cannot allocate the one-dimensional temporal reduction'
    if(.not.ok) return
    allocate(local(n,c+3),global(n,c+3),owners(n),total(n),line(n,c),xyz(3,n),stat=status)
    bad=merge(1,0,status/=0)
    call MPI_Allreduce(bad,all_bad,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,err);call mpi_ok(err)
    ok=all_bad==0
    if(.not.ok) return
    local=0;owners=0
    do i=1,size(values,1)
      slot=first(axis)-int(definition%ranges(1,axis))+i
      local(slot,:3)=coordinates(:,i);local(slot,4:)=values(i,:);owners(slot)=owners(slot)+1
    enddo
    call MPI_Allreduce(owners,total,n,MPI_INTEGER,MPI_SUM,MPI_COMM_WORLD,err);call mpi_ok(err)
    ok=all(total==1);message='each global temporal node must have exactly one MPI owner'
    if(.not.ok) return
    call MPI_Comm_rank(MPI_COMM_WORLD,rank,err);call mpi_ok(err)
    call MPI_Reduce(local,global,size(local),mpi_real_type,MPI_SUM,0,MPI_COMM_WORLD,err);call mpi_ok(err)
    if(rank==0) then
      xyz=transpose(global(:,:3));line=global(:,4:)
    endif
    message=''
  end subroutine

  subroutine publish_csv(path,status)
    character(*),intent(in) :: path
    integer,intent(out) :: status
    integer :: cleanup
    interface
      integer(c_int) function hard_link(old,new) bind(C,name='link')
        import c_int,c_char
        character(c_char),intent(in) :: old(*),new(*)
      end function
      integer(c_int) function remove_file(path) bind(C,name='unlink')
        import c_int,c_char
        character(c_char),intent(in) :: path(*)
      end function
      subroutine print_io_error(path) bind(C,name='perror')
        import c_char
        character(c_char),intent(in) :: path(*)
      end subroutine
    end interface
    status=hard_link(path//'.writing'//c_null_char,path//c_null_char)
    if(status/=0) then
      call print_io_error(path//c_null_char);return
    endif
    cleanup=remove_file(path//'.writing'//c_null_char)
    if(cleanup/=0) print '(a)','ASTR_TEMPORAL_CSV_TEMPORARY_RETAINED '//path//'.writing'
  end subroutine

  subroutine write_metadata(unit,definition,fields,generation,samples,first_step,first_time,last_step,last_time,status)
    integer,intent(in) :: unit
    character(*),intent(in) :: definition
    integer,intent(in) :: fields(:)
    integer(int64),intent(in) :: generation,samples,first_step,last_step
    real(real_kind),intent(in) :: first_time,last_time
    integer,intent(out) :: status
    character(:),allocatable :: system,time_unit,coordinate_unit,frequency_unit,quantity_unit
    character(32),parameter :: reference_names(12)=[character(32)::'rho_inf','u_inf','v_inf','w_inf','p_inf','T_inf', &
      'reference_density','reference_velocity','reference_temperature','reference_length','reference_time','reference_viscosity']
    real(real_kind) :: references(12)
    integer :: c
    write(unit,'(a)',iostat=status) '# phase=completed_step; definition='//definition
    if(status==0) write(unit,'(a,i0,a,i0,a,i0,a,es26.17e3)',iostat=status) &
      '# history_generation=',generation,'; sampled_endpoints=',samples,'; last_sample_step=',last_step, &
      '; last_sample_time=',last_time
    if(status==0) write(unit,'(a,i0,a,es26.17e3)',iostat=status) &
      '# first_sample_step=',first_step,'; first_sample_time=',first_time
    if(status==0) write(unit,'(a,l1,a,i0)',iostat=status) '# nondimensional=',nondimen,'; source_bytes=',real_bytes
    system='si';time_unit='s';coordinate_unit='m';frequency_unit='Hz'
    if(nondimen) then
      system='dimensionless';time_unit='native_time';coordinate_unit='solver-dimensionless';frequency_unit='1/native_time'
    endif
    if(status==0) write(unit,'(a)',iostat=status,advance='no') &
      '# units={"time":"'//time_unit//'","coordinate":"'//coordinate_unit//'","frequency":"'//frequency_unit//'"'
    do c=1,size(fields)
      quantity_unit=catalog_quantity_units(fields(c),system)
      if(status==0) write(unit,'(a)',iostat=status,advance='no') &
        ',"'//trim(catalog_field_names(fields(c)))//'":"'//quantity_unit//'"'
    enddo
    if(status==0) write(unit,'(a)',iostat=status) '}'
    if(status==0) write(unit,'(a)',iostat=status,advance='no') '# psd_units={'
    do c=1,size(fields)
      quantity_unit=catalog_quantity_units(fields(c),system)
      if(c>1.and.status==0) write(unit,'(a)',iostat=status,advance='no') ','
      if(status==0) write(unit,'(a)',iostat=status,advance='no') &
        '"'//trim(catalog_field_names(fields(c)))//'":"('//quantity_unit//')^2*'//time_unit//'"'
    enddo
    if(status==0) write(unit,'(a)',iostat=status) '}'
    references=[roinf,uinf,vinf,winf,pinf,tinf,ref_den,ref_vel,ref_tem,ref_len,ref_tim,ref_miu]
    if(status==0) write(unit,'(a)',iostat=status,advance='no') '# reference_state={'
    do c=1,size(references)
      if(c>1.and.status==0) write(unit,'(a)',iostat=status,advance='no') ','
      if(status==0) write(unit,'(a,es26.17e3)',iostat=status,advance='no') &
        '"'//trim(reference_names(c))//'":',references(c)
    enddo
    if(status==0) write(unit,'(a)',iostat=status) '}'
  end subroutine

  subroutine publish_block(directory,s,snapshot,ok,message)
    character(*),intent(in) :: directory
    type(temporal_history),intent(inout) :: s
    logical,intent(in) :: snapshot
    logical,intent(out) :: ok
    character(*),intent(out) :: message
    character(160) :: filename
    integer :: unit,status,closed,t,j,c
    logical :: gap
    real(8) :: started
    ok=.true.;message=''
    if(s%block%filled==0) return
    if(s%definition%view%kind/='time_space'.and..not.s%definition%temporal%archive_samples) return
    started=insitu_clock()
    if(snapshot) then
      write(filename,'("block.g",i0.12,".n",i0.12,".partial.step",i0.12,".csv")') &
        s%generation,s%block%number,s%last_step
    else
      write(filename,'("block.g",i0.12,".n",i0.12,".complete.csv")') s%generation,s%block%number
    endif
    open(newunit=unit,file=directory//'/'//trim(filename)//'.writing',status='new',action='write',iostat=status)
    if(status==0) then
      call write_metadata(unit,s%definition%record,s%definition%fields,s%generation,s%samples, &
        s%first_step,s%first_time,s%last_step,s%last_time,status)
      if(status==0) write(unit,'(a,i0,a,i0,a,l1)',iostat=status) '# block=',s%block%number, &
        '; samples=',s%block%filled,'; snapshot=',snapshot
      if(status==0) write(unit,'(a)',iostat=status,advance='no') 'step,time,gap_before,node,x,y,z'
      do c=1,size(s%definition%fields)
        if(status==0) write(unit,'(a)',iostat=status,advance='no') ','//trim(catalog_field_names(s%definition%fields(c)))
      enddo
      if(allocated(s%block%mean_coverage).and.status==0) then
        write(unit,'(a)',iostat=status,advance='no') ',mean_window_start,mean_window_end,mean_measure'
      endif
      if(status==0) write(unit,'(a)',iostat=status) ''
      do t=1,s%block%filled
        gap=s%block%gaps(t)
        if(t==1.and.s%archived_step>=0) gap=gap.or. &
          s%block%steps(t)>s%archived_step+s%definition%temporal%sample_steps
        do j=1,s%block%nodes
          if(status/=0) exit
          write(unit,'(i0,a,es26.17e3,a,i0,a,i0,*(a,es26.17e3))',iostat=status,advance='no') &
            s%block%steps(t),',',s%block%times(t),',',merge(1,0,gap),',',j, &
            (',',s%block%coordinates(c,j),c=1,3),(',',s%block%values(j,c,t),c=1,s%block%fields)
          if(allocated(s%block%mean_coverage).and.status==0) write(unit,'(*(a,es26.17e3))',iostat=status,advance='no') &
            (',',s%block%mean_coverage(c,t),c=1,3)
          if(status==0) write(unit,'(a)',iostat=status) ''
        enddo
      enddo
      close(unit,iostat=closed)
      if(closed/=0) status=closed
      if(status==0) call publish_csv(directory//'/'//trim(filename),status)
    endif
    ok=status==0
    if(ok) s%archived_step=s%block%steps(s%block%filled)
    if(.not.ok) write(message,'(a,i0)') 'time-block CSV write/close/publication failed, status=',status
    if(snapshot) then
      call report_product_time('archive_partial',s%definition%id,s%last_step,started)
    else
      call report_product_time('archive_full',s%definition%id,s%last_step,started)
    endif
  end subroutine

  subroutine publish_output(index,state,identity,directory,options,available,ok,message)
    integer,intent(in) :: index
    type(runtime_product_state),intent(in) :: state
    type(checkpoint_state_identity),intent(in) :: identity
    character(*),intent(in) :: directory
    type(insitu_options),intent(in) :: options
    logical,intent(out) :: available,ok
    character(*),intent(out) :: message
    real(real_kind),allocatable :: frequencies(:),spectrum(:,:)
    integer :: rank,err,status,unit,closed,i,c,filled
    integer(int64) :: segments,generation,bytes
    real(real_kind) :: spacing
    logical :: uniform
    character(80) :: filename
    call MPI_Comm_rank(MPI_COMM_WORLD,rank,err);call mpi_ok(err)
    available=.false.;ok=.true.;status=0;message=''
    if(rank==0.and.retained(index)) then
      associate(s=>histories(index))
      if(s%definition%view%kind=='time_space') then
        available=s%block%filled>0
        if(state%definition%view%channel=='data') available=s%samples>0
      else
        call read_welch(s%spectrum,frequencies,spectrum,available)
        call welch_info(s%spectrum,segments,filled,generation,spacing,uniform,bytes)
        write(filename,'("spectrum.step",i0.12,".csv")') identity%step
        open(newunit=unit,file=directory//'/'//trim(filename)//'.writing',status='new',action='write',iostat=status)
        if(status==0) then
          call write_metadata(unit,histories(index)%definition%record,s%definition%fields,s%generation,s%samples, &
            s%first_step,s%first_time,s%last_step,s%last_time,status)
          if(status==0) write(unit,'(a,es26.17e3,a,es26.17e3)',iostat=status) &
            '# requested_spectral_window_start=',options%statistics_window(1), &
            '; requested_spectral_window_end=',options%statistics_window(2)
          if(status==0) write(unit,'(a,i0,a,i0,a,i0,a)',iostat=status) &
            '# welch_sampled_endpoints=',s%welch_samples,'; welch_first_sample_step=',s%welch_first_step, &
            '; welch_last_sample_step=',s%welch_last_step,';'
          if(status==0) write(unit,'(a,es26.17e3,a,es26.17e3)',iostat=status) &
            '# welch_first_sample_time=',s%welch_first_time,'; welch_last_sample_time=',s%welch_last_time
          if(status==0) write(unit,'(a,l1,a,i0,a,i0,a,es26.17e3,a,i0)',iostat=status) &
            '# available=',available,'; valid_segments=',segments,'; incomplete_segment_samples=',filled, &
            '; sampling_interval=',spacing,'; frequency_generation=',generation
          if(status==0) write(unit,'(a)',iostat=status) '# periodic Hann; segment demean; one-sided PSD; automatic/unequal physical sampling has no estimate'
          if(status==0) write(unit,'(a)',iostat=status,advance='no') 'frequency'
          do c=1,size(s%definition%fields)
            if(status==0) write(unit,'(a)',iostat=status,advance='no') ','//trim(catalog_field_names(s%definition%fields(c)))//'_PSD'
          enddo
          if(status==0) write(unit,'(a)',iostat=status) ''
          if(available) then
            do i=1,size(frequencies)
              if(status==0) write(unit,'(es26.17e3,*(a,es26.17e3))',iostat=status) &
                frequencies(i),(',',spectrum(i,c),c=1,size(spectrum,2))
            enddo
          endif
          close(unit,iostat=closed)
          if(closed/=0) status=closed
          if(status==0) call publish_csv(directory//'/'//trim(filename),status)
        endif
        ok=status==0
        if(.not.ok) write(message,'(a,i0)') 'Welch CSV write/close/publication failed, status=',status
      endif
      end associate
    endif
    status=merge(0,1,ok)
    call MPI_Bcast(status,1,MPI_INTEGER,0,MPI_COMM_WORLD,err);call mpi_ok(err)
    call MPI_Bcast(available,1,MPI_LOGICAL,0,MPI_COMM_WORLD,err);call mpi_ok(err)
    call MPI_Bcast(message,len(message),MPI_CHARACTER,0,MPI_COMM_WORLD,err);call mpi_ok(err)
    ok=status==0
    if(.not.ok.or..not.available.or.state%definition%view%channel/='image') return
    call draw_temporal(index,state,identity,directory,options,frequencies,spectrum,ok,message)
  end subroutine

  subroutine draw_temporal(index,state,identity,directory,options,frequencies,spectrum,ok,message)
    integer,intent(in) :: index
    type(runtime_product_state),intent(in) :: state
    type(checkpoint_state_identity),intent(in) :: identity
    character(*),intent(in) :: directory
    type(insitu_options),intent(in) :: options
    real(real_kind),allocatable,intent(in) :: frequencies(:),spectrum(:,:)
    logical,intent(out) :: ok
    character(*),intent(out) :: message
    integer :: rank,err,n,rows,axis,status,i
#if defined(ASTR_WITH_CATALYST) && defined(ASTR_INSITU_DEVICE_RENDERING)
    character(32) :: labels(11)
    integer(c_int) :: words(7)
    real(real64) :: settings(31)
    real(real_kind),allocatable :: coordinates(:),times(:),values(:)
    integer(c_int),allocatable :: gaps(:)
    character(c_char) :: errors(512)
    interface
      integer(c_int) function render(pipeline,backend,script,comm,step,time,id,source,definition,directory, &
          version,labels,words,settings,nodes,rows,axis,coordinates,times,values,gaps) &
          bind(C,name='astr_insitu_render_temporal_product')
        import c_int,c_int64_t,c_char,c_double,c_real_kind
        character(c_char),intent(in) :: pipeline(*),backend(*),script(*),id(*),source(*),definition(*),directory(*),labels(*)
        integer(c_int),value :: comm,nodes,rows,axis
        integer(c_int64_t),value :: step,version
        real(c_real_kind),value :: time
        integer(c_int),intent(in) :: words(7),gaps(*)
        real(c_double),intent(in) :: settings(31)
        real(c_real_kind),intent(in) :: coordinates(*),times(*),values(*)
      end function
      subroutine render_error(errors,capacity) bind(C,name='astr_insitu_runtime_error')
        import c_char,c_int
        character(c_char),intent(out) :: errors(*)
        integer(c_int),value :: capacity
      end subroutine
    end interface
#endif
    ok=.false.;message='temporal images require the Catalyst device-rendering build'
#if defined(ASTR_WITH_CATALYST) && defined(ASTR_INSITU_DEVICE_RENDERING)
    if(.not.options%render) then
      message='temporal image products require render=t; numerical histories continue without image products';return
    endif
    call begin_catalog_renderer(ok,message)
    if(.not.ok) return
    call MPI_Comm_rank(MPI_COMM_WORLD,rank,err);call mpi_ok(err)
    n=0;rows=0;axis=0
    if(rank==0) then
      if(state%definition%view%kind=='time_space') then
        associate(block=>histories(index)%block)
        n=block%nodes;rows=block%filled
        axis=maxloc(state%definition%ranges(2,:)-state%definition%ranges(1,:),dim=1)
        allocate(coordinates(3*n),times(rows),values(n*rows),gaps(rows),stat=status)
        if(status==0) then
          coordinates=reshape(block%coordinates,[3*n]);times=block%times(:rows)
          values=reshape(block%values(:,1,:rows),[n*rows]);gaps=merge(1_c_int,0_c_int,block%gaps(:rows))
        endif
        end associate
      else
        n=size(frequencies);rows=1
        allocate(coordinates(n),times(1),values(n),gaps(1),stat=status)
        if(status==0) then
          coordinates=frequencies;times=identity%time;values=spectrum(:,1);gaps=0
        endif
      endif
    else
      allocate(coordinates(0),times(0),values(0),gaps(0),stat=status)
    endif
    call MPI_Allreduce(MPI_IN_PLACE,status,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,err);call mpi_ok(err)
    if(status/=0) then
      ok=.false.;message='cannot allocate one-dimensional temporal chart input';return
    endif
    call pack_catalog_render(state%definition%view,labels,words,settings)
    status=render(trim(options%rendering_pipeline)//c_null_char,trim(options%implementation_path)//c_null_char, &
      trim(options%pipeline_file)//c_null_char,int(MPI_COMM_WORLD,c_int),identity%step,identity%time, &
      state%definition%id//c_null_char,state%definition%source//c_null_char,state%definition%record//c_null_char, &
      directory//c_null_char,state%version,labels,words,settings,int(n,c_int),int(rows,c_int),int(axis,c_int), &
      coordinates,times,values,gaps)
    ok=status==0;message=''
    if(.not.ok) then
      call render_error(errors,int(size(errors),c_int))
      do i=1,size(errors)
        if(errors(i)==c_null_char) exit
        message=trim(message)//errors(i)
      enddo
    endif
#endif
  end subroutine

  subroutine finish_temporal_products()
    type(runtime_product_state) :: state
    type(checkpoint_state_identity) :: identity
    character(:),allocatable :: directory
    character(512) :: message
    integer :: p,rank,err,flag,status
    logical :: ok
    if(.not.temporal_catalog_present()) return
    call MPI_Comm_rank(MPI_COMM_WORLD,rank,err);call mpi_ok(err)
    identity=checkpoint_state_identity(int(nstep,int64),time,0.e0_real_kind,0.e0_real_kind)
    do p=1,runtime_catalog_count()
      call runtime_catalog_state(p,state)
      if(state%status/=runtime_active.and.state%status/=runtime_waiting) cycle
      if(state%definition%view%kind/='time_space'.and.state%definition%view%kind/='probe_spectrum') cycle
      flag=0
      if(rank==0.and.retained(p)) flag=merge(1,0,histories(p)%block%filled>0)
      call MPI_Bcast(flag,1,MPI_INTEGER,0,MPI_COMM_WORLD,err);call mpi_ok(err)
      if(flag==0) cycle
      call runtime_product_segment(p,state,directory,ok,message)
      status=merge(0,1,ok)
      if(rank==0.and.ok) then
        call publish_block(directory,histories(p),.true.,ok,message)
        status=merge(0,1,ok)
      endif
      call MPI_Bcast(status,1,MPI_INTEGER,0,MPI_COMM_WORLD,err);call mpi_ok(err)
      call MPI_Bcast(message,len(message),MPI_CHARACTER,0,MPI_COMM_WORLD,err);call mpi_ok(err)
      if(status/=0) call disable_product(p,identity,message)
    enddo
  end subroutine
end module
