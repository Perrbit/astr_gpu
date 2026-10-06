module output_archive
  use iso_fortran_env, only: int64,real64,iostat_end
  use iso_c_binding, only: c_int,c_char,c_null_char
  use mpi
  use commvar, only: ia,ja,ka,im,jm,km,numq,use_gpu,nondimen
  use commarray, only: x
  use parallel, only: ig0,jg0,kg0,mpirank
  use output_config, only: output_options,output_product_options,validate_output_grid,output_product_derived_indices
  use output_fields, only: write_basic_output,pack_basic_output_cpu,write_slice_output_xdmf,write_output_series_xdmf
  use output_fields, only: begin_derived_output_cpu,pack_derived_output_cpu,end_derived_output_cpu, &
    derived_output_workspace_cpu
  use output_lineage, only: prepare_output_lineage,stage_output_lineage
  use insitu_schedule, only: sample_schedule,configure_schedule,poll_schedule, &
    write_schedule_state,restore_schedule_state
  use adaptive_output, only: adaptive_clock,adaptive_binding,adaptive_binding_equal,adaptive_dependencies_reset, &
    configure_adaptive_clock,poll_adaptive_clock,report_adaptive_clock,adaptive_clock_file,adaptive_shared_config
  use insitu_checkpoint_batch, only: create_batch,copy_batch_file,file_fingerprint
  use checkpoint_state_io, only: checkpoint_state_identity,checkpoint_state_require
  use checkpoint_bundle, only: seal_checkpoint_bundle,publish_checkpoint_bundle,write_checkpoint_resource_refs, &
    checkpoint_stream_at_end
#ifdef _CUDA
  use output_fields_gpu, only: begin_basic_output_gpu,pack_basic_output_gpu,end_basic_output_gpu
  use output_fields_gpu, only: begin_derived_output_gpu,pack_derived_output_gpu,end_derived_output_gpu, &
    derived_output_workspace_gpu
  use insitu_statistics_gpu, only: statistics_device_bytes
  use production_statistics_gpu, only: compact_statistics_device_bytes,compact_statistics_host_bytes
#ifdef ASTR_AIR5_CHEMISTRY
  use chemistry_mean_statistics_gpu, only: air5_mean_statistics_device_bytes
  use chemistry_flow_solver_gpu, only: air5_conservation_device_bytes
#endif
#endif
  implicit none
  private
  public :: configure_archives,begin_archives,observe_archives,archive_control_file,archives_enabled
  type(output_options),save :: options
  type(sample_schedule),save :: schedules(2)
  type(adaptive_clock),save :: adaptive_clocks(2)
  type(output_product_options),save :: products(2)
  integer(int64),save :: last_steps(2)=-1,origin_steps(2)=0
  integer(int64),save :: segment_ids(2)=-1,segment_bytes(2)=0,segment_crc(2)=0
  integer(int64),save :: parent_ids(2)=-1,parent_bytes(2)=0,parent_crc(2)=0
  real(real64),save :: origin_times(2)=0
  logical,save :: effective_initial(2)=.false.
  logical,save :: lineage_compatible(2)=.false.
  character(1200),save :: roots(2)=''
  character(16),parameter :: labels(2)=[character(16) :: 'fields','slices']
  interface
    function archive_parent(current,parent,relative,capacity) bind(C,name='astr_output_archive_parent') result(status)
      import c_int,c_char
      character(c_char),intent(in) :: current(*),parent(*)
      character(c_char),intent(out) :: relative(*)
      integer(c_int),value :: capacity
      integer(c_int) :: status
    end function
    function archive_segment(root,reuse,generation,fresh) bind(C,name='astr_output_archive_segment') result(status)
      import c_int,c_char
      character(c_char),intent(in) :: root(*)
      integer(c_int),value :: reuse
      integer(c_int),intent(out) :: generation,fresh
      integer(c_int) :: status
    end function
    function hdf5_self_contained(path) bind(C,name='astr_checkpoint_hdf5_self_contained') result(status)
      import c_int,c_char
      character(c_char),intent(in) :: path(*)
      integer(c_int) :: status
    end function
    function replace_plain_file(root,temporary,target) bind(C,name='astr_checkpoint_replace_plain_file') result(status)
      import c_int,c_char
      character(c_char),intent(in) :: root(*),temporary(*),target(*)
      integer(c_int) :: status
    end function
  end interface
contains
  subroutine check(condition,message)
    logical,intent(in) :: condition
    character(*),intent(in) :: message
    call checkpoint_state_require(condition,MPI_COMM_WORLD,'output archive: '//message)
  end subroutine

  logical function archives_enabled()
    archives_enabled=products(1)%enabled.or.products(2)%enabled
  end function

  subroutine configure_archives(selected)
    use commvar, only: flowtype,num_species,lreadgrid,lcomb,limmbou,ndims, &
      conschm,difschm,lihomo,ljhomo,lkhomo,lfilter,diffterm,hm,num_modequ
    use bc, only: bctype,turbinf
    use parallel, only: mpisize
    type(output_options),intent(in) :: selected
    logical :: ok,derived_case,air5_case
    integer :: p
    character(256) :: message
    options=selected; products=[options%volume,options%slices]
    last_steps=-1; origin_steps=0; origin_times=0
    segment_ids=-1; segment_bytes=0; segment_crc=0
    parent_ids=-1; parent_bytes=0; parent_crc=0
    lineage_compatible=.false.
    call validate_output_grid(options,int([ia,ja,ka],int64),ok,message)
    call check(ok,trim(message))
    derived_case=trim(flowtype)=='tgv'.and..not.lreadgrid.and.all(bctype==1).and. &
      lihomo.and.ljhomo.and.lkhomo.and.trim(conschm)=='643e'.and.lfilter
    derived_case=derived_case.or.(trim(flowtype)=='channel'.and..not.lreadgrid.and. &
      all(bctype==[1,1,41,41,1,1]).and.jm==ja.and.trim(conschm)=='643e'.and.lfilter)
    derived_case=derived_case.or.(trim(flowtype)=='bl'.and.lreadgrid.and. &
      all(bctype==[11,21,41,51,1,1]).and.jm==ja.and.trim(conschm)=='543e'.and. &
      (trim(turbinf)=='prof'.or.trim(turbinf)=='intp'))
    derived_case=derived_case.and.numq==5.and.num_species==0.and..not.lcomb.and.nondimen
    air5_case=.false.
#ifdef ASTR_AIR5_CHEMISTRY
    air5_case=(trim(flowtype)=='air5hbl'.or.trim(flowtype)=='air5sbli').and. &
      numq==11.and.num_species==5.and.num_modequ==1.and.lcomb.and..not.nondimen.and..not.lreadgrid.and. &
      all(bctype==[11,50,41,51,1,1]).and.jm==ja.and.trim(conschm)=='643e'.and.lfilter.and. &
      ((trim(flowtype)=='air5hbl'.and.im==ia).or.(trim(flowtype)=='air5sbli'.and.km==ka))
#endif
    do p=1,2
      if(products(p)%enabled.and.(products(p)%velocity_gradient.or.products(p)%vorticity.or.products(p)%qcriterion)) &
        call check((derived_case.or.air5_case).and.ndims==3.and. &
          .not.limmbou.and.all([ia,ja,ka]==16).and.mpisize<=2.and. &
          hm>=3.and.min(im,jm,km)>=max(hm,5).and.trim(difschm)=='643e'.and. &
          diffterm,'derived fields require registered 16-cubed explicit TGV/channel/CURVE/AIR5 NP<=2')
      effective_initial(p)=products(p)%initial_frame
      if(products(p)%enabled) then
        call configure_schedule(schedules(p),trim(products(p)%mode),products(p)%interval_steps, &
          products(p)%interval_time,origin_steps(p),origin_times(p),effective_initial(p),.false.,ok)
        call check(ok,'invalid '//trim(labels(p))//' schedule')
        if(products(p)%adaptive%enabled) then
          call configure_adaptive_clock(adaptive_clocks(p),products(p)%adaptive,products(p)%mode, &
            products(p)%interval_steps,products(p)%interval_time,origin_steps(p),origin_times(p), &
            effective_initial(p),products(p)%final_frame,ok)
          call check(ok,'invalid adaptive '//trim(labels(p))//' clock')
        endif
      endif
    enddo
  end subroutine

  subroutine archive_control_file(path,writing,identity)
    character(*),intent(in) :: path
    logical,intent(in) :: writing
    type(checkpoint_state_identity),intent(in) :: identity
    type(output_product_options) :: saved(2)
    integer(int64) :: indices(256,3),saved_last(2),steps(2),clock_step
    real(real64) :: times(2),clock_time
    logical :: initial(2),checkpoint_enabled,ok,same,same_flags(2),clock_same
    integer :: unit,err,closed,p
    character(8) :: magic
    magic='ASTROA02'
    if(any(products%adaptive%enabled)) magic='ASTROA03'
    if(writing) then
      err=0; closed=0
      if(mpirank==0) then
        open(newunit=unit,file=path,status='new',access='stream',form='unformatted', &
          convert='little_endian',action='write',iostat=err)
        if(err==0) then
          write(unit,iostat=err) magic,identity%step,identity%time,options%checkpoint%enabled, &
            last_steps,origin_steps,origin_times,effective_initial
          do p=1,2
            if(err/=0) exit
            call write_product(unit,products(p),err,magic=='ASTROA03')
          enddo
          if(err==0) write(unit,iostat=err) options%i_indices,options%j_indices,options%k_indices
          do p=1,2
            if(err/=0) exit
            if(.not.products(p)%enabled) cycle
            call write_schedule_state(unit,schedules(p),trim(labels(p)),identity%step,identity%time,ok)
            if(.not.ok) err=1
          enddo
          if(magic=='ASTROA03') then
            do p=1,2
              if(err/=0) exit
              if(.not.products(p)%enabled.or..not.products(p)%adaptive%enabled) cycle
              call adaptive_clock_file(unit,.true.,adaptive_clocks(p),products(p)%adaptive, &
                identity%step,identity%time,clock_same,ok)
              if(.not.ok) err=1
            enddo
          endif
          if(err==0) write(unit,iostat=err) segment_ids,segment_bytes,segment_crc
          close(unit,iostat=closed)
        endif
      endif
      call check(err==0.and.closed==0,'write schedule history')
    else
      open(newunit=unit,file=path,status='old',access='stream',form='unformatted', &
        convert='little_endian',action='read',iostat=err)
      call check(err==0,'missing schedule history')
      read(unit,iostat=err) magic,clock_step,clock_time,checkpoint_enabled,saved_last,steps,times,initial
      call check(err==0.and.(magic=='ASTROA02'.or.magic=='ASTROA03').and. &
        clock_step==identity%step.and.clock_time==identity%time, &
        'schedule history clock/version')
      call check((checkpoint_enabled.eqv.options%checkpoint%enabled).or.options%restart_output=='override', &
        'checkpoint switch changed without override')
      do p=1,2
        read(unit,iostat=err) saved(p)%enabled,saved(p)%mode,saved(p)%interval_steps,saved(p)%interval_time, &
          saved(p)%initial_frame,saved(p)%final_frame,saved(p)%fields, &
          saved(p)%velocity_gradient,saved(p)%vorticity,saved(p)%qcriterion
        call check(err==0,'read product options')
        if(magic=='ASTROA03') then
          read(unit,iostat=err) saved(p)%adaptive%enabled,saved(p)%adaptive%dense_steps,saved(p)%adaptive%dense_time, &
            saved(p)%adaptive%events,saved(p)%adaptive%windows
          call check(err==0,'read adaptive product options')
        endif
      enddo
      read(unit,iostat=err) indices(:,1),indices(:,2),indices(:,3)
      call check(err==0,'read slice selections')
      do p=1,2
        if(saved(p)%enabled) then
          call configure_schedule(schedules(p),trim(saved(p)%mode),saved(p)%interval_steps, &
            saved(p)%interval_time,steps(p),times(p),initial(p),.false.,ok)
          call check(ok,'invalid saved product schedule')
          call restore_schedule_state(unit,schedules(p),trim(labels(p)),identity%step,identity%time,ok)
          call check(ok,'invalid saved product history')
        endif
        same=same_product(saved(p),products(p)).and..not.adaptive_dependencies_reset(products(p)%adaptive)
        if(p==2) same=same.and.all(indices(:,1)==options%i_indices).and. &
          all(indices(:,2)==options%j_indices).and.all(indices(:,3)==options%k_indices)
        call check(same.or.options%restart_output=='override','product options changed without override')
        same_flags(p)=same
        call check(saved_last(p)>=-1.and.saved_last(p)<=identity%step,'invalid last frame step')
        if(same) then
          last_steps(p)=saved_last(p); origin_steps(p)=steps(p); origin_times(p)=times(p)
          effective_initial(p)=initial(p)
        else
          last_steps(p)=identity%step; origin_steps(p)=identity%step; origin_times(p)=identity%time
          effective_initial(p)=.false.
          if(adaptive_shared_config%enabled.and.mpirank==0) write(*,'(3a)') &
            'ASTR_AP_RESET product ',trim(labels(p)),' reason=product_configuration_or_dependency_change'
          if(products(p)%enabled) then
            call configure_schedule(schedules(p),trim(products(p)%mode),products(p)%interval_steps, &
              products(p)%interval_time,origin_steps(p),origin_times(p),.false.,.false.,ok)
            call check(ok,'invalid overridden product schedule')
          endif
        endif
      enddo
      if(magic=='ASTROA03') then
        do p=1,2
          if(.not.saved(p)%enabled.or..not.saved(p)%adaptive%enabled) cycle
          call adaptive_clock_file(unit,.false.,adaptive_clocks(p),products(p)%adaptive, &
            identity%step,identity%time,clock_same,ok)
          call check(ok.and.(clock_same.or.options%restart_output=='override'),'invalid adaptive archive history')
        enddo
      endif
      do p=1,2
        if(.not.products(p)%enabled.or..not.products(p)%adaptive%enabled.or.same_flags(p)) cycle
        call configure_adaptive_clock(adaptive_clocks(p),products(p)%adaptive,products(p)%mode, &
          products(p)%interval_steps,products(p)%interval_time,identity%step,identity%time, &
          .false.,products(p)%final_frame,ok)
        call check(ok,'invalid overridden adaptive archive clock')
      enddo
      read(unit,iostat=err) parent_ids,parent_bytes,parent_crc
      call check(err==0,'missing saved output segment identity')
      do p=1,2
        call check((saved(p)%enabled.and.parent_ids(p)>=0.and.parent_ids(p)<=99999999.and.parent_bytes(p)>0).or. &
          (.not.saved(p)%enabled.and.parent_ids(p)==-1.and.parent_bytes(p)==0.and.parent_crc(p)==0), &
          'invalid saved output segment identity')
      enddo
      call check(checkpoint_stream_at_end(unit),'schedule history tail')
      close(unit,iostat=closed)
      call check(closed==0,'schedule history close')
    endif
  end subroutine

  subroutine write_product(unit,product,err,adaptive)
    integer,intent(in) :: unit
    type(output_product_options),intent(in) :: product
    integer,intent(out) :: err
    logical,intent(in) :: adaptive
    write(unit,iostat=err) product%enabled,product%mode,product%interval_steps,product%interval_time, &
      product%initial_frame,product%final_frame,product%fields,product%velocity_gradient,product%vorticity,product%qcriterion
    if(err==0.and.adaptive) write(unit,iostat=err) product%adaptive%enabled,product%adaptive%dense_steps, &
      product%adaptive%dense_time,product%adaptive%events,product%adaptive%windows
  end subroutine

  logical function same_product(a,b) result(same)
    type(output_product_options),intent(in) :: a,b
    same=(a%enabled.eqv.b%enabled).and.a%mode==b%mode.and.a%interval_steps==b%interval_steps.and. &
      a%interval_time==b%interval_time.and.(a%initial_frame.eqv.b%initial_frame).and. &
      (a%final_frame.eqv.b%final_frame).and.a%fields==b%fields.and. &
      (a%velocity_gradient.eqv.b%velocity_gradient).and.(a%vorticity.eqv.b%vorticity).and. &
      (a%qcriterion.eqv.b%qcriterion).and.adaptive_binding_equal(a%adaptive,b%adaptive)
  end function

  subroutine begin_archives(identity,reuse)
    type(checkpoint_state_identity),intent(in) :: identity
    logical,intent(in) :: reuse
    logical :: ok
    integer :: p,unit,err,closed,i,axes(768),indices(768),n,components,selected(14),nselected
    integer(int64) :: bytes,crc,input_bytes,input_crc,saved_bytes,saved_crc
    character(1200) :: source,resource_path,parent_path
    character(c_char) :: relative(1200)
    character(15) :: segment,parent_segment
    character(16) :: units
    integer(c_int) :: generation,fresh,status
    do p=1,2
      if(.not.products(p)%enabled) cycle
      roots(p)=trim(options%directory)//'/'//trim(labels(p))
      status=0; generation=-1; fresh=0
      if(mpirank==0) status=archive_segment(trim(roots(p))//c_null_char,merge(1_c_int,0_c_int,reuse),generation,fresh)
      call check(status==0,'cannot select exclusive '//trim(labels(p))//' segment')
      call MPI_Bcast(generation,1,MPI_INTEGER,0,MPI_COMM_WORLD,err)
      call check(err==MPI_SUCCESS,'archive generation broadcast')
      call MPI_Bcast(fresh,1,MPI_INTEGER,0,MPI_COMM_WORLD,err)
      call check(err==MPI_SUCCESS,'archive resource ownership broadcast')
      resource_path=trim(roots(p))//'/resources'
      if(fresh==1) then
        call create_batch(trim(resource_path),MPI_COMM_WORLD,ok)
        call check(ok,'cannot create shared archive geometry directory')
      else
        ok=.true.
        if(mpirank==0) ok=hdf5_self_contained(trim(resource_path)//'/data.h5'//c_null_char)==0
        call check(ok,'archive geometry has external HDF5 dependencies')
      endif
      call write_archive_geometry(p,trim(resource_path),identity,verify=fresh==0)
      write(segment,'("segment",i8.8)') generation
      roots(p)=trim(roots(p))//'/'//segment
      call create_batch(trim(roots(p)),MPI_COMM_WORLD,ok)
      call check(ok,'cannot create archive segment')
      ok=.true.
      if(mpirank==0) then
        call get_command_argument(2,source,status=err)
        ok=err==0
        if(ok) call copy_batch_file(trim(source),trim(roots(p))//'/input.txt',ok)
        bytes=0; crc=0
        if(ok.and.len_trim(options%restore_directory)>0) &
          call file_fingerprint(trim(options%restore_directory)//'/MANIFEST',bytes,crc,ok)
        parent_path='none'
        if(ok.and.parent_ids(p)>=0) then
          write(parent_segment,'("segment",i8.8)') parent_ids(p)
          source=trim(options%restore_directory)//'/../../'//trim(labels(p))//'/'//parent_segment
          status=archive_parent(trim(roots(p))//c_null_char,trim(source)//c_null_char,relative,1200_c_int)
          ok=status==0
          if(ok) then
            parent_path=''
            do i=1,1199
              if(relative(i)==c_null_char) exit
              parent_path(i:i)=relative(i)
            enddo
            saved_bytes=0; saved_crc=0
            call file_fingerprint(trim(source)//'/SEGMENT',saved_bytes,saved_crc,ok)
            ok=ok.and.saved_bytes==parent_bytes(p).and.saved_crc==parent_crc(p)
          endif
        endif
        if(ok) call file_fingerprint(trim(roots(p))//'/input.txt',input_bytes,input_crc,ok)
        if(ok) then
          open(newunit=unit,file=trim(roots(p))//'/SEGMENT',status='new',action='write',iostat=err)
          if(err==0) then
            write(unit,'(a,/,i0,1x,es24.16,/,i0,1x,z16.16,/,i0,1x,i0,1x,z16.16,/,i0,1x,z16.16,/,a)',iostat=err) &
              'ASTR_OUTPUT_SEGMENT_2',identity%step,identity%time,bytes,crc, &
              parent_ids(p),parent_bytes(p),parent_crc(p),input_bytes,input_crc,trim(parent_path)
            close(unit,iostat=closed)
            ok=err==0.and.closed==0
          else
            ok=.false.
          endif
        endif
      endif
      call check(ok,'cannot record segment input/parent identity')
      segment_ids(p)=int(generation,int64)
      if(mpirank==0) call file_fingerprint(trim(roots(p))//'/SEGMENT',segment_bytes(p),segment_crc(p),ok)
      call check(ok,'cannot fingerprint segment provenance')
      call MPI_Bcast(segment_bytes(p),1,MPI_INTEGER8,0,MPI_COMM_WORLD,err)
      call check(err==MPI_SUCCESS,'segment size broadcast')
      call MPI_Bcast(segment_crc(p),1,MPI_INTEGER8,0,MPI_COMM_WORLD,err)
      call check(err==MPI_SUCCESS,'segment fingerprint broadcast')
      err=0; closed=0
      if(mpirank==0) then
        open(newunit=unit,file=trim(roots(p))//'/series.frames',status='new',action='write',iostat=err)
        if(err==0) then
          write(unit,'(a)',iostat=err) 'ASTR_FRAME_SERIES_1'
          close(unit,iostat=closed)
        endif
      endif
      call check(err==0.and.closed==0,'cannot create bounded series ledger')
      call plane_selections(p,axes,indices,n)
      components=6
      if(numq==11) components=12
      units='dimensionless'
      if(.not.nondimen) units='si'
      call output_product_derived_indices(products(p),selected,nselected)
      ok=.true.
      if(mpirank==0) call prepare_output_lineage(trim(roots(p)),trim(labels(p)),[ia,ja,ka]+1, &
        axes(:n),indices(:n),components,trim(units),selected(:nselected),lineage_compatible(p),ok)
      call check(ok,'invalid or over-budget explicit archive parent chain')
      call MPI_Bcast(lineage_compatible(p),1,MPI_LOGICAL,0,MPI_COMM_WORLD,err)
      call check(err==MPI_SUCCESS,'archive lineage compatibility broadcast')
      if(mpirank==0.and..not.lineage_compatible(p)) write(*,'(a,a)') &
        'ASTR_OUTPUT_LINEAGE unavailable: valid layout override; per-segment indexes retained product=',trim(labels(p))
    enddo
  end subroutine

  subroutine plane_selections(product,axes,indices,n)
    integer,intent(in) :: product
    integer,intent(out) :: axes(768),indices(768),n
    integer :: d,i,index
    n=1; axes(1)=0; indices(1)=0
    if(product/=2) return
    n=0
    do d=1,3
      do i=1,256
        select case(d)
        case(1)
          index=int(options%i_indices(i))
        case(2)
          index=int(options%j_indices(i))
        case(3)
          index=int(options%k_indices(i))
        end select
        if(index<0) cycle
        n=n+1; axes(n)=d; indices(n)=index
      enddo
    enddo
  end subroutine

  subroutine write_archive_geometry(product,path,identity,verify)
    integer,intent(in) :: product
    character(*),intent(in) :: path
    type(checkpoint_state_identity),intent(in) :: identity
    logical,intent(in) :: verify
    integer :: axes(768),indices(768),n,i,components
    integer(int64) :: peak,bytes
    character(32) :: group
    character(16) :: units
    character(1),parameter :: tags(3)=['i','j','k']
    call plane_selections(product,axes,indices,n)
    components=6
    if(numq==11) components=12
    units='dimensionless'
    if(.not.nondimen) units='si'
    do i=1,n
      group=''
      if(product==2) write(group,'(a,i12.12)') tags(axes(i)),indices(i)
      if(product==1) then
        call write_basic_output(path,[ia,ja,ka]+1,[ig0,jg0,kg0],[im,jm,km],axes(i),indices(i),components, &
          x(0:im,0:jm,0:km,:),pack_basic_output_cpu,identity,options%buffer_bytes,options%host_budget_bytes, &
          MPI_COMM_WORLD,trim(units),peak,bytes,geometry_only=.true.,verify_geometry=verify)
      else
        call write_basic_output(path,[ia,ja,ka]+1,[ig0,jg0,kg0],[im,jm,km],axes(i),indices(i),components, &
          x(0:im,0:jm,0:km,:),pack_basic_output_cpu,identity,options%buffer_bytes,options%host_budget_bytes, &
          MPI_COMM_WORLD,trim(units),peak,bytes,group_name=trim(group),append=i>1,geometry_only=.true., &
          verify_geometry=verify)
      endif
      call check(bytes==0,'shared geometry unexpectedly packed flow fields')
    enddo
  end subroutine

  subroutine observe_archives(identity,is_final)
    use benchmark_runtime, only: insitu_clock,report_insitu_timing
    type(checkpoint_state_identity),intent(in) :: identity
    logical,intent(in) :: is_final
    logical :: emit,ok
    integer(int64) :: crossed
    integer :: p
    real(real64) :: started
    do p=1,2
      if(.not.products(p)%enabled) cycle
      ! Restored/overridden identities must not leave an unpublishable pending clock.
      if(last_steps(p)==identity%step) cycle
      if(products(p)%adaptive%enabled) then
        call poll_adaptive_clock(adaptive_clocks(p),products(p)%adaptive,identity%step,identity%time,is_final,emit,ok)
      else
        call poll_schedule(schedules(p),identity%step,identity%time,.false.,emit,crossed,ok)
      endif
      call check(ok,'invalid complete-step archive clock')
      emit=emit.or.(is_final.and.products(p)%final_frame)
      if(.not.emit.or.last_steps(p)==identity%step) cycle
      started=insitu_clock()
      call write_frame(p,identity)
      call report_insitu_timing('archive_'//trim(labels(p))//'_inclusive',started,int(identity%step))
      if(products(p)%adaptive%enabled) then
        if(mpirank==0) write(*,'(a,a,a,i0,a,es24.16,a,l1,a,i0,a,i0,a,es24.16)') &
          'ASTR_AP_PRODUCT id=',trim(labels(p)),' step=',identity%step,' time=',identity%time, &
          ' dense=',adaptive_clocks(p)%dense,' reason=',adaptive_clocks(p)%reason, &
          ' target_step=',adaptive_clocks(p)%requested_step,' target_time=',adaptive_clocks(p)%requested_time
        call report_adaptive_clock(adaptive_clocks(p),0,0,ok)
        call check(ok,'adaptive archive publication history')
      endif
      last_steps(p)=identity%step
    enddo
  end subroutine

  subroutine write_frame(product,identity)
    integer,intent(in) :: product
    type(checkpoint_state_identity),intent(in) :: identity
    integer :: axes(768),indices(768),n,d,i,components,unit,err,closed,selected(14),nselected,nfields
    integer(int64) :: peak,bytes,total_bytes,download,capacity,budget,resident,tile_buffer,host_peak, &
      global_bytes,global_download,device_peak,host_workspace,device_workspace,host_budget,reported_host,reported_device
    integer :: ierr
    character(1200) :: path
    character(32) :: name,group
    character(16) :: units
    character(1),parameter :: tags(3)=['i','j','k']
    logical :: ok
    components=6
    if(numq==11) components=12
    call output_product_derived_indices(products(product),selected,nselected)
    nfields=components+nselected
    units='dimensionless'
    if(.not.nondimen) units='si'
    call plane_selections(product,axes,indices,n)
    write(name,'("step",i12.12)') identity%step
    path=trim(roots(product))//'/'//trim(name)//'.tmp'
    call create_batch(trim(path),MPI_COMM_WORLD,ok)
    call check(ok,'cannot create exclusive frame candidate')
    host_workspace=0; device_workspace=0
#ifdef _CUDA
    if(use_gpu) then
      if(nselected>0) call derived_output_workspace_gpu(host_workspace,device_workspace)
    else
#endif
      if(nselected>0) call derived_output_workspace_cpu(host_workspace)
#ifdef _CUDA
    endif
#endif
    host_budget=options%host_budget_bytes-host_workspace
#ifdef _CUDA
    host_budget=host_budget-compact_statistics_host_bytes()
#endif
    call check(host_budget>0,'private derivative halo leaves no host packing budget')
    tile_buffer=min(options%buffer_bytes,host_budget)
    device_peak=0; download=0; host_peak=0
#ifdef _CUDA
    if(use_gpu) then
      resident=statistics_device_bytes()+compact_statistics_device_bytes()
#ifdef ASTR_AIR5_CHEMISTRY
      resident=resident+air5_mean_statistics_device_bytes()
      resident=resident+air5_conservation_device_bytes()
#endif
      budget=options%device_budget_bytes-resident-device_workspace
      call check(budget>0,'statistics leave no device packing budget')
      capacity=min(tile_buffer/8/(nfields+3),budget/8/nfields, &
        product_nodes(product),int(huge(i)/(nfields+3),int64))
      call check(capacity>0,'device budget cannot hold one basic output node')
      ! Use the same node limit for host tiles and the flat device packing buffer.
      tile_buffer=capacity*8*(nfields+3)
      capacity=capacity*nfields
      device_peak=device_workspace+capacity*8
      if(nselected>0) then
        call begin_derived_output_gpu(capacity,options%host_budget_bytes, &
          options%device_budget_bytes-resident,MPI_COMM_WORLD,selected(:nselected),reported_host,reported_device, &
          options%device_reserve_bytes)
        call check(reported_host==host_workspace.and.reported_device==device_workspace, &
          'private GPU workspace changed during frame entry')
      else
        call begin_basic_output_gpu(capacity,budget,MPI_COMM_WORLD,options%device_reserve_bytes)
      endif
    else
#endif
      if(nselected>0) then
        capacity=min(tile_buffer/8/(nfields+3),product_nodes(product),int(huge(i)/(nfields+3),int64))
        call check(capacity>0,'host budget cannot hold one derived output node')
        tile_buffer=capacity*8*(nfields+3)
        capacity=capacity*nfields
        call begin_derived_output_cpu(capacity,options%host_budget_bytes,MPI_COMM_WORLD, &
          selected(:nselected),reported_host)
        call check(reported_host==host_workspace,'private CPU workspace changed during frame entry')
      endif
#ifdef _CUDA
    endif
#endif
    total_bytes=0
    do i=1,n
      group=''
      if(product==2) write(group,'(a,i12.12)') tags(axes(i)),indices(i)
#ifdef _CUDA
      if(use_gpu) then
        if(nselected>0) then
          call transfer(pack_derived_output_gpu)
        else
          call transfer(pack_basic_output_gpu)
        endif
      else
#endif
        if(nselected>0) then
          call transfer(pack_derived_output_cpu)
        else
          call transfer(pack_basic_output_cpu)
        endif
#ifdef _CUDA
      endif
#endif
      total_bytes=total_bytes+bytes
      host_peak=max(host_peak,host_workspace+peak)
    enddo
#ifdef _CUDA
    if(use_gpu) then
      if(nselected>0) then
        call end_derived_output_gpu(download)
      else
        call end_basic_output_gpu(download)
      endif
      call check(download==total_bytes,'device download count differs from selected fields')
    else
#endif
      if(nselected>0) call end_derived_output_cpu()
#ifdef _CUDA
    endif
#endif
    call MPI_Allreduce(total_bytes,global_bytes,1,MPI_INTEGER8,MPI_SUM,MPI_COMM_WORLD,ierr)
    call check(ierr==MPI_SUCCESS,'frame field byte reduction')
    call MPI_Allreduce(download,global_download,1,MPI_INTEGER8,MPI_SUM,MPI_COMM_WORLD,ierr)
    call check(ierr==MPI_SUCCESS,'frame download byte reduction')
    if(product==2) call write_slice_output_xdmf(trim(path)//'/data.xdmf',[ia,ja,ka]+1, &
      axes(:n),indices(:n),components,identity,trim(units),MPI_COMM_WORLD, &
      geometry_file='../../resources/data.h5',derived_indices=selected(:nselected))
    err=0; closed=0
    if(mpirank==0) then
      open(newunit=unit,file=trim(path)//'/FRAME',status='new',action='write',iostat=err)
      if(err==0) then
        if(nselected>0) then
          write(unit,'(a,/,i0,1x,es24.16,/,i0,1x,a,/,i0,1x,i0,/,14(i0,1x))',iostat=err) &
            'ASTR_DERIVED_FRAME_1',identity%step,identity%time,components,trim(units), &
            global_bytes,global_download,selected
        else
          write(unit,'(a,/,i0,1x,es24.16,/,i0,1x,a,/,i0,1x,i0)',iostat=err) 'ASTR_BASIC_FRAME_1', &
            identity%step,identity%time,components,trim(units),global_bytes,global_download
        endif
        close(unit,iostat=closed)
      endif
    endif
    call check(err==0.and.closed==0,'cannot write frame identity')
    call write_checkpoint_resource_refs(trim(path),[character(128) :: 'SEGMENT','data.h5'],MPI_COMM_WORLD,ok)
    call check(ok,'cannot record shared archive coordinates')
    call seal_checkpoint_bundle(trim(path),[character(128) :: 'data.h5','data.xdmf','FRAME','RESOURCES'],MPI_COMM_WORLD,ok)
    call check(ok,'cannot seal frame')
    call publish_checkpoint_bundle(trim(roots(product)),trim(name),MPI_COMM_WORLD,ok)
    call check(ok,'cannot publish frame')
    call publish_series(product,identity)
    write(*,'(a,i0,a,a,a,i0,a,i0,a,i0,a,i0)') 'ASTR_OUTPUT_ARCHIVE rank=',mpirank, &
      ' product=',trim(labels(product)),' field_bytes=',total_bytes,' download_bytes=',download, &
      'host_peak_bytes=',host_peak,' device_peak_bytes=',device_peak
  contains
    subroutine transfer(pack)
      interface
        subroutine pack(start,extent,components,values)
          import real64
          integer,intent(in) :: start(3),extent(3),components
          real(real64),intent(out) :: values(:)
        end subroutine
      end interface
      if(product==1) then
        call write_basic_output(trim(path),[ia,ja,ka]+1,[ig0,jg0,kg0],[im,jm,km],0,0,components, &
          x(0:im,0:jm,0:km,:),pack,identity,tile_buffer,host_budget, &
          MPI_COMM_WORLD,trim(units),peak,bytes,geometry_file='../../resources/data.h5', &
          derived_indices=selected(:nselected))
      else
        call write_basic_output(trim(path),[ia,ja,ka]+1,[ig0,jg0,kg0],[im,jm,km],axes(i),indices(i),components, &
          x(0:im,0:jm,0:km,:),pack,identity,tile_buffer,host_budget, &
          MPI_COMM_WORLD,trim(units),peak,bytes,group_name=trim(group),append=i>1, &
          geometry_file='../../resources/data.h5',derived_indices=selected(:nselected))
      endif
    end subroutine
  end subroutine

  subroutine publish_series(product,identity)
    integer,intent(in) :: product
    type(checkpoint_state_identity),intent(in) :: identity
    integer :: axes(768),indices(768),n,components,source,unit,err,closed,status,selected(14),nselected
    character(256) :: line
    character(16) :: units
    logical :: ok
    err=0; closed=0
    if(mpirank==0) then
      open(newunit=source,file=trim(roots(product))//'/series.frames',status='old',action='read',iostat=err)
      if(err==0) then
        open(newunit=unit,file=trim(roots(product))//'/series.frames.tmp',status='new',action='write',iostat=err)
        if(err==0) then
          do
            read(source,'(a)',iostat=status) line
            if(status==iostat_end) exit
            if(status/=0) then
              err=status; exit
            endif
            write(unit,'(a)',iostat=err) trim(line)
            if(err/=0) exit
          enddo
          if(err==0) write(unit,'(i0,1x,es24.16)',iostat=err) identity%step,identity%time
          close(unit,iostat=closed)
        endif
        close(source,iostat=status)
        if(status/=0) err=status
      endif
    endif
    call check(err==0.and.closed==0,'cannot stage series ledger')
    call plane_selections(product,axes,indices,n)
    components=6
    if(numq==11) components=12
    units='dimensionless'
    if(.not.nondimen) units='si'
    call output_product_derived_indices(products(product),selected,nselected)
    call write_output_series_xdmf(trim(roots(product))//'/series.xdmf.tmp', &
      trim(roots(product))//'/series.frames.tmp',[ia,ja,ka]+1,axes(:n),indices(:n),components,trim(units), &
      MPI_COMM_WORLD,selected(:nselected))
    err=0
    if(mpirank==0) err=replace_plain_file(trim(roots(product))//c_null_char, &
      'series.frames.tmp'//c_null_char,'series.frames'//c_null_char)
    call check(err==0,'cannot publish series ledger')
    if(mpirank==0) err=replace_plain_file(trim(roots(product))//c_null_char, &
      'series.xdmf.tmp'//c_null_char,'series.xdmf'//c_null_char)
    call check(err==0,'cannot publish time-series XDMF')
    if(.not.lineage_compatible(product)) return
    ok=.true.
    if(mpirank==0) call stage_output_lineage(trim(roots(product)),ok)
    call check(ok,'cannot stage explicit parent time series')
    call write_output_series_xdmf(trim(roots(product))//'/lineage.xdmf.tmp', &
      trim(roots(product))//'/lineage.frames.tmp',[ia,ja,ka]+1,axes(:n),indices(:n),components,trim(units), &
      MPI_COMM_WORLD,selected(:nselected))
    err=0
    if(mpirank==0) err=replace_plain_file(trim(roots(product))//c_null_char, &
      'lineage.frames.tmp'//c_null_char,'lineage.frames'//c_null_char)
    call check(err==0,'cannot publish explicit parent ledger')
    if(mpirank==0) err=replace_plain_file(trim(roots(product))//c_null_char, &
      'lineage.xdmf.tmp'//c_null_char,'lineage.xdmf'//c_null_char)
    call check(err==0,'cannot publish explicit parent XDMF')
  end subroutine

  integer(int64) function product_nodes(product) result(nodes)
    integer,intent(in) :: product
    integer(int64) :: dimensions(3)
    dimensions=int([im,jm,km],int64)+1
    nodes=product_of(dimensions)
    if(product==2) nodes=max(product_of(dimensions(1:2)),dimensions(1)*dimensions(3),dimensions(2)*dimensions(3))
  end function

  integer(int64) function product_of(values) result(result)
    integer(int64),intent(in) :: values(:)
    result=product(values)
  end function
end module
