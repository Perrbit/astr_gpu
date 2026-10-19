module output_runtime
  use iso_fortran_env, only: int64,real64,compiler_version,compiler_options
  use iso_c_binding, only: c_int,c_char,c_null_char
  use ieee_arithmetic, only: ieee_is_finite
  use mpi
  use commvar, only: ia,ja,ka,im,jm,km,hm,ndims,numq,num_species,num_modequ, &
    nstep,time,deltat,maxstep,use_gpu,flowtype,lcomb,lavg,lcracon,limmbou,lrestart, &
    feqchkpt,feqwsequ,feqslice,feqlist,feqavg,conschm,difschm,rkscheme,lihomo,ljhomo,lkhomo, &
    lreadgrid,gridfile,ninit,diffterm,lfilter,nondimen,turbmode,recon_schem,lchardecomp
  use commarray, only: q,rho,vel,prs,tmp,x,jacob,dxi,crinod
  use parallel, only: ig0,jg0,kg0,mpirank,irk,mpisize
  use bc, only: bctype,turbinf,ninflowslice,complete_inflow_cpu_file, &
    configure_wall_blowing,wall_blowing_legacy_random
  use output_input_resources, only: set_inflow_resource_root,inflow_source_path, &
    inflow_source_name,discover_inflow_sources,set_initial_resource_root,initial_source_path,initial_source_name, &
    set_wall_resource_root
  use wall_blowing_random, only: legacy_random_state_size,get_legacy_random_state, &
    put_legacy_random_state,legacy_random_runtime_signature
  use output_config, only: output_options
  use compact_runtime, only: global_compact_enabled
  use adaptive_output, only: configure_adaptive,adaptive_monitor_due,observe_adaptive,feed_adaptive_signal, &
    adaptive_shared_config,adaptive_shared_state,adaptive_shared_file
  use output_config_collective, only: read_output_options_collective
  use output_archive, only: configure_archives,begin_archives,observe_archives,archive_control_file,archives_enabled
  use checkpoint_state_io
  use checkpoint_bundle
  use insitu_checkpoint_batch, only: create_batch,copy_batch_file
  use insitu_session, only: configure_output_statistics,output_statistics_file, &
    output_render_file,begin_output_insitu,complete_output_insitu,output_render_repartition_allowed, &
    restart_output_statistics
  use insitu_schedule, only: sample_schedule,configure_schedule,poll_schedule, &
    write_schedule_state,restore_schedule_state
  use statistic, only: channel_driver_state,complete_mean_statistics_file,nsamples
  use benchmark_runtime, only: benchmark_field_io_disabled
#ifdef ASTR_AIR5_CHEMISTRY
  use commarray, only: tve,spc
  use chemistry_compensation, only: air5_carry,air5_compensated
  use chemistry_flow_runtime, only: air5_step_history,air5_history_enabled,air5_tolerance_choice, &
    air5_conservative_filter,air5_filter_identity
  use chemistry_hbl_boundary, only: complete_air5_top_contract
#endif
#ifdef _CUDA
  use statistic_gpu, only: gpu_adaptive_energy_sum
  use checkpoint_state_gpu, only: checkpoint_perfect_gas_gpu,checkpoint_crashfix_gpu, &
    checkpoint_lean_perfect_gas_gpu,rebuild_checkpoint_primitives_gpu
  use production_statistics_gpu, only: complete_compact_statistics_file,compact_statistics_host_bytes
  use inflow_timeseries_gpu, only: complete_inflow_gpu_file
#ifdef ASTR_AIR5_CHEMISTRY
  use checkpoint_state_gpu, only: checkpoint_air5_gpu
  use chemistry_mean_statistics_gpu, only: initialize_air5_mean_statistics_gpu,complete_air5_mean_statistics_file, &
    air5_mean_statistics_device_bytes
  use chemistry_flow_solver_gpu, only: air5_conservation_state_gpu,restore_air5_conservation_state_gpu, &
    begin_air5_conservation_file_gpu,configure_air5_checkpoint_conservation_gpu
#endif
#endif
  implicit none
  private
  public :: configure_output_runtime,begin_output_runtime,completed_output_runtime,new_output_enabled
  public :: initial_output_runtime
  public :: observe_output_adaptive
  public :: bootstrap_output_resources,output_resource_path
  public :: prepare_preview_paths,configure_preview_runtime,restore_preview_runtime
  logical,save :: enabled=.false.
  logical,save :: options_loaded=.false.
  logical,save :: statistics_active=.false.
  logical,save :: compact_statistics=.false.
  logical,save :: mean_statistics=.false.
  logical,save :: conservation_statistics=.false.
  logical,save :: repartitioning=.false.
  logical,save :: restored_statistics_saved=.true.
  ! Bounded x/z channel repartition passed the approved frozen-field matrix.
  logical,parameter :: channel_repartition_validated=.true.
  type(output_options),save :: options
  type(checkpoint_retention),save :: ledger
  type(sample_schedule),save :: checkpoint_schedule
  integer(int64),save :: contract(8),last_step=-1
  integer(int64),save :: schedule_origin_step=0
  real(real64),save :: schedule_origin_time=0
  real(real64),save :: initial_dt_used=0
  integer,save :: inflow_count=0
  character(64),parameter :: air5_resources(4)=[character(64) :: &
    'air5_hbl_domain.dat','air5_hbl_profile.dat','air5_hbl_initial_field.dat','air5_incident_shock.dat']
  interface
    function preview_output(batch,output) bind(C,name='astr_checkpoint_preview_output') result(status)
      import c_int,c_char
      character(c_char),intent(in) :: batch(*),output(*)
      integer(c_int) :: status
    end function
    function reuse_run(root,restore,reuse) bind(C,name='astr_checkpoint_reuse_run') result(status)
      import c_int,c_char
      character(c_char),intent(in) :: root(*),restore(*)
      integer(c_int),intent(out) :: reuse
      integer(c_int) :: status
    end function
  end interface
contains
  subroutine prepare_preview_paths()
    use insitu_run_config, only: insitu_options
    use insitu_config_collective, only: read_insitu_options_collective
    type(insitu_options) :: preview
    character(1024) :: filename,message
    integer :: status
    logical :: ok
    call load_output_options()
    filename=''
    call get_environment_variable('ASTR_INSITU_CONFIG',filename,status=status)
    call check(status==0.and.len_trim(filename)>0,'preview requires an explicit in-situ configuration')
    call read_insitu_options_collective(trim(filename),MPI_COMM_WORLD,preview,ok,message)
    call check(ok,trim(message))
    call check(preview_output(trim(options%restore_directory)//c_null_char, &
      trim(preview%output_directory)//c_null_char)==0, &
      'preview requires an empty output directory and working directory outside the source run')
  end subroutine

  subroutine configure_preview_runtime()
    use insitu_session, only: configure_preview_insitu
    character(1024) :: directory
    call configure_output_runtime()
    call check(use_gpu.and. &
      ((trim(flowtype)=='tgv'.and.all([ia,ja,ka]==32).and.all(bctype==1).and.ninit==0.and.trim(conschm)=='643e').or. &
       (trim(flowtype)=='bl'.and.all([ia,ja,ka]==[450,130,96]).and.all(bctype==[11,21,41,50,1,1]).and. &
        ninit==2.and.trim(conschm)=='743e'.and.recon_schem==5.and.lchardecomp)).and. &
      .not.lreadgrid.and.nondimen.and.numq==5.and.num_species==0.and.num_modequ==0.and. &
      .not.lcomb.and..not.lavg.and.trim(difschm)=='643e', &
      'preview requires approved FP64 TGV 32^3 or frozen M12 450x130x96 state')
    call check(len_trim(options%restore_directory)>0,'preview requires an existing new-format checkpoint')
    call check(.not.options%checkpoint%enabled.and..not.options%volume%enabled.and. &
      .not.options%slices%enabled.and..not.options%adaptive%enabled,'preview forbids production output and adaptive clocks')
    call configure_preview_insitu(directory)
    call check(preview_output(trim(options%restore_directory)//c_null_char,trim(directory)//c_null_char)==0, &
      'preview requires an empty output directory and working directory outside the source run')
  end subroutine

  subroutine restore_preview_runtime()
    use iso_c_binding, only: c_int64_t,c_double
#ifdef _CUDA
    use gpu_runtime, only: gpu_prepare_preview_geometry
#endif
    type(checkpoint_state_identity) :: identity,expected,geometry_identity
    character(8) :: magic
    integer(int64) :: saved(8),remaining,header(13)
    real(real64),allocatable :: buffer(:,:,:,:)
    integer :: unit,status,closed,a,b,m
    logical :: ok,matched
    real(real64) :: restore_started,restore_seconds,restore_max
#ifdef ASTR_WITH_CATALYST
    interface
      integer(c_int) function preview_source_set(path,step,t,fcomm) &
          bind(C,name='astr_insitu_preview_source_set')
        import c_char,c_int,c_int64_t,c_double
        character(c_char),intent(in) :: path(*)
        integer(c_int64_t),value :: step
        real(c_double),value :: t
        integer(c_int),value :: fcomm
      end function
    end interface
#endif
    restore_started=MPI_Wtime()
    magic='';saved=0;matched=.false.
    call validate_checkpoint_bundle(trim(options%restore_directory),MPI_COMM_WORLD,ok)
    call check(ok,'invalid preview checkpoint bundle')
    call checkpoint_state_header(trim(options%restore_directory)//'/state.h5',header,comm=MPI_COMM_WORLD)
    call check(header(13)==2,'preview requires a full checkpoint written with exact_restart=t')
    ! Only read the control prefix. Production clocks, RNG and driver history stay untouched.
    open(newunit=unit,file=trim(options%restore_directory)//'/control.bin',status='old', &
      access='stream',form='unformatted',convert='little_endian',action='read',iostat=status)
    call check(status==0,'open preview control metadata')
    call read_control_contract(unit,magic,saved,status)
    if(status==0) read(unit,iostat=status) identity%step,identity%time,identity%dt_used,identity%dt_next
    close(unit,iostat=closed)
    call check(status==0.and.closed==0.and. &
      ((trim(flowtype)=='tgv'.and.(magic=='ASTROC04'.or.magic=='ASTROC07')).or. &
       (trim(flowtype)=='bl'.and.(magic=='ASTROC06'.or.magic=='ASTROC09'))), &
      'preview control prefix differs from the admitted source; adaptive/RNG restoration is not performed')
    call check(saved(6)==1,'preview requires a GPU checkpoint')
    call check(.not.statistics_active.or.btest(saved(8),0),'preview checkpoint lacks native statistics')
    expected=identity
    geometry_identity=checkpoint_state_identity(0_int64,0.d0,0.d0,1.d0)
    call allocate_checkpoint_buffer([im,jm,km],hm,13,state_host_budget(),buffer,remaining,MPI_COMM_WORLD)
    call checkpoint_state_transfer(trim(options%restore_directory)//'/../../resources/geometry.h5',.false.,.true., &
      [ia,ja,ka]+1,[ig0,jg0,kg0],[im,jm,km],hm,buffer,geometry_identity,remaining,MPI_COMM_WORLD, &
      allow_repartition=.false.,restored_exact=matched)
    call check(matched.and.geometry_identity%step==0.and.geometry_identity%time==0.d0.and. &
      geometry_identity%dt_used==0.d0.and.geometry_identity%dt_next==1.d0,'preview geometry identity mismatch')
    call check(.not.allocated(jacob).and..not.allocated(dxi),'preview geometry was already initialized')
    allocate(jacob(-hm:im+hm,-hm:jm+hm,-hm:km+hm), &
      dxi(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3,3),stat=status)
    call check(status==0,'cannot allocate saved preview metrics')
    x=buffer(:,:,:,1:3)
    jacob=buffer(:,:,:,4)
    m=4
    do b=1,3
      do a=1,3
        m=m+1
        dxi(:,:,:,a,b)=buffer(:,:,:,m)
      enddo
    enddo
    deallocate(buffer)
#ifdef _CUDA
    call gpu_prepare_preview_geometry()
    call checkpoint_perfect_gas_gpu(trim(options%restore_directory)//'/state.h5',.false., &
      [ia,ja,ka]+1,[ig0,jg0,kg0],[im,jm,km],hm,identity,state_host_budget(),MPI_COMM_WORLD, &
      allow_repartition=.false.,restored_exact=matched)
#else
    call check(.false.,'device preview requires a CUDA build')
#endif
    call check(matched.and.identity%step==expected%step.and.identity%time==expected%time.and. &
      identity%dt_used==expected%dt_used.and.identity%dt_next==expected%dt_next,'preview flow/control clock mismatch')
    call check(identity%step<=huge(nstep),'preview step exceeds solver integer range')
    nstep=int(identity%step);time=identity%time;deltat=identity%dt_next
#ifdef ASTR_WITH_CATALYST
    call check(preview_source_set(trim(options%restore_directory)//c_null_char,identity%step,identity%time, &
      int(MPI_COMM_WORLD,c_int))==0,'cannot register preview source identity')
#endif
    call begin_output_insitu()
    if(statistics_active) call output_statistics_file(trim(options%restore_directory)//'/statistics.h5', &
      .false.,identity,state_host_budget(),allow_repartition=.false.)
    restore_seconds=MPI_Wtime()-restore_started
    call MPI_Allreduce(restore_seconds,restore_max,1,MPI_DOUBLE_PRECISION,MPI_MAX,MPI_COMM_WORLD,status)
    call check(status==MPI_SUCCESS,'preview restore timing consensus')
    if(mpirank==0) then
      write(*,'(a,es24.16)') 'ASTR_INSITU_PREVIEW_RESTORE seconds=',restore_max
      write(*,'(a,i0,a,es24.16,a)') 'ASTR_INSITU_PREVIEW restored_step=',nstep,' time=',time,' advances=0'
    endif
  end subroutine

  logical function dynamic_output_case()
    dynamic_output_case=trim(flowtype)=='bl'.and.trim(turbinf)=='intp'
  end function
  logical function air5_output_case()
    air5_output_case=.false.
#ifdef ASTR_AIR5_CHEMISTRY
    air5_output_case=(trim(flowtype)=='air5hbl'.or.trim(flowtype)=='air5sbli').and.lcomb.and.numq==11.and. &
      num_species==5.and.num_modequ==1.and..not.lreadgrid.and. &
      all(bctype==[11,50,41,51,1,1])
#endif
  end function
  integer function air5_resource_count() result(count)
    count=3
    if(trim(flowtype)=='air5sbli') count=4
  end function
  function air5_checkpoint_resources() result(names)
    character(64),allocatable :: names(:)
    logical :: selected(4)
    selected=[.true.,.true.,.false.,trim(flowtype)=='air5sbli']
    inquire(file=trim(output_resource_path('air5_hbl_initial_field.dat', &
      'datin/air5_hbl_initial_field.dat')),exist=selected(3))
    names=pack(air5_resources,selected)
  end function
  logical function new_output_enabled()
    new_output_enabled=enabled
  end function

  logical function tgv_repartition_case()
    tgv_repartition_case=trim(flowtype)=='tgv'.and.all(bctype==1).and..not.lreadgrid.and. &
      ninit==0.and..not.mean_statistics.and..not.compact_statistics.and. &
      trim(conschm)=='643e'.and.trim(difschm)=='643e'.and.lfilter.and.diffterm.and. &
      output_render_repartition_allowed()
  end function

  logical function channel_repartition_case()
    character(128) :: mode,value
    real(real64) :: forcing
    integer :: status,ios
    channel_repartition_case=.false.
    if(.not.channel_repartition_validated) return
    if(trim(flowtype)/='channel'.or.any(bctype/=[1,1,41,41,1,1])) return
    if(lreadgrid.or.ninit/=3.or..not.nondimen.or.any([ia,ja,ka]/=16)) return
    if(jg0/=0.or.jm/=ja.or..not.lihomo.or.ljhomo.or..not.lkhomo) return
    if(compact_statistics.or.statistics_active) return
    if(trim(conschm)/='643e'.or.trim(difschm)/='643e'.or..not.lfilter.or..not.diffterm) return
    if(.not.output_render_repartition_allowed()) return
    call get_environment_variable('ASTR_CHANNEL_FORCE_MODE',mode,status=status)
    if(status/=0.or.trim(mode)/='fixed') return
    call get_environment_variable('ASTR_CHANNEL_FORCE_FIXED',value,status=status)
    if(status/=0) return
    read(value,*,iostat=ios) forcing
    if(ios/=0) return
    if(.not.ieee_is_finite(forcing)) return
    channel_repartition_case=forcing==1.d-4
  end function

  logical function registered_repartition_case()
    registered_repartition_case=tgv_repartition_case().or.channel_repartition_case().or. &
      curve_repartition_case().or.air5_repartition_case()
  end function

  logical function air5_repartition_case()
    character(64) :: value
    integer :: status,i
    character(32),parameter :: names(4)=[character(32) :: 'ASTR_AIR5_SOURCE_MODE', &
      'ASTR_AIR5_CONVECTION_LIMITER','ASTR_AIR5_DIFFUSION_LIMITER','ASTR_AIR5_TOP_MODE']
    character(32),parameter :: expected(4)=[character(32) :: 'coupled','symmetric_species','layered','characteristic']
    air5_repartition_case=.false.
#ifdef ASTR_AIR5_CHEMISTRY
    if(.not.air5_output_case()) return
    if(ninit/=0.or.nondimen.or.any([ia,ja,ka]/=16)) return
    if(jg0/=0.or.jm/=ja.or.lihomo.or.ljhomo.or..not.lkhomo) return
    if(trim(flowtype)=='air5hbl') then
      if(ig0/=0.or.im/=ia.or.recon_schem/=5) return
    else
      if(kg0/=0.or.km/=ka.or.recon_schem/=3) return
    endif
    if(mpisize<1.or.mpisize>2.or.compact_statistics.or.statistics_active) return
    if(.not.air5_compensated.or.trim(turbmode)/='none') return
    if(trim(conschm)/='643e'.or.trim(difschm)/='643e'.or..not.lfilter.or..not.diffterm) return
    if(lchardecomp.or..not.output_render_repartition_allowed()) return
    do i=1,size(names)
      call get_environment_variable(trim(names(i)),value,status=status)
      if(status/=0.or.trim(value)/=trim(expected(i))) return
    enddo
    air5_repartition_case=.true.
#endif
  end function

  logical function curve_repartition_case()
    curve_repartition_case=trim(flowtype)=='bl'.and.lreadgrid.and.ninit==0.and.nondimen.and. &
      mpisize>=1.and.mpisize<=2.and. &
      all([ia,ja,ka]==16).and.all(bctype==[11,21,41,51,1,1]).and. &
      (trim(turbinf)=='prof'.or.(dynamic_output_case().and.inflow_count==12)).and. &
      jg0==0.and.jm==ja.and..not.lihomo.and..not.ljhomo.and.lkhomo.and. &
      .not.statistics_active.and. &
      trim(conschm)=='543e'.and.trim(difschm)=='643e'.and.lfilter.and.diffterm.and. &
      trim(turbmode)=='none'.and.recon_schem==3.and..not.lchardecomp.and. &
      output_render_repartition_allowed()
  end function

  function registered_repartition_axes() result(axes)
    logical :: axes(3)
    axes=[.true.,.not.channel_repartition_case(),.true.]
    if(curve_repartition_case()) axes=[.true.,.false.,.true.]
    if(air5_repartition_case()) then
      axes=[.false.,.false.,.true.]
      if(trim(flowtype)=='air5sbli') axes=[.true.,.false.,.false.]
    endif
  end function

  integer(int64) function state_host_budget() result(bytes)
    bytes=options%host_budget_bytes
#ifdef _CUDA
    bytes=bytes-compact_statistics_host_bytes()
#endif
  end function

  subroutine check(ok,message)
    logical,intent(in) :: ok
    character(*),intent(in) :: message
    call checkpoint_state_require(ok,MPI_COMM_WORLD,message)
  end subroutine

  function last_runtime_checkpoint() result(path)
    character(1200) :: path
    path=last_published_checkpoint(ledger)
    if(trim(path)=='none in this run'.and.len_trim(options%restore_directory)>0) path=options%restore_directory
  end function

  logical function extruded_profile_output_case()
    extruded_profile_output_case=trim(flowtype)=='bl'.and..not.lreadgrid.and.ninit==2.and.nondimen.and. &
      trim(conschm)=='743e'.and.trim(difschm)=='643e'.and.recon_schem==5.and.lchardecomp.and. &
      .not.lfilter.and.trim(turbinf)=='prof'.and. &
      (all(bctype==[11,21,41,50,1,1]).or.all(bctype==[11,22,41,52,1,1]))
  end function

  function profile_grid_resource_name() result(name)
    character(128) :: name
    name='grid.h5'
    if(extruded_profile_output_case()) name='grid.2d'
  end function

  function profile_grid_resource_source() result(path)
    character(1200) :: path
    path=gridfile
    if(extruded_profile_output_case()) path='datin/grid.2d'
  end function

  subroutine check_capability()
    logical :: supported_case
#ifdef ASTR_BUILD_TESTING
    character(16) :: validation_mode
    integer :: validation_status
#endif
    call check(options%host_budget_bytes>0,'new output requires an explicit host buffer budget')
    if(options%volume%enabled.or.options%slices%enabled) call check(ndims==3.and. &
      ((numq==5.and.num_species==0.and.num_modequ==0.and..not.lcomb).or.air5_output_case()), &
      'field archives require a registered three-dimensional field layout')
    ! Exact continuation has a stricter contract than ordinary solver execution.
    if(.not.options%checkpoint%enabled.and.len_trim(options%restore_directory)==0) return
    call check(.not.air5_output_case().or.options%exact_restart, &
      'AIR5 checkpoint currently requires exact_restart=t to preserve chemical continuation state')
    supported_case=(trim(flowtype)=='tgv'.and.all(bctype==1)).or. &
      (trim(flowtype)=='tgv'.and.lreadgrid.and. &
      (all([ia,ja,ka]==32).or.all([ia,ja,ka]==64).or.all([ia,ja,ka]==128).or.all([ia,ja,ka]==256)).and. &
      all(bctype==[1,1,41,41,1,1]).and.trim(conschm)=='643e'.and.trim(difschm)=='643e').or. &
      (trim(flowtype)=='channel'.and.all(bctype==[1,1,41,41,1,1])).or. &
      (trim(flowtype)=='bl'.and.lreadgrid.and.(trim(turbinf)=='prof'.or.trim(turbinf)=='intp').and. &
      all(bctype==[11,21,41,51,1,1])).or.extruded_profile_output_case()
#ifdef ASTR_BUILD_TESTING
    ! This bounded RK probe has no output/restart contract to publish.
    call get_environment_variable('ASTR_VALIDATION_NSCBC_RK',validation_mode,status=validation_status)
    if(validation_status==0.and.trim(validation_mode)=='1') then
      supported_case=trim(flowtype)=='openshock'.and.use_gpu.and. &
        all([ia,ja,ka]==[64,8,8]).and.all(bctype==[12,22,1,1,1,1]).and. &
        .not.lreadgrid.and.ninit==0.and.nondimen.and..not.lavg.and. &
        .not.lfilter.and..not.diffterm.and..not.lchardecomp.and. &
        trim(conschm)=='543e'.and.trim(difschm)=='643e'.and.recon_schem==3.and. &
        maxstep==5.and.deltat==1.d-5.and. &
        .not.options%checkpoint%enabled.and..not.options%volume%enabled.and..not.options%slices%enabled.and. &
        .not.options%adaptive%enabled.and.len_trim(options%restore_directory)==0
    endif
#endif
    supported_case=supported_case.and.numq==5.and.num_species==0.and.num_modequ==0.and. &
      .not.lcomb.and.(.not.lavg.or..not.use_gpu.or.trim(flowtype)=='bl')
    call check((supported_case.or.air5_output_case()).and..not.limmbou.and. &
      .not.lrestart.and.trim(rkscheme)=='rk3', &
      'new output admits TGV, bc41 channel, profile/dynamic flatplate or fixed AIR5 HBL/SBLI RK3')
    call check(.not.lcracon.or.(numq==5.and.num_species==0.and..not.lcomb), &
      'checkpoint crashfix supports only non-reacting five-variable flow')
    if(extruded_profile_output_case().and.bctype(2)==22) then
      call check(use_gpu.and..not.options%adaptive%enabled.and..not.lavg, &
        'profile joint NSCBC admits GPU archives and same-topology restart; means/adaptive not validated')
    endif
    call check(ninit>=0.and.ninit<=3,'new output initialization dimension must be 0:3')
    if(air5_output_case()) call check(ninit==0,'AIR5 external initialization is not registered')
    if(air5_output_case().and.lavg) call check(diffterm.and.feqavg>0, &
      'AIR5 mean statistics currently require viscous gradients and a positive sample interval')
  end subroutine

  subroutine load_output_options()
    character(1024) :: config
    character(256) :: message
    integer :: status,length,flag,minflag,maxflag,ierr
    logical :: ok
    if(options_loaded) return
    options_loaded=.true.
    config=''
    call get_environment_variable('ASTR_OUTPUT_CONFIG',config,length=length,status=status)
    flag=0
    if (status==0.and.length>0) flag=1
    call MPI_Allreduce(flag,minflag,1,MPI_INTEGER,MPI_MIN,MPI_COMM_WORLD,ierr)
    call check(ierr==MPI_SUCCESS,'output configuration override reduce')
    call MPI_Allreduce(flag,maxflag,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
    call check(ierr==MPI_SUCCESS.and.minflag==maxflag.and.(status==0.or.status==1), &
      'inconsistent or truncated output configuration override')
    if(maxflag==0) config='datin/input.output'
    enabled=.true.
    call read_output_options_collective(trim(config),options,MPI_COMM_WORLD,ok,message)
    call check(ok,'cannot load required output configuration '//trim(config)//': '//trim(message))
    call check(.not.lrestart,'legacy checkpoint restart is disabled; use lrestart=f and output restore_directory')
    if(mpirank==0) write(*,'(2a)') 'ASTR_OUTPUT_CONFIG_FILE ',trim(config)
  end subroutine

  function output_resource_path(name,fallback) result(path)
    character(*),intent(in) :: name,fallback
    character(1200) :: path
    path=fallback
    if(enabled.and.(trim(flowtype)=='bl'.or.trim(flowtype)=='air5hbl'.or.trim(flowtype)=='air5sbli'.or. &
      (trim(flowtype)=='tgv'.and.lreadgrid)).and. &
      len_trim(options%restore_directory)>0) &
      path=trim(options%restore_directory)//'/../../resources/'//name
  end function

  subroutine bootstrap_output_resources()
    call load_output_options()
    if(.not.enabled) return
    if(len_trim(options%restore_directory)>0) then
      if(extruded_profile_output_case()) &
        call set_wall_resource_root(trim(options%restore_directory)//'/../../resources')
    endif
    if(trim(flowtype)=='bl'.and.bctype(3)==41) call configure_wall_blowing()
    if(extruded_profile_output_case().and.len_trim(options%restore_directory)>0) &
      call check(wall_blowing_legacy_random,'extruded profile restart requires legacy_random checkpoint')
    if(wall_blowing_legacy_random.and.(options%checkpoint%enabled.or.len_trim(options%restore_directory)>0)) then
      call check(extruded_profile_output_case(), 'random wall checkpoint requires admitted extruded profile case')
      call check(.not.options%adaptive%enabled, 'random wall checkpoint adaptive scheduling is not admitted')
    endif
    if(ninit>=1.and.ninit<=3) then
      if(len_trim(options%restore_directory)>0) &
        call set_initial_resource_root(trim(options%restore_directory)//'/../../resources')
    endif
    if(dynamic_output_case()) call bootstrap_inflow_resources()
  end subroutine

  subroutine bootstrap_inflow_resources()
    character(1400) :: path,root
    character(8) :: magic
    integer :: ierr,unit,err,closed
    logical :: ok,restoring
    restoring=len_trim(options%restore_directory)>0
    root='inflow'
    if(restoring) root=trim(options%restore_directory)//'/../../resources'
    ok=.true.
    if(mpirank==0) then
      if(restoring) then
        path=trim(root)//'/inflow_index.bin'
        open(newunit=unit,file=trim(path),status='old',access='stream',form='unformatted', &
          convert='little_endian',action='read',iostat=err)
        ok=err==0
        if(ok) then
          read(unit,iostat=err) magic,inflow_count
          close(unit,iostat=closed)
          ok=err==0.and.closed==0.and.(magic=='ASTRIF01'.or.magic=='ASTRIF02')
        endif
      else
        call discover_inflow_sources(trim(root),inflow_count,ok)
      endif
    endif
    call check(ok,'dynamic inflow requires a frozen contiguous regular-file sequence from 00000')
    call MPI_Bcast(inflow_count,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)
    call check(ierr==MPI_SUCCESS.and.inflow_count>=4.and.inflow_count<=100000,'dynamic inflow source count')
    ! Resource names coexist while a new checkpoint is published.
    call check(288_int64*(int(inflow_count,int64)+6)+131072_int64<=options%host_budget_bytes, &
      'dynamic inflow resource metadata budget')
    if(restoring) call set_inflow_resource_root(trim(root),inflow_count)
  end subroutine

  subroutine configure_output_runtime()
    character(128) :: value
    integer :: status
    logical :: ok
    call load_output_options()
    if(.not.enabled) return
    if(benchmark_field_io_disabled()) then
      options%checkpoint%enabled=.false.
      options%volume%enabled=.false.
      options%slices%enabled=.false.
      if(mpirank==0) write(*,'(a)') 'ASTR_OUTPUT no-field-I/O overrides checkpoint, volume and slices'
    endif
    call check_capability()
    if(air5_output_case()) then
      value=''
      call get_environment_variable('ASTR_AIR5_C4_CONSERVATION',value,status=status)
      call check(status==0.or.status==1,'AIR5 conservation option is truncated')
      select case(trim(value))
      case('1','t','T','true','TRUE','yes','YES','on','ON')
        conservation_statistics=.true.
      case('','0','f','F','false','FALSE','no','NO','off','OFF')
        conservation_statistics=.false.
      case default
        call check(.false.,'invalid AIR5 conservation diagnostic option')
      end select
      call check(.not.conservation_statistics.or.use_gpu,'AIR5 conservation diagnostic is GPU-only')
    endif
    call configure_adaptive(options%adaptive,0_int64,0.d0,ok)
    call check(ok,'invalid adaptive monitor configuration')
    if(options%adaptive%enabled) call check(trim(flowtype)=='tgv'.and.all(bctype==1).and. &
      .not.lreadgrid.and.numq==5.and.num_species==0.and..not.lcomb.and.nondimen.and. &
      (all([ia,ja,ka]==16).or.all([ia,ja,ka]==32)), &
      'adaptive output currently admits only registered 16/32-cubed periodic Cartesian nonreacting TGV')
    call configure_output_statistics(statistics_active)
    compact_statistics=use_gpu.and.lavg.and.trim(flowtype)=='bl'
    mean_statistics=lavg.and.(.not.use_gpu.or.air5_output_case())
    call check(.not.(statistics_active.and.mean_statistics),'formal and CPU legacy statistics need separate transactions')
    call configure_archives(options)
  end subroutine

  subroutine begin_output_runtime(counter,rkfirst_pending)
    integer,intent(inout) :: counter
    logical,intent(inout) :: rkfirst_pending
    character(1024) :: path,source
    character(256) :: value,root_probe
    integer :: status,ierr,i
    logical :: ok
    type(checkpoint_state_identity) :: identity
    integer(c_int) :: reuse,status_reuse
    if (.not.enabled) return
    call check_capability()
    call check(lavg.eqv.(compact_statistics.or.mean_statistics),'legacy statistics activation changed during the run')
    call begin_output_insitu()
#if defined(_CUDA) && defined(ASTR_AIR5_CHEMISTRY)
    if(use_gpu.and.mean_statistics) call initialize_air5_mean_statistics_gpu(options%device_budget_bytes)
    if(use_gpu.and.air5_output_case()) then
      call configure_air5_checkpoint_conservation_gpu( &
        options%device_budget_bytes-air5_mean_statistics_device_bytes(),ok)
      call check(ok,'AIR5 conservation diagnostic device budget/allocation')
    endif
#endif
    call get_command_argument(2,source,status=status)
    call check(status==0,'input path unavailable')
    contract=0
    contract(1:6)=[int(feqchkpt,int64),int(feqwsequ,int64),int(feqslice,int64), &
      int(feqlist,int64),int(feqavg,int64),int(merge(1,0,use_gpu),int64)]
    contract(8)=merge(1_int64,0_int64,statistics_active)+merge(2_int64,0_int64,compact_statistics)+ &
      merge(4_int64,0_int64,mean_statistics)
    if(.not.options%checkpoint_statistics) contract(8)=ibset(contract(8),3)
    ! Only exact checkpoint consumers require the validated FP64/sync contract.
    if(options%checkpoint%enabled.or.len_trim(options%restore_directory)>0) then
    do i=1,3
      value=''
      select case(i)
      case(1)
        call get_environment_variable('ASTR_GPU_PRECISION_MODE',value,status=status)
        call check(status==1.or.trim(value)==''.or.trim(value)=='fp64','new restart requires FP64')
      case(2)
        call get_environment_variable('ASTR_GPU_SYNC_MODE',value,status=status)
        call check(status==1.or.trim(value)==''.or.trim(value)=='explicit','new restart first gate requires explicit sync')
      case(3)
        call get_environment_variable('ASTR_GPU_FILTER_WORKSPACE',value,status=status)
        contract(7)=merge(1_int64,0_int64,trim(value)=='full')
        value=''
        call get_environment_variable('ASTR_GPU_MP_POSITIVITY',value,status=status)
        if(trim(value)=='flux') contract(7)=ibset(contract(7),1)
        if(global_compact_enabled()) contract(7)=ibclr(contract(7),0)
      end select
    enddo
    endif
    if(options%checkpoint%enabled) then
      call configure_schedule(checkpoint_schedule,trim(options%checkpoint%mode), &
        options%checkpoint%interval_steps,options%checkpoint%interval_time, &
        schedule_origin_step,schedule_origin_time,options%checkpoint%initial_frame,.false.,ok)
      call check(ok,'invalid checkpoint schedule')
    endif
    if (len_trim(options%restore_directory)>0) then
      path=trim(options%restore_directory)
      call validate_checkpoint_bundle(trim(path),MPI_COMM_WORLD,ok)
      call check(ok,'invalid new checkpoint bundle')
      call control_file(trim(path)//'/control.bin',.false.,counter,rkfirst_pending,identity)
      call archive_control_file(trim(path)//'/archives.bin',.false.,identity)
      call output_render_file(trim(path)//'/insitu_control.bin',.false.,identity,state_host_budget(), &
        options%restart_output=='override')
#ifdef ASTR_AIR5_CHEMISTRY
      if(air5_output_case()) call air5_contract_file(trim(path)//'/air5_config.bin',.false.)
#ifdef _CUDA
      if(air5_output_case().and.use_gpu) call conservation_file(trim(path)//'/air5_conservation.bin',.false.,identity)
#endif
#endif
      call flow_file(trim(path)//'/state.h5',.false.,identity)
      value=''
      call get_environment_variable('ASTR_CHECKPOINT_TEST_RESTORE_PROBE',value,status=status)
      call check(status==0.or.status==1,'restore probe option is truncated')
      root_probe=value
      call MPI_Bcast(root_probe,len(root_probe),MPI_CHARACTER,0,MPI_COMM_WORLD,ierr)
      call check(ierr==MPI_SUCCESS.and.value==root_probe,'restore probe option differs between ranks')
      if(len_trim(value)>0) then
        call check(trim(value)=='1'.and.air5_repartition_case(),'restore probe requires bounded AIR5 HBL/SBLI')
        ! Test-only reserialization of the actual host/device state before any RK stage.
        call flow_file('outdat/restore_probe.h5',.true.,identity)
      endif
      if(dynamic_output_case()) call inflow_file(trim(path)//'/inflow.h5',.false.,identity)
      call check(identity%step<=huge(nstep),'restored step exceeds solver integer range')
      if(options%checkpoint_statistics.and.restored_statistics_saved) then
      if(statistics_active) call output_statistics_file(trim(path)//'/statistics.h5',.false., &
        identity,state_host_budget(),allow_repartition=repartitioning)
      if(mean_statistics) call mean_statistics_file(trim(path)//'/statistics.h5',.false., &
        identity,state_host_budget())
#ifdef _CUDA
      if(compact_statistics) call complete_compact_statistics_file(trim(path)//'/statistics.h5',.false., &
        identity,state_host_budget(),allow_repartition=curve_repartition_case())
#endif
      else
        nsamples=0
        call restart_output_statistics(int(identity%step),identity%time)
      endif
      if(len_trim(root_probe)>0) then
        if(mean_statistics) call mean_statistics_file('outdat/restore_statistics_probe.h5',.true., &
          identity,state_host_budget())
#if defined(_CUDA) && defined(ASTR_AIR5_CHEMISTRY)
        if(conservation_statistics) call conservation_file('outdat/restore_conservation_probe.bin',.true.,identity)
#endif
      endif
      nstep=int(identity%step)
      time=identity%time
      deltat=identity%dt_next
      initial_dt_used=identity%dt_used
      call check(nstep<=maxstep+1,'restored step lies past requested stop')
    endif
    identity=checkpoint_state_identity(int(nstep,int64),time,initial_dt_used,deltat)
    if(archives_enabled().or.options%checkpoint%enabled) then
      status_reuse=0; reuse=0
      if(mpirank==0) status_reuse=reuse_run(trim(options%directory)//c_null_char, &
        trim(options%restore_directory)//c_null_char,reuse)
      call MPI_Bcast(reuse,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)
      call check(ierr==MPI_SUCCESS.and.status_reuse==0,'existing output root requires its own validated restore source')
    endif
    if(archives_enabled()) call begin_archives(identity,reuse==1)
#if defined(_CUDA) && defined(ASTR_AIR5_CHEMISTRY)
    if(conservation_statistics) then
      write(value,'("air5_conservation_from_step",i12.12,".dat")') identity%step
      call begin_air5_conservation_file_gpu(trim(options%directory)//'/'//trim(value),ok)
      call check(ok,'cannot create exclusive AIR5 conservation diagnostic segment')
    endif
#endif
    if (.not.options%checkpoint%enabled) return
    if(reuse==1) return
    ! The caller prepares the output root; exclusive subdirectories prevent reuse.
    call create_batch(trim(options%directory)//'/resources',MPI_COMM_WORLD,ok)
    call check(ok,'cannot create new run resources directory')
    call create_batch(trim(options%directory)//'/checkpoints',MPI_COMM_WORLD,ok)
    call check(ok,'cannot create new run checkpoints directory')
    if(options%exact_restart) call write_geometry_resource(trim(options%directory)//'/resources/geometry.h5')
    if (mpirank==0) call copy_batch_file(trim(source),trim(options%directory)//'/resources/input.txt',ok)
    call check(ok,'cannot freeze primary input resource')
    if(trim(flowtype)=='tgv'.and.lreadgrid) &
      call freeze_static_resource('grid.h5',trim(gridfile))
    if(trim(flowtype)=='bl') then
      call freeze_static_resource(trim(profile_grid_resource_name()),trim(profile_grid_resource_source()))
      call freeze_static_resource('inlet.prof','datin/inlet.prof')
      if(wall_blowing_legacy_random) &
        call freeze_static_resource('wallbs.dat','datin/wallbs.dat')
      if(dynamic_output_case()) call freeze_inflow_resources()
    endif
    if(air5_output_case()) then
      block
        character(64),allocatable :: names(:)
        names=air5_checkpoint_resources()
        do i=1,size(names)
          call freeze_static_resource(trim(names(i)),'datin/'//trim(names(i)))
        enddo
      end block
    endif
    if(ninit>=1.and.ninit<=3) &
      call freeze_static_resource(initial_source_name(ninit),trim(initial_source_path(ninit)))
  end subroutine

  subroutine mean_statistics_file(path,writing,identity,budget)
    character(*),intent(in) :: path
    logical,intent(in) :: writing
    type(checkpoint_state_identity),intent(inout) :: identity
    integer(int64),intent(in) :: budget
#if defined(_CUDA) && defined(ASTR_AIR5_CHEMISTRY)
    if(use_gpu.and.air5_output_case()) then
      call complete_air5_mean_statistics_file(path,writing,identity,budget,allow_repartition=air5_repartition_case(), &
        repartition_axes=registered_repartition_axes())
      return
    endif
#endif
    call complete_mean_statistics_file(path,writing,identity,budget, &
      allow_repartition=channel_repartition_case().or.curve_repartition_case().or.air5_repartition_case(), &
      repartition_axes=registered_repartition_axes())
  end subroutine

  subroutine freeze_inflow_resources()
    integer :: i,found,unit,err,closed
    logical :: ok
    character(1400) :: root,path
    root='inflow'
    if(len_trim(options%restore_directory)>0) root=trim(options%restore_directory)//'/../../resources'
    do i=1,inflow_count
      path=inflow_source_path(i-1)
      call freeze_static_resource(inflow_source_name(i-1),trim(path))
    enddo
    ok=.true.
    if(mpirank==0) then
      call discover_inflow_sources(trim(root),found,ok)
      ok=ok.and.found==inflow_count
      if(ok) then
        path=trim(options%directory)//'/resources/inflow_index.bin'
        open(newunit=unit,file=trim(path),status='new',access='stream',form='unformatted', &
          convert='little_endian',action='write',iostat=err)
        ok=err==0
        if(ok) then
          write(unit,iostat=err) 'ASTRIF02',inflow_count
          close(unit,iostat=closed)
          ok=err==0.and.closed==0
        endif
      endif
    endif
    call check(ok,'cannot freeze dynamic inflow sequence')
    call set_inflow_resource_root(trim(options%directory)//'/resources',inflow_count)
  end subroutine

  subroutine inflow_file(path,writing,identity)
    character(*),intent(in) :: path
    logical,intent(in) :: writing
    type(checkpoint_state_identity),intent(inout) :: identity
    type(checkpoint_state_identity) :: expected
    expected=identity
#ifdef _CUDA
    if(use_gpu) then
      call complete_inflow_gpu_file(path,writing,identity,state_host_budget(), &
        allow_repartition=curve_repartition_case())
    else
#endif
      call complete_inflow_cpu_file(path,writing,identity,state_host_budget(), &
        allow_repartition=curve_repartition_case())
#ifdef _CUDA
    endif
#endif
    call check(irk/=0.or.ninflowslice<inflow_count,'dynamic inflow cache cursor outside frozen sources')
    call check(identity%step==expected%step.and.identity%time==expected%time.and. &
      identity%dt_used==expected%dt_used.and.identity%dt_next==expected%dt_next,'inflow/control clock mismatch')
  end subroutine

  subroutine freeze_static_resource(name,fallback)
    character(*),intent(in) :: name,fallback
    character(1200) :: path
    logical :: ok
    path=output_resource_path(name,fallback)
    ok=.true.
    if(mpirank==0) then
      call copy_batch_file(trim(path),trim(options%directory)//'/resources/'//name,ok)
    endif
    call check(ok,'cannot copy static resource '//name)
  end subroutine

  subroutine initial_output_runtime(counter,pending)
    integer,intent(inout) :: counter
    logical,intent(inout) :: pending
    if(.not.enabled) return
    call completed_output_runtime(initial_dt_used,counter,pending,initial=.true.)
  end subroutine

  subroutine flow_file(path,writing,identity)
    use fludyna, only: updatefvar
    use parallel, only: qswap
    character(*),intent(in) :: path
    logical,intent(in) :: writing
    type(checkpoint_state_identity),intent(inout) :: identity
    type(checkpoint_state_identity) :: expected
    real(real64),allocatable :: buffer(:,:,:,:)
    integer(int64) :: header(13),remaining
    logical :: matched,lean
    expected=identity
    lean=.not.options%exact_restart
    if(.not.writing) then
      call checkpoint_state_header(path,header,comm=MPI_COMM_WORLD)
      if(options%exact_restart) call check(header(13)/=11, &
        'exact_restart=t cannot restore a non-exact checkpoint')
      lean=header(13)==11
    endif
#ifdef ASTR_AIR5_CHEMISTRY
    if(air5_output_case()) then
#ifdef _CUDA
      if(use_gpu) then
        call checkpoint_air5_gpu(path,writing,[ia,ja,ka]+1,[ig0,jg0,kg0],[im,jm,km],hm, &
          identity,state_host_budget(),MPI_COMM_WORLD,allow_repartition=air5_repartition_case(),restored_exact=matched, &
          refresh_halos=refresh_checkpoint_halos,repartition_axes=registered_repartition_axes())
      else
#endif
        call checkpoint_air5_state(path,writing,[ia,ja,ka]+1,[ig0,jg0,kg0],[im,jm,km],hm, &
          q,air5_carry,air5_compensated,rho,vel,prs,tmp,tve,spc,identity,state_host_budget(),MPI_COMM_WORLD, &
          allow_repartition=air5_repartition_case(),restored_exact=matched,refresh_halos=refresh_checkpoint_halos, &
          repartition_axes=registered_repartition_axes(),step_history=air5_step_history)
#ifdef _CUDA
      endif
#endif
    else
#endif
#ifdef _CUDA
    if (use_gpu) then
      if(lean) then
      call checkpoint_lean_perfect_gas_gpu(path,writing,[ia,ja,ka]+1,[ig0,jg0,kg0],[im,jm,km], &
        identity,state_host_budget(),options%device_budget_bytes,MPI_COMM_WORLD, &
        registered_repartition_case(),matched,registered_repartition_axes())
      else
      call checkpoint_perfect_gas_gpu(path,writing,[ia,ja,ka]+1,[ig0,jg0,kg0],[im,jm,km],hm, &
        identity,state_host_budget(),MPI_COMM_WORLD,allow_repartition=registered_repartition_case(),restored_exact=matched, &
        refresh_halos=refresh_checkpoint_halos,repartition_axes=registered_repartition_axes())
      if(.not.writing.and..not.options%exact_restart) call rebuild_checkpoint_primitives_gpu()
      endif
      call checkpoint_crashfix_gpu(path,writing,state_host_budget(),options%device_budget_bytes,MPI_COMM_WORLD,matched, &
        allow_inexact=.not.options%exact_restart)
    else
#endif
      if(lean) then
        call allocate_checkpoint_buffer([im,jm,km],0,5,state_host_budget(),buffer,remaining,MPI_COMM_WORLD)
        buffer=0.d0
        if(writing) buffer=q(0:im,0:jm,0:km,:)
        call checkpoint_state_transfer(path,writing,.false.,[ia,ja,ka]+1,[ig0,jg0,kg0],[im,jm,km],0, &
          buffer,identity,remaining,MPI_COMM_WORLD,role=11,allow_repartition=registered_repartition_case(), &
          restored_exact=matched,repartition_axes=registered_repartition_axes(),physical_only=.true.)
        if(.not.writing) then
          q=0.d0
          q(0:im,0:jm,0:km,:)=buffer
        endif
        deallocate(buffer)
      else
        call checkpoint_perfect_gas_state(path,writing,[ia,ja,ka]+1,[ig0,jg0,kg0],[im,jm,km],hm, &
        q,rho,vel,prs,tmp,identity,state_host_budget(),MPI_COMM_WORLD, &
        allow_repartition=registered_repartition_case(),restored_exact=matched,refresh_halos=refresh_checkpoint_halos, &
        repartition_axes=registered_repartition_axes())
      endif
      if(.not.writing.and..not.options%exact_restart) then
        call updatefvar()
        call qswap()
      endif
      call checkpoint_crashfix_state(path,writing,crinod,lcracon,state_host_budget(),MPI_COMM_WORLD,matched, &
        allow_inexact=.not.options%exact_restart)
#ifdef _CUDA
    endif
#endif
#ifdef ASTR_AIR5_CHEMISTRY
    endif
#endif
    if (.not.writing) then
      repartitioning=.not.matched
      call check(identity%step==expected%step.and.identity%time==expected%time.and. &
        identity%dt_used==expected%dt_used.and.identity%dt_next==expected%dt_next,'flow/control clock mismatch')
      if(repartitioning.and.mpirank==0) write(*,'(3a)') 'ASTR_OUTPUT_REPARTITION ',trim(flowtype), &
        ' physical-node restore; communication halos rebuilt'
    endif
  end subroutine

#ifdef ASTR_AIR5_CHEMISTRY
  subroutine canonical_air5_modes(values)
    character(128),intent(inout) :: values(10)
    character(128),parameter :: defaults(10)=[character(128) :: &
      'coupled','full_state','full_state','off','prescribed','', &
      'off','baseline','off','off']
    integer :: i
    do i=1,10
      values(i)=trim(adjustl(values(i)))
      if(len_trim(values(i))==0) values(i)=defaults(i)
    enddo
    ! The parsed relaxation rate is compared bitwise in the separate top contract.
    values(6)='inactive'
    if(values(5)=='characteristic') values(6)='top-contract'
    select case(trim(values(10)))
    case('1','t','T','true','TRUE','yes','YES','on','ON')
      values(10)='on'
    case('0','f','F','false','FALSE','no','NO','off','OFF')
      values(10)='off'
    end select
  end subroutine

  subroutine air5_contract_file(path,writing)
    character(*),intent(in) :: path
    logical,intent(in) :: writing
    character(40),parameter :: names(10)=[character(40) :: &
      'ASTR_AIR5_SOURCE_MODE','ASTR_AIR5_CONVECTION_LIMITER','ASTR_AIR5_DIFFUSION_LIMITER', &
      'ASTR_AIR5_COMPENSATION','ASTR_AIR5_TOP_MODE','ASTR_AIR5_TOP_TAU', &
      'ASTR_AIR5_PRIMITIVE_REUSE','ASTR_AIR5_CHEMISTRY_REDUCTIONS', &
      'ASTR_AIR5_FILTER_VALIDATION','ASTR_AIR5_C4_CONSERVATION']
    character(128) :: values(10),root_values(10),saved_values(10)
    character(128) :: source_evaluation,root_source_evaluation,saved_source_evaluation
    character(8) :: magic,expected_magic,filter_magic
    real(real64) :: top(40)
    integer(int64) :: bits(40),root_bits(40),saved_bits(40),legacy_crc(4)
    integer(int64) :: integration_control(2),saved_integration_control(2)
    integer(int64) :: filter_control(5),saved_filter_control(5)
    integer :: i,status,unit,err,closed,ierr,count
    count=air5_resource_count()
    expected_magic='ASTRA503'
    if(count==4) expected_magic='ASTRA504'
    do i=1,10
      values(i)=''
      call get_environment_variable(trim(names(i)),values(i),status=status)
      call check(status==0.or.status==1,'AIR5 configuration is truncated')
    enddo
    call canonical_air5_modes(values)
    source_evaluation=''
    call get_environment_variable('ASTR_AIR5_SOURCE_EVALUATION',source_evaluation,status=status)
    call check(status==0.or.status==1,'AIR5 source evaluation is truncated')
    source_evaluation=trim(adjustl(source_evaluation))
    if(len_trim(source_evaluation)==0) source_evaluation='joint'
    call check(source_evaluation=='joint'.or.source_evaluation=='source-only','invalid AIR5 source evaluation')
    call check(use_gpu.or.source_evaluation=='joint','AIR5 source-only evaluation is GPU-only')
    integration_control=[int(air5_tolerance_choice,int64),merge(1_int64,0_int64,air5_history_enabled)]
    filter_control=0
    if(air5_conservative_filter) filter_control=int(air5_filter_identity,int64)
    call complete_air5_top_contract(top)
    bits=transfer(top,bits)
    root_values=values; root_bits=bits
    call MPI_Bcast(root_values,1280,MPI_CHARACTER,0,MPI_COMM_WORLD,ierr)
    call check(ierr==MPI_SUCCESS.and.all(root_values==values),'AIR5 runtime modes differ between ranks')
    root_source_evaluation=source_evaluation
    call MPI_Bcast(root_source_evaluation,128,MPI_CHARACTER,0,MPI_COMM_WORLD,ierr)
    call check(ierr==MPI_SUCCESS.and.root_source_evaluation==source_evaluation, &
      'AIR5 source evaluation differs between ranks')
    call MPI_Bcast(root_bits,40,MPI_INTEGER8,0,MPI_COMM_WORLD,ierr)
    call check(ierr==MPI_SUCCESS.and.all(root_bits==bits),'AIR5 top contract differs between ranks')
    err=0; closed=0
    if(writing) then
      if(mpirank==0) then
        open(newunit=unit,file=path,status='new',access='stream',form='unformatted', &
          convert='little_endian',action='write',iostat=err)
        if(err==0) then
          write(unit,iostat=err) expected_magic,values,bits,source_evaluation
          if(err==0.and.(any(integration_control/=0).or.air5_conservative_filter)) &
            write(unit,iostat=err) integration_control
          if(err==0.and.air5_conservative_filter) write(unit,iostat=err) 'ASTRF001',filter_control
          close(unit,iostat=closed)
        endif
      endif
      call check(err==0.and.closed==0,'write AIR5 configuration')
    else
      open(newunit=unit,file=path,status='old',access='stream',form='unformatted', &
        convert='little_endian',action='read',iostat=err)
      call check(err==0,'open AIR5 configuration')
      read(unit,iostat=err) magic,saved_values,saved_bits
      if(err==0.and.(magic=='ASTRA501'.or.magic=='ASTRA502')) read(unit,iostat=err) legacy_crc(:count)
      call check(err==0,'AIR5 configuration size/tail')
      ! Batches predating the appended field used the joint evaluator exclusively.
      saved_source_evaluation='joint'
      if(.not.checkpoint_stream_at_end(unit)) read(unit,iostat=err) saved_source_evaluation
      call check(err==0,'AIR5 configuration size/tail')
      ! No numeric extension means baseline tolerance with history disabled.
      saved_integration_control=0
      if(.not.checkpoint_stream_at_end(unit)) read(unit,iostat=err) saved_integration_control
      call check(err==0,'AIR5 configuration size/tail')
      saved_filter_control=0
      if(.not.checkpoint_stream_at_end(unit)) then
        read(unit,iostat=err) filter_magic,saved_filter_control
        call check(err==0.and.filter_magic=='ASTRF001','AIR5 filter control version/size')
        call check(all(saved_filter_control==int(air5_filter_identity,int64)),'AIR5 filter control contract mismatch')
      endif
      call check(err==0.and.checkpoint_stream_at_end(unit),'AIR5 configuration size/tail')
      close(unit,iostat=closed)
      call check(err==0.and.closed==0,'read AIR5 configuration')
      call canonical_air5_modes(saved_values)
      call check(saved_source_evaluation==source_evaluation,'AIR5 source evaluation contract mismatch')
      call check(all(saved_integration_control==integration_control),'AIR5 integration control contract mismatch')
      call check(all(saved_filter_control==filter_control),'AIR5 filter control contract mismatch')
      call check((magic==expected_magic.or.(count==3.and.magic=='ASTRA501').or. &
        (count==4.and.magic=='ASTRA502')).and.all(saved_values==values).and.all(saved_bits==bits), &
        'AIR5 source/limiter/compensation/top contract mismatch')
    endif
  end subroutine

#ifdef _CUDA
  subroutine conservation_file(path,writing,identity)
    use ieee_arithmetic, only: ieee_is_finite
    character(*),intent(in) :: path
    logical,intent(in) :: writing
    type(checkpoint_state_identity),intent(in) :: identity
    integer(int64) :: metadata(5),root_metadata(5),bits(11),root_bits(11),step,file_bytes
    real(real64) :: baseline(11),clock(3)
    character(8) :: magic
    integer :: unit,err,closed,ierr
    call check(state_host_budget()>=512,'AIR5 conservation scalar state budget')
    call air5_conservation_state_gpu(metadata,baseline)
    call check(metadata(1)==merge(1_int64,0_int64,conservation_statistics), &
      'AIR5 conservation activation differs from output configuration')
    if(writing) then
      root_metadata=metadata; bits=transfer(baseline,bits); root_bits=bits
      call MPI_Bcast(root_metadata,5,MPI_INTEGER8,0,MPI_COMM_WORLD,ierr)
      call check(ierr==MPI_SUCCESS.and.all(metadata==root_metadata),'AIR5 conservation history differs between ranks')
      call MPI_Bcast(root_bits,11,MPI_INTEGER8,0,MPI_COMM_WORLD,ierr)
      call check(ierr==MPI_SUCCESS.and.all(bits==root_bits),'AIR5 conservation baseline differs between ranks')
    else
      open(newunit=unit,file=path,status='old',access='stream',form='unformatted', &
        convert='little_endian',action='read',iostat=err)
      call check(err==0,'missing AIR5 conservation state')
      inquire(unit=unit,size=file_bytes,iostat=err)
      call check(err==0.and.file_bytes==168,'AIR5 conservation state size')
      read(unit,iostat=err) magic,step,clock,metadata,baseline
      call check(err==0.and.magic=='ASTRA5C1'.and.step==identity%step.and. &
        all(clock==[identity%time,identity%dt_used,identity%dt_next]),'AIR5 conservation state clock/version')
      call check(checkpoint_stream_at_end(unit),'AIR5 conservation state tail')
      close(unit,iostat=closed)
      call check(closed==0,'AIR5 conservation state close')
    endif
    call check(metadata(1)==merge(1_int64,0_int64,conservation_statistics).and. &
      metadata(2)>=0.and.metadata(2)<=metadata(1).and.all(ieee_is_finite(baseline)), &
      'AIR5 conservation activation/nonfinite baseline')
    if(metadata(2)==1) then
      call check(metadata(3)>=0.and.metadata(3)<identity%step.and.metadata(4)==identity%step.and. &
        metadata(5)==metadata(4)-metadata(3),'AIR5 conservation sample identity')
    else
      call check(all(metadata(3:)==0).and.all(baseline==0).and. &
        (.not.conservation_statistics.or.identity%step==0),'AIR5 conservation empty history')
    endif
    if(writing) then
      err=0; closed=0
      if(mpirank==0) then
        open(newunit=unit,file=path,status='new',access='stream',form='unformatted', &
          convert='little_endian',action='write',iostat=err)
        if(err==0) then
          write(unit,iostat=err) 'ASTRA5C1',identity%step,[identity%time,identity%dt_used,identity%dt_next],metadata,baseline
          close(unit,iostat=closed)
        endif
      endif
      call check(err==0.and.closed==0,'write AIR5 conservation state')
    else
      call restore_air5_conservation_state_gpu(metadata,baseline)
    endif
  end subroutine
#endif
#endif

  subroutine write_geometry_resource(path)
    character(*),intent(in) :: path
    real(real64),allocatable :: buffer(:,:,:,:)
    integer(int64) :: remaining
    integer :: a,b,m
    type(checkpoint_state_identity) :: identity
    identity=checkpoint_state_identity(0_int64,0.0_real64,0.0_real64,1.0_real64)
    call allocate_checkpoint_buffer([im,jm,km],hm,13,state_host_budget(),buffer,remaining,MPI_COMM_WORLD)
    do m=1,3
      call pack_geometry_field(buffer(:,:,:,m),x(:,:,:,m),.false.)
    enddo
    call pack_geometry_field(buffer(:,:,:,4),jacob,.true.)
    m=4
    do b=1,3
      do a=1,3
        m=m+1
        call pack_geometry_field(buffer(:,:,:,m),dxi(:,:,:,a,b),.true.)
      enddo
    enddo
    call checkpoint_state_transfer(path,.true.,.true.,[ia,ja,ka]+1,[ig0,jg0,kg0], &
      [im,jm,km],hm,buffer,identity,remaining,MPI_COMM_WORLD, &
      allow_repartition=registered_repartition_case(), &
      repartition_axes=registered_repartition_axes())
  end subroutine

  subroutine refresh_checkpoint_halos(buffer,cells,halo,budget,comm)
    use parallel, only: dataswap
    real(real64),contiguous,intent(inout) :: buffer(:,:,:,:)
    integer,intent(in) :: cells(3),halo,comm
    integer(int64),intent(in) :: budget
    integer(int64) :: face,ncomp
    call checkpoint_state_require(comm==MPI_COMM_WORLD.and.all(cells==[im,jm,km]).and.halo==hm.and. &
      registered_repartition_case().and.minval(cells)>=halo,comm,'repartition halo needs registered solver layout')
    ncomp=size(buffer,4,kind=int64)
    face=max(int(cells(1)+1,int64)*(cells(2)+1),int(cells(1)+1,int64)*(cells(3)+1), &
      int(cells(2)+1,int64)*(cells(3)+1))
    ! Four MPI face arrays plus conservative room for two assignment temporaries.
    call checkpoint_state_require(face<=budget/8/6/max(1,halo)/ncomp,comm,'repartition halo scratch budget')
    call dataswap(buffer)
  end subroutine

  subroutine geometry_defined_bounds(metric,lo,hi)
    logical,intent(in) :: metric
    integer,intent(out) :: lo(3),hi(3)
    lo=1
    hi=[im,jm,km]+2*hm+1
    if(.not.metric) return
    ! dataswap does not define metric halos outside a nonperiodic physical face.
    where(.not.[lihomo,ljhomo,lkhomo].and.[ig0,jg0,kg0]==0) lo=hm+1
    where(.not.[lihomo,ljhomo,lkhomo].and.[ig0,jg0,kg0]+[im,jm,km]==[ia,ja,ka]) hi=[im,jm,km]+hm+1
  end subroutine

  subroutine pack_geometry_field(saved,actual,metric)
    real(real64),intent(out) :: saved(:,:,:)
    real(real64),intent(in) :: actual(:,:,:)
    logical,intent(in) :: metric
    integer :: h,lo(3),hi(3)
    h=hm+1
    call geometry_defined_bounds(metric,lo,hi)
    saved=0.d0
    saved(lo(1):hi(1),h:h+jm,h:h+km)=actual(lo(1):hi(1),h:h+jm,h:h+km)
    saved(h:h+im,lo(2):hi(2),h:h+km)=actual(h:h+im,lo(2):hi(2),h:h+km)
    saved(h:h+im,h:h+jm,lo(3):hi(3))=actual(h:h+im,h:h+jm,lo(3):hi(3))
  end subroutine

  subroutine read_control_contract(unit,magic,saved,status)
    integer,intent(in) :: unit
    character(8),intent(out) :: magic
    integer(int64),intent(out) :: saved(8)
    integer,intent(out) :: status
    integer(int64) :: legacy(14)
    read(unit,iostat=status) magic
    if(status/=0) return
    select case(magic)
    case('ASTROC04','ASTROC05','ASTROC06')
      ! Existing production batches carry obsolete fingerprints around these fields.
      read(unit,iostat=status) legacy
      if(status==0) saved=legacy(5:12)
    case('ASTROC07','ASTROC08','ASTROC09')
      read(unit,iostat=status) saved
    case default
      status=1
    end select
  end subroutine

  subroutine control_file(path,writing,counter,pending,identity)
    character(*),intent(in) :: path
    logical,intent(in) :: writing
    integer,intent(inout) :: counter
    logical,intent(inout) :: pending
    type(checkpoint_state_identity),intent(inout) :: identity
    integer(int64) :: saved(8),interval,saved_last,origin_step
    real(real64) :: dt_interval,origin_time
    integer :: unit,err,closed,saved_counter,status,ierr
    logical :: saved_pending,saved_initial,ok,same_schedule
    real(real64) :: driver(6),root_driver(6)
    character(128) :: driver_config(3),saved_driver_config(3),root_config(3)
    type(sample_schedule) :: restored_schedule
    character(8) :: magic
    character(16) :: mode
    integer,allocatable :: wall_state(:,:),local_wall_state(:)
    integer(int64) :: rng_signature(8),saved_signature(8),saved_wall_crc
    integer :: packet_size,saved_packet,saved_ranks
    character(16384) :: rng_compiler,rng_options,saved_compiler,saved_options
    character(8) :: wall_magic,expected_magic
    magic='ASTROC07'
    if(options%adaptive%enabled) magic='ASTROC08'
    if(wall_blowing_legacy_random) magic='ASTROC09'
    expected_magic=magic
    if(wall_blowing_legacy_random) then
      packet_size=legacy_random_state_size()+6
      call check(packet_size>7.and.packet_size<=4096,'random wall RNG packet size')
      call check(4_int64*packet_size*(int(mpisize,int64)+1)+65536<=options%host_budget_bytes, &
        'random wall RNG metadata budget')
      allocate(wall_state(packet_size,mpisize),local_wall_state(packet_size),stat=err)
      call check(err==0,'random wall RNG metadata allocation')
      local_wall_state(:6)=[ig0,jg0,kg0,im,jm,km]
      call get_legacy_random_state(local_wall_state(7:))
      call legacy_random_runtime_signature(rng_signature)
      call check(len(compiler_version())<=len(rng_compiler).and.len(compiler_options())<=len(rng_options), &
        'random wall compiler identity length')
      rng_compiler=compiler_version(); rng_options=compiler_options()
      if(writing) then
        call MPI_Gather(local_wall_state,packet_size,MPI_INTEGER,wall_state,packet_size,MPI_INTEGER, &
          0,MPI_COMM_WORLD,ierr)
        call check(ierr==MPI_SUCCESS,'random wall RNG state gather')
      endif
    endif
    driver=0
    driver_config=''
    if(trim(flowtype)=='channel') then
      call get_environment_variable('ASTR_CHANNEL_FORCE_MODE',driver_config(1),status=status)
      call check(status==0.or.status==1,'channel force mode is truncated')
      call get_environment_variable('ASTR_CHANNEL_FORCE_FIXED',driver_config(2),status=status)
      call check(status==0.or.status==1,'channel fixed force is truncated')
      if(writing) then
        call channel_driver_state(driver,.true.,ok)
        call check(ok,'invalid channel driver state')
        root_driver=driver
        call MPI_Bcast(root_driver,6,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
        call check(ierr==MPI_SUCCESS,'channel driver state broadcast')
        call check(all(transfer(root_driver,[0_int64],6)==transfer(driver,[0_int64],6)), &
          'channel driver state differs between ranks')
      endif
    endif
    if(trim(flowtype)=='bl') then
      call get_environment_variable('ASTR_PROFILE_INFLOW_MODE',driver_config(3),status=status)
      call check(status==0.or.status==1,'profile inflow mode is truncated')
    endif
    root_config=driver_config
    call MPI_Bcast(root_config,384,MPI_CHARACTER,0,MPI_COMM_WORLD,ierr)
    call check(ierr==MPI_SUCCESS.and.all(root_config==driver_config),'boundary/driver configuration differs between ranks')
    ! Small control metadata is read by all ranks; no field data is gathered.
    if (writing) then
      err=0
      closed=0
      if (mpirank==0) then
        open(newunit=unit,file=path,status='new',access='stream',form='unformatted', &
          convert='little_endian',action='write',iostat=err)
        if (err==0) then
          write(unit,iostat=err) magic,contract,identity%step,identity%time,identity%dt_used,identity%dt_next, &
            counter,pending,last_step,options%checkpoint%mode,options%checkpoint%interval_steps, &
            options%checkpoint%interval_time,options%checkpoint%initial_frame,schedule_origin_step,schedule_origin_time, &
            driver_config,driver
          if(err==0) then
            call write_schedule_state(unit,checkpoint_schedule,'checkpoint',identity%step,identity%time,ok)
            if(.not.ok) err=1
          endif
          if(err==0.and.magic=='ASTROC08') then
            call adaptive_shared_file(unit,.true.,identity%step,identity%time,.false.,ok)
            if(.not.ok) err=1
          endif
          if(err==0.and.magic=='ASTROC09') write(unit,iostat=err) 'ASTRWR02',mpisize,packet_size, &
            rng_compiler,rng_options,rng_signature,wall_state
          if(err==0) call compact_control_file(unit,.true.,err)
          close(unit,iostat=closed)
        endif
      endif
      call check(err==0.and.closed==0,'write control metadata')
    else
      open(newunit=unit,file=path,status='old',access='stream',form='unformatted', &
        convert='little_endian',action='read',iostat=err)
      call check(err==0,'open control metadata')
      call read_control_contract(unit,magic,saved,err)
      call check(err==0,'read control contract')
      read(unit,iostat=err) identity%step,identity%time,identity%dt_used,identity%dt_next, &
        saved_counter,saved_pending,saved_last,mode,interval,dt_interval,saved_initial,origin_step,origin_time, &
        saved_driver_config,driver
      call check(err==0,'read control metadata')
      restored_statistics_saved=.not.btest(saved(8),3)
      saved(8)=ibclr(saved(8),3)
      if(global_compact_enabled()) saved(7)=ibclr(saved(7),0)
      call check(all(saved(1:7)==contract(1:7)).and.saved(8)==ibclr(contract(8),3), &
        'numerical/controller contract mismatch')
      call check((magic=='ASTROC06'.or.magic=='ASTROC09').eqv.(expected_magic=='ASTROC09'), &
        'random wall checkpoint RNG mode/version mismatch')
      call check(all(saved_driver_config==driver_config),'boundary/driver configuration mismatch')
      if(trim(flowtype)=='channel') then
        call channel_driver_state(driver,.false.,ok)
        call check(ok,'invalid restored channel driver state')
      endif
      call configure_schedule(restored_schedule,trim(mode),interval,dt_interval, &
        origin_step,origin_time,saved_initial,.false.,ok)
      call check(ok,'invalid saved checkpoint schedule')
      call restore_schedule_state(unit,restored_schedule,'checkpoint',identity%step,identity%time,ok)
      call check(ok,'checkpoint schedule history')
      if(magic=='ASTROC05'.or.magic=='ASTROC08') then
        call adaptive_shared_file(unit,.false.,identity%step,identity%time,options%restart_output=='override',ok)
      else
        call check(.not.options%adaptive%enabled.or.options%restart_output=='override', &
          'adaptive monitoring enabled without explicit override')
        call configure_adaptive(options%adaptive,identity%step,identity%time,ok)
      endif
      call check(ok,'adaptive monitor/control metadata')
      if(magic=='ASTROC06'.or.magic=='ASTROC09') then
        read(unit,iostat=err) wall_magic,saved_ranks,saved_packet
        call check(err==0,'missing random wall RNG metadata')
        if(wall_magic=='ASTRWR01') read(unit,iostat=err) saved_wall_crc
        if(err==0) read(unit,iostat=err) saved_compiler,saved_options,saved_signature
        call check(err==0,'read random wall RNG metadata')
        call check((wall_magic=='ASTRWR01'.or.wall_magic=='ASTRWR02').and. &
          saved_ranks==mpisize.and.saved_packet==packet_size, &
          'random wall RNG format/rank count mismatch')
        call check(saved_compiler==rng_compiler.and.saved_options==rng_options.and.all(saved_signature==rng_signature), &
          'random wall compiler/runtime mismatch')
        read(unit,iostat=err) wall_state
        call check(err==0,'missing random wall RNG rank state')
        call check(all(wall_state(:6,mpirank+1)==local_wall_state(:6)), 'random wall topology mismatch')
        call put_legacy_random_state(wall_state(7:,mpirank+1),ok)
        call check(ok,'invalid random wall RNG rank state')
      endif
      call compact_control_file(unit,.false.,err)
      call check(err==0.and.ok.and.checkpoint_stream_at_end(unit),'adaptive monitor/control metadata tail')
      close(unit,iostat=closed)
      call check(ok.and.closed==0,'read checkpoint schedule state')
      counter=saved_counter
      pending=saved_pending
      last_step=saved_last
      same_schedule=mode==options%checkpoint%mode.and.interval==options%checkpoint%interval_steps.and. &
        dt_interval==options%checkpoint%interval_time.and.(saved_initial.eqv.options%checkpoint%initial_frame)
      if (options%restart_output=='saved') &
        call check(same_schedule,'saved output schedule differs; select explicit override')
      if(same_schedule) then
        checkpoint_schedule=restored_schedule
        schedule_origin_step=origin_step
        schedule_origin_time=origin_time
      else
        schedule_origin_step=identity%step
        schedule_origin_time=identity%time
        if(options%checkpoint%enabled) then
          call configure_schedule(checkpoint_schedule,trim(options%checkpoint%mode), &
            options%checkpoint%interval_steps,options%checkpoint%interval_time, &
            schedule_origin_step,schedule_origin_time,options%checkpoint%initial_frame,.false.,ok)
          call check(ok,'invalid overridden checkpoint schedule')
        endif
      endif
    endif
  end subroutine

  subroutine compact_control_file(unit,writing,status)
    use commvar, only: alfa_filter,bfacmpld,shkcrt
    integer,intent(in) :: unit
    logical,intent(in) :: writing
    integer,intent(out) :: status
    character(8) :: tag
    character(64) :: operator_id,saved_id
    character(4) :: schemes(2),saved_schemes(2)
    integer(int64) :: settings(17),saved_settings(17)
    real(real64) :: parameters(3),saved_parameters(3)
    logical :: active,present,matched
    active=global_compact_enabled()
    operator_id='astr-global-compact-v1'
    schemes=[conschm,difschm]
    settings=[1_int64,int(hm,int64),int(ndims,int64),int(numq,int64),int(recon_schem,int64), &
      int(merge(1,0,diffterm),int64),int(merge(1,0,lfilter),int64), &
      int(merge(1,0,lchardecomp),int64),int(merge(1,0,lihomo),int64), &
      int(merge(1,0,ljhomo),int64),int(merge(1,0,lkhomo),int64),int(bctype,int64)]
    parameters=0.d0
    if(lfilter) parameters(1)=alfa_filter
    if(recon_schem==5) parameters(2)=bfacmpld
    if(lchardecomp) parameters(3)=shkcrt
    status=0
    if(writing) then
      if(active) write(unit,iostat=status) 'ASTRCM01',operator_id,schemes,settings,parameters
      return
    endif
    ! Old new-format batches have no tail; only non-exact compact migration is allowed.
    present=.not.checkpoint_stream_at_end(unit)
    matched=.false.
    if(present) then
      read(unit,iostat=status) tag
      if(status/=0.or.tag/='ASTRCM01') then
        status=1
        return
      endif
      read(unit,iostat=status) saved_id,saved_schemes,saved_settings,saved_parameters
      call check(status==0,'invalid compact operator identity record')
      matched=active.and.saved_id==operator_id.and.all(saved_schemes==schemes).and. &
        all(saved_settings==settings).and.all(saved_parameters==parameters)
    endif
    if(.not.active.and..not.present) return
    if(options%exact_restart) then
      call check(present,'compact operator identity missing; select exact_restart=f for migration')
      call check(matched,'compact operator identity mismatch; select exact_restart=f for migration')
      if(mpirank==0) write(*,'(A)') 'ASTR_COMPACT_RESTART exact operator identity matched'
    else
      if(mpirank==0) then
        if(.not.present) then
          write(*,'(A)') 'ASTR_COMPACT_RESTART non-exact; source operator identity unknown; derived state rebuilt'
        else if(.not.matched) then
          write(*,'(A)') 'ASTR_COMPACT_RESTART non-exact; operator identity changed; derived state rebuilt'
        else
          write(*,'(A)') 'ASTR_COMPACT_RESTART non-exact; operator identity matched; derived state rebuilt'
        endif
      endif
    endif
  end subroutine

  subroutine observe_output_adaptive(step,t)
    use commvar, only: roinf,uinf
    use benchmark_runtime, only: insitu_clock,report_insitu_timing
    integer,intent(in) :: step
    real(real64),intent(in) :: t
    logical :: due,ok
    real(real64) :: local,global,value,density,reference,start,phase_started
    integer :: i,j,k,ierr
    if(.not.adaptive_shared_config%enabled) return
    if(adaptive_shared_state%last_step==int(step,int64).and.adaptive_shared_state%last_time==t) return
    start=MPI_Wtime()
    phase_started=insitu_clock()
    call observe_adaptive(int(step,int64),t,ok)
    call check(ok,'invalid adaptive complete-step/window clock')
    call adaptive_monitor_due(int(step,int64),t,due,ok)
    call check(ok,'invalid adaptive monitoring clock')
    call report_insitu_timing('adaptive_event_clock',phase_started,step)
    if(.not.due) return
    phase_started=insitu_clock()
    local=0; ok=.true.
#ifdef _CUDA
    if(use_gpu) then
      call gpu_adaptive_energy_sum(local,ok)
    else
#endif
      do k=0,km-1
      do j=0,jm-1
      do i=0,im-1
        density=q(i,j,k,1)
        if(.not.ieee_is_finite(density).or.density<=0) then
          ok=.false.; cycle
        endif
        value=(q(i,j,k,2)**2+q(i,j,k,3)**2+q(i,j,k,4)**2)/density
        if(.not.ieee_is_finite(value).or.value<0) then
          ok=.false.; cycle
        endif
        local=local+value
      enddo
      enddo
      enddo
#ifdef _CUDA
    endif
#endif
    call check(ok.and.ieee_is_finite(local),'invalid adaptive kinetic energy node or local sum')
    call MPI_Allreduce(local,global,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierr)
    call check(ierr==MPI_SUCCESS.and.ieee_is_finite(global),'adaptive kinetic energy reduction')
    reference=roinf*uinf
    call check(ieee_is_finite(reference).and.reference>0,'adaptive reference density/velocity product')
    reference=reference*uinf
    call check(ieee_is_finite(reference).and.reference>0,'adaptive reference density/velocity square')
    value=0.5d0*(global/(real(ia,real64)*real(ja,real64)*real(ka,real64)))/reference
    call check(ieee_is_finite(value),'nonfinite normalized adaptive kinetic energy')
    call report_insitu_timing('adaptive_monitor',phase_started,step)
    phase_started=insitu_clock()
    call feed_adaptive_signal(value,t,ok)
    call check(ok,'adaptive finite difference, scale or event hold overflow')
    call report_insitu_timing('adaptive_event_update',phase_started,step)
    if(mpirank==0) write(*,'(a,i0,a,es24.16,a,es24.16,a,8(es24.16,1x),a,8(l1,1x),a,8(i0,1x),a,es16.8)') &
      'ASTR_AP_MONITOR step=',step,' time=',t,' kinetic_energy=',value,' rates=',adaptive_shared_state%r, &
      ' active=',adaptive_shared_state%active,' windows=',adaptive_shared_state%window_phase, &
      ' wall_seconds=',MPI_Wtime()-start
  end subroutine

  subroutine completed_output_runtime(dt_used,counter,pending,initial)
    real(real64),intent(in) :: dt_used
    integer,intent(inout) :: counter
    logical,intent(inout) :: pending
    logical,optional,intent(in) :: initial
    logical :: due,ok,scheduled
    integer(int64) :: crossed
    character(64) :: name
    character(1200) :: path
    character(128),allocatable :: resources(:)
    character(128) :: files(9)
    integer :: file_count,resource_count,i,err
    type(checkpoint_state_identity) :: identity
    if (.not.enabled) return
    call check_capability()
    call check(lavg.eqv.(compact_statistics.or.mean_statistics),'legacy statistics activation changed during the run')
    identity=checkpoint_state_identity(int(nstep,int64),time,dt_used,deltat)
    call observe_output_adaptive(nstep,time)
    call complete_output_insitu(nstep,time,nstep>maxstep)
    if(archives_enabled()) call observe_archives(identity,nstep>maxstep)
    if (.not.options%checkpoint%enabled) return
    due=nstep>maxstep
    if (present(initial)) due=due.or.(initial.and.options%checkpoint%initial_frame.and.last_step==-1)
    call poll_schedule(checkpoint_schedule,int(nstep,int64),time,.false.,scheduled,crossed,ok)
    call check(ok,'invalid checkpoint schedule clock or unrepresentable time targets')
    due=due.or.scheduled
    if (.not.due.or.last_step==int(nstep,int64)) return
    write(name,'("step",i12.12)') nstep
    path=trim(options%directory)//'/checkpoints/'//trim(name)//'.tmp'
    call checkpoint_state_context(trim(options%directory)//'/checkpoints/'//trim(name),last_runtime_checkpoint())
    call create_batch(trim(path),MPI_COMM_WORLD,ok)
    call check(ok,'cannot create checkpoint candidate')
    identity=checkpoint_state_identity(int(nstep,int64),time,dt_used,deltat)
    call flow_file(trim(path)//'/state.h5',.true.,identity)
    if(dynamic_output_case()) call inflow_file(trim(path)//'/inflow.h5',.true.,identity)
    if(options%checkpoint_statistics) then
    if(statistics_active) call output_statistics_file(trim(path)//'/statistics.h5',.true., &
      identity,state_host_budget())
    if(mean_statistics) call mean_statistics_file(trim(path)//'/statistics.h5',.true., &
      identity,state_host_budget())
#ifdef _CUDA
    if(compact_statistics) call complete_compact_statistics_file(trim(path)//'/statistics.h5',.true., &
      identity,state_host_budget(),allow_repartition=curve_repartition_case())
#endif
    endif
    last_step=nstep
    call control_file(trim(path)//'/control.bin',.true.,counter,pending,identity)
    call archive_control_file(trim(path)//'/archives.bin',.true.,identity)
    call output_render_file(trim(path)//'/insitu_control.bin',.true.,identity,state_host_budget(),.false.)
    resource_count=2
    if(trim(flowtype)=='tgv'.and.lreadgrid) resource_count=3
    if(trim(flowtype)=='bl') resource_count=4
    if(dynamic_output_case()) resource_count=5+inflow_count
    if(air5_output_case()) resource_count=2+size(air5_checkpoint_resources())
    if(ninit>=1.and.ninit<=3) resource_count=resource_count+1
    if(wall_blowing_legacy_random) resource_count=resource_count+1
    allocate(resources(resource_count),stat=err)
    call check(err==0,'checkpoint resource-name allocation')
    resources(1:2)=[character(128) :: 'geometry.h5','input.txt']
    if(trim(flowtype)=='tgv'.and.lreadgrid) resources(3)='grid.h5'
    if(trim(flowtype)=='bl') then
      resources(3)=profile_grid_resource_name()
      resources(4)='inlet.prof'
    endif
    if(dynamic_output_case()) then
      resources(5)='inflow_index.bin'
      do i=1,inflow_count
        resources(5+i)=inflow_source_name(i-1)
      enddo
    endif
    if(air5_output_case()) then
#ifdef ASTR_AIR5_CHEMISTRY
      call air5_contract_file(trim(path)//'/air5_config.bin',.true.)
#ifdef _CUDA
      if(use_gpu) call conservation_file(trim(path)//'/air5_conservation.bin',.true.,identity)
#endif
#endif
      resources(3:resource_count)=air5_checkpoint_resources()
    endif
    i=resource_count
    if(wall_blowing_legacy_random) then
      resources(i)='wallbs.dat'
      i=i-1
    endif
    if(ninit>=1.and.ninit<=3) resources(i)=initial_source_name(ninit)
    if(.not.options%exact_restart) resources=pack(resources,resources/='geometry.h5')
    resource_count=size(resources)
    call write_checkpoint_resource_refs(trim(path),resources(:resource_count),MPI_COMM_WORLD,ok)
    call check(ok,'cannot record checkpoint resources')
    files=''
    files(1:4)=[character(128) :: 'state.h5','control.bin','RESOURCES','archives.bin']
    file_count=4
    file_count=file_count+1
    files(file_count)='insitu_control.bin'
    if(options%checkpoint_statistics.and.(statistics_active.or.compact_statistics.or.mean_statistics)) then
      file_count=file_count+1
      files(file_count)='statistics.h5'
    endif
    if(air5_output_case()) then
      file_count=file_count+1
      files(file_count)='air5_config.bin'
#if defined(_CUDA) && defined(ASTR_AIR5_CHEMISTRY)
      if(use_gpu) then
        file_count=file_count+1
        files(file_count)='air5_conservation.bin'
      endif
#endif
    endif
    if(dynamic_output_case()) then
      file_count=file_count+1
      files(file_count)='inflow.h5'
    endif
    call seal_checkpoint_bundle(trim(path),files(:file_count),MPI_COMM_WORLD,ok)
    call check(ok,'cannot seal checkpoint')
    call publish_retained_checkpoint(trim(options%directory)//'/checkpoints',trim(name),options%keep, &
      ledger,MPI_COMM_WORLD,ok)
    call checkpoint_state_context(trim(options%directory)//'/checkpoints/'//trim(name),last_runtime_checkpoint())
    call check(ok,'cannot publish/retain checkpoint')
    call checkpoint_state_context('','')
    if (mpirank==0) write(*,'(a,i0,a,es24.16)') 'ASTR_OUTPUT complete_step=',nstep,' time=',time
  end subroutine
end module
