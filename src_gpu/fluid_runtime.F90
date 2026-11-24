module fluid_runtime
  use iso_c_binding, only: c_ptr,c_null_ptr,c_int,c_double,c_size_t,c_associated,c_sizeof,c_int8_t,c_loc
  use iso_fortran_env, only: int64
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use astr_precision, only: real_kind,clock_kind,real_bytes
  use astr_precision_mpi, only: mpi_real_type
  use mpi
  use commvar, only: im,jm,km,hm,ia,ja,ka,numq,num_species,num_modequ,use_gpu,flowtype,ndims, &
    lihomo,ljhomo,lkhomo,lreadgrid,nondimen,lcomb,lfilter,diffterm,conschm,difschm,rkscheme,turbmode, &
    const2,const5,const6,mach,gamma,rgas,reynolds,prandtl,tempconst,tempconst1,nstep,deltat,lcracon,lavg, &
    roinf,uinf,xmax,rkstep,is,ie,js,je,ks,ke,force,limmbou,recon_schem,lchardecomp,bfacmpld,shkcrt
  use commarray, only: q,rho,vel,prs,tmp,jacob,dxi,x,crinod,nodestat
  use parallel, only: mpirank,lio,ig0,jg0,kg0,isize,jsize,ksize, &
    mpileft,mpiright,mpidown,mpiup,mpiback,mpifront
  use device_state
  use halo_transport
  use fluid_explicit
  use upwind_operator_cpp
  use mp_positivity_operator_cpp
  use shock_sensor_cpp
  use fluid_step, only: fluid_prepare_stage,fluid_advance_prepared_stage
  use compact_flow_cpp
  use numerical_forms, only: convection_form,viscous_form
  use boundary_state_cpp
  use sponge_operator_cpp
  use inflow_timeseries_cpp
  use crashfix_operator_cpp
  use crashfix_runtime_cpp, only: repair_crashfix_cpp,repair_reconciled_cpp
  use keep_sbp63_runtime, only: keep_sbp63_enabled,keep_sbp63_wall_temperature, &
    keep_w1_enabled,keep_w1_inverse,keep_w1_configure_grid,keep_w1_end_widths
  use me6_flow_cpp, only: prepare_me6_cpp,me6_diffusion_cpp,report_me6_cpp
#ifdef ASTR_AIR5_CHEMISTRY
  use chemistry_runtime_cpp, only: initialize_air5_cpp,advance_air5_cpp,check_air5_cpp_state,transfer_air5_aux_cpp
  use chemistry_mean_statistics_cpp, only: release_air5_mean_statistics_cpp
  use chemistry_conservation_cpp, only: close_air5_conservation_file_cpp
#endif
  implicit none
  private
  public :: cpp_after_flowinit,cpp_before_finalize,cpp_time_integration,cpp_check_state
  public :: cpp_export_host_state,cpp_import_host_state
  public :: cpp_export_host_primitives
  public :: cpp_device_context
  public :: cpp_wall_forcing_active
  public :: cpp_refresh_mean_fields
  public :: cpp_prepare_preview_state
  public :: collect_timestep_rates_cpp,collect_cfl_cpp
  public :: cpp_crashfix_supported
  public :: cpp_adaptive_energy_sum
  type(c_ptr),save :: context=c_null_ptr
  logical,save :: host_primitives_current=.false.
  integer(c_int),save :: physical_low(3)=0,physical_high(3)=0,axis_types(3)=3,body_first(3)=0,body_last(3)=0
  logical,save :: physical_boundaries=.false.
  logical,save :: upwind_convection=.false.,upwind_sensor=.false.
  logical,save :: mp_positivity=.false.
  logical,save :: profile_mach_pressure=.false.
  logical,save :: x_characteristic_boundaries=.false.
  logical,save :: profile_characteristic_outlet=.false.
  logical,save :: profile_staged_outlet=.false.
  logical,save :: top_characteristic_boundary=.false.,top_self_filter=.false.,top_viscous_source=.false.
  logical,save :: top_shock=.false.,top_keep_corner=.false.
  logical,save :: top_incident=.false.
  logical,save :: high_open=.false.
  integer(c_int),save :: top_policy=0
  real(real_kind),save :: top_target(5)=0.e0_real_kind,top_mach_squared=0.e0_real_kind
  logical,save :: evolved_wall=.false.
  integer(c_int),save :: evolved_wall_axis=-1
  integer(c_int),save :: evolved_wall_kind=41
  logical,save :: sponge_enabled(2)=.false.
  logical,save :: wall_forcing_enabled=.false.
  logical,save :: crashfix_ready=.false.
  real(real_kind),save :: crashfix_pressure_floor=0.e0_real_kind
  logical,save :: mixed_viscous_flux=.false.,mixed_interface_flux=.false.,mixed_characteristic_flux=.false.,mixed_derivative=.false.
  logical,save :: packed_viscous_halo=.false.,halo_axis_timing=.false.
  logical,save :: device_memory_steps=.false.
  integer(c_int),save :: viscous_overlap_axis=0_c_int
  real(real_kind),allocatable,save :: lower_wall_velocity(:,:)
  interface
    integer(c_int) function w1_prepare_cpp(handle) bind(C,name='astr_keep_w1_prepare')
      import c_int,c_ptr
      type(c_ptr),value :: handle
    end function
    integer(c_int) function w1_diffusion_cpp(handle,first_y,total_y,split,inverse,wall,re,pr,c5,tc,tc1,send_width,recv_width) &
        bind(C,name='astr_keep_w1_diffusion')
      import c_int,c_ptr,c_double
      type(c_ptr),value :: handle
      integer(c_int),value :: first_y,total_y
      integer(c_int),intent(in) :: split(3)
      real(c_double),intent(in) :: inverse(3),wall(2)
      real(c_double),value :: re,pr,c5,tc,tc1
      integer(c_int),intent(in) :: send_width(2),recv_width(2)
    end function
  end interface
#ifdef ASTR_BUILD_TESTING
  integer(c_int),save :: exchange_sequence=0
#endif
contains
  subroutine cpp_prepare_preview_state()
    integer :: node,local_rank,local_size,ierr,status,sync_mode
    integer(c_int) :: visible
    character(32) :: sync
    call require(.not.c_associated(context),'preview device state is already initialized')
    call require(real_bytes==8,'C++ read-only preview currently supplies the existing FP64 path')
    call MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,node,ierr)
    call check(ierr,'preview node communicator')
    call MPI_Comm_rank(node,local_rank,ierr);call check(ierr,'preview local rank')
    call MPI_Comm_size(node,local_size,ierr);call check(ierr,'preview local size')
    call check(device_count(visible),'preview visible devices')
    call require(local_size<=visible,'preview needs one visible device per local rank')
    call check(device_bind(int(local_rank,c_int)),'preview device binding')
    call MPI_Comm_free(node,ierr);call check(ierr,'release preview node communicator')
    sync=''
    call get_environment_variable('ASTR_GPU_SYNC_MODE',sync,status=status)
    if(status==1) sync='dependency'
    sync_mode=-1
    if(sync=='dependency') sync_mode=0
    if(sync=='explicit') sync_mode=1
    call require((status==0.or.status==1).and.sync_mode>=0,'preview sync mode must be dependency or explicit')
    call check(device_state_create(int(local_rank,c_int),int(im,c_int),int(jm,c_int),int(km,c_int), &
      int(hm,c_int),int(sync_mode,c_int),context),'create read-only preview state')
    call check(fluid_prepare(context,0_c_int),'prepare preview fields without solver gradients')
    ! Restore only saved fields. Do not initialize boundary targets, RNG, drivers,
    ! repairs or time stepping as ordinary continuation does.
    call check(device_state_upload(context,device_q,q,bytes(size(q,kind=c_size_t))),'preview q upload')
    call check(device_state_upload(context,device_rho,rho,bytes(size(rho,kind=c_size_t))),'preview density upload')
    call check(device_state_upload(context,device_velocity,vel,bytes(size(vel,kind=c_size_t))),'preview velocity upload')
    call check(device_state_upload(context,device_pressure,prs,bytes(size(prs,kind=c_size_t))),'preview pressure upload')
    call check(device_state_upload(context,device_temperature,tmp,bytes(size(tmp,kind=c_size_t))),'preview temperature upload')
    call check(device_state_upload(context,device_metric,dxi,bytes(size(dxi,kind=c_size_t))),'preview metric upload')
    ! Saved x/Jacobian stay on host; instantaneous TGV tracing consumes no
    ! solver flux/Jacobian or derivative work arrays.
    host_primitives_current=.true.
    if(lio) write(*,'(a)') 'ASTR_GPU_CPP preview_state=read-only; solver/RNG initialization skipped'
  end subroutine
  logical function cpp_wall_forcing_active() result(active)
    active=wall_forcing_enabled
  end function
  subroutine cpp_adaptive_energy_sum(local,ok)
    real(real_kind),intent(out) :: local
    logical,intent(out) :: ok
    real(real_kind) :: values(2)
    call check(fluid_adaptive_energy_sum(context,values),'adaptive energy reduction')
    local=values(1)
    ok=values(2)==0.and.ieee_is_finite(local).and.local>=0
  end subroutine
  function cpp_device_context() result(handle)
    type(c_ptr) :: handle
    handle=context
  end function
  subroutine require(condition,message)
    logical,intent(in) :: condition
    character(*),intent(in) :: message
    integer :: ignored
    if(condition) return
    write(*,'(a,i0,2a)') 'ASTR_GPU_CPP rank=',mpirank,': ',message
    call MPI_Abort(MPI_COMM_WORLD,1,ignored)
  end subroutine
  subroutine check(status,label)
    integer,intent(in) :: status
    character(*),intent(in) :: label
    if(status/=0) write(*,'(a,i0,a,i0)') 'ASTR_GPU_CPP rank=',mpirank,' status=',status
    call require(status==0,label)
  end subroutine
  subroutine cpp_after_flowinit()
    use bc, only: bctype,configure_wall_blowing,wall_blowing_amplitude,wall_blowing_legacy_random,turbinf,configure_profile_inflow, &
      profile_inflow_mach_pressure_enabled,validate_high_order_wall_geometry,profile_nonreflecting_corner_enabled, &
      profile_low_order_nscbc_enabled,profile_nscbc_outlet_enabled,explicit_profile_nscbc_corrections_enabled
    use boundary_high_order, only: high_order_boundary_enabled,high_order_wall_enabled,high_order_wall_axis
#ifdef ASTR_BUILD_TESTING
    use keep_sbp63_runtime, only: keep_sbp63_mms_enabled
    use boundary_high_order, only: h1_internal_candidate,h1_near_candidate,h1_blend_candidate,h1_stage_budget_candidate, &
      h1_outlet_filter_candidate,h1_inlet_limited_candidate,h1_inlet_compatible_candidate, &
      h1_compression_sensor_candidate,h1_metric_eps_candidate
#endif
    use keep_geometry, only: validate_keep_wall_geometry
    use sponge_layer, only: spg_i0,spg_im,spg_j0,spg_jm,spg_k0,spg_km,lspg_im,lspg_jm
    use commvar, only: spg_def
    use numerical_forms, only: effective_filter_scheme,compact_filter_selected
    use case_model_options, only: get_case_option
    use conservative_boundary_runtime, only: conservative_boundary
    use benchmark_runtime, only: benchmark_gpu_rk_timing_enabled
    integer :: local_rank,ierr,choice,low,high,env_status,filter_components,workspace_choice
    integer :: halo_choice(2),halo_min(2),halo_max(2)
    integer(c_size_t) :: sigma_bytes,heat_bytes
    integer(c_int) :: transport_mode,halo_timing,halo_components
    logical :: nodal_fluxes,characteristic_periodic,characteristic_plate,characteristic_xphysical,derivative_periodic,derivative_plate
    type(device_view) :: filter_view
    character(32) :: sync,value,precision,candidate
    call get_environment_variable('ASTR_CPP_VISCOUS_HALO',value,status=env_status)
    if(env_status==1) value='separate'
    halo_choice=-1
    if(env_status==0.or.env_status==1) then
      if(value=='separate') halo_choice(1)=0
      if(value=='packed') halo_choice(1)=1
    endif
    call get_environment_variable('ASTR_CPP_VISCOUS_OVERLAP_AXIS',value,status=env_status)
    if(env_status==1) value='x'
    if(env_status==0.or.env_status==1) then
      if(value=='x') halo_choice(2)=0
      if(value=='z') halo_choice(2)=2
    endif
    call MPI_Allreduce(halo_choice,halo_min,2,MPI_INTEGER,MPI_MIN,MPI_COMM_WORLD,ierr)
    call check(ierr,'viscous halo options minimum')
    call MPI_Allreduce(halo_choice,halo_max,2,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
    call check(ierr,'viscous halo options maximum')
    call require(halo_min(1)>=0.and.halo_min(1)==halo_max(1), &
      'ASTR_CPP_VISCOUS_HALO requires consistent separate or packed')
    call require(halo_min(2)>=0.and.halo_min(2)==halo_max(2), &
      'ASTR_CPP_VISCOUS_OVERLAP_AXIS requires consistent x or z')
    packed_viscous_halo=halo_choice(1)==1
    viscous_overlap_axis=int(halo_choice(2),c_int)
    call require(viscous_overlap_axis==0.or.packed_viscous_halo,'z overlap requires ASTR_CPP_VISCOUS_HALO=packed')
    call get_environment_variable('ASTR_CPP_HALO_AXIS_TIMING',value,status=env_status)
    if(env_status==1) value='0'
    call require((env_status==0.or.env_status==1).and.(value=='0'.or.value=='1'), &
      'ASTR_CPP_HALO_AXIS_TIMING requires 0 or 1')
    halo_axis_timing=value=='1'
    call require(.not.halo_axis_timing.or.benchmark_gpu_rk_timing_enabled(), &
      'ASTR_CPP_HALO_AXIS_TIMING requires ASTR_GPU_RK_TIMING=1')
    call get_environment_variable('ASTR_CPP_DEVICE_MEMORY_STEPS',value,status=env_status)
    if(env_status==1) value='0'
    call require((env_status==0.or.env_status==1).and.(value=='0'.or.value=='1'), &
      'ASTR_CPP_DEVICE_MEMORY_STEPS requires 0 or 1')
    device_memory_steps=value=='1'
#ifdef ASTR_AIR5_CHEMISTRY
    if(lcomb)then
      call require(.not.packed_viscous_halo.and..not.halo_axis_timing,'use the AIR5 private halo controls for reacting flow')
      call create_solver_context(transport_mode,local_rank,sync)
      call initialize_air5_cpp(context,transport_mode)
      if(device_memory_steps) call report_device_memory('initialized')
      write(*,'(a,i0,2a,a,i0,2a)') 'ASTR_GPU_CPP rank=',mpirank,' backend=',ASTR_GPU_CPP_NAME, &
        ' device=',local_rank,' sync=',trim(sync)
      return
    endif
#endif
    call require((trim(flowtype)=='tgv'.or.trim(flowtype)=='hit'.or.trim(flowtype)=='2dvort'.or. &
      trim(flowtype)=='sod'.or.trim(flowtype)=='channel'.or.trim(flowtype)=='bl'.or. &
      trim(flowtype)=='openshock'.or.trim(flowtype)=='shuosher'.or.trim(flowtype)=='rti'.or.trim(flowtype)=='ldcavity').and.ndims==3.and.nondimen.and. &
      numq==5.and.num_species==0.and.num_modequ==0.and..not.lcomb.and.trim(turbmode)=='none', &
      'C++ flow requires nonreacting 3D TGV/HIT, vortex, Sod, channel, profile plate, OpenShock, Shu-Osher, RTI or LDC')
    if(trim(flowtype)=='bl') call require(turbinf=='prof'.or.turbinf=='intp', &
      'C++ plate inlet supports profile or existing time-series interpolation')
    x_characteristic_boundaries=all(bctype==[12,22,1,1,1,1])
    high_open=high_order_boundary_enabled
    profile_characteristic_outlet=profile_nonreflecting_corner_enabled()
    profile_characteristic_outlet=profile_characteristic_outlet.or. &
      (profile_nscbc_outlet_enabled().and.difschm=='643e'.and..not.explicit_profile_nscbc_corrections_enabled())
    profile_staged_outlet=.not.profile_low_order_nscbc_enabled().and. &
      (explicit_profile_nscbc_corrections_enabled().or.(profile_nscbc_outlet_enabled().and.difschm=='643c'))
    profile_staged_outlet=profile_staged_outlet.or.high_open
    if(high_open.and.diffterm) call require(viscous_form=='standard', &
      'C++ H1 connects nodal viscosity; H1 ME6 wall-band assembly is not connected')
    if(x_characteristic_boundaries) call require(conschm=='543e'.and.difschm=='643e'.and. &
      convection_form=='standard'.and.recon_schem==3.and..not.lchardecomp.and..not.lfilter.and..not.diffterm, &
      'C++ x-NSCBC stages currently connect inviscid MP7; other boundary/operator combinations remain unconnected')
#ifdef ASTR_BUILD_TESTING
    if(high_open) call require(.not.any([h1_internal_candidate,h1_near_candidate,h1_blend_candidate,h1_stage_budget_candidate, &
      h1_outlet_filter_candidate,h1_inlet_limited_candidate,h1_inlet_compatible_candidate,h1_compression_sensor_candidate, &
      h1_metric_eps_candidate]),'explicit H1 test candidates are not connected to the C++ production operator')
    if(keep_sbp63_enabled) call require(.not.keep_sbp63_mms_enabled, &
      'the SBP63 test-only steady manufactured source is not connected to the C++ main')
#endif
    if(high_open.and.lio) write(*,'(a)') 'ASTR_GPU_CPP high_open=production-h1'
    if(keep_sbp63_enabled.and.lio) write(*,'(a)') 'ASTR_GPU_CPP sbp63_wall=on'
    evolved_wall=high_order_wall_enabled
    if(evolved_wall) then
      call require(viscous_form=='standard', &
        'high-wall ME6 assembly is not connected to the C++ main')
      evolved_wall_axis=int(high_order_wall_axis-1,c_int)
      evolved_wall_kind=int(bctype(2*high_order_wall_axis-1),c_int)
      call validate_high_order_wall_geometry()
      if(lio) write(*,'(a,i0,a,i0)') 'ASTR_GPU_CPP high_wall_axis=',high_order_wall_axis,' kind=',evolved_wall_kind
    endif
    call require(.not.limmbou,'immersed-boundary operators are not connected to the C++ main')
    call require(spg_def=='layer'.and.all([spg_i0,spg_j0,spg_k0,spg_km]==0), &
      'C++ sponge currently implements the existing x+ and y+ layer operators')
    sponge_enabled=[lspg_im,lspg_jm]
    if(any(bctype(3)==[41,42,411,421])) then
      call configure_wall_blowing()
      wall_forcing_enabled=abs(wall_blowing_amplitude)>1.e-10_real_kind
      if(wall_blowing_legacy_random.and.wall_blowing_amplitude<=1.e-10_real_kind) wall_forcing_enabled=.false.
      if(wall_forcing_enabled) call require(bctype(3)==41.and..not.evolved_wall.and..not.keep_sbp63_enabled, &
        'C++ active blowing requires the prescribed lower bc41 state')
    endif
    physical_low=int(merge(1,0,bctype([1,3,5])/=1.and.[ig0,jg0,kg0]==0),c_int)
    physical_high=int(merge(1,0,bctype([2,4,6])/=1.and.[ig0+im,jg0+jm,kg0+km]==[ia,ja,ka]),c_int)
    physical_boundaries=.not.all(bctype==1)
    nodal_fluxes=diffterm.and.(viscous_form=='standard'.or.physical_boundaries).and..not.keep_w1_enabled
    axis_types=3
    where(physical_low/=0) axis_types=1
    where(physical_high/=0) axis_types=2
    where(physical_low/=0.and.physical_high/=0) axis_types=4
    body_first=int([is,js,ks],c_int);body_last=int([ie,je,ke],c_int)
    if(convection_form=='keep') call validate_keep_wall_geometry()
    upwind_convection=conschm=='543e'.or.conschm=='743e'
    upwind_sensor=upwind_convection.and.(lchardecomp.or.recon_schem==5)
    call require((conschm=='643e'.or.conschm=='643c'.or.upwind_convection).and.(difschm=='643e'.or.difschm=='643c').and.rkscheme=='rk3'.and. &
      (convection_form=='standard'.or.convection_form=='keep').and. &
      (viscous_form=='standard'.or.viscous_form=='me6_opti').and. &
      (effective_filter_scheme(conschm,difschm,.true.)=='explicit10'.or.compact_filter_selected(conschm,difschm)), &
      'C++ flow requires central/KEEP 643e/643c or MP7 543e/743e, nodal/ME6 viscosity, explicit10/compact10 and RK3')
    if(upwind_convection) then
      call require(convection_form=='standard'.and.(recon_schem==3.or.recon_schem==5), &
        'C++ upwind implements standard MP7 or MP7-LD reconstruction')
    endif
    call get_case_option('gpu_precision_mode',precision,status=env_status)
    if(env_status==1) precision='fp64'
    call require((env_status==0.or.env_status==1).and.(precision=='fp64'.or.precision=='mixed_workspace'), &
      'C++ precision mode must be fp64 or an implemented mixed workspace')
    call get_case_option('gpu_mixed_candidate',candidate,status=env_status)
    if(env_status==1) candidate='flux'
    call require(env_status==0.or.env_status==1,'mixed workspace candidate is truncated')
    workspace_choice=0
    if(precision=='mixed_workspace') then
      select case(candidate)
      case('derivative')
        derivative_periodic=trim(flowtype)=='tgv'.and.lihomo.and.ljhomo.and.lkhomo.and.conschm=='643e'
        derivative_plate=trim(flowtype)=='bl'.and.turbinf=='prof'.and.conschm=='543e'.and.recon_schem==3.and. &
          .not.lihomo.and..not.ljhomo.and.lkhomo.and.all(bctype==[11,21,41,51,1,1]).and..not.any(sponge_enabled)
        call require(diffterm.and..not.lchardecomp.and.difschm=='643e'.and. &
          convection_form=='standard'.and.viscous_form=='standard'.and. &
          .not.compact_filter_selected(conschm,difschm).and.(derivative_periodic.or.derivative_plate), &
          'C++ mixed diagnostic gradient connects the existing explicit viscous periodic TGV or MP7 profile plate')
        workspace_choice=4
      case('viscous_flux')
        call require(diffterm.and.difschm=='643e'.and.viscous_form=='standard'.and..not.keep_sbp63_enabled, &
          'C++ mixed viscous flux requires the connected standard explicit viscous pipeline')
        workspace_choice=1
      case('flux')
        call require(conschm=='543e'.and.recon_schem==3.and..not.lchardecomp.and. &
          lihomo.and.ljhomo.and.lkhomo,'C++ mixed flux connects existing periodic physical-space MP7 only')
        workspace_choice=2
      case('characteristic_flux')
        characteristic_periodic=lihomo.and.ljhomo.and.lkhomo.and..not.diffterm
        characteristic_plate=trim(flowtype)=='bl'.and.trim(turbinf)=='prof'.and.diffterm.and.difschm=='643e'.and. &
          .not.lihomo.and..not.ljhomo.and.lkhomo.and.all(bctype==[11,21,41,51,1,1]).and..not.any(sponge_enabled)
        characteristic_xphysical=flowtype=='shuosher'.and..not.diffterm.and.difschm=='643e'.and. &
          .not.lihomo.and.ljhomo.and.lkhomo.and.all(bctype==[50,50,1,1,1,1]).and. &
          jsize==1.and.ksize==1.and.(isize==1.or.isize==2)
        call require(conschm=='543e'.and.recon_schem==3.and.lchardecomp.and..not.lfilter.and. &
          (characteristic_periodic.or.characteristic_plate.or.characteristic_xphysical), &
          'C++ MP3 connects periodic/xphysical Euler or existing viscous profile-plate MP7')
        workspace_choice=3
      case default
        call require(.false.,'C++ mixed candidate is not implemented')
      end select
    else
      call require(candidate=='flux','a non-default mixed candidate requires mixed_workspace')
    endif
    call MPI_Allreduce(workspace_choice,low,1,MPI_INTEGER,MPI_MIN,MPI_COMM_WORLD,ierr)
    call check(ierr,'mixed workspace minimum')
    call MPI_Allreduce(workspace_choice,high,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
    call check(ierr,'mixed workspace maximum')
    call require(low==high,'mixed workspace mode differs across ranks')
    mixed_viscous_flux=workspace_choice==1
    mixed_interface_flux=workspace_choice==2
    mixed_characteristic_flux=workspace_choice==3
    mixed_derivative=workspace_choice==4
    call get_case_option('gpu_mp_positivity',value,status=env_status)
    choice=-1
    if(env_status==1.or.(env_status==0.and.(value==''.or.value=='off'))) choice=0
    if(env_status==0.and.value=='flux') choice=1
    call MPI_Allreduce(choice,low,1,MPI_INTEGER,MPI_MIN,MPI_COMM_WORLD,ierr)
    call check(ierr,'MP positivity minimum')
    call MPI_Allreduce(choice,high,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
    call check(ierr,'MP positivity maximum')
    call require(low>=0.and.low==high,'MP positivity requires consistent off or flux')
    mp_positivity=choice==1
    if(mp_positivity) call require(conschm=='743e'.and.recon_schem==5.and.lchardecomp.and. &
      .not.lfilter.and.lkhomo.and.workspace_choice==0.and..not.conservative_boundary%enabled.and..not.limmbou, &
      'MP positivity requires five-variable MP7-LD RK3, periodic z, no filter or mixed workspace')
    if(lcracon) call require(cpp_crashfix_supported(), &
      'C++ crashfix needs five-variable central643e/KEEP643c/MP7-LD743e; mixed supports FP64 viscous_flux only')
    filter_components=0
    if(lfilter) then
      value=''
      call get_environment_variable('ASTR_GPU_FILTER_WORKSPACE',value,status=env_status)
      if(env_status==1) value='scalar'
      call require((env_status==0.or.env_status==1).and.(value=='full'.or.value=='scalar'), &
        'ASTR_GPU_FILTER_WORKSPACE must be scalar or full')
      filter_components=merge(5,1,value=='full')
      call MPI_Allreduce(filter_components,low,1,MPI_INTEGER,MPI_MIN,MPI_COMM_WORLD,ierr)
      call check(ierr,'filter storage minimum')
      call MPI_Allreduce(filter_components,high,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
      call check(ierr,'filter storage maximum')
      call require(low==high,'filter storage differs across ranks')
      if(lio) write(*,'(2a)') 'ASTR_GPU_CPP filter_workspace=',trim(value)
    endif
    call create_solver_context(transport_mode,local_rank,sync)
    if(packed_viscous_halo) call require(transport_mode==halo_device.and..not.physical_boundaries.and. &
      all([lihomo,ljhomo,lkhomo]).and.diffterm.and.conschm=='643e'.and.difschm=='643e'.and. &
      convection_form=='standard'.and.viscous_form=='standard'.and..not.mixed_viscous_flux, &
      'packed viscous halo requires periodic device MPI with standard 643e/643e and native-precision fluxes')
    if(lio) write(*,'(a,a)') 'ASTR_GPU_CPP viscous_halo=',merge('packed  ','separate',packed_viscous_halo)
    if(packed_viscous_halo.and.lio) write(*,'(a,a)') &
      'ASTR_GPU_CPP viscous_overlap_axis=',merge('z','x',viscous_overlap_axis==2)
    ! TGV scalar enstrophy/dissipation consume gradients even for inviscid runs.
    call check(fluid_prepare(context,1_c_int),'prepare primitive and gradient fields')
    if(mixed_derivative) then
      call check(fluid_prepare_diagnostic_gradient(context),'prepare separate FP32 diagnostic gradients')
      call check(device_state_view(context,device_diagnostic_dvelocity,filter_view),'diagnostic velocity-gradient view')
      sigma_bytes=product(int(filter_view%extent,c_size_t))*filter_view%element_bytes
      call check(device_state_view(context,device_diagnostic_dtemperature,filter_view),'diagnostic thermal-gradient view')
      heat_bytes=product(int(filter_view%extent,c_size_t))*filter_view%element_bytes
      if(lio) write(*,'(a,i0,a,i0,a)') 'ASTR_GPU_CPP mixed_workspace=derivative dvel_bytes=',sigma_bytes, &
        ' dtmp_bytes=',heat_bytes,' solver_gradient_element_bytes=8'
    endif
    if(mixed_viscous_flux) then
      call check(fluid_rhs_prepare_mixed_viscous(context),'prepare FP32 stored viscous flux')
      call check(device_state_view(context,device_stress,filter_view),'mixed stress view')
      sigma_bytes=product(int(filter_view%extent,c_size_t))*filter_view%element_bytes
      call check(device_state_view(context,device_heat_flux,filter_view),'mixed heat-flux view')
      heat_bytes=product(int(filter_view%extent,c_size_t))*filter_view%element_bytes
      if(lio) write(*,'(a,i0,a,i0,a)') 'ASTR_GPU_CPP mixed_workspace=viscous_flux sigma_bytes=',sigma_bytes, &
        ' heat_flux_bytes=',heat_bytes,' transport_element_bytes=8'
    else
      call check(fluid_rhs_prepare(context,int(merge(1,0,nodal_fluxes),c_int)),'prepare RHS fields')
    endif
    if(keep_w1_enabled) then
      call keep_w1_configure_grid()
      call check(w1_prepare_cpp(context),'prepare W1 coefficient/flux workspace')
    endif
    if(upwind_convection) then
      if(mixed_interface_flux.or.mixed_characteristic_flux) then
        if(mixed_characteristic_flux) then
          call check(upwind_prepare_mixed_characteristic_cpp(context),'prepare MP3 interface workspace')
        else
          call check(upwind_prepare_mixed_flux_cpp(context),'prepare MP1 interface workspace')
        endif
        call check(device_state_view(context,device_interface_flux,filter_view),'mixed interface view')
        if(lio) write(*,'(3a,i0,a)') 'ASTR_GPU_CPP mixed_workspace=',trim(candidate),' interface_bytes=', &
          product(int(filter_view%extent,c_size_t))*int(filter_view%element_bytes,c_size_t),' authority_element_bytes=8'
      else
        call check(upwind_prepare_cpp(context),'prepare upwind interface workspace')
      endif
    endif
    if(upwind_sensor) call check(shock_prepare_cpp(context),'prepare upwind sensor')
    if(lfilter.and..not.compact_filter_selected(conschm,difschm)) then
      if(filter_components==1) then
        call check(fluid_filter_prepare_scalar(context),'prepare scalar filter workspace')
      else
        call check(fluid_filter_prepare(context),'prepare full filter workspace')
      endif
      call check(device_state_view(context,device_filter_work,filter_view),'filter storage view')
      if(lio) write(*,'(a,i0)') 'ASTR_GPU_CPP filter_workspace_bytes=', &
        bytes(product(int(filter_view%extent,c_size_t)))
    endif
    halo_components=int(merge(9,merge(6,5,nodal_fluxes),packed_viscous_halo),c_int)
    if(keep_w1_enabled) halo_components=14_c_int
    halo_timing=int(merge(1,0,benchmark_gpu_rk_timing_enabled()),c_int)
    if(halo_axis_timing) halo_timing=2_c_int
    call check(device_transport_prepare(context,int(MPI_COMM_WORLD,c_int), &
      int([mpileft,mpidown,mpiback],c_int),int([mpiright,mpiup,mpifront],c_int), &
      int([isize,jsize,ksize],c_int),halo_components,transport_mode,halo_timing, &
      int(merge(1,0,[lihomo,ljhomo,lkhomo]),c_int)),'prepare halo transport')
    if(mp_positivity) then
      call check(mp_positivity_prepare_cpp(context,int(MPI_COMM_WORLD,c_int), &
        int([mpileft,mpidown,mpiback],c_int),int([mpiright,mpiup,mpifront],c_int),transport_mode), &
        'prepare MP positivity workspace/transport')
      if(lio) write(*,'(a)') 'ASTR_GPU_CPP mp_positivity=flux budget=roundoff-interior-v1'
    endif
    call check(device_state_upload(context,device_q,q,bytes(size(q,kind=c_size_t))),'initial q upload')
    call check(device_state_upload(context,device_metric,dxi,bytes(size(dxi,kind=c_size_t))),'metric upload')
    call upload_slip_mask_cpp()
    call check(device_state_upload(context,device_jacobian,jacob,bytes(size(jacob,kind=c_size_t))),'Jacobian upload')
    call check(fluid_initialize_exterior(context,physical_low,physical_high), &
      'initial unused exterior definition')
    if(trim(flowtype)=='channel') then
      call check(fluid_channel_geometry_prepare(context),'channel y-coordinate allocation')
      call check(device_state_upload(context,device_coordinate_y,x(0:im,0:jm,0:km,2), &
        bytes(int((im+1)*(jm+1)*(km+1),c_size_t))),'static channel y-coordinate upload')
    endif
    call check(fluid_primitives(context,const2,const6),'initial primitive reconstruction')
    if(wall_forcing_enabled.and.physical_low(2)/=0) then
      allocate(lower_wall_velocity(0:im,0:km))
      call check(wall_forcing_prepare(context),'prepare lower-wall forcing plane')
      write(*,'(a,i0,a,i0)') 'ASTR_GPU_CPP wall_forcing rank=',mpirank,' plane_bytes=',bytes(size(lower_wall_velocity,kind=c_size_t))
    endif
    if((x_characteristic_boundaries.or.profile_characteristic_outlet.or.profile_staged_outlet).and.physical_high(1)/=0) &
      call check(nscbc_outlet_prepare(context),'prepare characteristic outlet plane')
    if(any(bctype==11)) then
      call configure_profile_inflow()
      profile_mach_pressure=profile_inflow_mach_pressure_enabled()
      call upload_profile_target_cpp()
      call init_dynamic_inflow_cpp(context)
    endif
    if(trim(flowtype)=='bl') call upload_flat_geometry_cpp()
    call prepare_top_boundary_cpp()
    call prepare_compact_flow_cpp(context,filter_components)
    if(any(sponge_enabled)) call prepare_sponge_cpp()
    if(diffterm.and.viscous_form=='me6_opti') call prepare_me6_cpp(context,transport_mode)
    ! The steploop initializes/restores crinod after flowinit; do not import
    ! that host history before its lifecycle has begun.
    call check(fluid_check(context),'initial finite positive physical state')
    if(benchmark_gpu_rk_timing_enabled().or.device_memory_steps) call report_device_memory('initialized')
    write(*,'(a,i0,2a,a,i0,2a)') 'ASTR_GPU_CPP rank=',mpirank,' backend=',ASTR_GPU_CPP_NAME, &
      ' device=',local_rank,' sync=',trim(sync)
    write(*,'(a,i0,3(a,3(i0,1x)),a,i0,3(a,3(i0,1x)))') 'ASTR_GPU_CPP_LAYOUT rank=',mpirank, &
      ' global=',ia,ja,ka,' origin=',ig0,jg0,kg0,' upper=',im,jm,km,' halo=',hm, &
      ' topology=',isize,jsize,ksize,' lower=',mpileft,mpidown,mpiback,' higher=',mpiright,mpiup,mpifront
  end subroutine
  subroutine create_solver_context(transport_mode,local_rank,sync)
    integer(c_int),intent(out) :: transport_mode
    integer,intent(out) :: local_rank
    character(*),intent(out) :: sync
    integer :: node,local_size,ierr,choice,low,high,env_status
    integer(c_int) :: visible
    character(32) :: value
    value=''
    call get_environment_variable('ASTR_GPU_HALO_TRANSPORT',value,status=env_status)
    if(env_status==1) value='pinned'
    call require((env_status==0.or.env_status==1).and.(value=='pinned'.or.value=='device'), &
      'ASTR_GPU_HALO_TRANSPORT must be pinned or device; device requires GPU-aware MPI')
    transport_mode=merge(halo_device,halo_pinned,value=='device')
    call MPI_Allreduce(transport_mode,low,1,MPI_INTEGER,MPI_MIN,MPI_COMM_WORLD,ierr)
    call check(ierr,'transport minimum')
    call MPI_Allreduce(transport_mode,high,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
    call check(ierr,'transport maximum')
    call require(low==high,'transport mode differs across ranks')
    if(lio) write(*,'(3a)') 'ASTR_GPU_CPP transport=',trim(value), &
      '; shared reconciliation then halo-only refresh'
    call MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,node,ierr)
    call check(ierr,'node communicator')
    call MPI_Comm_rank(node,local_rank,ierr)
    call check(ierr,'local rank')
    call MPI_Comm_size(node,local_size,ierr)
    call check(ierr,'local size')
    call check(device_count(visible),'visible devices')
    call require(local_size<=visible,'one visible device per local rank is required')
    call check(device_bind(int(local_rank,c_int)),'bind device')
    call MPI_Comm_free(node,ierr)
    call check(ierr,'release node communicator')
    sync=''
    call get_environment_variable('ASTR_GPU_SYNC_MODE',sync,status=env_status)
    if(env_status==1) sync='dependency'
    choice=-1
    if(sync=='dependency') choice=0
    if(sync=='explicit') choice=1
    call require((env_status==0.or.env_status==1).and.choice>=0,'ASTR_GPU_SYNC_MODE must be dependency or explicit')
    call MPI_Allreduce(choice,low,1,MPI_INTEGER,MPI_MIN,MPI_COMM_WORLD,ierr)
    call check(ierr,'sync configuration minimum')
    call MPI_Allreduce(choice,high,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
    call check(ierr,'sync configuration maximum')
    call require(low==high,'sync mode differs across ranks')
    call check(device_state_create(int(local_rank,c_int),int(im,c_int),int(jm,c_int),int(km,c_int), &
      int(hm,c_int),int(choice,c_int),context),'create state')
  end subroutine
  subroutine prepare_top_boundary_cpp()
    use bc, only: bctype,configure_nscbc_farfield,nscbc_farfield_policy,nscbc_farfield_target_state, &
      nscbc_farfield_sbli_shock_enabled,nscbc_farfield_target_states,profile_nonreflecting_corner_enabled, &
      nscbc_farfield_viscous_source_enabled,NSCBC_FARFIELD_POLICY_NONREFLECTING, &
      nscbc_farfield_grcbc_enabled,low_order_incident_enabled,nscbc_incident_parameters
    use nscbc_incident, only: incident_reference
    use commvar, only: ymax
    real(real_kind) :: density,u,v,w,temperature,shock_x
    real(real_kind) :: pre(5),post(5),entry,beta,normal(3),hn,position(3),external(5),face_reference(5)
    real(real_kind),allocatable,target :: plane(:,:,:)
    integer :: i,k,s
    logical :: shock,crosses_front
    top_characteristic_boundary=bctype(4)==52
    if(.not.top_characteristic_boundary) return
    call configure_nscbc_farfield()
    top_policy=int(nscbc_farfield_policy,c_int)
    top_incident=nscbc_farfield_grcbc_enabled()
    call require(top_policy<=NSCBC_FARFIELD_POLICY_NONREFLECTING.or.(top_incident.and.(low_order_incident_enabled().or.high_open)), &
      'C++ GRCBC connects the existing low-order KEEP/ME6 or production H1 boundary layout')
    top_self_filter=.not.lfilter.and.top_policy/=NSCBC_FARFIELD_POLICY_NONREFLECTING.and..not.top_incident
    top_viscous_source=nscbc_farfield_viscous_source_enabled()
    top_keep_corner=profile_nonreflecting_corner_enabled()
    top_shock=nscbc_farfield_sbli_shock_enabled()
    call nscbc_farfield_target_state(density,u,v,w,temperature)
    top_target=[density,density*u,density*v,density*w,density*temperature/const2*const6+ &
      .5_real_kind*density*(u*u+v*v+w*w)]
    if(physical_high(2)/=0) then
      if(top_self_filter) call check(nscbc_top_filter_prepare(context),'prepare upper-y transverse filter plane')
      if(high_open) then
        call nscbc_incident_parameters(pre,post,entry,beta)
        allocate(plane(0:im,0:km,42))
        do k=0,km
        do i=0,im
          normal=dxi(i,jm,k,2,:)/sqrt(sum(dxi(i,jm,k,2,:)**2))
          hn=sum((x(i,jm,k,:)-x(i,jm-1,k,:))*normal)
          call incident_reference(x(i,jm,k,:),entry,ymax,beta,pre,post,face_reference)
          crosses_front=.false.
          do s=1,7
            position=x(i,jm,k,:)+real(s,real_kind)*hn*normal
            call incident_reference(position,entry,ymax,beta,pre,post,external)
            crosses_front=crosses_front.or.any(external/=face_reference)
            plane(i,k,5*s-4:5*s)=[external(1),external(2:4)/external(1), &
              (gamma-1._real_kind)*(external(5)-.5_real_kind*sum(external(2:4)**2)/external(1))]
          enddo
          plane(i,k,36:40)=external;plane(i,k,41)=hn;plane(i,k,42)=merge(1.e0_real_kind,0.e0_real_kind,crosses_front)
        enddo
        enddo
        call check(nscbc_high_open_prepare(context,c_loc(plane),bytes(size(plane,kind=c_size_t))),'static H1 exterior lines upload')
      elseif(top_incident) then
        call nscbc_incident_parameters(pre,post,entry,beta)
        allocate(plane(0:im,0:km,6))
        do k=0,km
        do i=0,im
          normal=dxi(i,jm,k,2,:)/sqrt(sum(dxi(i,jm,k,2,:)**2))
          hn=sum((x(i,jm,k,:)-x(i,jm-1,k,:))*normal)
          position=x(i,jm,k,:)+hn*normal
          call incident_reference(position,entry,ymax,beta,pre,post,external)
          plane(i,k,1:5)=external;plane(i,k,6)=hn
        enddo
        enddo
        call check(nscbc_incident_prepare(context,plane,bytes(size(plane,kind=c_size_t))),'static GRCBC exterior/spacing upload')
      elseif(top_shock) then
        allocate(plane(0:im,0:km,5))
        do k=0,km
        do i=0,im
          call nscbc_farfield_target_states(x(i,jm,k,1),density,u,v,w,temperature,shock_x,shock)
          plane(i,k,:)=[density,density*u,density*v,density*w,density*temperature/const2*const6+ &
            .5_real_kind*density*(u*u+v*v+w*w)]
        enddo
        enddo
        call check(nscbc_top_target_upload(context,plane,bytes(size(plane,kind=c_size_t))),'static upper-y shock target upload')
      endif
    elseif(high_open) then
      call check(nscbc_high_open_prepare(context,c_null_ptr,0_c_size_t),'H1 boundary status preparation')
    endif
    if(lio) write(*,'(a,i0,3(a,l1))') 'ASTR_GPU_CPP top_policy=',top_policy, &
      ' transverse_filter=',top_self_filter,' viscous_source=',top_viscous_source,' shock=',top_shock
  end subroutine
  subroutine prepare_sponge_cpp()
    use sponge_layer, only: spg_im_beg,spg_im_end,spg_jm_beg,spg_jm_end, &
      sponge_damp_coef_im,sponge_damp_coef_jm
    integer(c_int) :: begin(2),ending(2),first(3),last(3)
    integer :: axis,lo(3),hi(3),counts(2)
    real(real_kind),allocatable :: xcoef(:),ycoef(:)
    begin=-1;ending=-1
    if(sponge_enabled(1)) then
      begin(1)=spg_im_beg;ending(1)=spg_im_end
    endif
    if(sponge_enabled(2)) then
      begin(2)=spg_jm_beg;ending(2)=spg_jm_end
    endif
    first=body_first;last=body_last;counts=0
    do axis=1,2
      if(begin(axis)<0) cycle
      lo=first;hi=last;lo(axis)=max(lo(axis),begin(axis));hi(axis)=min(hi(axis),ending(axis))
      counts(axis)=product(max(0,hi-lo+1))
    enddo
    allocate(xcoef(counts(1)),ycoef(counts(2)))
    if(counts(1)>0) then
      lo=first;hi=last;lo(1)=max(lo(1),begin(1));hi(1)=min(hi(1),ending(1))
      xcoef=reshape(sponge_damp_coef_im(lo(1):hi(1),lo(2):hi(2),lo(3):hi(3)),[counts(1)])
    endif
    if(counts(2)>0) then
      lo=first;hi=last;lo(2)=max(lo(2),begin(2));hi(2)=min(hi(2),ending(2))
      ycoef=reshape(sponge_damp_coef_jm(lo(1):hi(1),lo(2):hi(2),lo(3):hi(3)),[counts(2)])
    endif
    call check(sponge_prepare_cpp(context,first,last,begin,ending,xcoef,ycoef),'prepare packed sponge layers')
    write(*,'(a,i0,a,2(l1,1x),a,i0,a,i0)') 'ASTR_GPU_CPP sponge rank=',mpirank,' global_axes=',sponge_enabled, &
      ' coefficient_bytes=',8_int64*sum(int(counts,int64)),' packed_work_bytes=',40_int64*maxval(int(counts,int64))
  end subroutine
  subroutine upload_slip_mask_cpp()
    use bc, only: bctype,xslip
    integer(c_int8_t),allocatable :: mask(:,:)
    if(bctype(3)/=411.or.physical_low(2)==0) return
    allocate(mask(0:im,0:km))
    mask=merge(1_c_int8_t,0_c_int8_t,x(0:im,0,0:km,1)<=xslip)
    call check(slip_mask_upload(context,mask,size(mask,kind=c_size_t)),'static lower-wall slip mask upload')
  end subroutine
  subroutine upload_profile_target_cpp()
    use bc, only: bctype,initialize_x_inflow_target,rho_in,vel_in,prs_in,tmp_in,rho_prof,vel_prof,tmp_prof
    use commarray, only: bnorm_i0
    integer :: k
    real(real_kind),allocatable :: target(:,:,:)
    if(bctype(1)/=11.or.physical_low(1)==0) return
    if(.not.allocated(rho_in)) call initialize_x_inflow_target()
    allocate(target(0:jm,0:km,9))
    target(:,:,1)=rho_in;target(:,:,2:4)=vel_in;target(:,:,5)=prs_in;target(:,:,6)=tmp_in
    target(:,:,7:9)=bnorm_i0
    if(dynamic_inflow_requested_cpp()) then
      do k=0,km
        target(:,k,1)=rho_prof;target(:,k,2:4)=vel_prof;target(:,k,6)=tmp_prof
      enddo
      target(:,:,5)=target(:,:,1)*target(:,:,6)/const2
    endif
    call check(profile_inflow_prepare(context),'prepare static inlet target plane')
    call check(device_state_upload(context,device_inflow_target,target,bytes(size(target,kind=c_size_t))), &
      'static inlet target upload')
  end subroutine
  subroutine upload_flat_geometry_cpp()
    use commarray, only: cell
    use commfunc, only: arquad3d
    real(real_kind),allocatable :: volumes(:,:,:),areas(:,:,:)
    integer :: i,j,k
    call check(fluid_flat_geometry_prepare(context,int(merge(1,0,jg0==0),c_int)),'flat statistics geometry')
    allocate(volumes(1:im,1:jm,1:km))
    do k=1,km
    do j=1,jm
    do i=1,im
      volumes(i,j,k)=cell(i,j,k)%vol
    enddo
    enddo
    enddo
    call check(device_state_upload(context,device_cell_volume,volumes,bytes(size(volumes,kind=c_size_t))), &
      'static cell-volume upload')
    if(jg0==0) then
      allocate(areas(1:im,1,1:km))
      do k=1,km
      do i=1,im
        areas(i,1,k)=arquad3d(x(i-1,0,k-1,:),x(i,0,k-1,:),x(i,0,k,:),x(i-1,0,k,:))
      enddo
      enddo
      call check(device_state_upload(context,device_wall_cell_area,areas,bytes(size(areas,kind=c_size_t))), &
        'static wall-area upload')
    endif
  end subroutine
  integer(c_size_t) function bytes(count)
    integer(c_size_t),intent(in) :: count
    bytes=count*real_bytes
  end function
  subroutine report_device_memory(phase)
    character(*),intent(in) :: phase
    integer(c_size_t) :: free_bytes,total_bytes
    call check(device_memory_info(context,free_bytes,total_bytes),'device memory observation')
    write(*,'(a,1x,a,3(1x,i0))') 'ASTR_CPP_DEVICE_MEMORY',trim(phase),mpirank,free_bytes,total_bytes
  end subroutine
  subroutine report_step_device_memory()
    character(32) :: phase
    if(.not.device_memory_steps) return
    write(phase,'(a,i0)') 'step',nstep
    call report_device_memory(phase)
  end subroutine
  subroutine cpp_check_state()
#ifdef ASTR_AIR5_CHEMISTRY
    if(lcomb)then
      call check_air5_cpp_state(context)
      return
    endif
#endif
    if(lcracon) then
      call prepare_crashfix_state()
      call check(repair_input_selected(context),'initial crashfix input')
    endif
    call check(fluid_check(context),'finite positive physical state')
  end subroutine
  logical function cpp_crashfix_supported()
    cpp_crashfix_supported=numq==5.and.num_species==0.and..not.lcomb.and. &
      (conschm=='643e'.or.(conschm=='743e'.and.recon_schem==5).or. &
        (convection_form=='keep'.and.conschm=='643c')).and. &
      .not.(mixed_interface_flux.or.mixed_characteristic_flux.or.mixed_derivative).and. &
      (.not.mixed_viscous_flux.or.real_bytes==8)
  end function
  subroutine prepare_crashfix_state()
    use fludyna, only: thermal
    integer(c_int8_t),allocatable :: mask(:,:,:)
    if(crashfix_ready) return
    call require(cpp_crashfix_supported(),'crashfix history is unavailable for this C++ numerical mode')
    crashfix_pressure_floor=thermal(density=1.e-5_real_kind,temperature=1.e-5_real_kind)
    allocate(mask(0:im,0:jm,0:km))
    mask=int(merge(1,0,nodestat(0:im,0:jm,0:km)<=0),c_int8_t)
    call check(crashfix_prepare_cpp(context,mask,size(mask,kind=c_size_t)),'prepare crashfix state')
    crashfix_ready=.true.
    call transfer_crashfix_history(.false.)
  end subroutine
  subroutine transfer_crashfix_history(writing)
    logical,intent(in) :: writing
    integer(c_int8_t),allocatable :: mask(:,:,:)
    allocate(mask(-hm:im+hm,-hm:jm+hm,-hm:km+hm))
    if(writing) then
      call check(crashfix_history_download_cpp(context,mask,size(mask,kind=c_size_t)),'checkpoint critical-history download')
      crinod=mask/=0
    else
      mask=int(merge(1,0,crinod),c_int8_t)
      call check(crashfix_history_upload_cpp(context,mask,size(mask,kind=c_size_t)),'checkpoint critical-history upload')
    endif
  end subroutine
  integer(c_int) function repair_filtered_selected(handle) result(status)
    type(c_ptr),intent(in) :: handle
    type(crashfix_report) :: report
    status=0
    if(.not.lcracon) return
    call prepare_crashfix_state()
    status=repair_reconciled_cpp(handle,int([ig0,jg0,kg0],c_int),int([ia,ja,ka],c_int), &
      int(merge(1,0,[lihomo,ljhomo,lkhomo]),c_int),const2,const6,crashfix_pressure_floor,MPI_COMM_WORLD,report)
  end function
  integer(c_int) function repair_input_selected(handle) result(status)
    type(c_ptr),intent(in) :: handle
    logical :: repaired
    status=repair_input_state(handle,repaired)
  end function
  integer(c_int) function repair_input_state(handle,repaired) result(status)
    type(c_ptr),intent(in) :: handle
    logical,intent(out) :: repaired
    integer(c_int) :: local_invalid
    integer :: global_invalid,ierr
    status=0
    repaired=.false.
    if(.not.lcracon) return
    call prepare_crashfix_state()
    status=crashfix_input_invalid_cpp(handle,local_invalid)
    if(status/=0) return
    call MPI_Allreduce(local_invalid,global_invalid,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
    status=ierr
    if(status==0.and.global_invalid/=0) then
      status=repair_filtered_selected(handle)
      repaired=status==0
    endif
  end function
  integer(c_int) function repair_update_selected(handle) result(status)
    type(c_ptr),intent(in) :: handle
    type(crashfix_report) :: report
    integer :: rounds,expanded,diagnostics(2),repaired,ierr
    status=0
    if(.not.lcracon) return
    call prepare_crashfix_state()
    status=repair_crashfix_cpp(handle,int([ig0,jg0,kg0],c_int),int([ia,ja,ka],c_int), &
      int(merge(1,0,[lihomo,ljhomo,lkhomo]),c_int),const2,const6,crashfix_pressure_floor,MPI_COMM_WORLD, &
      report,rounds,expanded,diagnostics)
    if(status/=0) return
    call MPI_Allreduce(report%repaired,repaired,1,MPI_INTEGER,MPI_SUM,MPI_COMM_WORLD,ierr)
    status=ierr
    if(status/=0) return
    if(repaired>0) status=repair_reconciled_cpp(handle,int([ig0,jg0,kg0],c_int),int([ia,ja,ka],c_int), &
      int(merge(1,0,[lihomo,ljhomo,lkhomo]),c_int),const2,const6,crashfix_pressure_floor,MPI_COMM_WORLD,report, &
      already_repaired=.true.)
    if(lio) write(*,'(a,3(1x,i0),5(1x,es24.16e3))') 'ASTR_CPP_CRASHFIX',nstep,rkstep,repaired,report%max_change
  end function
  subroutine cpp_export_host_state(include_caches)
    logical,intent(in) :: include_caches
    integer(int64) :: downloaded
    call require(c_associated(context),'missing device state for checkpoint publication')
    ! Preserve stored primitive caches, including their complete-step halo phase.
    call check(device_state_download(context,device_q,q,bytes(size(q,kind=c_size_t))),'checkpoint q download')
#ifdef ASTR_AIR5_CHEMISTRY
    if(lcomb)then
      ! Both AIR5 restart modes retain the full thermochemical state.
      call cpp_export_host_primitives(downloaded)
      call transfer_air5_aux_cpp(context,.true.)
      return
    endif
#endif
    if(include_caches) call cpp_export_host_primitives(downloaded)
    if(crashfix_ready) call transfer_crashfix_history(.true.)
  end subroutine
  subroutine cpp_export_host_primitives(downloaded)
#ifdef ASTR_AIR5_CHEMISTRY
    use commarray, only: tve,spc
#endif
    integer(int64),intent(out) :: downloaded
    downloaded=0
    call require(c_associated(context),'missing device state for field publication')
    if(host_primitives_current) return
    call check(device_state_download(context,device_rho,rho,bytes(size(rho,kind=c_size_t))),'checkpoint density download')
    call check(device_state_download(context,device_velocity,vel,bytes(size(vel,kind=c_size_t))),'checkpoint velocity download')
    call check(device_state_download(context,device_pressure,prs,bytes(size(prs,kind=c_size_t))),'checkpoint pressure download')
    call check(device_state_download(context,device_temperature,tmp,bytes(size(tmp,kind=c_size_t))),'checkpoint temperature download')
    downloaded=int(bytes(size(rho,kind=c_size_t))+bytes(size(vel,kind=c_size_t))+ &
      bytes(size(prs,kind=c_size_t))+bytes(size(tmp,kind=c_size_t)),int64)
#ifdef ASTR_AIR5_CHEMISTRY
    if(lcomb)then
      call check(device_state_download(context,device_air5_tv,tve,bytes(size(tve,kind=c_size_t))),'AIR5 Tv download')
      call check(device_state_download(context,device_air5_species,spc,bytes(size(spc,kind=c_size_t))),'AIR5 species download')
      downloaded=downloaded+int(bytes(size(tve,kind=c_size_t))+bytes(size(spc,kind=c_size_t)),int64)
    endif
#endif
    host_primitives_current=.true.
  end subroutine
  subroutine cpp_import_host_state()
#ifdef ASTR_AIR5_CHEMISTRY
    use commarray, only: tve,spc
#endif
    integer :: local_history,global_history,ierr
    call require(c_associated(context),'missing device state for checkpoint restoration')
    call require(all(ieee_is_finite(q(0:im,0:jm,0:km,:))).and. &
      all(ieee_is_finite(rho(0:im,0:jm,0:km))).and.all(rho(0:im,0:jm,0:km)>0).and. &
      all(ieee_is_finite(vel(0:im,0:jm,0:km,:))).and. &
      all(ieee_is_finite(prs(0:im,0:jm,0:km))).and.all(prs(0:im,0:jm,0:km)>0).and. &
      all(ieee_is_finite(tmp(0:im,0:jm,0:km))).and.all(tmp(0:im,0:jm,0:km)>0), &
      'checkpoint physical state must be finite with positive density, pressure and temperature')
    call check(device_state_upload(context,device_q,q,bytes(size(q,kind=c_size_t))),'checkpoint q upload')
    call check(device_state_upload(context,device_rho,rho,bytes(size(rho,kind=c_size_t))),'checkpoint density upload')
    call check(device_state_upload(context,device_velocity,vel,bytes(size(vel,kind=c_size_t))),'checkpoint velocity upload')
    call check(device_state_upload(context,device_pressure,prs,bytes(size(prs,kind=c_size_t))),'checkpoint pressure upload')
    call check(device_state_upload(context,device_temperature,tmp,bytes(size(tmp,kind=c_size_t))),'checkpoint temperature upload')
    call check(device_state_upload(context,device_metric,dxi,bytes(size(dxi,kind=c_size_t))),'checkpoint metric upload')
    call check(device_state_upload(context,device_jacobian,jacob,bytes(size(jacob,kind=c_size_t))),'checkpoint Jacobian upload')
#ifdef ASTR_AIR5_CHEMISTRY
    if(lcomb)then
      call require(all(ieee_is_finite(tve(0:im,0:jm,0:km))).and.all(tve(0:im,0:jm,0:km)>0).and. &
        all(ieee_is_finite(spc(0:im,0:jm,0:km,:))).and.all(spc(0:im,0:jm,0:km,:)>=0), &
        'AIR5 checkpoint physical Tv/composition')
      call check(device_state_upload(context,device_air5_tv,tve,bytes(size(tve,kind=c_size_t))),'checkpoint Tv upload')
      call check(device_state_upload(context,device_air5_species,spc,bytes(size(spc,kind=c_size_t))),'checkpoint species upload')
      call transfer_air5_aux_cpp(context,.false.)
      host_primitives_current=.true.
      return
    endif
#endif
    call upload_slip_mask_cpp()
    call upload_profile_target_cpp()
    if(trim(flowtype)=='bl') call upload_flat_geometry_cpp()
    call prepare_top_boundary_cpp()
    local_history=merge(1,0,lcracon.or.any(crinod))
    call MPI_Allreduce(local_history,global_history,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
    call check(ierr,'restored critical-history presence')
    if(global_history/=0.or.crashfix_ready) then
      if(crashfix_ready) then
        call transfer_crashfix_history(.false.)
      else
        call prepare_crashfix_state()
      endif
      call check(crashfix_history_exchange_cpp(context),'restored critical-history ownership')
    endif
    host_primitives_current=.true.
  end subroutine
  subroutine collect_timestep_rates_cpp(values,valid)
    real(real_kind),intent(out) :: values(3)
    logical,intent(out) :: valid
    integer :: status
    status=fluid_cfl(context,int(merge(1,0,nondimen),c_int),mach,gamma,rgas,values)
    valid=status==0
    if(.not.valid) write(*,'(a,i0,a,i0)') 'ASTR_GPU_CPP wave collector rank=',mpirank,' status=',status
  end subroutine
  subroutine collect_cfl_cpp(values,keys,yvalues,by_plane,ny)
    integer,intent(in) :: ny
    real(real_kind),intent(out) :: values(4),yvalues(4,0:ny-1)
    integer(int64),intent(out) :: keys(4)
    logical,intent(in) :: by_plane
    yvalues=0
    call check(fluid_cfl_diagnostics(context,int(merge(1,0,nondimen),c_int),mach,gamma,rgas, &
      int([ia,ja,ka],c_int),int([ig0,jg0,kg0],c_int),int(merge(ny,0,by_plane),c_int), &
      values,keys,yvalues),'CFL diagnostic')
  end subroutine
  subroutine cpp_time_integration(statistics_started)
    use benchmark_runtime, only: benchmark_gpu_rk_timing_enabled
    use commvar, only: time
    real(clock_kind),intent(in) :: statistics_started
    logical :: timing
    real(clock_kind) :: started,prepared,statistics_begin,statistics_seconds,elapsed
    real(c_double) :: halo_times(6)
    real(c_double) :: axis_times(7,5,3)
    character(14),parameter :: groups(5)=[character(14) :: 'q','filter_work','stress','heat','viscous_packed']
    integer :: axis,group
#ifdef ASTR_AIR5_CHEMISTRY
    if(lcomb)then
      host_primitives_current=.false.
      call advance_air5_cpp(context)
      call report_step_device_memory()
      return
    endif
#endif
    timing=benchmark_gpu_rk_timing_enabled()
    statistics_seconds=0.e0_real_kind
    prepared=0.e0_real_kind
    if(timing) then
      call check(device_state_synchronize(context),'timing start dependency')
      started=MPI_Wtime()
    endif
    ! Host publication caches are reusable only until the next RK advancement.
    host_primitives_current=.false.
    call prepare_dynamic_inflow_cpp(real(time,real_kind))
    do rkstep=1,3
#ifdef ASTR_BUILD_TESTING
      exchange_sequence=0
#endif
      if(x_characteristic_boundaries) then
        call check(prepare_x_characteristic_stage(context),'prepare x-NSCBC RK stage')
      elseif(profile_staged_outlet) then
        call check(prepare_staged_profile(handle=context),'prepare staged profile NSCBC RK stage')
      elseif(top_characteristic_boundary) then
        call check(prepare_top_stage(context),'prepare upper-y NSCBC RK stage')
      elseif(physical_boundaries) then
        call check(fluid_prepare_stage(context,const2,const6,lfilter,cpp_exchange_axis,filter_selected, &
          basic_boundaries,primitive_selected,repair_filtered_selected,repair_input_selected),'prepare physical RK stage')
      elseif(cpp_compact_filter) then
        call check(fluid_prepare_stage(context,const2,const6,lfilter,cpp_exchange_axis,compact_filter_cpp, &
          filtered_recovery_operator=repair_filtered_selected,input_recovery_operator=repair_input_selected),'prepare RK stage')
      else
        call check(fluid_prepare_stage(context,const2,const6,lfilter,cpp_exchange_axis, &
          filtered_recovery_operator=repair_filtered_selected,input_recovery_operator=repair_input_selected),'prepare RK stage')
      endif
      if(timing.and.rkstep==1) then
        call check(device_state_synchronize(context),'timing stage-one preparation')
        prepared=MPI_Wtime()-started
      endif
#ifdef ASTR_BUILD_TESTING
      call snapshot_prepared_stage()
#endif
      if(rkstep==1.and.nstep>0) then
        if(timing) statistics_begin=MPI_Wtime()
        call publish_flow_statistics(statistics_started)
        if(timing) statistics_seconds=MPI_Wtime()-statistics_begin
      endif
      if(evolved_wall) then
        call check(fluid_advance_prepared_stage(context,rkstep,deltat,const2,const6,reynolds,prandtl, &
          const5,tempconst,tempconst1,diffterm,cpp_exchange_axis,device_exchange_begin,cpp_exchange_end, &
          gradient_selected,convection_selected,viscous_selected,source_operator=source_selected, &
          post_update_operator=post_update_selected),'advance evolved-density high-wall stage')
      elseif(keep_w1_enabled) then
        call check(fluid_advance_prepared_stage(context,rkstep,deltat,const2,const6,reynolds,prandtl, &
          const5,tempconst,tempconst1,diffterm,cpp_exchange_axis,device_exchange_begin,cpp_exchange_end, &
          gradient_selected,convection_selected,integrated_viscous_operator=w1_selected,source_operator=source_selected, &
          post_update_operator=post_update_selected),'advance W1 paired-wall RK stage')
      elseif(viscous_form=='me6_opti') then
        if(upwind_sensor) then
          call check(fluid_advance_prepared_stage(context,rkstep,deltat,const2,const6,reynolds,prandtl, &
            const5,tempconst,tempconst1,diffterm,cpp_exchange_axis,device_exchange_begin,cpp_exchange_end, &
            gradient_selected,convection_selected,integrated_viscous_operator=me6_selected,source_operator=source_selected, &
            sensor_operator=sensor_selected,post_update_operator=post_update_selected),'advance upwind sensor/RK stage')
        else
          call check(fluid_advance_prepared_stage(context,rkstep,deltat,const2,const6,reynolds,prandtl, &
            const5,tempconst,tempconst1,diffterm,cpp_exchange_axis,device_exchange_begin,cpp_exchange_end, &
            gradient_selected,convection_selected,integrated_viscous_operator=me6_selected,source_operator=source_selected, &
            post_update_operator=post_update_selected), &
            'advance RK stage')
        endif
      elseif(upwind_sensor) then
        call check(fluid_advance_prepared_stage(context,rkstep,deltat,const2,const6,reynolds,prandtl, &
          const5,tempconst,tempconst1,diffterm,cpp_exchange_axis,device_exchange_begin,cpp_exchange_end, &
          gradient_selected,convection_selected,viscous_selected,source_operator=source_selected, &
          sensor_operator=sensor_selected,post_update_operator=post_update_selected),'advance upwind sensor/RK stage')
      else
        call check(fluid_advance_prepared_stage(context,rkstep,deltat,const2,const6,reynolds,prandtl, &
          const5,tempconst,tempconst1,diffterm,cpp_exchange_axis,device_exchange_begin,cpp_exchange_end, &
          gradient_selected,convection_selected,viscous_selected,source_operator=source_selected, &
          post_update_operator=post_update_selected, &
          packed_viscous_axis=merge(viscous_overlap_axis,-1_c_int,packed_viscous_halo)),'advance RK stage')
      endif
#ifdef ASTR_BUILD_TESTING
      call snapshot_updated_stage()
#endif
    enddo
    if(timing) then
      call check(device_state_synchronize(context),'timing complete RK')
      elapsed=MPI_Wtime()-started-statistics_seconds
      if(lio) write(*,'(a,1x,i0,3(1x,es24.16e3))') &
        'ASTR_GPU_RK_TIMING',nstep,prepared,elapsed-prepared,elapsed
      write(*,'(a,2(1x,i0),3(1x,es24.16e3))') &
        'ASTR_GPU_RANK_RK_TIMING',mpirank,nstep,prepared,elapsed-prepared,elapsed
      call check(device_transport_metrics(context,halo_times,1_c_int),'RK halo timing')
      write(*,'(a,2(1x,i0),6(1x,es24.16e3))') 'ASTR_CPP_HALO_TIMING',mpirank,nstep,halo_times
      if(halo_axis_timing) then
        call check(device_transport_axis_metrics(context,axis_times),'RK halo axis timing')
        do axis=1,3
          do group=1,5
            if(axis_times(7,group,axis)>0) write(*,'(a,3(1x,i0),1x,a,1x,i0,6(1x,es24.16e3))') &
              'ASTR_CPP_HALO_AXIS_TIMING',mpirank,nstep,axis,trim(groups(group)), &
              int(axis_times(7,group,axis)),axis_times(1:6,group,axis)
          enddo
        enddo
      endif
    endif
    call report_compact_phases_cpp()
    call report_step_device_memory()
  end subroutine
  integer(c_int) function prepare_staged_profile(handle) result(status)
    use bc, only: twall,bctype
    use commvar, only: xmin,ymin,ymax,pinf,vinf,winf
    use bc, only: pout
    type(c_ptr),intent(in) :: handle
    real(real_kind) :: local_mach
    integer :: ierr
    logical :: repaired,rebuild_rhs
    rebuild_rhs=.false.
    if(lfilter) then
      status=filter_selected(handle)
      if(status/=0) return
      status=repair_filtered_selected(handle)
      if(status/=0) return
    endif
    status=fluid_primitives(handle,const2,const6)
    if(status/=0) return
    status=fluid_rhs_clear(handle)
    if(status/=0) return
    status=exchange_characteristic_solution(handle)
    if(status/=0) return
    status=primitive_selected(handle)
    if(status/=0) return
    if(physical_low(1)/=0) status=profile_inlet_state_cpp(handle)
    if(status/=0) return
    if(physical_low(2)/=0.and.wall_forcing_enabled) status=prepare_wall_forcing_cpp(handle)
    if(status/=0) return
    if(physical_low(2)/=0.and.physical_high(1)/=0.and.bctype(3)==41) then
      if(high_open) then
        status=high_open_wall_y(handle,41_c_int,int(merge(1,0,wall_forcing_enabled),c_int),1_c_int,twall(3),const2,const6)
      else
        status=wall41_outlet_y(handle,int(merge(1,0,wall_forcing_enabled),c_int),twall(3),const2,const6)
      endif
    endif
    if(status/=0) return
    status=repair_input_state(handle,repaired)
    if(status/=0) return
    if(physical_high(1)/=0) then
      if(high_open) then
        status=nscbc_outlet_filter_high_open(handle,axis_types(2))
      else
        status=nscbc_outlet_filter(handle,axis_types(2))
      endif
    endif
    if(status/=0) return
    status=repair_input_state(handle,repaired)
    if(status/=0) return
    status=fluid_primitives(handle,const2,const6)
    if(status/=0) return
    status=exchange_characteristic_solution(handle)
    if(status/=0) return
    status=primitive_selected(handle)
    if(status/=0) return
    if(physical_high(1)/=0.and..not.high_open) status=nscbc_outlet_x(handle,axis_types(2),axis_types(3),1_c_int, &
      gamma,mach,xmax-xmin,roinf,uinf,vinf,winf,pinf,pout)
    if(status/=0) return
    if(physical_low(2)/=0) then
      if(high_open) then
        status=high_open_wall_y(handle,int(bctype(3),c_int),int(merge(1,0,wall_forcing_enabled),c_int), &
          0_c_int,twall(3),const2,const6)
      elseif(bctype(3)==421) then
        status=slip_wall_y(handle,421_c_int,0_c_int,twall(3),const2,const6)
      elseif(wall_forcing_enabled) then
        status=wall41_forcing_y(handle,twall(3),const2,const6)
      else
        status=wall41_basic_state(handle,1_c_int,0_c_int,twall(3),const2,const6)
      endif
    endif
    if(status/=0) return
    status=repair_input_state(handle,repaired)
    if(status/=0) return
    rebuild_rhs=repaired
    local_mach=0.e0_real_kind
    if(physical_high(2)/=0) status=nscbc_top_mach2(handle,mach,local_mach)
    if(status/=0) return
    call MPI_Allreduce(local_mach,top_mach_squared,1,mpi_real_type,MPI_MAX,MPI_COMM_WORLD,ierr)
    status=ierr
    if(status/=0) return
    if(top_self_filter) then
      if(physical_high(2)/=0) status=nscbc_top_filter(handle,0_c_int,axis_types(1))
      if(status/=0) return
      status=repair_input_state(handle,repaired)
      if(status/=0) return
      rebuild_rhs=rebuild_rhs.or.repaired
      status=exchange_characteristic_solution(handle)
      if(status/=0) return
      status=primitive_selected(handle)
      if(status/=0) return
      if(physical_high(2)/=0) status=nscbc_top_filter(handle,2_c_int,axis_types(3))
      if(status/=0) return
      status=repair_input_state(handle,repaired)
      if(status/=0) return
      rebuild_rhs=rebuild_rhs.or.repaired
    endif
    status=fluid_primitives(handle,const2,const6)
    if(status/=0) return
    if(top_self_filter) then
      status=exchange_characteristic_solution(handle)
      if(status/=0) return
      status=primitive_selected(handle)
      if(status/=0) return
    endif
    if(physical_high(2)/=0.and..not.high_open) status=nscbc_top_rhs(handle,axis_types(1),axis_types(3),top_policy, &
      int(merge(1,0,top_shock),c_int),0_c_int,gamma,mach,ymax-ymin,pinf,top_target,top_mach_squared)
    if(status/=0) return
    if(.not.top_self_filter) status=exchange_characteristic_solution(handle)
    if(status/=0) return
    status=primitive_selected(handle)
    if(status/=0) return
    status=repair_input_state(handle,repaired)
    if(status/=0) return
    rebuild_rhs=rebuild_rhs.or.repaired
    if(rebuild_rhs) then
      ! An actual existing repair invalidates the earlier boundary RHS, not its filters/RNG.
      status=fluid_rhs_clear(handle)
      if(status/=0) return
      status=primitive_selected(handle)
      if(status/=0) return
      local_mach=0.e0_real_kind
      if(physical_high(2)/=0) status=nscbc_top_mach2(handle,mach,local_mach)
      if(status/=0) return
      call MPI_Allreduce(local_mach,top_mach_squared,1,mpi_real_type,MPI_MAX,MPI_COMM_WORLD,ierr)
      status=ierr
      if(status/=0) return
      if(physical_high(1)/=0.and..not.high_open) status=nscbc_outlet_x(handle,axis_types(2),axis_types(3),1_c_int, &
        gamma,mach,xmax-xmin,roinf,uinf,vinf,winf,pinf,pout)
      if(status/=0) return
      if(physical_high(2)/=0.and..not.high_open) status=nscbc_top_rhs(handle,axis_types(1),axis_types(3),top_policy, &
        int(merge(1,0,top_shock),c_int),0_c_int,gamma,mach,ymax-ymin,pinf,top_target,top_mach_squared)
      if(status/=0) return
    endif
#ifdef ASTR_BUILD_TESTING
    call snapshot_operator_rhs(handle,'boundary')
#endif
  end function
  integer(c_int) function prepare_top_stage(handle) result(status)
    use bc, only: NSCBC_FARFIELD_POLICY_NONREFLECTING
    type(c_ptr),intent(in) :: handle
    real(real_kind) :: local_mach
    integer :: ierr
    logical :: repaired,top_repaired
    status=fluid_prepare_stage(handle,const2,const6,lfilter,cpp_exchange_axis,filter_selected, &
      basic_boundaries,primitive_selected,repair_filtered_selected,repair_input_selected)
    if(status/=0) return
    if(profile_characteristic_outlet) then
      if(physical_high(1)/=0) status=nscbc_outlet_filter(handle,axis_types(2))
      if(status/=0) return
      status=repair_input_state(handle,repaired)
      if(status/=0) return
      status=exchange_characteristic_solution(handle)
      if(status/=0) return
      status=repair_input_state(handle,repaired)
      if(status/=0) return
      status=primitive_selected(handle)
      if(status/=0) return
    endif
    top_mach_squared=0.e0_real_kind
    if(top_policy/=NSCBC_FARFIELD_POLICY_NONREFLECTING) then
      local_mach=0.e0_real_kind
      if(physical_high(2)/=0) status=nscbc_top_mach2(handle,mach,local_mach)
      if(status/=0) return
      call MPI_Allreduce(local_mach,top_mach_squared,1,mpi_real_type,MPI_MAX,MPI_COMM_WORLD,ierr)
      status=ierr
      if(status/=0) return
    endif
    if(top_self_filter) then
      top_repaired=.false.
      if(physical_high(2)/=0) status=nscbc_top_filter(handle,0_c_int,axis_types(1))
      if(status/=0) return
      status=repair_input_state(handle,repaired)
      if(status/=0) return
      top_repaired=top_repaired.or.repaired
      status=exchange_characteristic_solution(handle)
      if(status/=0) return
      status=repair_input_state(handle,repaired)
      if(status/=0) return
      top_repaired=top_repaired.or.repaired
      if(physical_high(2)/=0) status=nscbc_top_filter(handle,2_c_int,axis_types(3))
      if(status/=0) return
      status=repair_input_state(handle,repaired)
      if(status/=0) return
      top_repaired=top_repaired.or.repaired
      status=exchange_characteristic_solution(handle)
      if(status/=0) return
      status=repair_input_state(handle,repaired)
      if(status/=0) return
      top_repaired=top_repaired.or.repaired
      status=primitive_selected(handle)
      if(status/=0) return
      if(top_repaired) then
        local_mach=0.e0_real_kind
        if(physical_high(2)/=0) status=nscbc_top_mach2(handle,mach,local_mach)
        if(status/=0) return
        call MPI_Allreduce(local_mach,top_mach_squared,1,mpi_real_type,MPI_MAX,MPI_COMM_WORLD,ierr)
        status=ierr
      endif
    endif
  end function
  integer(c_int) function prepare_x_characteristic_stage(handle) result(status)
    type(c_ptr),value :: handle
    status=fluid_primitives(handle,const2,const6)
    if(status/=0) return
    status=exchange_characteristic_solution(handle)
    if(status/=0) return
    if(physical_high(1)/=0) then
      status=nscbc_outlet_filter(handle,axis_types(2))
      if(status/=0) return
    endif
    status=exchange_characteristic_solution(handle)
    if(status/=0) return
    status=primitive_selected(handle)
  end function
  integer(c_int) function exchange_characteristic_solution(handle) result(status)
    type(c_ptr),value :: handle
    integer(c_int) :: axis,reconcile
    ! Call the imported C ABI directly, as in the sensor/stress consumers;
    ! NVHPC misoptimizes direct calls to this module's BIND(C) wrapper.
    do reconcile=1,0,-1
    do axis=0,2
      status=device_exchange_axis(handle,device_q,axis,reconcile)
      if(status/=0) return
#ifdef ASTR_BUILD_TESTING
      call snapshot_exchange(handle,device_q,axis,reconcile)
#endif
    enddo
    enddo
  end function
  integer(c_int) function filter_selected(handle) result(status)
    use fluid_step, only: fluid_filter_explicit
    type(c_ptr),value :: handle
    if(cpp_compact_filter) then
      status=compact_filter_cpp(handle)
    elseif(evolved_wall) then
      status=fluid_filter_explicit(handle,cpp_exchange_axis,physical_low,physical_high,[evolved_wall_axis,evolved_wall_kind])
    else
      status=fluid_filter_explicit(handle,cpp_exchange_axis,physical_low,physical_high)
    endif
  end function
  integer(c_int) function primitive_selected(handle) result(status)
    type(c_ptr),value :: handle
    status=fluid_refresh_primitives_boundary(handle,const2,const6,physical_low,physical_high)
  end function
  integer(c_int) function basic_boundaries(handle) result(status)
    use bc, only: bctype,twall,pout
    use commvar, only: pinf,time
    type(c_ptr),value :: handle
    integer :: a,side,face
    status=0
    ! Weak SAT leaves wall unknowns free; do not apply the strong bc41 state.
    if(keep_sbp63_enabled) return
    do a=1,3
    do side=0,1
      if(side==0.and.physical_low(a)==0) cycle
      if(side==1.and.physical_high(a)==0) cycle
      face=2*a-1+side
      select case(bctype(face))
      case(0)
        status=device_invalid_argument
        if(face==4.and.trim(flowtype)=='ldcavity') status=ldc_top_lid_y(handle,const2,const6)
      case(11)
        status=device_invalid_argument
        if(face==1) status=profile_inlet_state_cpp(handle)
      case(21)
        status=device_invalid_argument
        if(face==2) status=outflow21_x_basic(handle,mach,const2,const6,pinf*pout)
      case(22)
        status=merge(0_c_int,device_invalid_argument,face==2.and.(profile_characteristic_outlet.or.profile_staged_outlet))
      case(41)
        if(evolved_wall) then
          status=wall41_evolved_density_state(handle,int(a-1,c_int),int(side,c_int),twall(face),const2,const6)
        elseif(a==2.and.side==0.and.wall_forcing_enabled) then
          ! Exactly one prescribed-plane evaluation per actual lower wall call.
          status=prepare_wall_forcing_cpp(handle)
          if(status==0) status=wall41_forcing_y(handle,twall(face),const2,const6)
        else
          status=wall41_basic_state(handle,int(a-1,c_int),int(side,c_int),twall(face),const2,const6)
        endif
      case(42)
        if(evolved_wall) then
          status=wall_adiabatic_evolved_state(handle,int(a-1,c_int),int(side,c_int),0_c_int,const2,const6)
        else
          status=wall42_basic_state(handle,int(a-1,c_int),int(side,c_int),const2,const6)
        endif
      case(411,421)
        status=device_invalid_argument
        if(evolved_wall) then
          status=wall_adiabatic_evolved_state(handle,int(a-1,c_int),int(side,c_int),1_c_int,const2,const6)
        elseif(a==2) then
          status=slip_wall_y(handle,int(bctype(face),c_int),int(side,c_int),twall(face),const2,const6)
        endif
      case(31)
        status=device_invalid_argument
        if(a==2) status=fixed_rti_y(handle,int(side,c_int),const2,const6)
      case(50)
        status=zeroextrap_basic(handle,int(a-1,c_int),int(side,c_int),const2,const6)
      case(51)
        status=device_invalid_argument
        if(face==4) status=farfield51_upper_basic(handle,const2,const6)
      case(52)
        status=merge(0_c_int,device_invalid_argument,face==4)
      case(60)
        status=symmetry_basic(handle,int(a-1,c_int),int(side,c_int),const2,const6)
      case default
        write(*,'(a,2(i0,1x))') 'C++ boundary state is not implemented: face/type=',face,bctype(face)
        status=device_invalid_argument
      end select
      if(status/=0) return
    enddo
    enddo
  end function
  integer(c_int) function profile_inlet_state_cpp(handle) result(status)
    use commvar, only: time
    type(c_ptr),intent(in) :: handle
    if(dynamic_inflow_requested_cpp()) then
      status=apply_dynamic_inflow_cpp(real(time,real_kind),profile_mach_pressure,mach,const2,const6)
    else
      status=profile_inflow_x(handle,int(merge(1,0,profile_mach_pressure),c_int),mach,const2,const6)
    endif
  end function
  integer(c_int) function prepare_wall_forcing_cpp(handle) result(status)
    use bc, only: wall_blowing_velocity
    type(c_ptr),intent(in) :: handle
    lower_wall_velocity=wall_blowing_velocity()
    status=wall_forcing_upload(handle,lower_wall_velocity,bytes(size(lower_wall_velocity,kind=c_size_t)))
  end function
  integer(c_int) function post_update_selected(handle) result(status)
    type(c_ptr),value :: handle
    integer :: layer,axis
    status=0
    do layer=1,2
      if(.not.sponge_enabled(layer)) cycle
      do axis=0,2
        status=device_exchange_axis(handle,device_q,int(axis,c_int),0_c_int)
        if(status/=0) return
      enddo
      status=sponge_apply_cpp(handle,int(layer-1,c_int))
      if(status/=0) return
    enddo
    if(evolved_wall) status=project_evolved_wall(handle)
    if(status==0) status=repair_update_selected(handle)
  end function
  integer(c_int) function project_evolved_wall(handle) result(status)
    use bc, only: twall
    type(c_ptr),value :: handle
    integer :: side,face
    status=0
    do side=0,1
      if(side==0.and.physical_low(evolved_wall_axis+1)==0) cycle
      if(side==1.and.physical_high(evolved_wall_axis+1)==0) cycle
      face=2*evolved_wall_axis+1+side
      if(evolved_wall_kind==41) then
        status=wall41_evolved_density_state(handle,evolved_wall_axis,int(side,c_int),twall(face),const2,const6)
      else
        status=wall_adiabatic_evolved_state(handle,evolved_wall_axis,int(side,c_int), &
          int(merge(1,0,evolved_wall_kind==421),c_int),const2,const6)
      endif
      if(status/=0) return
    enddo
  end function
  subroutine cpp_refresh_mean_fields()
    integer :: axis
    host_primitives_current=.false.
    do axis=0,2
      call check(device_exchange_axis(context,device_velocity,int(axis,c_int),0_c_int),'mean velocity halo refresh')
    enddo
    do axis=0,2
      call check(device_exchange_axis(context,device_temperature,int(axis,c_int),0_c_int),'mean temperature halo refresh')
    enddo
    call check(gradient_selected(context),'completed-step mean gradient')
    if(mixed_derivative) call check(fluid_diagnostic_gradient(context,physical_low,physical_high), &
      'completed-step separate diagnostic mean gradients')
  end subroutine
  integer(c_int) function gradient_selected(handle) result(status)
    type(c_ptr),value :: handle
    if(high_open) then
      status=fluid_gradient_high_open(handle,physical_low,physical_high)
    elseif(keep_sbp63_enabled) then
      status=fluid_sbp63_gradient(handle,int(ja,c_int),int(jg0,c_int),keep_sbp63_wall_temperature)
    elseif(evolved_wall) then
      status=fluid_gradient_high_wall(handle,evolved_wall_axis,physical_low,physical_high)
    elseif(difschm=='643c') then
      status=compact_gradient_cpp(handle)
    elseif(physical_boundaries) then
      status=fluid_gradient_basic_boundary(handle,physical_low,physical_high)
    else
      status=fluid_gradient(handle)
    endif
  end function
  integer(c_int) function convection_selected(handle) result(status)
    use commvar, only: xmin,ymin,ymax,pinf,vinf,winf
    use bc, only: pout,profile_low_order_nscbc_enabled
    type(c_ptr),value :: handle
    integer(c_int) :: axis,high_first(3),high_last(3)
    if(mp_positivity) then
      call require(.not.lfilter,'MP positivity does not admit runtime enabling of filtering')
      status=0
      if(.not.profile_staged_outlet) status=fluid_rhs_clear(handle)
      if(status==0) status=mp_positivity_begin_cpp(handle)
      do axis=0,2
        if(status/=0) return
        status=mp_positivity_axis_cpp(handle,axis,int(merge(1,0,high_open),c_int),gamma,mach,bfacmpld, &
          physical_low,physical_high,body_first,body_last)
      enddo
      if(status==0) status=mp_positivity_faces_complete_cpp(handle)
    elseif(high_open) then
      status=fluid_rhs_clear(handle)
      high_first=physical_low;high_last=int([im,jm,km],c_int)
      do axis=0,2
        if(status/=0) return
        status=upwind_high_order_axis_cpp(handle,axis,gamma,mach,bfacmpld,physical_low,physical_high,high_first,high_last)
      enddo
    elseif(keep_sbp63_enabled) then
      status=fluid_rhs_clear(handle)
      if(status==0) status=fluid_sbp63_convection(handle,int(ja,c_int),int(jg0,c_int),gamma)
    elseif(evolved_wall) then
      status=fluid_rhs_clear(handle)
      if(status==0) status=fluid_flux_divergence_high_wall(handle,0_c_int,evolved_wall_axis, &
        physical_low,physical_high,body_first,body_last,int(merge(1,0,evolved_wall_kind/=41),c_int))
    elseif(upwind_convection) then
      ! Gradients, sensor and viscous-flux preparation leave this earlier boundary RHS intact.
      status=0
      if(.not.profile_staged_outlet) status=fluid_rhs_clear(handle)
      do axis=0,2
        if(status/=0) exit
        status=upwind_axis_cpp(handle,axis,int(recon_schem,c_int),int(merge(1,0,lchardecomp),c_int), &
          gamma,mach,bfacmpld,0_c_int,physical_low,physical_high,body_first,body_last)
      enddo
    elseif(convection_form=='keep') then
      if(conschm=='643c') then
        status=compact_keep_convection_cpp(handle,profile_staged_outlet)
      else
        if(physical_boundaries) then
          status=0
          if(.not.profile_staged_outlet) status=fluid_rhs_clear(handle)
          if(status==0) status=fluid_keep_accumulate_basic(handle,gamma,axis_types,body_first,body_last)
        else
          status=fluid_keep_convection(handle,gamma)
        endif
      endif
    elseif(conschm=='643c') then
      status=compact_convection_cpp(handle,profile_staged_outlet)
    elseif(physical_boundaries) then
      status=0
      if(.not.profile_staged_outlet) status=fluid_rhs_clear(handle)
      if(status==0) status=fluid_flux_divergence_basic(handle,0_c_int,physical_low,physical_high,body_first,body_last)
    else
      status=fluid_convection(handle)
    endif
    if(status/=0) return
    if(.not.profile_staged_outlet.and.top_characteristic_boundary.and.physical_high(2)/=0) then
      status=nscbc_top_rhs(handle,axis_types(1),axis_types(3),top_policy,int(merge(1,0,top_shock),c_int), &
        int(merge(1,0,top_keep_corner),c_int),gamma,mach,ymax-ymin,pinf,top_target,top_mach_squared)
      if(status/=0) return
      if(top_viscous_source) status=nscbc_top_prediff_capture(handle)
    endif
    if(status/=0) return
    if(profile_characteristic_outlet.and.physical_high(1)/=0) then
      status=nscbc_outlet_x(handle,axis_types(2),axis_types(3), &
        int(merge(0,1,profile_low_order_nscbc_enabled()),c_int), &
        gamma,mach,xmax-xmin,roinf,uinf,vinf,winf,pinf,pout)
      if(status/=0) return
      if(physical_high(2)/=0.and.top_keep_corner) status=nscbc_top_outlet_corner(handle,gamma,mach)
    endif
#ifdef ASTR_BUILD_TESTING
    if(status==0) call snapshot_operator_rhs(handle,'conv')
#endif
  end function
  integer(c_int) function sensor_selected(handle) result(status)
    type(c_ptr),value :: handle
    integer(c_int) :: axis
    status=0
    if(.not.diffterm) status=gradient_selected(handle)
    if(status/=0) return
    status=shock_raw_cpp(handle,physical_low,physical_high)
    if(status/=0) return
    do axis=0,2
      status=device_exchange_axis(handle,device_shock_sensor,axis,0_c_int)
      if(status/=0) return
    enddo
    status=shock_expand_cpp(handle,physical_low,physical_high,shkcrt)
    if(status==0.and.conschm=='743e'.and.difschm=='643c') status=shock_mask_exchange_cpp(handle)
#ifdef ASTR_BUILD_TESTING
    if(status==0) call snapshot_current_sensor(handle)
#endif
  end function
  integer(c_int) function viscous_selected(handle) result(status)
    type(c_ptr),value :: handle
    if(high_open) then
      status=fluid_viscous_high_open(handle,physical_low,physical_high)
    elseif(keep_sbp63_enabled) then
      status=fluid_sbp63_diffusion(handle,int(ja,c_int),int(jg0,c_int),keep_sbp63_wall_temperature, &
        reynolds,prandtl,const5,tempconst,tempconst1)
    elseif(evolved_wall) then
      status=fluid_flux_divergence_high_wall(handle,1_c_int,evolved_wall_axis, &
        physical_low,physical_high,body_first,body_last,int(merge(1,0,evolved_wall_kind/=41),c_int))
    elseif(difschm=='643c') then
      status=compact_viscous_cpp(handle)
    elseif(physical_boundaries) then
      status=fluid_flux_divergence_basic(handle,1_c_int,physical_low,physical_high,body_first,body_last)
    else
      status=fluid_viscous_divergence(handle)
    endif
    if(status==0.and.top_viscous_source.and.physical_high(2)/=0) status=nscbc_top_viscous_source(handle,gamma,mach)
#ifdef ASTR_BUILD_TESTING
    if(status==0.and..not.evolved_wall.and..not.top_incident) call snapshot_operator_rhs(handle,'full')
#endif
  end function
  integer(c_int) function w1_selected(handle) result(status)
    type(c_ptr),value :: handle
    integer(c_int) :: send_width(2),recv_width(2)
    call keep_w1_end_widths(send_width,recv_width)
    status=w1_diffusion_cpp(handle,int(jg0,c_int),int(ja,c_int),int(merge(1,0,[isize,jsize,ksize]>1),c_int), &
      keep_w1_inverse,real(keep_sbp63_wall_temperature,c_double), &
      real(reynolds,c_double),real(prandtl,c_double),real(const5,c_double),real(tempconst,c_double), &
      real(tempconst1,c_double),send_width,recv_width)
#ifdef ASTR_BUILD_TESTING
    if(status==0) call snapshot_operator_rhs(handle,'full')
#endif
  end function
  integer(c_int) function me6_selected(handle) result(status)
    type(c_ptr),value :: handle
    integer(c_int) :: axis
    status=0
    if(physical_boundaries) then
      status=fluid_viscous_flux(handle,reynolds,prandtl,const5,tempconst,tempconst1)
      if(status/=0) then
        write(*,'(a,i0)') 'ASTR_CPP_ME6 nodal flux status=',status
        return
      endif
      do axis=0,2
        ! Direct C ABI avoids NVHPC mis-inlining this module's BIND(C) callback.
        status=device_exchange_axis(handle,device_stress,axis,0_c_int)
        if(status/=0) then
          write(*,'(a,2(i0,1x))') 'ASTR_CPP_ME6 stress exchange axis/status=',axis,status
          return
        endif
#ifdef ASTR_BUILD_TESTING
        call snapshot_exchange(handle,device_stress,axis,0_c_int)
#endif
        status=device_exchange_axis(handle,device_heat_flux,axis,0_c_int)
        if(status/=0) then
          write(*,'(a,2(i0,1x))') 'ASTR_CPP_ME6 heat exchange axis/status=',axis,status
          return
        endif
#ifdef ASTR_BUILD_TESTING
        call snapshot_exchange(handle,device_heat_flux,axis,0_c_int)
#endif
      enddo
      if(difschm=='643c') then
        status=compact_viscous_cpp(handle,wall_band=.true.)
      else
        status=fluid_viscous_wall_band(handle,physical_low,physical_high,body_first,body_last)
      endif
      if(status/=0) return
    endif
    status=me6_diffusion_cpp(handle,reynolds,prandtl,const5,tempconst,tempconst1)
    if(status==0.and.top_viscous_source.and.physical_high(2)/=0) status=nscbc_top_viscous_source(handle,gamma,mach)
#ifdef ASTR_BUILD_TESTING
    if(status==0.and..not.top_incident) call snapshot_operator_rhs(handle,'full')
#endif
  end function
  integer(c_int) function source_selected(handle) result(status)
    use bc, only: pout,twall
    use commvar, only: xmin,vinf,winf,pinf
    type(c_ptr),value :: handle
    integer(c_int) :: incident_status,global_incident_status
    integer :: ierr
    integer(c_int) :: limited
    real(real_kind),parameter :: rk_a(3)=[1.e0_real_kind,.75_real_kind,1.e0_real_kind/3.e0_real_kind], &
      rk_b(3)=[0.e0_real_kind,.25_real_kind,2.e0_real_kind/3.e0_real_kind], &
      rk_c(3)=[1.e0_real_kind,.25_real_kind,2.e0_real_kind/3.e0_real_kind]
    status=0
#ifdef ASTR_BUILD_TESTING
    if(.not.diffterm.and..not.top_incident) call snapshot_operator_rhs(handle,'full')
#endif
    if(trim(flowtype)=='channel'.and.lihomo) status=fluid_uniform_force(handle,force)
    if(trim(flowtype)=='rti') status=fluid_rti_source(handle)
    if(status/=0) return
    if(x_characteristic_boundaries) then
      if(physical_low(1)/=0) then
        status=nscbc_inlet_x(handle,axis_types(2),axis_types(3),gamma,mach,xmax-xmin,roinf,uinf,vinf,winf,pinf)
        if(status/=0) return
      endif
      if(physical_high(1)/=0) then
        status=nscbc_outlet_x(handle,axis_types(2),axis_types(3),1_c_int,gamma,mach,xmax-xmin, &
          roinf,uinf,vinf,winf,pinf,pout)
        if(status/=0) return
      endif
    endif
    if(evolved_wall) then
      status=fluid_wall_rhs(handle,evolved_wall_axis,physical_low,physical_high, &
        twall(2*evolved_wall_axis+1:2*evolved_wall_axis+2),const2,const6,evolved_wall_kind)
      if(status/=0) return
    endif
    if(top_incident) then
#ifdef ASTR_BUILD_TESTING
      call snapshot_operator_rhs(handle,'incident_pre')
#endif
      incident_status=0
      if(high_open) then
        status=nscbc_high_open_complete(handle,physical_low,physical_high,gamma,mach,xmax-xmin, &
          roinf,uinf,vinf,winf,pinf,pout,incident_status)
      elseif(physical_high(2)/=0) then
        status=nscbc_incident_complete(handle,axis_types(1),gamma,mach,incident_status)
      endif
      if(status/=0) return
      call MPI_Allreduce(incident_status,global_incident_status,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
      status=ierr
      if(status/=0) return
      if(global_incident_status/=0) then
        if(lio) write(*,'(a,i0)') 'GRCBC boundary status (4=subsonic outlet/top corner): ',global_incident_status
        status=device_invalid_argument
        return
      endif
    endif
#ifdef ASTR_BUILD_TESTING
    if(status==0) call snapshot_operator_rhs(handle,'post_source')
    if(status==0.and.(evolved_wall.or.top_incident)) call snapshot_operator_rhs(handle,'full')
#endif
    if(status==0.and.mp_positivity) then
      status=mp_positivity_limit_cpp(handle,deltat,rk_a(rkstep),rk_b(rkstep),rk_c(rkstep), &
        int(rkstep,c_int),body_first,body_last,limited)
      if(status/=0) return
      if(lio) write(*,'(a,i0,a,i0,a,i0)') 'ASTR_CPP_MP_POSITIVITY step=',nstep,' stage=',rkstep,' limited_rank_nodes=',limited
#ifdef ASTR_BUILD_TESTING
      call snapshot_operator_rhs(handle,'mp_limited')
#endif
    endif
  end function
  integer(c_int) function cpp_exchange_axis(handle,field,axis,reconcile) bind(C,name='astr_cpp_exchange_axis') result(status)
    type(c_ptr),value :: handle
    integer(c_int),value :: field,axis,reconcile
    status=device_exchange_axis(handle,field,axis,reconcile)
#ifdef ASTR_BUILD_TESTING
    if(status==0) call snapshot_exchange(handle,field,axis,reconcile)
#endif
  end function
  integer(c_int) function cpp_exchange_end(handle,field,axis,reconcile) bind(C,name='astr_cpp_exchange_end') result(status)
    type(c_ptr),value :: handle
    integer(c_int),value :: field,axis,reconcile
    status=device_exchange_end(handle,field,axis,reconcile)
#ifdef ASTR_BUILD_TESTING
    if(status==0) call snapshot_exchange(handle,field,axis,reconcile)
#endif
  end function
#ifdef ASTR_BUILD_TESTING
  subroutine snapshot_current_sensor(handle)
    use validation_io, only: rhs_validation_requested,write_sensor_validation_snapshot
    type(c_ptr),intent(in) :: handle
    real(c_double),allocatable :: sensor(:,:,:)
    integer(c_int8_t),allocatable :: mask(:,:,:)
    logical,save :: dump_configured=.false.,dump_enabled=.false.,dumped=.false.
    character(1024),save :: dump_path=''
    character(1100) :: output_path
    character(32) :: suffix
    logical :: binary_requested,text_requested
    integer :: status,length,unit,err,i,j,k
    if(.not.dump_configured) then
      call get_environment_variable('ASTR_SHOCK_SENSOR_DUMP',dump_path,length=length,status=status)
      call require(status==0.or.status==1,'initial sensor dump path is truncated or unreadable')
      dump_enabled=status==0.and.length>0
      dump_configured=.true.
    endif
    binary_requested=rhs_validation_requested()
    text_requested=dump_enabled.and..not.dumped
    if(.not.binary_requested.and..not.text_requested) return
    allocate(sensor(-hm:im+hm,-hm:jm+hm,-hm:km+hm),mask(0:im,0:jm,0:km))
    call check(device_state_snapshot_axis(handle,device_shock_sensor,0_c_int,physical_low,physical_high, &
      sensor,size(sensor,kind=c_size_t)*c_sizeof(0.0_c_double)),'sensor validation physical values')
    call check(shock_mask_download_cpp(handle,mask,int(size(mask),c_size_t)),'sensor validation integer mask')
    if(binary_requested) call write_sensor_validation_snapshot('sensor',real(sensor(0:im,0:jm,0:km),real_kind),mask,nstep,rkstep)
    if(text_requested) then
      output_path=trim(dump_path)
      if(isize*jsize*ksize>1) then
        write(suffix,'(".rank",i0)') mpirank
        output_path=trim(output_path)//trim(suffix)
      endif
      open(newunit=unit,file=trim(output_path),status='replace',action='write',iostat=err)
      call check(err,'open initial sensor dump')
      write(unit,'(a,6(1x,i0))') '# shock_sensor',im,jm,km,ig0,jg0,kg0
      do k=0,km
      do j=0,jm
      do i=0,im
        write(unit,'(3(i0,1x),es24.16e3,1x,i1)') i,j,k,sensor(i,j,k),merge(1,0,mask(i,j,k)/=0_c_int8_t)
      enddo
      enddo
      enddo
      close(unit,iostat=err);call check(err,'close initial sensor dump')
      dumped=.true.
    endif
  end subroutine
  subroutine snapshot_updated_stage()
    use validation_io, only: rhs_validation_requested,write_q_validation_snapshot
    if(.not.rhs_validation_requested()) return
    call check(device_state_download(context,device_q,q,bytes(size(q,kind=c_size_t))),'updated-stage validation q')
    call write_q_validation_snapshot('post_update',nstep,rkstep)
  end subroutine
  subroutine snapshot_operator_rhs(handle,label)
    use validation_io, only: rhs_validation_requested,write_rhs_validation_snapshot
    type(c_ptr),intent(in) :: handle
    character(*),intent(in) :: label
    real(real_kind),allocatable :: values(:,:,:,:)
    if(.not.rhs_validation_requested()) return
    allocate(values(0:im,0:jm,0:km,5))
    call check(device_state_download(handle,device_rhs,values,bytes(size(values,kind=c_size_t))),'operator RHS observation')
    call write_rhs_validation_snapshot(label,nstep,rkstep,values)
  end subroutine
  subroutine snapshot_exchange(handle,field,axis,reconcile)
    use validation_io, only: rhs_validation_requested
    type(c_ptr),intent(in) :: handle
    integer(c_int),intent(in) :: field,axis,reconcile
    type(device_view) :: view
    real(c_double),allocatable :: data(:)
    character(1024) :: prefix
    character(1400) :: path
    integer :: unit,ignored
    integer(c_int) :: header(9)
    if(.not.rhs_validation_requested()) return
    exchange_sequence=exchange_sequence+1
    call check(device_state_view(handle,field,view),'exchange observation view')
    allocate(data(product(int(view%extent,c_size_t))))
    call check(device_state_snapshot_axis(handle,field,axis,physical_low,physical_high,data, &
      size(data,kind=c_size_t)*c_sizeof(0.0_c_double)), &
      'exchange observation readback')
    call get_environment_variable('ASTR_VALIDATION_RHS_PREFIX',prefix,status=ignored)
    write(path,'(a,".cpp_halo.step",i8.8,".rk",i2.2,".exchange",i2.2,".rank",i8.8,".bin")') &
      trim(prefix),nstep,rkstep,exchange_sequence,mpirank
    header=int([im,jm,km,hm,view%extent(4)*view%extent(5),field,axis,reconcile,exchange_sequence],c_int)
    open(newunit=unit,file=trim(path),access='stream',form='unformatted',status='new',action='write')
    write(unit) header
    write(unit) data
    close(unit)
    deallocate(data)
  end subroutine
  subroutine snapshot_prepared_stage()
    use validation_io, only: rhs_validation_requested,write_q_validation_snapshot,write_primitive_validation_snapshot
    use commarray, only: rho,vel,prs,tmp
    if(.not.rhs_validation_requested()) return
    call check(device_state_download(context,device_q,q,bytes(size(q,kind=c_size_t))),'stage validation q')
    call check(device_state_download(context,device_rho,rho,bytes(size(rho,kind=c_size_t))),'stage validation density')
    call check(device_state_download(context,device_velocity,vel,bytes(size(vel,kind=c_size_t))),'stage validation velocity')
    call check(device_state_download(context,device_pressure,prs,bytes(size(prs,kind=c_size_t))),'stage validation pressure')
    call check(device_state_download(context,device_temperature,tmp,bytes(size(tmp,kind=c_size_t))),'stage validation T')
    call write_q_validation_snapshot('pre_rhs',nstep,rkstep)
    call write_primitive_validation_snapshot('pre_rhs_primitives',nstep,rkstep)
  end subroutine
#endif
  subroutine publish_flow_statistics(started)
    use statistic, only: kenergy,enstophy,dissipation,max_q,min_q,statout,massflux,fbcx,wrms,massflux_target,chanfoce,wallheatflux
    use commvar, only: voldom
    real(clock_kind),intent(in) :: started
    real(real_kind) :: local(13),global(13),nodes,length
    real(real_kind) :: flat_local(5),flat_global(5)
    integer :: ierr
    call check(gradient_selected(context),'statistics gradient')
    if(mixed_derivative) call check(fluid_diagnostic_gradient(context,physical_low,physical_high), &
      'same-phase separate diagnostic gradients')
    call check(fluid_statistics(context,reynolds,tempconst,tempconst1,local),'field scalar statistics and extrema')
    if(trim(flowtype)=='channel') call check(fluid_channel_statistics(context,reynolds,tempconst,tempconst1, &
      int(merge(1,0,jg0==0),c_int),int(merge(1,0,jg0+jm==ja),c_int),local(:3)),'channel scalar statistics')
    if(trim(flowtype)=='bl') then
      call check(fluid_flat_statistics(context,reynolds,tempconst,tempconst1,prandtl,const5, &
        int(merge(1,0,jg0==0),c_int),flat_local),'flat-plate scalar statistics')
      call MPI_Allreduce(flat_local,flat_global,5,mpi_real_type,MPI_SUM,MPI_COMM_WORLD,ierr)
      call check(ierr,'flat-plate volume and wall sums')
    endif
    call MPI_Allreduce(local(1:3),global(1:3),3,mpi_real_type,MPI_SUM,MPI_COMM_WORLD,ierr)
    call check(ierr,'statistics sums')
    call MPI_Allreduce(local(4:8),global(4:8),5,mpi_real_type,MPI_MAX,MPI_COMM_WORLD,ierr)
    call check(ierr,'statistics maxima')
    call MPI_Allreduce(local(9:13),global(9:13),5,mpi_real_type,MPI_MIN,MPI_COMM_WORLD,ierr)
    call check(ierr,'statistics minima')
    nodes=real(ia,real_kind)*real(ja,real_kind)*real(ka,real_kind)
    if(trim(flowtype)=='channel') then
      massflux=global(1)/real(ia*ka,real_kind);fbcx=global(2)/real(ia*ka,real_kind)
      wrms=sqrt(global(3)/nodes)
      if(abs(massflux_target)<1.e-16_real_kind) then
        massflux_target=massflux;force=0.e0_real_kind
        if(lio) print *, ' ** target mass flux= ',massflux_target
      endif
      force(1)=chanfoce(force(1),massflux,fbcx,massflux_target)
      call MPI_Bcast(force,3,mpi_real_type,0,MPI_COMM_WORLD,ierr)
      call check(ierr,'channel driver broadcast')
    elseif(trim(flowtype)=='bl') then
      massflux=flat_global(1)/voldom;fbcx=flat_global(2)/flat_global(5)
      wrms=sqrt(flat_global(3)/nodes);wallheatflux=flat_global(4)/flat_global(5)
    elseif(trim(flowtype)/='rti'.and.trim(flowtype)/='ldcavity'.and.trim(flowtype)/='shuosher') then
      length=xmax/(2.e0_real_kind*acos(-1.e0_real_kind))
      kenergy=.5e0_real_kind*global(1)/nodes/(roinf*uinf*uinf)
      enstophy=.5e0_real_kind*global(2)/nodes/(roinf*(uinf/length)**2)
      dissipation=global(3)/nodes
    endif
    if(.not.allocated(max_q)) allocate(max_q(5),min_q(5))
    max_q=global(4:8);min_q=global(9:13)
    call statout(started)
  end subroutine
  subroutine cpp_before_finalize()
    use benchmark_runtime, only: benchmark_gpu_rk_timing_enabled
#ifdef ASTR_BUILD_TESTING
    use validation_io, only: rhs_validation_requested,write_q_validation_snapshot
    if(rhs_validation_requested()) then
      call check(device_state_download(context,device_q,q,bytes(size(q,kind=c_size_t))),'terminal validation q')
      call write_q_validation_snapshot('cpp_final',nstep,1)
    endif
#endif
    if(benchmark_gpu_rk_timing_enabled().or.device_memory_steps) call report_device_memory('final')
    call require(c_associated(context),'missing C++ device context at finalize')
    call release_compact_flow_cpp()
    call release_dynamic_inflow_cpp()
    if(diffterm.and.viscous_form=='me6_opti') call report_me6_cpp(context)
#ifdef ASTR_AIR5_CHEMISTRY
    if(lcomb) then
      call release_air5_mean_statistics_cpp()
      call close_air5_conservation_file_cpp()
    endif
#endif
    call check(device_state_destroy(context),'release device state')
    if(allocated(lower_wall_velocity)) deallocate(lower_wall_velocity)
    host_primitives_current=.false.
    crashfix_ready=.false.
  end subroutine
end module
