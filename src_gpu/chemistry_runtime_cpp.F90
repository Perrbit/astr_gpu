module chemistry_runtime_cpp
  use iso_c_binding, only: c_ptr,c_int,c_double,c_size_t,c_int8_t
  use iso_fortran_env, only: int64
  use mpi
  use astr_precision, only: real_bytes
  use commvar, only: im,jm,km,hm,is,ie,js,je,ks,ke,numq,num_species,num_modequ, &
    flowtype,lfilter,diffterm,nondimen,ndims,conschm,difschm,rkscheme,turbmode,lchardecomp,recon_schem, &
    lihomo,ljhomo,lkhomo,limmbou,lrestart,shkcrt,lcracon,deltat,nstep,rkstep,feqchkpt
  use commarray, only: q,dxi,jacob,x
  use parallel, only: isize,jsize,ksize,mpileft,mpiright,mpidown,mpiup,mpiback,mpifront,mpirank,lio
  use chemistry_air5_data, only: air5_temperature_min_k
  use chemistry_thermo, only: air5_species_vibrational_energy
  use chemistry_flow_runtime
  use chemistry_flow_solver, only: air5_filter_configuration
  use chemistry_compensation, only: air5_compensated,air5_carry
  use chemistry_postshock_boundary, only: get_air5_postshock_boundary
  use chemistry_hbl_boundary, only: get_air5_hbl_boundary,get_air5_incident_top,air5_dynamic_top, &
    get_air5_top_rate,get_air5_top_reference_sound
  use case_model_options, only: get_case_option
  use boundary_high_order, only: high_order_wall_enabled,high_order_wall_axis,high_order_adiabatic_wall_enabled
  use device_state
  use halo_transport, only: device_transport_prepare
  use shock_sensor_cpp, only: shock_prepare_cpp
  use chemistry_device_properties
  use chemistry_device_coupling
  use chemistry_device_boundary
  use chemistry_device_transport
  use chemistry_device_step
  use chemistry_mean_statistics_cpp, only: accumulate_air5_mean_statistics_cpp,set_air5_mean_sampling_cpp
  use chemistry_conservation_cpp, only: sample_air5_conservation_cpp
  implicit none
  private
  public :: initialize_air5_cpp,advance_air5_cpp,check_air5_cpp_state,transfer_air5_aux_cpp
  public :: sample_air5_mean_cpp
  type(air5_rk_options),save :: options
  integer(c_int),save :: first(3,2),last(3,2),regions,physical_low(3),physical_high(3)
  integer(c_int),save :: recovered_core(3,2)
  logical,save :: reacts,open_x,hbl,normal,reports_only,packed,activity,reuse_primitives
  real(c_double),save :: outlet_pressure,wall_temperature
contains
  subroutine initialize_air5_cpp(handle,transport_mode)
    type(c_ptr),intent(in) :: handle
    integer(c_int),intent(in) :: transport_mode
    type(air5_device_options) :: chemistry
    integer(c_int) :: lower(3),upper(3),periodic(3),scalar,flux_mode
    integer :: s,status
    integer :: filter_global(3),filter_offset(3)
    character(32) :: value
    real(c_double) :: left_q(11),right_q(11),farfield_q(11),incident_q(11,2),origin,incident_x
    real(c_double) :: filter_tr(5),filter_v(5)
    real(c_double),allocatable :: inlet(:,:)
    logical :: incident

    call require(real_bytes==8.and.numq==11.and.num_species==5.and.num_modequ==1.and..not.nondimen, &
      'fixed AIR5 requires dimensional FP64 eleven-variable state')
    call require(ndims==3.and.trim(turbmode)=='none'.and.rkscheme=='rk3'.and..not.limmbou.and..not.lcracon, &
      'fixed AIR5 requires 3D laminar RK3 without immersed boundaries or crashfix')
    call require(conschm=='643e'.and.difschm=='643e'.and..not.lchardecomp.and. &
      any(recon_schem==[0,3,5]).and.hm>=3.and.min(im,jm,km)>=6,'fixed AIR5 explicit stencil configuration')
    call require(all(jacob(0:im,0:jm,0:km)>0.0_c_double),'fixed AIR5 requires positive grid Jacobian')
    call require(.not.lrestart,'AIR5 C++ exact and non-exact restart are not yet connected')
    open_x=air5_open_x_flowtype(flowtype);hbl=air5_hbl_flowtype(flowtype)
    normal=air5_normal_shock_flowtype(flowtype);reacts=air5_reacting_flowtype(flowtype)
    periodic=merge(1,0,[lihomo,ljhomo,lkhomo])
    call require(all(periodic==[merge(0,1,open_x.or.hbl),merge(0,1,hbl),1]),'fixed AIR5 boundary topology')
    reuse_primitives=env_choice('ASTR_AIR5_PRIMITIVE_REUSE',[character(9)::'off','chemistry'])==1

    lower=[mpileft,mpidown,mpiback];upper=[mpiright,mpiup,mpifront]
    physical_low=merge(1,0,lower==MPI_PROC_NULL.and.periodic==0)
    physical_high=merge(1,0,upper==MPI_PROC_NULL.and.periodic==0)
    options%upper=[im,jm,km];options%ntypes=3
    where(physical_low==1) options%ntypes=1
    where(physical_high==1) options%ntypes=2
    where(physical_low==1.and.physical_high==1) options%ntypes=4
    first(:,1)=[is,js,ks];last(:,1)=[ie,je,ke]
    options%characteristic_top=hbl.and.air5_dynamic_top()
    regions=1
    if(options%characteristic_top)then
      options%top_rate=get_air5_top_rate()
      options%top_reference_sound=get_air5_top_reference_sound()
      if(physical_high(2)==1)then
        regions=2
        first(:,2)=[physical_low(1),jm,0_c_int];last(:,2)=[im,jm,km]
      endif
    endif
    recovered_core(:,1)=max(first(:,1),1_c_int)
    recovered_core(:,2)=min(last(:,1),options%upper-1_c_int)
    if(reuse_primitives.and.reacts)call require(all(recovered_core(:,2)>=recovered_core(:,1)), &
      'AIR5 chemistry reuse requires a nonempty recovered interior')
    options%halo=hm;options%comm=MPI_COMM_WORLD
    options%symmetric_convection=air5_symmetric_species_convection()
    options%preserve=merge(1_c_int,0_c_int,options%symmetric_convection)
    options%convection_mode=air5_conv_full_state
    if(air5_convection_species_budget())options%convection_mode=air5_conv_species_budget
    if(air5_consistent_species_convection())options%convection_mode=air5_conv_consistent_species
    options%capturing=air5_shock_capturing_enabled();options%shock_threshold=shkcrt
    options%filtering=lfilter;options%conservative_filter=air5_conservative_filter
    options%viscous=diffterm;options%layered=air5_layered_diffusion()
    options%zero_lower_y=hbl.and.physical_low(2)==1
    options%compensated=air5_compensated
    options%wall_axis=-1;options%wall_family=0
    if(high_order_wall_enabled)then
      options%wall_axis=high_order_wall_axis-1
      options%wall_family=merge(1,0,high_order_adiabatic_wall_enabled)
    endif
    do s=1,5
      options%floor(s)=air5_species_vibrational_energy(s,air5_temperature_min_k)
    enddo
    value=''
    call get_case_option('source_evaluation',value,status=status)
    call require(status==1.or.(status==0.and.any(value==[character(32)::'joint','source-only'])), &
      'AIR5 source_evaluation must be joint or source-only')
    chemistry%source_only=merge(1,0,value=='source-only')
    chemistry%source_mode=air5_active_source_mode()
    chemistry%compensated=merge(1,0,options%compensated)
    chemistry%history_enabled=merge(1,0,air5_history_enabled)
    chemistry%preserve_all=options%preserve
    chemistry%rtol=air5_chemistry_rtol;chemistry%atol_factor=air5_chemistry_atol_factor
    chemistry%activity_threshold=air5_activity_threshold
    chemistry%block_size=merge(128,32,env_choice('ASTR_AIR5_CHEMISTRY_BLOCK_SIZE',[character(3)::'128','32'])==0)
    chemistry%specialized_dispatch=env_choice('ASTR_AIR5_CHEMISTRY_DISPATCH',[character(11)::'general','specialized'])
    reports_only=env_choice('ASTR_AIR5_CHEMISTRY_DIAGNOSTICS',[character(12)::'every-step','report-steps'])==1
    packed=env_choice('ASTR_AIR5_CHEMISTRY_REDUCTIONS',[character(8)::'baseline','packed'])==1
    activity=chemistry%activity_threshold>0.0_c_double
    call check(air5_device_prepare(handle),'resident AIR5 fields')
    call check(air5_geometry_prepare(handle),'AIR5 geometry')
    call check(air5_chemistry_prepare(handle,chemistry),'AIR5 chemistry workspace')
    call check(air5_rk_prepare(handle),'AIR5 RK workspace')
    call check(device_transport_prepare(handle,options%comm,lower,upper,[isize,jsize,ksize], &
      11_c_int,transport_mode,0_c_int,periodic),'AIR5 halo transport')
    if(options%symmetric_convection.or.options%characteristic_top)then
      call check(air5_faces_prepare(handle),'AIR5 convection faces')
      call check(air5_symmetric_prepare(handle,options%floor,merge(1_c_int,0_c_int,options%characteristic_top)), &
        'AIR5 symmetric convection workspace')
    endif
    if(.not.options%symmetric_convection)then
      call check(air5_convection_budget_prepare(handle,merge(1,0,options%convection_mode==air5_conv_consistent_species)), &
        'AIR5 convection workspaces')
    endif
    if(options%viscous)then
      flux_mode=env_choice('ASTR_AIR5_FLUX_HALO',[character(14)::'separate','packed','packed-overlap'])
      call check(air5_transport_prepare(handle,merge(1,0,options%layered)),'AIR5 viscous fields')
      call check(air5_diffusion_prepare(handle),'AIR5 diffusion workspaces')
      call check(air5_flux_transport_prepare(handle,flux_mode),'AIR5 flux transport')
    elseif(options%capturing)then
      call check(air5_velocity_gradients_prepare(handle),'AIR5 inviscid gradients')
    endif
    if(options%capturing)call check(shock_prepare_cpp(handle),'AIR5 sensor fields')
    if(options%filtering)then
      scalar=1-env_choice('ASTR_GPU_FILTER_WORKSPACE',[character(6)::'scalar','full'])
      if(options%conservative_filter)then
        call air5_filter_configuration(filter_global,filter_offset,filter_tr,filter_v)
        call check(air5_filter_prepare(handle,filter_tr,filter_v,int(filter_global,c_int), &
          int(filter_offset,c_int),periodic,scalar),'AIR5 conservative filter workspace')
      else
        call check(air5_legacy_linear_prepare(handle,scalar),'AIR5 explicit filter workspace')
      endif
    endif
    if(open_x)then
      call get_air5_postshock_boundary(left_q,right_q,outlet_pressure)
      call check(air5_fixed_boundary_prepare(handle,left_q,right_q),'AIR5 x boundary targets')
    endif
    if(hbl)then
      call get_air5_hbl_boundary(inlet,farfield_q,wall_temperature,origin)
      call get_air5_incident_top(incident,incident_x,incident_q)
      if(physical_low(1)==1)call check(air5_profile_boundary_prepare(handle,inlet,size(inlet,kind=c_size_t)), &
        'AIR5 inlet profile')
      if(physical_high(2)==1)call check(air5_top_boundary_prepare(handle,farfield_q,incident_q,origin,incident_x, &
        merge(1,0,incident),x(0:im,jm,0:km,1),int(im+1,c_size_t)*int(km+1,c_size_t)),'AIR5 top targets')
      if(regions==2)call check(air5_characteristic_top_prepare(handle,x(0:im,jm,0:km,2)-x(0:im,jm-1,0:km,2), &
        int(im+1,c_size_t)*int(km+1,c_size_t)),'AIR5 characteristic top geometry')
    endif
    call check(device_state_upload(handle,device_q,q,8_c_size_t*size(q,kind=c_size_t)),'AIR5 initial q upload')
    call check(device_state_upload(handle,device_metric,dxi,8_c_size_t*size(dxi,kind=c_size_t)),'AIR5 metric upload')
    call check(device_state_upload(handle,device_jacobian,jacob,8_c_size_t*size(jacob,kind=c_size_t)),'AIR5 Jacobian upload')
    call transfer_air5_aux_cpp(handle,.false.)
    call check(air5_prepare_spatial_state(handle,options%upper,options%halo,options%preserve, &
      options%compensated,options%comm,state_boundary),'AIR5 initial spatial state')
    if(lio)write(*,'(a,l1,a,l1,a,l1)')'ASTR_GPU_CPP AIR5 resident RK3; Strang=',reacts, &
      ' compensation=',air5_compensated,' primitive_reuse=',reuse_primitives
  end subroutine

  subroutine transfer_air5_aux_cpp(handle,writing)
    type(c_ptr),intent(in) :: handle
    logical,intent(in) :: writing
    integer(c_size_t) :: bytes
    if(air5_compensated)then
      bytes=8_c_size_t*size(air5_carry,kind=c_size_t)
      if(writing)then
        call check(device_state_download(handle,device_air5_carry,air5_carry,bytes),'AIR5 low part download')
      else
        call check(device_state_upload(handle,device_air5_carry,air5_carry,bytes),'AIR5 low part upload')
      endif
    endif
    if(air5_history_enabled)then
      bytes=8_c_size_t*size(air5_step_history,kind=c_size_t)
      if(writing)then
        call check(device_state_download(handle,device_air5_history,air5_step_history,bytes),'AIR5 chemistry history download')
      else
        call check(device_state_upload(handle,device_air5_history,air5_step_history,bytes),'AIR5 chemistry history upload')
      endif
    endif
  end subroutine

  subroutine sample_air5_mean_cpp()
    call set_air5_mean_sampling_cpp(.true.)
    call accumulate_air5_mean_statistics_cpp(reshape([physical_low,physical_high],[3,2]))
  end subroutine

  subroutine advance_air5_cpp(handle)
    type(c_ptr),intent(in) :: handle
    type(air5_rk_diagnostics) :: diagnostics
    options%dt=deltat
    options%collect=report_step()
    ! Match the complete-step preparation preceding the old chemistry call.
    call check(air5_prepare_spatial_state(handle,options%upper,options%halo,options%preserve, &
      options%compensated,options%comm,state_boundary),'AIR5 complete-step input state')
    if(reacts)then
      call snapshot(handle,'pre_chemistry',1)
      call chemistry_half(handle,1)
      call snapshot(handle,'post_chemistry',1)
    endif
    call sample_air5_conservation_cpp(handle,int(nstep,int64),0)
    do rkstep=1,3
      call check(air5_rk_transport_stage(handle,rkstep,.true.,options,first(:,:regions),last(:,:regions),diagnostics, &
        state_boundary,no_boundary_rhs &
#ifdef ASTR_BUILD_TESTING
        ,observe_raw_convection,observe_conservative_filter &
#endif
        ),'AIR5 transport stage')
      call snapshot(handle,'post_update',rkstep)
    enddo
    call sample_air5_conservation_cpp(handle,int(nstep,int64)+1,1)
    if(reacts)then
      call snapshot(handle,'post_transport',1)
      call chemistry_half(handle,2)
      call snapshot(handle,'post_chemistry',2)
    endif
  end subroutine

  subroutine chemistry_half(handle,half)
    type(c_ptr),intent(in) :: handle
    integer,intent(in) :: half
    type(air5_device_report) :: local
    type(air5_chemistry_global_report) :: global
    integer :: status
    status=air5_chemistry_apply_regions(handle,regions,first(:,:regions),last(:,:regions),0.5_c_double*options%dt, &
      .not.reports_only.or.report_step(),packed,activity,options%comm,local,global)
    if(status/=0.and.local%failure(1)/=0)then
      write(*,'(a,8(i0,1x))') 'AIR5_CHEMISTRY_FAILURE rank/step/half/i/j/k/status/phase=', &
        mpirank,nstep,half,local%failure(2:6)
      write(*,'(a,11(es25.17,1x))') 'AIR5_CHEMISTRY_FAILURE q=',local%failure_q
      write(*,'(a,6(es25.17,1x))') 'AIR5_CHEMISTRY_FAILURE carry=',local%failure_carry
    endif
    call check(status,'AIR5 chemistry half-step')
    if(reuse_primitives)then
      call refresh_after_chemistry(handle,half,recovered_core)
    else
      call refresh_after_chemistry(handle,half)
    endif
    if(lio.and.report_step())then
      write(*,'(a,i0,a,5(es12.4,1x),a,4(i0,1x))') 'AIR5_CHEMISTRY half=',half, &
        ' min_rhos/min_T/max_T/min_Tv/max_Tv=',global%minimums(1:2),global%maximums(1), &
        global%minimums(3),global%maximums(2),' max_acc/rej/rhs/jac=',global%max_counts
      if(activity)write(*,'(a,i0,a,i0,a,i0)') 'AIR5_ACTIVITY half=',half,' skipped_calls=',global%skipped, &
        ' total_calls=',global%total
    endif
  end subroutine

  subroutine refresh_after_chemistry(handle,half,recovered)
    type(c_ptr),intent(in) :: handle
    integer,intent(in) :: half
    integer(c_int),optional,intent(in) :: recovered(3,2)
    if(half==1.or.options%compensated)then
      call check(air5_prepare_spatial_state(handle,options%upper,options%halo,options%preserve, &
        options%compensated,options%comm,state_boundary,recovered),'AIR5 post-chemistry spatial state')
    else
      if(hbl)call check(state_boundary(handle),'AIR5 post-chemistry HBL boundary')
      call check(air5_refresh_physical_primitives(handle,options%upper,options%preserve,options%comm,recovered), &
        'AIR5 post-chemistry physical primitives')
    endif
    if(present(recovered).and.report_step())write(*,'(a,i0,a,i0,a,i0,a,i0,a,i0)') &
      'ASTR_AIR5_PRIMITIVE_SHELL rank=',mpirank,' step=',nstep,' half=',half, &
      ' recovered_nodes=',product(int(recovered(:,2)-recovered(:,1)+1,int64)), &
      ' physical_nodes=',product(int(options%upper+1,int64))
  end subroutine

  integer(c_int) function state_boundary(handle) result(status)
    type(c_ptr),intent(in) :: handle
    integer(c_int) :: failed_stage
    status=0
    if(open_x)status=air5_postshock_boundary_apply(handle,physical_low(1),physical_high(1),normal, &
      outlet_pressure,options%preserve,options%comm)
    if(hbl)status=air5_hbl_boundary_apply(handle,physical_low,physical_high,wall_temperature,options%characteristic_top, &
      options%preserve,options%compensated,options%comm,failed_stage)
  end function

  integer(c_int) function no_boundary_rhs(handle) result(status)
#ifdef ASTR_BUILD_TESTING
    use validation_io, only: rhs_validation_requested,write_scalar_validation_snapshot
#endif
    type(c_ptr),intent(in) :: handle
#ifdef ASTR_BUILD_TESTING
    real(c_double),allocatable :: ratio(:,:,:)
#endif
    status=0
#ifdef ASTR_BUILD_TESTING
    if(options%conservative_filter.and.rhs_validation_requested())then
      call snapshot_rhs(handle,'conv')
      if(.not.options%symmetric_convection)then
        allocate(ratio(-hm:im+hm,-hm:jm+hm,-hm:km+hm))
        call check(device_state_download(handle,device_air5_diffusion_ratio,ratio,8_c_size_t*size(ratio,kind=c_size_t)), &
          'AIR5 validation convection ratio')
        call write_scalar_validation_snapshot('convection_ratio',ratio,rkstep)
      endif
    endif
#endif
  end function

#ifdef ASTR_BUILD_TESTING
  subroutine observe_conservative_filter(handle,label)
    use validation_io, only: rhs_validation_requested,write_air5_filter_validation_snapshot
    type(c_ptr),intent(in) :: handle
    character(*),intent(in) :: label
    real(c_double),allocatable :: high(:,:,:,:),carry(:,:,:,:)
    if(nstep/=0.or.rkstep/=2.or..not.options%compensated)return
    if(.not.rhs_validation_requested())return
    allocate(high(-hm:im+hm,-hm:jm+hm,-hm:km+hm,11),carry(0:im,0:jm,0:km,11))
    call check(device_state_download(handle,device_q,high,8_c_size_t*size(high,kind=c_size_t)),'AIR5 filter validation q')
    call check(device_state_download(handle,device_air5_carry,carry,8_c_size_t*size(carry,kind=c_size_t)), &
      'AIR5 filter validation carry')
    call write_air5_filter_validation_snapshot(label,rkstep,high(0:im,0:jm,0:km,:),carry)
  end subroutine

  integer(c_int) function observe_raw_convection(handle) result(status)
    use validation_io, only: rhs_validation_requested
    type(c_ptr),intent(in) :: handle
    status=0
    if(.not.options%conservative_filter.or..not.rhs_validation_requested())return
    call snapshot(handle,'pre_rhs',rkstep)
    call snapshot_rhs(handle,'conv_raw')
    if(.not.options%symmetric_convection.and.nstep==0.and.rkstep==2)call snapshot_convection_replay(handle)
  end function

  subroutine snapshot_convection_replay(handle)
    use commvar, only: ia,ja,ka
    use shock_sensor_cpp, only: shock_mask_download_cpp
    type(c_ptr),intent(in) :: handle
    real(c_double),allocatable :: primitive(:,:,:,:),save(:,:,:,:)
    integer(c_int8_t),allocatable :: mask(:,:,:)
    integer(c_size_t) :: word_bytes
    integer :: unit
    character(1024) :: prefix
    character(1200) :: name
    allocate(primitive(-hm:im+hm,-hm:jm+hm,-hm:km+hm,10),save(0:im,0:jm,0:km,11))
    word_bytes=8_c_size_t*size(jacob,kind=c_size_t)
    call check(device_state_download(handle,device_rho,primitive(:,:,:,1),word_bytes),'replay density')
    call check(device_state_download(handle,device_velocity,primitive(:,:,:,2:4),3*word_bytes),'replay velocity')
    call check(device_state_download(handle,device_pressure,primitive(:,:,:,5),word_bytes),'replay pressure')
    call check(device_state_download(handle,device_air5_species,primitive(:,:,:,6:10),5*word_bytes),'replay species')
    call check(device_state_download(handle,device_save,save,8_c_size_t*size(save,kind=c_size_t)),'replay RK origin')
    call get_environment_variable('ASTR_VALIDATION_RHS_PREFIX',prefix)
    write(name,'(a,".rk2_replay.rank",i8.8,".bin")')trim(prefix),mpirank
    open(newunit=unit,file=trim(name),access='stream',form='unformatted',status='new',action='write')
    write(unit) [ia,ja,ka],merge(1,0,[lihomo,ljhomo,lkhomo]),options%floor
    write(unit) [im,jm,km,hm],merge(1,0,options%capturing),0.25_c_double*options%dt,q,save,jacob,dxi,primitive
    if(options%capturing)then
      allocate(mask(0:im,0:jm,0:km))
      call check(shock_mask_download_cpp(handle,mask,size(mask,kind=c_size_t)),'replay shock mask')
      write(unit) mask
    endif
    close(unit)
  end subroutine

  subroutine snapshot_rhs(handle,label)
    use commarray, only: qrhs
    use validation_io, only: write_rhs_validation_snapshot
    type(c_ptr),intent(in) :: handle
    character(*),intent(in) :: label
    call check(device_state_download(handle,device_rhs,qrhs,8_c_size_t*size(qrhs,kind=c_size_t)), &
      'AIR5 validation convection RHS')
    call write_rhs_validation_snapshot(label,nstep,rkstep)
  end subroutine
#endif

  subroutine check_air5_cpp_state(handle)
    type(c_ptr),intent(in) :: handle
    integer :: local,global,ierr
    call check(air5_device_check(handle,local),'AIR5 state status')
    call MPI_Allreduce(local,global,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
    call check(ierr,'AIR5 state status reduction')
    call check(global,'AIR5 admissible state')
  end subroutine

  logical function report_step()
    report_step=nstep==0
    if(feqchkpt>0)report_step=report_step.or.mod(nstep,feqchkpt)==0
  end function

  integer function env_choice(name,values) result(choice)
    character(*),intent(in) :: name,values(:)
    character(64) :: value
    integer :: status,i,local(2),range(2),ierr
    call get_environment_variable(name,value,status=status)
    choice=-1
    if(status==1)choice=0
    if(status==0)then
      do i=1,size(values)
        if(value==values(i))choice=i-1
      enddo
    endif
    local=[choice,-choice]
    call MPI_Allreduce(local,range,2,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
    call check(ierr,name)
    call require(range(1)>=0.and.range(1)==-range(2),'invalid or rank-inconsistent '//name)
  end function

  subroutine snapshot(handle,label,stage)
    use validation_io, only: rhs_validation_requested,write_q_validation_snapshot
    type(c_ptr),intent(in) :: handle
    character(*),intent(in) :: label
    integer,intent(in) :: stage
    if(.not.rhs_validation_requested())return
    call check(device_state_download(handle,device_q,q,8_c_size_t*size(q,kind=c_size_t)),'AIR5 validation q')
    call write_q_validation_snapshot(label,nstep,stage)
  end subroutine

  subroutine require(ok,label)
    logical,intent(in) :: ok
    character(*),intent(in) :: label
    if(.not.ok)call check(1,label)
  end subroutine

  subroutine check(status,label)
    integer,intent(in) :: status
    character(*),intent(in) :: label
    integer :: ignored
    if(status==0)return
    write(*,'(a,i0,3a,i0)') 'ASTR_GPU_CPP AIR5 rank=',mpirank,' ',label,' status=',status
    call MPI_Abort(MPI_COMM_WORLD,abs(status),ignored)
  end subroutine
end module
