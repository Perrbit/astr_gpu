module chemistry_device_step
  use iso_c_binding, only: c_ptr,c_int,c_int64_t,c_double
  use mpi, only: MPI_Allreduce,MPI_Abort,MPI_INTEGER,MPI_DOUBLE_PRECISION,MPI_MIN,MPI_MAX,MPI_SUM
  use device_state, only: device_q,device_shock_sensor,device_air5_diffusion_ratio,device_air5_diffusion_energy_ratio, &
    device_air5_convection_fluid_ratio,device_invalid_argument,c_device_real_kind
  use halo_transport, only: device_exchange_axis
  use fluid_explicit, only: fluid_gradient_basic_boundary
  use shock_sensor_cpp, only: shock_raw_cpp,shock_expand_cpp
  use chemistry_device_properties, only: air5_device_primitives,air5_device_primitives_regions,air5_device_check
  use chemistry_device_transport, only: air5_filter_apply,air5_legacy_filter_apply,air5_compensation_axis
  use chemistry_device_transport, only: air5_gradients,air5_diffusive_flux,air5_flux_exchange, &
    air5_diffusion_begin,air5_diffusion_budget,air5_diffusion_rhs,air5_diffusion_report,air5_diffusion_read_report
  use chemistry_device_transport, only: air5_raw_convection,air5_convection_budget_begin,air5_convection_budget, &
    air5_convection_rhs,air5_convection_report,air5_convection_read_report,air5_rk_update, &
    air5_conv_full_state,air5_conv_species_budget,air5_conv_consistent_species
  implicit none
  private
  public :: air5_prepare_spatial_state,air5_prepare_rk_stage
  public :: air5_add_viscous_rhs,air5_limit_viscous_rhs
  public :: air5_prepare_shock_mask,air5_assemble_raw_convection,air5_limit_convection_rhs
  public :: air5_rk_transport_stage
  type,public :: air5_rk_options
    integer(c_int) :: upper(3),ntypes(3),halo,comm,convection_mode,preserve,wall_axis,wall_family
    real(c_double) :: dt,floor(5),shock_threshold
    logical :: capturing,viscous,layered,zero_lower_y,collect,filtering,conservative_filter,compensated
  end type
  type,public :: air5_rk_diagnostics
    type(air5_convection_report) :: convection
    type(air5_diffusion_report) :: diffusion
    integer(c_int) :: convection_active(10),diffusion_active
    real(c_double) :: convection_minimum(10),diffusion_minimum
  end type
  ! All ranks call the boundary phase; a failure status is already collective.
  abstract interface
    integer(c_int) function boundary_step(handle) result(status)
      import c_ptr,c_int
      type(c_ptr),intent(in) :: handle
    end function
  end interface
contains
  ! The first range is the interior transport domain; additional disjoint
  ! ranges are boundary nodes whose convection is supplied by boundary_rhs.
  integer(c_int) function air5_rk_transport_stage(handle,stage,prepared,options,first,last,diagnostics, &
      state_boundary,boundary_rhs) result(status)
    type(c_ptr),intent(in) :: handle
    integer(c_int),intent(in) :: stage,first(:,:),last(:,:)
    logical,intent(in) :: prepared
    type(air5_rk_options),intent(in) :: options
    type(air5_rk_diagnostics),intent(out) :: diagnostics
    procedure(boundary_step) :: state_boundary,boundary_rhs
    real(c_double),parameter :: a(3)=[1.0_c_double,0.75_c_double,1.0_c_double/3.0_c_double]
    real(c_double),parameter :: b(3)=[0.0_c_double,0.25_c_double,2.0_c_double/3.0_c_double]
    real(c_double),parameter :: c(3)=[1.0_c_double,0.25_c_double,2.0_c_double/3.0_c_double]
    integer(c_int) :: region,comm
    logical :: initial_stage

    comm=options%comm
    initial_stage=stage==1
    status=air5_prepare_rk_stage(handle,options%upper,options%halo,stage,prepared,options%filtering, &
      options%conservative_filter,options%floor,options%wall_axis,options%wall_family,options%preserve, &
      options%compensated,comm,state_boundary)
    if(status/=0)return
    if(options%capturing)call air5_prepare_shock_mask(handle,options%ntypes,options%shock_threshold,comm)
    status=air5_assemble_raw_convection(handle,options%ntypes,first(:,1),last(:,1),options%capturing,.false.,comm)
    if(status/=0)return
    status=air5_limit_convection_rhs(handle,options%ntypes,first(:,1),last(:,1),options%floor,a(stage),b(stage), &
      c(stage)*options%dt,initial_stage,options%capturing,options%convection_mode,options%collect,comm, &
      diagnostics%convection_active,diagnostics%convection_minimum,diagnostics%convection)
    if(status/=0)return
    status=boundary_rhs(handle)
    if(status/=0)return
    if(options%viscous)then
      status=air5_add_viscous_rhs(handle,options%ntypes,int(size(first,2),c_int),first,last,options%floor, &
        a(stage),b(stage),c(stage)*options%dt,initial_stage,options%zero_lower_y,options%layered,options%collect,comm, &
        diagnostics%diffusion_active,diagnostics%diffusion_minimum,diagnostics%diffusion)
      if(status/=0)return
    endif
    do region=1,size(first,2)
      call check_api(air5_rk_update(handle,first(:,region),last(:,region),options%dt,a(stage),b(stage),c(stage), &
        merge(1_c_int,0_c_int,initial_stage)),comm,'transport RK update')
    enddo
    status=state_boundary(handle)
    if(status/=0.or.stage/=3)return
    call check_api(air5_device_primitives(handle,[0_c_int,0_c_int,0_c_int],options%upper,options%preserve), &
      comm,'post-transport interior primitives')
    status=collect_status(handle,comm)
  end function

  integer(c_int) function air5_prepare_rk_stage(handle,last,halo,rkstep,prepared,filtering,conservative_filter, &
      floor,wall_axis,wall_family,preserve,compensated,comm,boundary) result(status)
    type(c_ptr),intent(in) :: handle
    integer(c_int),intent(in) :: last(3),halo,rkstep,wall_axis,wall_family,preserve,comm
    logical,intent(in) :: prepared,filtering,conservative_filter,compensated
    real(c_double),intent(in) :: floor(5)
    procedure(boundary_step) :: boundary

    status=0_c_int
    if(filtering)then
      if(conservative_filter)then
        call check_api(air5_filter_apply(handle,preserve,status),comm,'conservative filter')
      else
        call check_api(air5_legacy_filter_apply(handle,floor,wall_axis,wall_family,status),comm,'legacy filter')
      endif
      if(status/=0_c_int)return
    elseif(rkstep==1_c_int.and.prepared)then
      return
    endif
    status=air5_prepare_spatial_state(handle,last,halo,preserve,compensated,comm,boundary)
  end function

  subroutine air5_prepare_shock_mask(handle,ntypes,threshold,comm)
    type(c_ptr),intent(in) :: handle
    integer(c_int),intent(in) :: ntypes(3),comm
    real(c_double),intent(in) :: threshold
    integer(c_int) :: low(3),high(3),axis

    low=merge(1_c_int,0_c_int,ntypes==1.or.ntypes==4)
    high=merge(1_c_int,0_c_int,ntypes==2.or.ntypes==4)
    call check_api(fluid_gradient_basic_boundary(handle,low,high),comm,'sensor gradient')
    call check_api(shock_raw_cpp(handle,low,high),comm,'raw shock sensor')
    do axis=0_c_int,2_c_int
      call check_api(device_exchange_axis(handle,device_shock_sensor,axis,0_c_int),comm,'shock sensor halo')
    enddo
    call check_api(shock_expand_cpp(handle,low,high,real(threshold,c_device_real_kind)),comm,'shock sensor expansion')
  end subroutine

  integer(c_int) function air5_assemble_raw_convection(handle,ntypes,first,last,capturing,symmetric,comm) result(status)
    type(c_ptr),intent(in) :: handle
    integer(c_int),intent(in) :: ntypes(3),first(3),last(3),comm
    logical,intent(in) :: capturing,symmetric
    call check_api(air5_raw_convection(handle,ntypes,first,last,merge(1_c_int,0_c_int,capturing), &
      merge(1_c_int,0_c_int,symmetric)),comm,'raw convection')
    status=collect_status(handle,comm)
  end function

  integer(c_int) function air5_limit_convection_rhs(handle,ntypes,first,last,floor,a,b,cdt, &
      first_stage,capturing,mode,collect,comm,active,minimum,report) result(status)
    type(c_ptr),intent(in) :: handle
    integer(c_int),intent(in) :: ntypes(3),first(3),last(3),mode,comm
    real(c_double),intent(in) :: floor(5),a,b,cdt
    logical,intent(in) :: first_stage,capturing,collect
    integer(c_int),intent(out) :: active(10)
    real(c_double),intent(out) :: minimum(10)
    type(air5_convection_report),intent(out) :: report
    integer(c_int) :: phase,start_phase,end_phase,species_budget,axis,field,shock
    real(c_double) :: local_minimum(10)
    integer :: ierr

    if(mode<air5_conv_full_state.or.mode>air5_conv_consistent_species) &
      call check_api(device_invalid_argument,comm,'convection limiter mode')
    active=0_c_int;minimum=huge(1.0_c_double)
    report%claimed=0_c_int;report%location=0_c_int;report%active=0_c_int
    report%minimum_bits=transfer(huge(1.0_c_double),0_c_int64_t);report%values=0.0_c_double
    shock=merge(1_c_int,0_c_int,capturing)
    start_phase=merge(1_c_int,0_c_int,mode==air5_conv_consistent_species)
    end_phase=merge(2_c_int,0_c_int,mode==air5_conv_consistent_species)
    do phase=start_phase,end_phase
      species_budget=merge(1_c_int,0_c_int,mode==air5_conv_species_budget.or.phase==2)
      call check_api(air5_convection_budget_begin(handle,phase),comm,'convection budget reset')
      call check_api(air5_convection_budget(handle,ntypes,first,last,floor,a,b,cdt, &
        merge(1_c_int,0_c_int,first_stage),shock,species_budget,phase, &
        merge(1_c_int,0_c_int,collect.and.phase==end_phase)),comm,'convection budget')
      status=collect_status(handle,comm)
      if(status/=0_c_int)then
        call check_api(air5_convection_read_report(handle,report),comm,'convection baseline failure')
        return
      endif
      if(collect.and.phase==end_phase)then
        call check_api(air5_convection_read_report(handle,report),comm,'convection limiter report')
        local_minimum=transfer(report%minimum_bits,local_minimum)
        call MPI_Allreduce(report%active,active,10,MPI_INTEGER,MPI_SUM,int(comm),ierr)
        call check_api(int(ierr,c_int),comm,'convection limiter counts')
        call MPI_Allreduce(local_minimum,minimum,10,MPI_DOUBLE_PRECISION,MPI_MIN,int(comm),ierr)
        call check_api(int(ierr,c_int),comm,'convection limiter minima')
      endif
      field=merge(device_air5_convection_fluid_ratio,device_air5_diffusion_ratio,phase==1)
      do axis=0_c_int,2_c_int
        call check_api(device_exchange_axis(handle,field,axis,0_c_int),comm,'convection ratio halo')
      enddo
    enddo
    call check_api(air5_convection_rhs(handle,ntypes,first,last,shock,end_phase),comm,'limited convection RHS')
    status=collect_status(handle,comm)
  end function

  integer(c_int) function air5_prepare_spatial_state(handle,last,halo,preserve,compensated,comm,boundary) result(status)
    type(c_ptr),intent(in) :: handle
    integer(c_int),intent(in) :: last(3),halo,preserve,comm
    logical,intent(in) :: compensated
    procedure(boundary_step) :: boundary
    integer(c_int) :: axis,lo(3,6),hi(3,6)
    integer(c_int),parameter :: origin(3)=0_c_int

    status=boundary(handle)
    if(status/=0_c_int)return
    call check_api(air5_device_primitives(handle,origin,last,preserve),comm,'interior primitives')
    status=collect_status(handle,comm)
    if(status/=0_c_int)return

    do axis=1_c_int,3_c_int
      if(compensated)call check_api(air5_compensation_axis(handle,axis,0_c_int),comm,'carry exchange')
      call check_api(device_exchange_axis(handle,device_q,axis-1_c_int,1_c_int),comm,'shared node exchange')
      if(compensated)call check_api(air5_compensation_axis(handle,axis,1_c_int),comm,'carry normalization')
    enddo
    ! Later shared-node updates can change a transverse face. Refresh halos
    ! from the final aliases before rebuilding primitives at faces and ghosts.
    do axis=1_c_int,3_c_int
      call check_api(device_exchange_axis(handle,device_q,axis-1_c_int,0_c_int),comm,'final halo exchange')
      lo(:,2*axis-1)=origin;hi(:,2*axis-1)=last
      lo(:,2*axis)=origin;hi(:,2*axis)=last
      lo(axis,2*axis-1)=-halo;hi(axis,2*axis-1)=0_c_int
      lo(axis,2*axis)=last(axis);hi(axis,2*axis)=last(axis)+halo
    enddo
    call check_api(air5_device_primitives_regions(handle,6_c_int,lo,hi,preserve),comm,'face primitives')
    status=collect_status(handle,comm)
  end function

  ! The input RHS contains the already limited convection, including the
  ! optional dynamic-top convection. No RK state is committed here.
  integer(c_int) function air5_add_viscous_rhs(handle,ntypes,regions,first,last,floor,a,b,cdt, &
      first_stage,zero_lower_y,layered,collect,comm,limited,minimum,report) result(status)
    type(c_ptr),intent(in) :: handle
    integer(c_int),intent(in) :: ntypes(3),regions,first(3,regions),last(3,regions),comm
    real(c_double),intent(in) :: floor(5),a,b,cdt
    logical,intent(in) :: first_stage,zero_lower_y,layered,collect
    integer(c_int),intent(out) :: limited
    real(c_double),intent(out) :: minimum
    type(air5_diffusion_report),intent(out) :: report
    integer(c_int) :: low(3),high(3)

    limited=0_c_int;minimum=huge(minimum)
    report=air5_diffusion_report(0_c_int,[0_c_int,0_c_int,0_c_int,0_c_int],0_c_int, &
      transfer(minimum,report%minimum_bits),[0.0_c_double,0.0_c_double,0.0_c_double,0.0_c_double,0.0_c_double])
    low=merge(1_c_int,0_c_int,ntypes==1.or.ntypes==4)
    high=merge(1_c_int,0_c_int,ntypes==2.or.ntypes==4)
    call check_api(air5_gradients(handle,low,high,merge(1_c_int,0_c_int,zero_lower_y)),comm,'viscous gradients')
    call check_api(air5_diffusive_flux(handle),comm,'viscous flux')
    status=collect_status(handle,comm)
    if(status/=0_c_int)return
    call check_api(air5_flux_exchange(handle),comm,'viscous flux halo')
    status=air5_limit_viscous_rhs(handle,ntypes,regions,first,last,floor,a,b,cdt, &
      first_stage,layered,collect,comm,limited,minimum,report)
  end function

  integer(c_int) function air5_limit_viscous_rhs(handle,ntypes,regions,first,last,floor,a,b,cdt, &
      first_stage,layered,collect,comm,limited,minimum,report) result(status)
    type(c_ptr),intent(in) :: handle
    integer(c_int),intent(in) :: ntypes(3),regions,first(3,regions),last(3,regions),comm
    real(c_double),intent(in) :: floor(5),a,b,cdt
    logical,intent(in) :: first_stage,layered,collect
    integer(c_int),intent(out) :: limited
    real(c_double),intent(out) :: minimum
    type(air5_diffusion_report),intent(out) :: report
    integer(c_int) :: axis,part,phase,field
    real(c_double) :: local_minimum
    integer :: ierr

    limited=0_c_int;minimum=huge(minimum)
    report=air5_diffusion_report(0_c_int,[0_c_int,0_c_int,0_c_int,0_c_int],0_c_int, &
      transfer(minimum,report%minimum_bits),[0.0_c_double,0.0_c_double,0.0_c_double,0.0_c_double,0.0_c_double])
    call check_api(air5_diffusion_begin(handle),comm,'diffusion budget reset')
    do phase=0_c_int,merge(1_c_int,0_c_int,layered)
      do part=1_c_int,regions
        call check_api(air5_diffusion_budget(handle,ntypes,first(:,part),last(:,part),floor,a,b,cdt, &
          merge(1_c_int,0_c_int,first_stage),phase,merge(1_c_int,0_c_int,collect)),comm,'diffusion budget')
      enddo
      if(phase==0_c_int)then
        status=collect_status(handle,comm)
        if(collect.or.status/=0_c_int) &
          call check_api(air5_diffusion_read_report(handle,report),comm,'diffusion report')
        if(status/=0_c_int)return
        if(collect)then
          local_minimum=transfer(report%minimum_bits,minimum)
          call MPI_Allreduce(report%limited,limited,1,MPI_INTEGER,MPI_SUM,int(comm),ierr)
          call check_api(int(ierr,c_int),comm,'diffusion limiter count')
          call MPI_Allreduce(local_minimum,minimum,1,MPI_DOUBLE_PRECISION,MPI_MIN,int(comm),ierr)
          call check_api(int(ierr,c_int),comm,'diffusion limiter minimum')
        endif
      endif
      field=merge(device_air5_diffusion_ratio,device_air5_diffusion_energy_ratio,phase==0_c_int)
      do axis=0_c_int,2_c_int
        call check_api(device_exchange_axis(handle,field,axis,0_c_int),comm,'diffusion ratio halo')
      enddo
    enddo
    do part=1_c_int,regions
      do axis=1_c_int,3_c_int
        call check_api(air5_diffusion_rhs(handle,ntypes,first(:,part),last(:,part),axis),comm,'diffusion divergence')
      enddo
    enddo
  end function

  integer(c_int) function collect_status(handle,comm) result(status)
    type(c_ptr),intent(in) :: handle
    integer(c_int),intent(in) :: comm
    integer(c_int) :: local
    integer :: ierr
    call check_api(air5_device_check(handle,local),comm,'physics status')
    call MPI_Allreduce(local,status,1,MPI_INTEGER,MPI_MAX,int(comm),ierr)
    call check_api(int(ierr,c_int),comm,'physics status reduction')
  end function

  subroutine check_api(code,comm,phase)
    integer(c_int),intent(in) :: code,comm
    character(*),intent(in) :: phase
    integer :: ierr
    if(code==0_c_int)return
    write(*,'(A,A,A,I0)')'AIR5 ',phase,' failed, code=',code
    call MPI_Abort(int(comm),abs(int(code)),ierr)
  end subroutine
end module
