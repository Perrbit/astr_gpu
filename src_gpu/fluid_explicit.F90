module fluid_explicit
  use iso_c_binding, only: c_int,c_ptr,c_double,c_int64_t
  use device_state, only: c_device_real_kind
  implicit none
  private
  public :: fluid_prepare,fluid_primitives,fluid_conservative,fluid_gradient,fluid_check,fluid_cfl
  public :: fluid_gradient_basic_boundary
  public :: fluid_gradient_high_wall
  public :: fluid_gradient_high_open
  public :: fluid_viscous_high_open
  public :: fluid_flux_divergence_high_wall,fluid_wall_rhs
  public :: fluid_sbp63_gradient,fluid_sbp63_convection,fluid_sbp63_diffusion
  public :: fluid_keep_accumulate_basic
  public :: fluid_rhs_clear,fluid_flux_divergence_basic,fluid_refresh_primitives_boundary
  public :: fluid_channel_geometry_prepare,fluid_uniform_force,fluid_channel_statistics
  public :: fluid_rti_source
  public :: fluid_flat_geometry_prepare,fluid_flat_statistics
  public :: fluid_filter_axis_basic,fluid_filter_component_axis_basic
  public :: fluid_filter_axis_high_wall
  public :: fluid_refresh_primitives
  public :: fluid_initialize_exterior
  public :: fluid_rhs_prepare,fluid_convection,fluid_viscous_flux,fluid_viscous_divergence
  public :: fluid_rhs_prepare_mixed_viscous
  public :: fluid_prepare_diagnostic_gradient,fluid_diagnostic_gradient
  public :: fluid_keep_convection
  public :: fluid_filter_prepare,fluid_filter_axis,fluid_filter_commit,fluid_rk_save,fluid_rk_update
  public :: fluid_statistics,fluid_cfl_diagnostics
  public :: fluid_adaptive_energy_sum
  public :: fluid_filter_prepare_scalar,fluid_filter_workspace_components
  public :: fluid_filter_component_axis,fluid_filter_component_commit
  public :: fluid_viscous_wall_band,fluid_me6_prepare,fluid_me6_diffusion,fluid_me6_metrics
  interface
    integer(c_int) function fluid_viscous_high_open(handle,low,high) bind(C,name='astr_device_fluid_viscous_high_open')
      import c_int,c_ptr
      type(c_ptr),value :: handle
      integer(c_int),intent(in) :: low(3),high(3)
    end function
    integer(c_int) function fluid_gradient_high_open(handle,low,high) bind(C,name='astr_device_fluid_gradient_high_open')
      import c_int,c_ptr
      type(c_ptr),value :: handle
      integer(c_int),intent(in) :: low(3),high(3)
    end function
    integer(c_int) function fluid_prepare_diagnostic_gradient(handle) &
        bind(C,name='astr_device_fluid_prepare_diagnostic_gradient')
      import c_int,c_ptr
      type(c_ptr),value :: handle
    end function
    integer(c_int) function fluid_diagnostic_gradient(handle,low,high) bind(C,name='astr_device_fluid_diagnostic_gradient')
      import c_int,c_ptr
      type(c_ptr),value :: handle
      integer(c_int),intent(in) :: low(3),high(3)
    end function
    integer(c_int) function fluid_rhs_prepare_mixed_viscous(handle) bind(C,name='astr_device_fluid_rhs_prepare_mixed_viscous')
      import c_int,c_ptr
      type(c_ptr),value :: handle
    end function
    integer(c_int) function fluid_sbp63_gradient(handle,total,start,temperature) &
        bind(C,name='astr_device_fluid_sbp63_gradient')
      import c_ptr,c_int,c_device_real_kind
      type(c_ptr),value :: handle
      integer(c_int),value :: total,start
      real(c_device_real_kind),intent(in) :: temperature(2)
    end function
    integer(c_int) function fluid_sbp63_convection(handle,total,start,gamma) &
        bind(C,name='astr_device_keep_sbp63_convection')
      import c_ptr,c_int,c_device_real_kind
      type(c_ptr),value :: handle
      integer(c_int),value :: total,start
      real(c_device_real_kind),value :: gamma
    end function
    integer(c_int) function fluid_sbp63_diffusion(handle,total,start,temperature,re,pr,c5,tc,tc1) &
        bind(C,name='astr_device_fluid_sbp63_diffusion')
      import c_ptr,c_int,c_device_real_kind
      type(c_ptr),value :: handle
      integer(c_int),value :: total,start
      real(c_device_real_kind),intent(in) :: temperature(2)
      real(c_device_real_kind),value :: re,pr,c5,tc,tc1
    end function
    integer(c_int) function fluid_filter_axis_high_wall(handle,axis,component,lower,upper,adiabatic) &
        bind(C,name='astr_device_fluid_filter_axis_high_wall')
      import c_int,c_ptr
      type(c_ptr),value :: handle
      integer(c_int),value :: axis,component,lower,upper,adiabatic
    end function
    integer(c_int) function fluid_flux_divergence_high_wall(handle,viscous,axis,low,high,first,last,adiabatic) &
        bind(C,name='astr_device_fluid_flux_divergence_high_wall')
      import c_int,c_ptr
      type(c_ptr),value :: handle
      integer(c_int),value :: viscous,axis,adiabatic
      integer(c_int),intent(in) :: low(3),high(3),first(3),last(3)
    end function
    integer(c_int) function fluid_wall_rhs(handle,axis,low,high,temperature,c2,c6,kind) &
        bind(C,name='astr_device_fluid_wall_rhs')
      import c_int,c_ptr,c_device_real_kind
      type(c_ptr),value :: handle
      integer(c_int),value :: axis,kind
      integer(c_int),intent(in) :: low(3),high(3)
      real(c_device_real_kind),intent(in) :: temperature(2)
      real(c_device_real_kind),value :: c2,c6
    end function
    integer(c_int) function fluid_gradient_high_wall(handle,axis,low,high) bind(C,name='astr_device_fluid_gradient_high_wall')
      import c_int,c_ptr
      type(c_ptr),value :: handle
      integer(c_int),value :: axis
      integer(c_int),intent(in) :: low(3),high(3)
    end function
    integer(c_int) function fluid_rti_source(handle) bind(C,name='astr_device_fluid_rti_source')
      import c_int,c_ptr
      type(c_ptr),value :: handle
    end function
    integer(c_int) function fluid_flat_geometry_prepare(handle,lower_wall) bind(C,name='astr_device_fluid_flat_geometry_prepare')
      import c_ptr,c_int
      type(c_ptr),value :: handle
      integer(c_int),value :: lower_wall
    end function
    integer(c_int) function fluid_flat_statistics(handle,reynolds,tc,tc1,prandtl,c5,lower_wall,values) &
        bind(C,name='astr_device_fluid_flat_statistics')
      import c_ptr,c_int,c_device_real_kind
      type(c_ptr),value :: handle
      real(c_device_real_kind),value :: reynolds,tc,tc1,prandtl,c5
      integer(c_int),value :: lower_wall
      real(c_device_real_kind),intent(out) :: values(5)
    end function
    integer(c_int) function fluid_initialize_exterior(handle,low,high) bind(C,name='astr_device_fluid_initialize_exterior')
      import c_int,c_ptr
      type(c_ptr),value :: handle
      integer(c_int),intent(in) :: low(3),high(3)
    end function
    integer(c_int) function fluid_me6_metrics(parent,values) bind(C,name='astr_device_me6_metrics')
      import c_ptr,c_int,c_int64_t
      type(c_ptr),value :: parent
      integer(c_int64_t),intent(out) :: values(4)
    end function
    integer(c_int) function fluid_me6_prepare(parent,communicator,lower,upper,parts,periodic,first,last,inner_first,inner_last, &
        mode,ci,cd,cp) bind(C,name='astr_device_me6_prepare')
      import c_ptr,c_int,c_device_real_kind
      type(c_ptr),value :: parent
      integer(c_int),value :: communicator,mode
      integer(c_int),intent(in) :: lower(3),upper(3),parts(3),periodic(3),first(3),last(3),inner_first(3),inner_last(3)
      real(c_device_real_kind),intent(in) :: ci(9,6),cd(9,6),cp(9,6)
    end function
    integer(c_int) function fluid_me6_diffusion(parent,re,pr,c5,tc,tc1) bind(C,name='astr_device_me6_diffusion')
      import c_ptr,c_int,c_device_real_kind
      type(c_ptr),value :: parent
      real(c_device_real_kind),value :: re,pr,c5,tc,tc1
    end function
    integer(c_int) function fluid_viscous_wall_band(handle,low,high,first,last) &
        bind(C,name='astr_device_fluid_viscous_wall_band')
      import c_ptr,c_int
      type(c_ptr),value :: handle
      integer(c_int),intent(in) :: low(3),high(3),first(3),last(3)
    end function
    integer(c_int) function fluid_filter_axis_basic(handle,axis,lower,upper) bind(C,name='astr_device_fluid_filter_axis_basic')
      import c_ptr,c_int
      type(c_ptr),value :: handle
      integer(c_int),value :: axis,lower,upper
    end function
    integer(c_int) function fluid_filter_component_axis_basic(handle,axis,component,lower,upper) &
        bind(C,name='astr_device_fluid_filter_component_axis_basic')
      import c_ptr,c_int
      type(c_ptr),value :: handle
      integer(c_int),value :: axis,component,lower,upper
    end function
    integer(c_int) function fluid_channel_geometry_prepare(handle) bind(C,name='astr_device_fluid_channel_geometry_prepare')
      import c_int,c_ptr
      type(c_ptr),value :: handle
    end function
    integer(c_int) function fluid_uniform_force(handle,force) bind(C,name='astr_device_fluid_uniform_force')
      import c_int,c_ptr,c_device_real_kind
      type(c_ptr),value :: handle
      real(c_device_real_kind),intent(in) :: force(3)
    end function
    integer(c_int) function fluid_channel_statistics(handle,reynolds,tc,tc1,lower,upper,values) &
        bind(C,name='astr_device_fluid_channel_statistics')
      import c_int,c_ptr,c_device_real_kind
      type(c_ptr),value :: handle
      real(c_device_real_kind),value :: reynolds,tc,tc1
      integer(c_int),value :: lower,upper
      real(c_device_real_kind),intent(out) :: values(3)
    end function
    integer(c_int) function fluid_keep_convection(handle,gamma) bind(C,name='astr_device_keep_explicit')
      import c_int,c_ptr,c_device_real_kind
      type(c_ptr),value :: handle
      real(c_device_real_kind),value :: gamma
    end function
    integer(c_int) function fluid_keep_accumulate_basic(handle,gamma,types,first,last) &
        bind(C,name='astr_device_keep_explicit_accumulate_basic')
      import c_int,c_ptr,c_device_real_kind
      type(c_ptr),value :: handle
      real(c_device_real_kind),value :: gamma
      integer(c_int),intent(in) :: types(3),first(3),last(3)
    end function
    integer(c_int) function fluid_adaptive_energy_sum(handle,values) bind(C,name='astr_device_fluid_adaptive_energy_sum')
      import c_int,c_ptr,c_device_real_kind
      type(c_ptr),value :: handle
      real(c_device_real_kind),intent(out) :: values(2)
    end function
    integer(c_int) function fluid_statistics(handle,reynolds,tc,tc1,values) bind(C,name='astr_device_fluid_statistics')
      import c_int,c_ptr,c_device_real_kind
      type(c_ptr),value :: handle
      real(c_device_real_kind),value :: reynolds,tc,tc1
      real(c_device_real_kind),intent(out) :: values(13)
    end function
    integer(c_int) function fluid_cfl_diagnostics(handle,nondimensional,mach,gamma,rgas,global_upper,origin,ny, &
        values,keys,profile) bind(C,name='astr_device_fluid_cfl_diagnostics')
      import c_int,c_ptr,c_device_real_kind,c_int64_t
      type(c_ptr),value :: handle
      integer(c_int),value :: nondimensional,ny
      real(c_device_real_kind),value :: mach,gamma,rgas
      integer(c_int),intent(in) :: global_upper(3),origin(3)
      real(c_device_real_kind),intent(out) :: values(4)
      integer(c_int64_t),intent(out) :: keys(4)
      real(c_device_real_kind),intent(inout) :: profile(*)
    end function
    integer(c_int) function fluid_filter_prepare(handle) bind(C,name='astr_device_fluid_filter_prepare')
      import c_int,c_ptr
      type(c_ptr),value :: handle
    end function
    integer(c_int) function fluid_filter_prepare_scalar(handle) bind(C,name='astr_device_fluid_filter_prepare_scalar')
      import c_int,c_ptr
      type(c_ptr),value :: handle
    end function
    integer(c_int) function fluid_filter_workspace_components(handle) bind(C,name='astr_device_fluid_filter_workspace_components')
      import c_int,c_ptr
      type(c_ptr),value :: handle
    end function
    integer(c_int) function fluid_filter_component_axis(handle,axis,component) &
        bind(C,name='astr_device_fluid_filter_component_axis')
      import c_int,c_ptr
      type(c_ptr),value :: handle
      integer(c_int),value :: axis,component
    end function
    integer(c_int) function fluid_filter_component_commit(handle,component) &
        bind(C,name='astr_device_fluid_filter_component_commit')
      import c_int,c_ptr
      type(c_ptr),value :: handle
      integer(c_int),value :: component
    end function
    integer(c_int) function fluid_filter_axis(handle,axis) bind(C,name='astr_device_fluid_filter_axis')
      import c_int,c_ptr
      type(c_ptr),value :: handle
      integer(c_int),value :: axis
    end function
    integer(c_int) function fluid_filter_commit(handle) bind(C,name='astr_device_fluid_filter_commit')
      import c_int,c_ptr
      type(c_ptr),value :: handle
    end function
    integer(c_int) function fluid_rk_save(handle) bind(C,name='astr_device_fluid_rk_save')
      import c_int,c_ptr
      type(c_ptr),value :: handle
    end function
    integer(c_int) function fluid_rk_update(handle,dt,a,b,c) bind(C,name='astr_device_fluid_rk_update')
      import c_int,c_ptr,c_device_real_kind
      type(c_ptr),value :: handle
      real(c_device_real_kind),value :: dt,a,b,c
    end function
    integer(c_int) function fluid_rhs_prepare(handle,viscous) bind(C,name='astr_device_fluid_rhs_prepare')
      import c_int,c_ptr
      type(c_ptr),value :: handle
      integer(c_int),value :: viscous
    end function
    integer(c_int) function fluid_convection(handle) bind(C,name='astr_device_fluid_convection')
      import c_int,c_ptr
      type(c_ptr),value :: handle
    end function
    integer(c_int) function fluid_rhs_clear(handle) bind(C,name='astr_device_fluid_rhs_clear')
      import c_int,c_ptr
      type(c_ptr),value :: handle
    end function
    integer(c_int) function fluid_flux_divergence_basic(handle,viscous,low,high,first,last) &
        bind(C,name='astr_device_fluid_flux_divergence_basic')
      import c_int,c_ptr
      type(c_ptr),value :: handle
      integer(c_int),value :: viscous
      integer(c_int),intent(in) :: low(3),high(3),first(3),last(3)
    end function
    integer(c_int) function fluid_refresh_primitives_boundary(handle,c2,c6,low,high) &
        bind(C,name='astr_device_fluid_refresh_primitives_boundary')
      import c_int,c_ptr,c_device_real_kind
      type(c_ptr),value :: handle
      real(c_device_real_kind),value :: c2,c6
      integer(c_int),intent(in) :: low(3),high(3)
    end function
    integer(c_int) function fluid_viscous_flux(handle,reynolds,prandtl,c5,tc,tc1) &
        bind(C,name='astr_device_fluid_viscous_flux')
      import c_int,c_ptr,c_device_real_kind
      type(c_ptr),value :: handle
      real(c_device_real_kind),value :: reynolds,prandtl,c5,tc,tc1
    end function
    integer(c_int) function fluid_viscous_divergence(handle) bind(C,name='astr_device_fluid_viscous_divergence')
      import c_int,c_ptr
      type(c_ptr),value :: handle
    end function
    integer(c_int) function fluid_prepare(handle,gradients) bind(C,name='astr_device_fluid_prepare')
      import c_int,c_ptr
      type(c_ptr),value :: handle
      integer(c_int),value :: gradients
    end function
    integer(c_int) function fluid_primitives(handle,c2,c6) bind(C,name='astr_device_fluid_primitives')
      import c_int,c_ptr,c_device_real_kind
      type(c_ptr),value :: handle
      real(c_device_real_kind),value :: c2,c6
    end function
    integer(c_int) function fluid_refresh_primitives(handle,c2,c6) bind(C,name='astr_device_fluid_refresh_primitives')
      import c_int,c_ptr,c_device_real_kind
      type(c_ptr),value :: handle
      real(c_device_real_kind),value :: c2,c6
    end function
    integer(c_int) function fluid_conservative(handle,coefficient) bind(C,name='astr_device_fluid_conservative')
      import c_int,c_ptr,c_device_real_kind
      type(c_ptr),value :: handle
      real(c_device_real_kind),value :: coefficient
    end function
    integer(c_int) function fluid_gradient(handle) bind(C,name='astr_device_fluid_gradient')
      import c_int,c_ptr
      type(c_ptr),value :: handle
    end function
    integer(c_int) function fluid_gradient_basic_boundary(handle,low,high) bind(C,name='astr_device_fluid_gradient_basic_boundary')
      import c_int,c_ptr
      type(c_ptr),value :: handle
      integer(c_int),intent(in) :: low(3),high(3)
    end function
    integer(c_int) function fluid_check(handle) bind(C,name='astr_device_fluid_check')
      import c_int,c_ptr
      type(c_ptr),value :: handle
    end function
    integer(c_int) function fluid_cfl(handle,nondimensional,mach,gamma,rgas,rates) bind(C,name='astr_device_fluid_cfl')
      import c_int,c_ptr,c_device_real_kind
      type(c_ptr),value :: handle
      integer(c_int),value :: nondimensional
      real(c_device_real_kind),value :: mach,gamma,rgas
      real(c_device_real_kind),intent(out) :: rates(3)
    end function
  end interface
end module
