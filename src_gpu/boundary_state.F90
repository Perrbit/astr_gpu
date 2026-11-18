module boundary_state_cpp
  use iso_c_binding, only: c_int,c_ptr,c_size_t,c_int8_t
  use device_state, only: c_device_real_kind
  implicit none
  private
  public :: wall41_basic_state,wall42_basic_state,zeroextrap_basic,symmetry_basic
  public :: profile_inflow_prepare,profile_inflow_x,farfield51_upper_basic
  public :: outflow21_x_basic
  public :: nscbc_outlet_x,nscbc_inlet_x
  public :: nscbc_outlet_prepare,nscbc_outlet_filter
  public :: nscbc_top_rhs,nscbc_top_target_upload,nscbc_top_mach2
  public :: nscbc_incident_prepare,nscbc_incident_complete
  public :: nscbc_high_open_prepare,nscbc_high_open_complete,nscbc_outlet_filter_high_open,high_open_wall_y
  public :: nscbc_top_outlet_corner
  public :: wall41_outlet_y
  public :: nscbc_top_filter_prepare,nscbc_top_filter
  public :: nscbc_top_prediff_capture,nscbc_top_viscous_source
  public :: slip_wall_y,slip_mask_upload
  public :: fixed_rti_y
  public :: ldc_top_lid_y
  public :: wall41_evolved_density_state
  public :: wall_adiabatic_evolved_state
  public :: wall_forcing_prepare,wall_forcing_upload,wall41_forcing_y
  public :: inflow_slices_prepare,inflow_slice_upload,inflow_slice_download,dynamic_inflow_x
  interface
    integer(c_int) function nscbc_high_open_prepare(parent,target,bytes) bind(C,name='astr_device_nscbc_high_open_prepare')
      import c_ptr,c_int,c_size_t
      type(c_ptr),value :: parent,target
      integer(c_size_t),value :: bytes
    end function
    integer(c_int) function nscbc_high_open_complete(parent,low,high,gamma,mach,length,roinf,uinf,vinf,winf,pinf,pout, &
        boundary_status) bind(C,name='astr_device_nscbc_high_open_complete')
      import c_ptr,c_int,c_device_real_kind
      type(c_ptr),value :: parent
      integer(c_int),intent(in) :: low(3),high(3)
      real(c_device_real_kind),value :: gamma,mach,length,roinf,uinf,vinf,winf,pinf,pout
      integer(c_int),intent(out) :: boundary_status
    end function
    integer(c_int) function nscbc_outlet_filter_high_open(parent,npdcj) bind(C,name='astr_device_nscbc_outlet_filter_high_open')
      import c_ptr,c_int
      type(c_ptr),value :: parent
      integer(c_int),value :: npdcj
    end function
    integer(c_int) function high_open_wall_y(parent,code,forcing,outlet_only,tw,c2,c6) bind(C,name='astr_device_high_open_wall_y')
      import c_ptr,c_int,c_device_real_kind
      type(c_ptr),value :: parent
      integer(c_int),value :: code,forcing,outlet_only
      real(c_device_real_kind),value :: tw,c2,c6
    end function
    integer(c_int) function nscbc_incident_prepare(parent,target,bytes) bind(C,name='astr_device_nscbc_incident_prepare')
      import c_ptr,c_int,c_device_real_kind,c_size_t
      type(c_ptr),value :: parent
      real(c_device_real_kind),intent(in) :: target(*)
      integer(c_size_t),value :: bytes
    end function
    integer(c_int) function nscbc_incident_complete(parent,npdci,gamma,mach,boundary_status) &
        bind(C,name='astr_device_nscbc_incident_complete')
      import c_ptr,c_int,c_device_real_kind
      type(c_ptr),value :: parent
      integer(c_int),value :: npdci
      real(c_device_real_kind),value :: gamma,mach
      integer(c_int),intent(out) :: boundary_status
    end function
    integer(c_int) function wall41_outlet_y(parent,forcing,tw,c2,c6) bind(C,name='astr_device_wall41_outlet_y')
      import c_int,c_ptr,c_device_real_kind
      type(c_ptr),value :: parent
      integer(c_int),value :: forcing
      real(c_device_real_kind),value :: tw,c2,c6
    end function
    integer(c_int) function inflow_slices_prepare(parent) bind(C,name='astr_device_inflow_slices_prepare')
      import c_ptr,c_int
      type(c_ptr),value :: parent
    end function
    integer(c_int) function inflow_slice_upload(parent,slot,values,bytes) bind(C,name='astr_device_inflow_slice_upload')
      import c_ptr,c_int,c_device_real_kind,c_size_t
      type(c_ptr),value :: parent
      integer(c_int),value :: slot
      real(c_device_real_kind),intent(in) :: values(*)
      integer(c_size_t),value :: bytes
    end function
    integer(c_int) function inflow_slice_download(parent,slot,values,bytes) bind(C,name='astr_device_inflow_slice_download')
      import c_ptr,c_int,c_device_real_kind,c_size_t
      type(c_ptr),value :: parent
      integer(c_int),value :: slot
      real(c_device_real_kind),intent(out) :: values(*)
      integer(c_size_t),value :: bytes
    end function
    integer(c_int) function dynamic_inflow_x(parent,slots,time0,delta,time,mach_pressure,mach,c2,c6) &
        bind(C,name='astr_device_dynamic_inflow_x')
      import c_ptr,c_int,c_device_real_kind
      type(c_ptr),value :: parent
      integer(c_int),intent(in) :: slots(4)
      integer(c_int),value :: mach_pressure
      real(c_device_real_kind),value :: time0,delta,time,mach,c2,c6
    end function
    integer(c_int) function wall_forcing_prepare(parent) bind(C,name='astr_device_wall_forcing_prepare')
      import c_ptr,c_int
      type(c_ptr),value :: parent
    end function
    integer(c_int) function wall_forcing_upload(parent,velocity,bytes) bind(C,name='astr_device_wall_forcing_upload')
      import c_ptr,c_int,c_device_real_kind,c_size_t
      type(c_ptr),value :: parent
      real(c_device_real_kind),intent(in) :: velocity(*)
      integer(c_size_t),value :: bytes
    end function
    integer(c_int) function wall41_forcing_y(parent,tw,c2,c6) bind(C,name='astr_device_wall41_forcing_y')
      import c_ptr,c_int,c_device_real_kind
      type(c_ptr),value :: parent
      real(c_device_real_kind),value :: tw,c2,c6
    end function
    integer(c_int) function wall_adiabatic_evolved_state(parent,axis,side,slip,c2,c6) &
        bind(C,name='astr_device_wall_adiabatic_evolved_state')
      import c_int,c_ptr,c_device_real_kind
      type(c_ptr),value :: parent
      integer(c_int),value :: axis,side,slip
      real(c_device_real_kind),value :: c2,c6
    end function
    integer(c_int) function wall41_evolved_density_state(parent,axis,side,tw,c2,c6) &
        bind(C,name='astr_device_wall41_evolved_density_state')
      import c_int,c_ptr,c_device_real_kind
      type(c_ptr),value :: parent
      integer(c_int),value :: axis,side
      real(c_device_real_kind),value :: tw,c2,c6
    end function
    integer(c_int) function ldc_top_lid_y(parent,c2,c6) bind(C,name='astr_device_ldc_top_lid_y')
      import c_int,c_ptr,c_device_real_kind
      type(c_ptr),value :: parent
      real(c_device_real_kind),value :: c2,c6
    end function
    integer(c_int) function fixed_rti_y(parent,side,c2,c6) bind(C,name='astr_device_fixed_rti_y')
      import c_int,c_ptr,c_device_real_kind
      type(c_ptr),value :: parent
      integer(c_int),value :: side
      real(c_device_real_kind),value :: c2,c6
    end function
    integer(c_int) function slip_mask_upload(parent,mask,bytes) bind(C,name='astr_device_slip_mask_upload')
      import c_int,c_ptr,c_int8_t,c_size_t
      type(c_ptr),value :: parent
      integer(c_int8_t),intent(in) :: mask(*)
      integer(c_size_t),value :: bytes
    end function
    integer(c_int) function slip_wall_y(parent,code,side,tw,c2,c6) bind(C,name='astr_device_slip_wall_y')
      import c_int,c_ptr,c_device_real_kind
      type(c_ptr),value :: parent
      integer(c_int),value :: code,side
      real(c_device_real_kind),value :: tw,c2,c6
    end function
    integer(c_int) function nscbc_top_prediff_capture(parent) bind(C,name='astr_device_nscbc_top_prediff_capture')
      import c_int,c_ptr
      type(c_ptr),value :: parent
    end function
    integer(c_int) function nscbc_top_outlet_corner(parent,gamma,mach) bind(C,name='astr_device_nscbc_top_outlet_corner')
      import c_int,c_ptr,c_device_real_kind
      type(c_ptr),value :: parent
      real(c_device_real_kind),value :: gamma,mach
    end function
    integer(c_int) function nscbc_top_viscous_source(parent,gamma,mach) bind(C,name='astr_device_nscbc_top_viscous_source')
      import c_int,c_ptr,c_device_real_kind
      type(c_ptr),value :: parent
      real(c_device_real_kind),value :: gamma,mach
    end function
    integer(c_int) function nscbc_top_filter_prepare(parent) bind(C,name='astr_device_nscbc_top_filter_prepare')
      import c_int,c_ptr
      type(c_ptr),value :: parent
    end function
    integer(c_int) function nscbc_top_filter(parent,axis,ntype) bind(C,name='astr_device_nscbc_top_filter')
      import c_int,c_ptr
      type(c_ptr),value :: parent
      integer(c_int),value :: axis,ntype
    end function
    integer(c_int) function nscbc_top_rhs(parent,npdci,npdck,policy,shock,keep_corner, &
        gamma,mach,length,pressure_target,target,max_mach2) bind(C,name='astr_device_nscbc_top_rhs')
      import c_int,c_ptr,c_device_real_kind
      type(c_ptr),value :: parent
      integer(c_int),value :: npdci,npdck,policy,shock,keep_corner
      real(c_device_real_kind),value :: gamma,mach,length,pressure_target,max_mach2
      real(c_device_real_kind),intent(in) :: target(5)
    end function
    integer(c_int) function nscbc_top_target_upload(parent,target,bytes) bind(C,name='astr_device_nscbc_top_target_upload')
      import c_int,c_ptr,c_device_real_kind,c_size_t
      type(c_ptr),value :: parent
      real(c_device_real_kind),intent(in) :: target(*)
      integer(c_size_t),value :: bytes
    end function
    integer(c_int) function nscbc_top_mach2(parent,mach,value) bind(C,name='astr_device_nscbc_top_mach2')
      import c_int,c_ptr,c_device_real_kind
      type(c_ptr),value :: parent
      real(c_device_real_kind),value :: mach
      real(c_device_real_kind),intent(out) :: value
    end function
    integer(c_int) function nscbc_inlet_x(parent,npdcj,npdck,gamma,mach,length,roinf,uinf,vinf,winf,pinf) &
        bind(C,name='astr_device_nscbc_inlet_x')
      import c_int,c_ptr,c_device_real_kind
      type(c_ptr),value :: parent
      integer(c_int),value :: npdcj,npdck
      real(c_device_real_kind),value :: gamma,mach,length,roinf,uinf,vinf,winf,pinf
    end function
    integer(c_int) function nscbc_outlet_prepare(parent) bind(C,name='astr_device_nscbc_outlet_prepare')
      import c_int,c_ptr
      type(c_ptr),value :: parent
    end function
    integer(c_int) function nscbc_outlet_filter(parent,npdcj) bind(C,name='astr_device_nscbc_outlet_filter')
      import c_int,c_ptr
      type(c_ptr),value :: parent
      integer(c_int),value :: npdcj
    end function
    integer(c_int) function nscbc_outlet_x(parent,npdcj,npdck,include_lower_z,gamma,mach,length, &
        roinf,uinf,vinf,winf,pinf,pout) bind(C,name='astr_device_nscbc_outlet_x')
      import c_int,c_ptr,c_device_real_kind
      type(c_ptr),value :: parent
      integer(c_int),value :: npdcj,npdck,include_lower_z
      real(c_device_real_kind),value :: gamma,mach,length,roinf,uinf,vinf,winf,pinf,pout
    end function
    integer(c_int) function outflow21_x_basic(parent,mach,c2,c6,pressure_target) bind(C,name='astr_device_outflow21_x_basic')
      import c_int,c_ptr,c_device_real_kind
      type(c_ptr),value :: parent
      real(c_device_real_kind),value :: mach,c2,c6,pressure_target
    end function
    integer(c_int) function profile_inflow_prepare(parent) bind(C,name='astr_device_profile_inflow_prepare')
      import c_int,c_ptr
      type(c_ptr),value :: parent
    end function
    integer(c_int) function profile_inflow_x(parent,mach_pressure,mach,c2,c6) bind(C,name='astr_device_profile_inflow_x')
      import c_int,c_ptr,c_device_real_kind
      type(c_ptr),value :: parent
      integer(c_int),value :: mach_pressure
      real(c_device_real_kind),value :: mach,c2,c6
    end function
    integer(c_int) function farfield51_upper_basic(parent,c2,c6) bind(C,name='astr_device_farfield51_upper_basic')
      import c_int,c_ptr,c_device_real_kind
      type(c_ptr),value :: parent
      real(c_device_real_kind),value :: c2,c6
    end function
    integer(c_int) function symmetry_basic(parent,axis,side,c2,c6) bind(C,name='astr_device_symmetry_basic')
      import c_int,c_ptr,c_device_real_kind
      type(c_ptr),value :: parent
      integer(c_int),value :: axis,side
      real(c_device_real_kind),value :: c2,c6
    end function
    integer(c_int) function zeroextrap_basic(parent,axis,side,c2,c6) bind(C,name='astr_device_zeroextrap_basic')
      import c_int,c_ptr,c_device_real_kind
      type(c_ptr),value :: parent
      integer(c_int),value :: axis,side
      real(c_device_real_kind),value :: c2,c6
    end function
    integer(c_int) function wall42_basic_state(parent,axis,side,c2,c6) bind(C,name='astr_device_wall42_basic_state')
      import c_int,c_ptr,c_device_real_kind
      type(c_ptr),value :: parent
      integer(c_int),value :: axis,side
      real(c_device_real_kind),value :: c2,c6
    end function
    integer(c_int) function wall41_basic_state(parent,axis,side,tw,c2,c6) bind(C,name='astr_device_wall41_basic_state')
      import c_int,c_ptr,c_device_real_kind
      type(c_ptr),value :: parent
      integer(c_int),value :: axis,side
      real(c_device_real_kind),value :: tw,c2,c6
    end function
  end interface
end module
