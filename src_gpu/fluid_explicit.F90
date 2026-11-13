module fluid_explicit
  use iso_c_binding, only: c_int,c_ptr,c_double,c_int64_t
  implicit none
  private
  public :: fluid_prepare,fluid_primitives,fluid_conservative,fluid_gradient,fluid_check,fluid_cfl
  public :: fluid_refresh_primitives
  public :: fluid_rhs_prepare,fluid_convection,fluid_viscous_flux,fluid_viscous_divergence
  public :: fluid_filter_prepare,fluid_filter_axis,fluid_filter_commit,fluid_rk_save,fluid_rk_update
  public :: fluid_statistics,fluid_cfl_diagnostics
  interface
    integer(c_int) function fluid_statistics(handle,reynolds,tc,tc1,values) bind(C,name='astr_device_fluid_statistics')
      import c_int,c_ptr,c_double
      type(c_ptr),value :: handle
      real(c_double),value :: reynolds,tc,tc1
      real(c_double),intent(out) :: values(13)
    end function
    integer(c_int) function fluid_cfl_diagnostics(handle,nondimensional,mach,gamma,rgas,global_upper,origin,ny, &
        values,keys,profile) bind(C,name='astr_device_fluid_cfl_diagnostics')
      import c_int,c_ptr,c_double,c_int64_t
      type(c_ptr),value :: handle
      integer(c_int),value :: nondimensional,ny
      real(c_double),value :: mach,gamma,rgas
      integer(c_int),intent(in) :: global_upper(3),origin(3)
      real(c_double),intent(out) :: values(4)
      integer(c_int64_t),intent(out) :: keys(4)
      real(c_double),intent(inout) :: profile(*)
    end function
    integer(c_int) function fluid_filter_prepare(handle) bind(C,name='astr_device_fluid_filter_prepare')
      import c_int,c_ptr
      type(c_ptr),value :: handle
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
      import c_int,c_ptr,c_double
      type(c_ptr),value :: handle
      real(c_double),value :: dt,a,b,c
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
    integer(c_int) function fluid_viscous_flux(handle,reynolds,prandtl,c5,tc,tc1) &
        bind(C,name='astr_device_fluid_viscous_flux')
      import c_int,c_ptr,c_double
      type(c_ptr),value :: handle
      real(c_double),value :: reynolds,prandtl,c5,tc,tc1
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
      import c_int,c_ptr,c_double
      type(c_ptr),value :: handle
      real(c_double),value :: c2,c6
    end function
    integer(c_int) function fluid_refresh_primitives(handle,c2,c6) bind(C,name='astr_device_fluid_refresh_primitives')
      import c_int,c_ptr,c_double
      type(c_ptr),value :: handle
      real(c_double),value :: c2,c6
    end function
    integer(c_int) function fluid_conservative(handle,coefficient) bind(C,name='astr_device_fluid_conservative')
      import c_int,c_ptr,c_double
      type(c_ptr),value :: handle
      real(c_double),value :: coefficient
    end function
    integer(c_int) function fluid_gradient(handle) bind(C,name='astr_device_fluid_gradient')
      import c_int,c_ptr
      type(c_ptr),value :: handle
    end function
    integer(c_int) function fluid_check(handle) bind(C,name='astr_device_fluid_check')
      import c_int,c_ptr
      type(c_ptr),value :: handle
    end function
    integer(c_int) function fluid_cfl(handle,nondimensional,mach,gamma,rgas,rates) bind(C,name='astr_device_fluid_cfl')
      import c_int,c_ptr,c_double
      type(c_ptr),value :: handle
      integer(c_int),value :: nondimensional
      real(c_double),value :: mach,gamma,rgas
      real(c_double),intent(out) :: rates(3)
    end function
  end interface
end module
