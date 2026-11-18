module upwind_operator_cpp
  use iso_c_binding, only: c_ptr,c_int
  use device_state, only: c_device_real_kind
  implicit none
  private
  public :: upwind_prepare_cpp,upwind_prepare_mixed_flux_cpp,upwind_axis_cpp
  public :: upwind_prepare_mixed_characteristic_cpp
  public :: upwind_high_order_axis_cpp
  interface
    integer(c_int) function upwind_high_order_axis_cpp(handle,axis,gamma,mach,weight,low,high,first,last) &
        bind(C,name='astr_device_upwind_high_order_axis')
      import c_ptr,c_int,c_device_real_kind
      type(c_ptr),value :: handle
      integer(c_int),value :: axis
      real(c_device_real_kind),value :: gamma,mach,weight
      integer(c_int),intent(in) :: low(3),high(3),first(3),last(3)
    end function
    integer(c_int) function upwind_prepare_cpp(handle) bind(C,name='astr_device_upwind_prepare')
      import c_ptr,c_int
      type(c_ptr),value :: handle
    end function
    integer(c_int) function upwind_prepare_mixed_flux_cpp(handle) bind(C,name='astr_device_upwind_prepare_mixed_flux')
      import c_ptr,c_int
      type(c_ptr),value :: handle
    end function
    integer(c_int) function upwind_prepare_mixed_characteristic_cpp(handle) &
        bind(C,name='astr_device_upwind_prepare_mixed_characteristic_flux')
      import c_ptr,c_int
      type(c_ptr),value :: handle
    end function
    integer(c_int) function upwind_axis_cpp(handle,axis,reconstruction,characteristic,gamma,mach,weight, &
        metric_consistent,low,high,first,last) bind(C,name='astr_device_upwind_axis')
      import c_ptr,c_int,c_device_real_kind
      type(c_ptr),value :: handle
      integer(c_int),value :: axis,reconstruction,characteristic,metric_consistent
      real(c_device_real_kind),value :: gamma,mach,weight
      integer(c_int),intent(in) :: low(3),high(3),first(3),last(3)
    end function
  end interface
end module
