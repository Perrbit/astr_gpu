module mp_positivity_operator_cpp
  use iso_c_binding, only: c_ptr,c_int
  use device_state, only: c_device_real_kind
  implicit none
  private
  public :: mp_positivity_prepare_cpp,mp_positivity_begin_cpp,mp_positivity_axis_cpp
  public :: mp_positivity_faces_complete_cpp,mp_positivity_limit_cpp
  interface
    integer(c_int) function mp_positivity_prepare_cpp(handle,communicator,lower,upper,mode) &
        bind(C,name='astr_device_mp_positivity_prepare')
      import c_ptr,c_int
      type(c_ptr),value :: handle
      integer(c_int),value :: communicator,mode
      integer(c_int),intent(in) :: lower(3),upper(3)
    end function
    integer(c_int) function mp_positivity_begin_cpp(handle) bind(C,name='astr_device_mp_positivity_begin')
      import c_ptr,c_int
      type(c_ptr),value :: handle
    end function
    integer(c_int) function mp_positivity_axis_cpp(handle,axis,high_order,gamma,mach,weight,low,high,first,last) &
        bind(C,name='astr_device_mp_positivity_axis')
      import c_ptr,c_int,c_device_real_kind
      type(c_ptr),value :: handle
      integer(c_int),value :: axis,high_order
      real(c_device_real_kind),value :: gamma,mach,weight
      integer(c_int),intent(in) :: low(3),high(3),first(3),last(3)
    end function
    integer(c_int) function mp_positivity_faces_complete_cpp(handle) bind(C,name='astr_device_mp_positivity_faces_complete')
      import c_ptr,c_int
      type(c_ptr),value :: handle
    end function
    integer(c_int) function mp_positivity_limit_cpp(handle,dt,a,b,c,stage,first,last,limited) &
        bind(C,name='astr_device_mp_positivity_limit')
      import c_ptr,c_int,c_device_real_kind
      type(c_ptr),value :: handle
      real(c_device_real_kind),value :: dt,a,b,c
      integer(c_int),value :: stage
      integer(c_int),intent(in) :: first(3),last(3)
      integer(c_int),intent(out) :: limited
    end function
  end interface
end module
