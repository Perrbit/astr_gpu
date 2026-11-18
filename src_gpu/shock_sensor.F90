module shock_sensor_cpp
  use iso_c_binding, only: c_ptr,c_int,c_int8_t,c_size_t
  use device_state, only: c_device_real_kind
  implicit none
  private
  public :: shock_prepare_cpp,shock_raw_cpp,shock_expand_cpp,shock_mask_download_cpp
  public :: shock_mask_exchange_cpp,shock_faces_download_cpp
  interface
    integer(c_int) function shock_prepare_cpp(handle) bind(C,name='astr_device_shock_prepare')
      import c_ptr,c_int
      type(c_ptr),value :: handle
    end function
    integer(c_int) function shock_raw_cpp(handle,low,high) bind(C,name='astr_device_shock_raw')
      import c_ptr,c_int
      type(c_ptr),value :: handle
      integer(c_int),intent(in) :: low(3),high(3)
    end function
    integer(c_int) function shock_expand_cpp(handle,low,high,threshold) bind(C,name='astr_device_shock_expand')
      import c_ptr,c_int,c_device_real_kind
      type(c_ptr),value :: handle
      integer(c_int),intent(in) :: low(3),high(3)
      real(c_device_real_kind),value :: threshold
    end function
    integer(c_int) function shock_mask_exchange_cpp(handle) bind(C,name='astr_device_shock_mask_exchange')
      import c_ptr,c_int
      type(c_ptr),value :: handle
    end function
    integer(c_int) function shock_faces_download_cpp(handle,host,bytes) bind(C,name='astr_device_shock_faces_download')
      import c_ptr,c_int,c_int8_t,c_size_t
      type(c_ptr),value :: handle
      integer(c_int8_t),intent(out) :: host(*)
      integer(c_size_t),value :: bytes
    end function
    integer(c_int) function shock_mask_download_cpp(handle,host,bytes) bind(C,name='astr_device_shock_mask_download')
      import c_ptr,c_int,c_int8_t,c_size_t
      type(c_ptr),value :: handle
      integer(c_int8_t),intent(out) :: host(*)
      integer(c_size_t),value :: bytes
    end function
  end interface
end module
