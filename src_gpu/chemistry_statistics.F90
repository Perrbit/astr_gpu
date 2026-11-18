module chemistry_device_statistics
  use iso_c_binding,only:c_ptr,c_int,c_double,c_size_t,c_int64_t
  implicit none
  private
  public :: air5_mean_prepare,air5_mean_reset,air5_mean_release,air5_mean_sample,air5_mean_check
  public :: air5_mean_refresh_velocity
  public :: air5_conservation_prepare,air5_conservation_collect
  public :: air5_monitor_prepare,air5_monitor_collect
  interface
    integer(c_int) function air5_conservation_prepare(handle,budget,bytes) &
      bind(C,name='astr_device_air5_conservation_prepare')
      import c_ptr,c_int,c_int64_t
      type(c_ptr),value :: handle
      integer(c_int64_t),value :: budget
      integer(c_int64_t),intent(out) :: bytes
    end function
    integer(c_int) function air5_conservation_collect(handle,totals) bind(C,name='astr_device_air5_conservation_collect')
      import c_ptr,c_int,c_double
      type(c_ptr),value :: handle
      real(c_double),intent(out) :: totals(11)
    end function
    integer(c_int) function air5_monitor_prepare(handle,count,nodes,xyz) bind(C,name='astr_device_air5_monitor_prepare')
      import c_ptr,c_int,c_double
      type(c_ptr),value :: handle
      integer(c_int),value :: count
      integer(c_int),intent(in) :: nodes(3,*)
      real(c_double),intent(in) :: xyz(3,*)
    end function
    integer(c_int) function air5_monitor_collect(handle,packed,bytes) bind(C,name='astr_device_air5_monitor_collect')
      import c_ptr,c_int,c_double,c_size_t
      type(c_ptr),value :: handle
      real(c_double),intent(out) :: packed(*)
      integer(c_size_t),value :: bytes
    end function
    integer(c_int) function air5_mean_prepare(handle) bind(C,name='astr_device_air5_mean_prepare')
      import c_ptr,c_int
      type(c_ptr),value :: handle
    end function
    integer(c_int) function air5_mean_reset(handle) bind(C,name='astr_device_air5_mean_reset')
      import c_ptr,c_int
      type(c_ptr),value :: handle
    end function
    integer(c_int) function air5_mean_release(handle) bind(C,name='astr_device_air5_mean_release')
      import c_ptr,c_int
      type(c_ptr),value :: handle
    end function
    integer(c_int) function air5_mean_refresh_velocity(handle,low,high) &
      bind(C,name='astr_device_air5_mean_refresh_velocity')
      import c_ptr,c_int
      type(c_ptr),value :: handle
      integer(c_int),intent(in) :: low(3),high(3)
    end function
    integer(c_int) function air5_mean_sample(handle,interval,has_previous) bind(C,name='astr_device_air5_mean_sample')
      import c_ptr,c_int,c_double
      type(c_ptr),value :: handle
      real(c_double),value :: interval
      integer(c_int),value :: has_previous
    end function
    integer(c_int) function air5_mean_check(handle,status) bind(C,name='astr_device_air5_mean_check')
      import c_ptr,c_int
      type(c_ptr),value :: handle
      integer(c_int),intent(out) :: status
    end function
  end interface
end module
