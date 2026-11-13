module device_halo
  use iso_c_binding, only: c_int,c_ptr
  implicit none
  private
  public :: device_periodic_axis
  interface
    integer(c_int) function device_periodic_axis(handle,field,axis,reconcile) bind(C,name='astr_device_periodic_axis')
      import c_int,c_ptr
      type(c_ptr),value :: handle
      integer(c_int),value :: field,axis,reconcile
    end function
  end interface
end module
