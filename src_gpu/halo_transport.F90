module halo_transport
  use iso_c_binding, only: c_int,c_ptr
  implicit none
  private
  public :: device_transport_prepare,device_exchange_axis
  interface
    integer(c_int) function device_transport_prepare(handle,communicator,lower,upper,sizes,max_components) &
        bind(C,name='astr_device_transport_prepare')
      import c_int,c_ptr
      type(c_ptr),value :: handle
      integer(c_int),value :: communicator,max_components
      integer(c_int),intent(in) :: lower(3),upper(3),sizes(3)
    end function
    integer(c_int) function device_exchange_axis(handle,field,axis,reconcile) bind(C,name='astr_device_exchange_axis')
      import c_int,c_ptr
      type(c_ptr),value :: handle
      integer(c_int),value :: field,axis,reconcile
    end function
  end interface
end module
