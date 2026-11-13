module halo_transport
  use iso_c_binding, only: c_int,c_ptr,c_double
  implicit none
  private
  public :: device_transport_prepare,device_exchange_axis,device_exchange_begin,device_exchange_end
  public :: halo_pinned,halo_device
  public :: device_transport_metrics
  integer(c_int),parameter :: halo_pinned=0,halo_device=1
  interface
    integer(c_int) function device_transport_prepare(handle,communicator,lower,upper,sizes,max_components,mode,timing) &
        bind(C,name='astr_device_transport_prepare')
      import c_int,c_ptr
      type(c_ptr),value :: handle
      integer(c_int),value :: communicator,max_components,mode,timing
      integer(c_int),intent(in) :: lower(3),upper(3),sizes(3)
    end function
    integer(c_int) function device_transport_metrics(handle,values,reset) bind(C,name='astr_device_transport_metrics')
      import c_int,c_ptr,c_double
      type(c_ptr),value :: handle
      real(c_double),intent(out) :: values(6)
      integer(c_int),value :: reset
    end function
    integer(c_int) function device_exchange_begin(handle,field,axis,reconcile) bind(C,name='astr_device_exchange_begin')
      import c_int,c_ptr
      type(c_ptr),value :: handle
      integer(c_int),value :: field,axis,reconcile
    end function
    integer(c_int) function device_exchange_end(handle,field,axis,reconcile) bind(C,name='astr_device_exchange_end')
      import c_int,c_ptr
      type(c_ptr),value :: handle
      integer(c_int),value :: field,axis,reconcile
    end function
    integer(c_int) function device_exchange_axis(handle,field,axis,reconcile) bind(C,name='astr_device_exchange_axis')
      import c_int,c_ptr
      type(c_ptr),value :: handle
      integer(c_int),value :: field,axis,reconcile
    end function
  end interface
end module
