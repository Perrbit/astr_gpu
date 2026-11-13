module device_state
  use iso_c_binding, only: c_ptr,c_int,c_int64_t,c_size_t,c_double
  implicit none
  private
  integer(c_int),parameter,public :: device_q=0,device_rhs=1,device_save=2
  integer(c_int),parameter,public :: device_rho=3,device_velocity=4,device_pressure=5, &
    device_temperature=6,device_metric=7,device_dvelocity=8,device_dtemperature=9
  integer(c_int),parameter,public :: device_jacobian=10,device_stress=11,device_heat_flux=12
  integer(c_int),parameter,public :: device_filter_work=13
  integer(c_int),parameter,public :: device_not_allocated=-2,device_nonphysical=-3,device_invalid_spectrum=-4
  type,bind(C),public :: device_view
    type(c_ptr) :: data
    integer(c_int) :: rank,element_bytes
    integer(c_int) :: lower(5),extent(5)
    integer(c_int64_t) :: stride(5)
  end type
  public :: device_bind,device_state_create,device_state_view
  public :: device_count
  public :: device_state_upload,device_state_download,device_state_synchronize,device_state_destroy
  public :: device_state_snapshot_axis
  public :: device_memory_info
  interface
    integer(c_int) function device_memory_info(handle,free_bytes,total_bytes) bind(C,name='astr_device_memory_info')
      import c_int,c_ptr,c_size_t
      type(c_ptr),value :: handle
      integer(c_size_t),intent(out) :: free_bytes,total_bytes
    end function
    integer(c_int) function device_state_snapshot_axis(handle,field,axis,host,bytes) &
        bind(C,name='astr_device_state_snapshot_axis')
      import c_int,c_ptr,c_double,c_size_t
      type(c_ptr),value :: handle
      integer(c_int),value :: field,axis
      real(c_double),intent(out) :: host(*)
      integer(c_size_t),value :: bytes
    end function
    integer(c_int) function device_count(count) bind(C,name='astr_device_count')
      import c_int
      integer(c_int),intent(out) :: count
    end function
    integer(c_int) function device_bind(device) bind(C,name='astr_device_bind')
      import c_int
      integer(c_int),value :: device
    end function
    integer(c_int) function device_state_create(device,im,jm,km,halo,debug,handle) &
        bind(C,name='astr_device_state_create')
      import c_int,c_ptr
      integer(c_int),value :: device,im,jm,km,halo,debug
      type(c_ptr),intent(out) :: handle
    end function
    integer(c_int) function device_state_view(handle,field,view) bind(C,name='astr_device_state_view')
      import c_int,c_ptr,device_view
      type(c_ptr),value :: handle
      integer(c_int),value :: field
      type(device_view),intent(out) :: view
    end function
    integer(c_int) function device_state_upload(handle,field,host,bytes) bind(C,name='astr_device_state_upload')
      import c_int,c_ptr,c_double,c_size_t
      type(c_ptr),value :: handle
      integer(c_int),value :: field
      real(c_double),intent(in) :: host(*)
      integer(c_size_t),value :: bytes
    end function
    integer(c_int) function device_state_download(handle,field,host,bytes) bind(C,name='astr_device_state_download')
      import c_int,c_ptr,c_double,c_size_t
      type(c_ptr),value :: handle
      integer(c_int),value :: field
      real(c_double),intent(out) :: host(*)
      integer(c_size_t),value :: bytes
    end function
    integer(c_int) function device_state_synchronize(handle) bind(C,name='astr_device_state_synchronize')
      import c_int,c_ptr
      type(c_ptr),value :: handle
    end function
    integer(c_int) function device_state_destroy(handle) bind(C,name='astr_device_state_destroy')
      import c_int,c_ptr
      type(c_ptr),intent(inout) :: handle
    end function
  end interface
end module
