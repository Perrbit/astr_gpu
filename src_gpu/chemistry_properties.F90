module chemistry_device_properties
  use iso_c_binding, only: c_ptr,c_int
  implicit none
  private
  public :: air5_device_prepare,air5_device_conservative,air5_device_primitives,air5_device_check
  public :: air5_device_primitives_regions
  public :: air5_device_primitives_shell
  interface
    integer(c_int) function air5_device_primitives_regions(handle,regions,first,last,preserve_all) &
        bind(C,name='astr_device_air5_primitives_regions')
      import c_int,c_ptr
      type(c_ptr),value :: handle
      integer(c_int),value :: regions,preserve_all
      integer(c_int),intent(in) :: first(3,*),last(3,*)
    end function
    integer(c_int) function air5_device_prepare(handle) bind(C,name='astr_device_air5_prepare')
      import c_int,c_ptr
      type(c_ptr),value :: handle
    end function
    integer(c_int) function air5_device_conservative(handle,first,last) bind(C,name='astr_device_air5_conservative')
      import c_int,c_ptr
      type(c_ptr),value :: handle
      integer(c_int),intent(in) :: first(3),last(3)
    end function
    integer(c_int) function air5_device_primitives(handle,first,last,preserve_all) bind(C,name='astr_device_air5_primitives')
      import c_int,c_ptr
      type(c_ptr),value :: handle
      integer(c_int),intent(in) :: first(3),last(3)
      integer(c_int),value :: preserve_all
    end function
    integer(c_int) function air5_device_primitives_shell(handle,first,last,preserve_all) &
        bind(C,name='astr_device_air5_primitives_shell')
      import c_int,c_ptr
      type(c_ptr),value :: handle
      integer(c_int),intent(in) :: first(3),last(3)
      integer(c_int),value :: preserve_all
    end function
    integer(c_int) function air5_device_check(handle,status) bind(C,name='astr_device_air5_check')
      import c_int,c_ptr
      type(c_ptr),value :: handle
      integer(c_int),intent(out) :: status
    end function
  end interface
end module
