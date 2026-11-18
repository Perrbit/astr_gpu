module sponge_operator_cpp
  use iso_c_binding, only: c_ptr,c_int
  use device_state, only: c_device_real_kind
  implicit none
  private
  public :: sponge_prepare_cpp,sponge_apply_cpp
  interface
    integer(c_int) function sponge_prepare_cpp(parent,first,last,begin,ending,xcoef,ycoef) &
        bind(C,name='astr_device_sponge_prepare')
      import c_ptr,c_int,c_device_real_kind
      type(c_ptr),value :: parent
      integer(c_int),intent(in) :: first(3),last(3),begin(2),ending(2)
      real(c_device_real_kind),intent(in) :: xcoef(*),ycoef(*)
    end function
    integer(c_int) function sponge_apply_cpp(parent,axis) bind(C,name='astr_device_sponge_apply')
      import c_ptr,c_int
      type(c_ptr),value :: parent
      integer(c_int),value :: axis
    end function
  end interface
end module
