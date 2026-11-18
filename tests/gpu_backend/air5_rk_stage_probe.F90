module air5_rk_stage_probe
  use iso_c_binding, only: c_ptr,c_int,c_double
  use chemistry_device_step, only: air5_prepare_rk_stage
  use chemistry_device_boundary, only: air5_postshock_boundary_apply
  implicit none
contains
  integer(c_int) function prepare_rk_probe(handle,last,edges,comm,compensated,stage,filtering,prepared, &
      conservative,floor,wall,family,boundary_calls) bind(C) result(status)
    type(c_ptr),value :: handle
    integer(c_int),intent(in) :: last(3),edges(2)
    integer(c_int),value :: comm,compensated,stage,filtering,prepared,conservative,wall,family
    real(c_double),intent(in) :: floor(5)
    integer(c_int),intent(out) :: boundary_calls
    boundary_calls=0_c_int
    status=air5_prepare_rk_stage(handle,last,5_c_int,stage,prepared/=0,filtering/=0,conservative/=0, &
      floor,wall,family,0_c_int,compensated/=0,comm,boundary)
  contains
    integer(c_int) function boundary(context) result(code)
      type(c_ptr),intent(in) :: context
      boundary_calls=boundary_calls+1_c_int
      code=0_c_int
      if(any(edges/=0)) code=air5_postshock_boundary_apply(context,edges(1),edges(2),.false.,0.0_c_double,1_c_int,comm)
    end function
  end function
end module
