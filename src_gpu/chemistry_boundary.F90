module chemistry_device_boundary
  use iso_c_binding,only:c_ptr,c_int,c_double,c_size_t
  use chemistry_device_properties,only:air5_device_check
  implicit none
  private
  public :: air5_wall_lower_y,air5_pressure_outlet_upper_x
  public :: air5_fixed_boundary_prepare,air5_fixed_boundary_apply
  public :: air5_profile_boundary_prepare,air5_profile_inflow_lower_x
  public :: air5_top_boundary_prepare,air5_hbl_outflow_upper_x,air5_farfield_upper_y,air5_hbl_boundary_carry_reset
  public :: air5_characteristic_top_prepare,air5_characteristic_top_rhs
  public :: air5_hbl_boundary_apply,air5_postshock_boundary_apply
  interface
    integer(c_int) function air5_characteristic_top_prepare(handle,spacing,count) &
        bind(C,name='astr_device_air5_characteristic_top_prepare')
      import c_ptr,c_int,c_double,c_size_t
      type(c_ptr),value :: handle
      real(c_double),intent(in) :: spacing(*)
      integer(c_size_t),value :: count
    end function
    integer(c_int) function air5_characteristic_top_rhs(handle,first_i,preserve,rate,reference_sound) &
        bind(C,name='astr_device_air5_characteristic_top_rhs')
      import c_ptr,c_int,c_double
      type(c_ptr),value :: handle
      integer(c_int),value :: first_i,preserve
      real(c_double),value :: rate,reference_sound
    end function
    integer(c_int) function air5_top_boundary_prepare(handle,farfield,incident,origin,incident_x,enabled,x_top,count) &
        bind(C,name='astr_device_air5_top_boundary_prepare')
      import c_ptr,c_int,c_double,c_size_t
      type(c_ptr),value :: handle
      real(c_double),intent(in) :: farfield(11),incident(11,2),x_top(*)
      real(c_double),value :: origin,incident_x
      integer(c_int),value :: enabled
      integer(c_size_t),value :: count
    end function
    integer(c_int) function air5_hbl_outflow_upper_x(handle,physical_top,characteristic_top,preserve) &
        bind(C,name='astr_device_air5_hbl_outflow_upper_x')
      import c_ptr,c_int
      type(c_ptr),value :: handle
      integer(c_int),value :: physical_top,characteristic_top,preserve
    end function
    integer(c_int) function air5_farfield_upper_y(handle,characteristic_top,preserve) &
        bind(C,name='astr_device_air5_farfield_upper_y')
      import c_ptr,c_int
      type(c_ptr),value :: handle
      integer(c_int),value :: characteristic_top,preserve
    end function
    integer(c_int) function air5_hbl_boundary_carry_reset(handle,lower,upper,characteristic_top) &
        bind(C,name='astr_device_air5_hbl_boundary_carry_reset')
      import c_ptr,c_int
      type(c_ptr),value :: handle
      integer(c_int),intent(in) :: lower(3),upper(3)
      integer(c_int),value :: characteristic_top
    end function
    integer(c_int) function air5_fixed_boundary_prepare(handle,left,right) bind(C,name='astr_device_air5_fixed_boundary_prepare')
      import c_ptr,c_int,c_double
      type(c_ptr),value :: handle
      real(c_double),intent(in) :: left(11),right(11)
    end function
    integer(c_int) function air5_profile_boundary_prepare(handle,profile,count) bind(C,name='astr_device_air5_profile_boundary_prepare')
      import c_ptr,c_int,c_double,c_size_t
      type(c_ptr),value :: handle
      real(c_double),intent(in) :: profile(*)
      integer(c_size_t),value :: count
    end function
    integer(c_int) function air5_fixed_boundary_apply(handle,left,right,normal,pressure,preserve) &
        bind(C,name='astr_device_air5_fixed_boundary_apply')
      import c_ptr,c_int,c_double
      type(c_ptr),value :: handle
      integer(c_int),value :: left,right,normal,preserve
      real(c_double),value :: pressure
    end function
    integer(c_int) function air5_profile_inflow_lower_x(handle) bind(C,name='astr_device_air5_profile_inflow_lower_x')
      import c_ptr,c_int
      type(c_ptr),value :: handle
    end function
    integer(c_int) function air5_wall_lower_y(handle,temperature,preserve) bind(C,name='astr_device_air5_wall_lower_y')
      import c_ptr,c_int,c_double
      type(c_ptr),value :: handle
      real(c_double),value :: temperature
      integer(c_int),value :: preserve
    end function
    integer(c_int) function air5_pressure_outlet_upper_x(handle,pressure,preserve) bind(C,name='astr_device_air5_pressure_outlet_upper_x')
      import c_ptr,c_int,c_double
      type(c_ptr),value :: handle
      real(c_double),value :: pressure
      integer(c_int),value :: preserve
    end function
  end interface
contains
  ! This is the original state-boundary order, not a transport/RK update.
  ! All communicator ranks call it; a failed stage prevents later faces/reset.
  integer(c_int) function air5_hbl_boundary_apply(handle,lower,upper,wall_temperature, &
      characteristic,preserve,compensated,comm,failed_stage) result(status)
    use mpi,only:MPI_Allreduce,MPI_INTEGER,MPI_MAX,MPI_SUCCESS,MPI_Abort
    type(c_ptr),intent(in) :: handle
    integer(c_int),intent(in) :: lower(3),upper(3),preserve,comm
    real(c_double),intent(in) :: wall_temperature
    logical,intent(in) :: characteristic,compensated
    integer(c_int),intent(out) :: failed_stage
    integer(c_int) :: local_status,dynamic
    integer :: ierr,ignored

    status=0_c_int;failed_stage=0_c_int;local_status=0_c_int
    dynamic=merge(1_c_int,0_c_int,characteristic)
    if(upper(1)/=0)then
      call check_boundary_api(air5_hbl_outflow_upper_x(handle,upper(2),dynamic,preserve),comm)
      call check_boundary_api(air5_device_check(handle,local_status),comm)
    endif
    call release_stage(1_c_int)
    if(status/=0)return
    if(lower(2)/=0)then
      call check_boundary_api(air5_wall_lower_y(handle,wall_temperature,preserve),comm)
      call check_boundary_api(air5_device_check(handle,local_status),comm)
    endif
    call release_stage(2_c_int)
    if(status/=0)return
    if(upper(2)/=0)then
      call check_boundary_api(air5_farfield_upper_y(handle,dynamic,preserve),comm)
      call check_boundary_api(air5_device_check(handle,local_status),comm)
    endif
    call release_stage(3_c_int)
    if(status/=0)return
    if(lower(1)/=0)call check_boundary_api(air5_profile_inflow_lower_x(handle),comm)
    if(compensated)call check_boundary_api(air5_hbl_boundary_carry_reset(handle,lower,upper,dynamic),comm)
  contains
    subroutine release_stage(stage)
      integer(c_int),intent(in) :: stage
      call MPI_Allreduce(local_status,status,1,MPI_INTEGER,MPI_MAX,int(comm),ierr)
      if(ierr/=MPI_SUCCESS)call MPI_Abort(int(comm),ierr,ignored)
      if(status/=0)failed_stage=stage
      local_status=0_c_int
    end subroutine
  end function

  integer(c_int) function air5_postshock_boundary_apply(handle,lower_x,upper_x,normal,pressure,preserve,comm) result(status)
    use mpi,only:MPI_Allreduce,MPI_INTEGER,MPI_MAX,MPI_SUCCESS,MPI_Abort
    type(c_ptr),intent(in) :: handle
    integer(c_int),intent(in) :: lower_x,upper_x,preserve,comm
    logical,intent(in) :: normal
    real(c_double),intent(in) :: pressure
    integer(c_int) :: local_status
    integer :: ierr,ignored

    call check_boundary_api(air5_fixed_boundary_apply(handle,lower_x,upper_x, &
      merge(1_c_int,0_c_int,normal),pressure,preserve),comm)
    status=0_c_int
    if(normal.and.pressure>0.0_c_double)then
      call check_boundary_api(air5_device_check(handle,local_status),comm)
      call MPI_Allreduce(local_status,status,1,MPI_INTEGER,MPI_MAX,int(comm),ierr)
      if(ierr/=MPI_SUCCESS)call MPI_Abort(int(comm),ierr,ignored)
    endif
  end function

  subroutine check_boundary_api(code,comm)
    use mpi,only:MPI_Abort
    integer(c_int),intent(in) :: code,comm
    integer :: ierr
    if(code==0)return
    write(*,'(A,I0)')'AIR5 boundary device API failed, code=',code
    call MPI_Abort(int(comm),abs(int(code)),ierr)
  end subroutine
end module
