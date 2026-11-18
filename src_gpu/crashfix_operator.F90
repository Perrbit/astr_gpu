module crashfix_operator_cpp
  use iso_c_binding, only: c_int,c_ptr,c_size_t,c_int8_t
  use device_state, only: c_device_real_kind
  implicit none
  private
  public :: crashfix_report,crashfix_prepare_cpp,crashfix_history_upload_cpp,crashfix_history_download_cpp
  public :: crashfix_ordered_pass_cpp,crashfix_expanded_pass_cpp
  public :: crashfix_recovery_halos_cpp,crashfix_reconciled_primitives_cpp,crashfix_input_invalid_cpp
  public :: crashfix_history_exchange_cpp
  type,bind(C) :: crashfix_report
    integer(c_int) :: candidates,repaired,invalid
    real(c_device_real_kind) :: max_change(5),minima(4)
  end type
  interface
    integer(c_int) function crashfix_history_exchange_cpp(parent) bind(C,name='astr_device_crashfix_history_exchange')
      import c_ptr,c_int
      type(c_ptr),value :: parent
    end function
    integer(c_int) function crashfix_recovery_halos_cpp(parent,c2,c6) bind(C,name='astr_device_crashfix_recovery_halos')
      import c_ptr,c_int,c_device_real_kind
      type(c_ptr),value :: parent
      real(c_device_real_kind),value :: c2,c6
    end function
    integer(c_int) function crashfix_reconciled_primitives_cpp(parent,c2,c6) &
        bind(C,name='astr_device_crashfix_reconciled_primitives')
      import c_ptr,c_int,c_device_real_kind
      type(c_ptr),value :: parent
      real(c_device_real_kind),value :: c2,c6
    end function
    integer(c_int) function crashfix_input_invalid_cpp(parent,invalid) bind(C,name='astr_device_crashfix_input_invalid')
      import c_ptr,c_int
      type(c_ptr),value :: parent
      integer(c_int),intent(out) :: invalid
    end function
    integer(c_int) function crashfix_expanded_pass_cpp(parent,lower,upper,c2,c6,eps_pressure,report,diagnostics) &
        bind(C,name='astr_device_crashfix_expanded_pass')
      import c_ptr,c_int,c_device_real_kind,crashfix_report
      type(c_ptr),value :: parent
      integer(c_int),intent(in) :: lower(3),upper(3)
      real(c_device_real_kind),value :: c2,c6,eps_pressure
      type(crashfix_report),intent(out) :: report
      integer(c_int),intent(out) :: diagnostics(4)
    end function
    integer(c_int) function crashfix_prepare_cpp(parent,fluid,bytes) bind(C,name='astr_device_crashfix_prepare')
      import c_ptr,c_int,c_int8_t,c_size_t
      type(c_ptr),value :: parent
      integer(c_int8_t),intent(in) :: fluid(*)
      integer(c_size_t),value :: bytes
    end function
    integer(c_int) function crashfix_history_upload_cpp(parent,mask,bytes) bind(C,name='astr_device_crashfix_history_upload')
      import c_ptr,c_int,c_int8_t,c_size_t
      type(c_ptr),value :: parent
      integer(c_int8_t),intent(in) :: mask(*)
      integer(c_size_t),value :: bytes
    end function
    integer(c_int) function crashfix_history_download_cpp(parent,mask,bytes) bind(C,name='astr_device_crashfix_history_download')
      import c_ptr,c_int,c_int8_t,c_size_t
      type(c_ptr),value :: parent
      integer(c_int8_t),intent(out) :: mask(*)
      integer(c_size_t),value :: bytes
    end function
    integer(c_int) function crashfix_ordered_pass_cpp(parent,origin,global,c2,c6,eps_pressure,report) &
        bind(C,name='astr_device_crashfix_ordered_pass')
      import c_ptr,c_int,c_device_real_kind,crashfix_report
      type(c_ptr),value :: parent
      integer(c_int),intent(in) :: origin(3),global(3)
      real(c_device_real_kind),value :: c2,c6,eps_pressure
      type(crashfix_report),intent(out) :: report
    end function
  end interface
end module
