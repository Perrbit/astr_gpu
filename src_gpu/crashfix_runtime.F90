module crashfix_runtime_cpp
  use iso_c_binding, only: c_ptr,c_int
  use mpi
  use device_state, only: device_view,device_state_view,device_q,device_nonphysical,c_device_real_kind
  use halo_transport, only: device_exchange_axis
  use crashfix_operator_cpp
  implicit none
  private
  public :: repair_crashfix_cpp,repair_reconciled_cpp
contains
  integer(c_int) function repair_crashfix_cpp(handle,origin,global,periodic,c2,c6,epsp,communicator, &
      report,rounds,expanded,diagnostics) result(status)
    type(c_ptr),intent(in) :: handle
    integer(c_int),intent(in) :: origin(3),global(3),periodic(3)
    real(c_device_real_kind),intent(in) :: c2,c6,epsp
    integer,intent(in) :: communicator
    type(crashfix_report),intent(out) :: report
    integer,intent(out) :: rounds,expanded,diagnostics(2)
    type(device_view) :: view
    integer(c_int) :: lower(3),upper(3),local_diagnostics(4),cells(3),halo
    integer :: global_invalid,progress,axis,err
    rounds=0;expanded=0;diagnostics=0
    status=device_state_view(handle,device_q,view)
    if(status/=0) return
    halo=-view%lower(1);cells=view%extent(1:3)-2*halo-1
    lower=max(-halo,-origin);upper=min(cells+halo,global-origin)
    where(periodic/=0) lower=-halo
    where(periodic/=0) upper=cells+halo
    status=crashfix_ordered_pass_cpp(handle,origin,global,c2,c6,epsp,report)
    if(status/=0) return
    call MPI_Allreduce(report%invalid,global_invalid,1,MPI_INTEGER,MPI_MAX,communicator,err)
    if(err/=MPI_SUCCESS) then
      status=err;return
    endif
    do while(global_invalid/=0)
      rounds=rounds+1
      ! A shared replica may still be NaN: exchange only the halo, not averages.
      do axis=0,2
        status=device_exchange_axis(handle,device_q,int(axis,c_int),0_c_int)
        if(status/=0) return
      enddo
      status=crashfix_expanded_pass_cpp(handle,lower,upper,c2,c6,epsp,report,local_diagnostics)
      if(status/=0) return
      call MPI_Allreduce(report%invalid,global_invalid,1,MPI_INTEGER,MPI_MAX,communicator,err)
      if(err/=MPI_SUCCESS) then
        status=err;return
      endif
      call MPI_Allreduce(local_diagnostics(2),progress,1,MPI_INTEGER,MPI_SUM,communicator,err)
      if(err/=MPI_SUCCESS) then
        status=err;return
      endif
      expanded=expanded+progress;diagnostics=local_diagnostics(3:4)
      if(progress==0) exit
    enddo
    call MPI_Allreduce(MPI_IN_PLACE,diagnostics,2,MPI_INTEGER,MPI_MAX,communicator,err)
    status=err
    if(err==MPI_SUCCESS.and.global_invalid/=0) status=device_nonphysical
    if(status==0) status=crashfix_history_exchange_cpp(handle)
  end function

  integer(c_int) function repair_reconciled_cpp(handle,origin,global,periodic,c2,c6,epsp,communicator, &
      report,already_repaired) result(status)
    type(c_ptr),intent(in) :: handle
    integer(c_int),intent(in) :: origin(3),global(3),periodic(3)
    real(c_device_real_kind),intent(in) :: c2,c6,epsp
    integer,intent(in) :: communicator
    type(crashfix_report),intent(inout) :: report
    logical,optional,intent(in) :: already_repaired
    integer(c_int) :: invalid
    integer :: axis,global_invalid,err,rounds,expanded,diagnostics(2)
    logical :: repair_required
    repair_required=.true.
    if(present(already_repaired)) repair_required=.not.already_repaired
    do
      if(repair_required) then
        do axis=0,2
          status=device_exchange_axis(handle,device_q,int(axis,c_int),0_c_int)
          if(status/=0) return
        enddo
        status=crashfix_recovery_halos_cpp(handle,c2,c6)
        if(status/=0) return
        status=repair_crashfix_cpp(handle,origin,global,periodic,c2,c6,epsp,communicator,report, &
          rounds,expanded,diagnostics)
        if(status/=0) return
      endif
      do axis=0,2
        status=device_exchange_axis(handle,device_q,int(axis,c_int),1_c_int)
        if(status/=0) return
      enddo
      do axis=0,2
        status=device_exchange_axis(handle,device_q,int(axis,c_int),0_c_int)
        if(status/=0) return
      enddo
      status=crashfix_reconciled_primitives_cpp(handle,c2,c6)
      if(status/=0) return
      status=crashfix_input_invalid_cpp(handle,invalid)
      if(status/=0) return
      call MPI_Allreduce(invalid,global_invalid,1,MPI_INTEGER,MPI_MAX,communicator,err)
      if(err/=MPI_SUCCESS) then
        status=err;return
      endif
      if(global_invalid==0) exit
      repair_required=.true.
    enddo
  end function
end module
