module fluid_step
  use iso_c_binding, only: c_int,c_ptr,c_double
  use device_state, only: device_q,device_filter_work,device_stress,device_heat_flux
  use fluid_explicit
  implicit none
  private
  public :: fluid_advance_stage,fluid_prepare_stage,fluid_advance_prepared_stage
  abstract interface
    integer(c_int) function exchange_axis(handle,field,axis,reconcile) bind(C)
      import c_int,c_ptr
      type(c_ptr),value :: handle
      integer(c_int),value :: field,axis,reconcile
    end function
  end interface
contains
  integer(c_int) function fluid_prepare_stage(handle,c2,c6,filtering,exchange) result(status)
    type(c_ptr),value :: handle
    real(c_double),intent(in) :: c2,c6
    logical,intent(in) :: filtering
    procedure(exchange_axis) :: exchange
    integer(c_int) :: axis,component,workspace_components
    if(filtering) then
      workspace_components=fluid_filter_workspace_components(handle)
      status=exchange(handle,device_q,0_c_int,0_c_int)
      if(status/=0) return
      if(workspace_components==1) then
        do component=1,5
          status=fluid_filter_component_axis(handle,0_c_int,component)
          if(status/=0) return
          status=exchange(handle,device_filter_work,1_c_int,0_c_int)
          if(status/=0) return
          status=fluid_filter_component_axis(handle,1_c_int,component)
          if(status/=0) return
        enddo
        ! Components are independent: finish all y updates before the single
        ! q z-halo exchange, then reuse the scalar buffer for each z result.
        status=exchange(handle,device_q,2_c_int,0_c_int)
        if(status/=0) return
        do component=1,5
          status=fluid_filter_component_axis(handle,2_c_int,component)
          if(status/=0) return
          status=fluid_filter_component_commit(handle,component)
          if(status/=0) return
        enddo
      elseif(workspace_components==5) then
        status=fluid_filter_axis(handle,0_c_int)
        if(status/=0) return
        status=exchange(handle,device_filter_work,1_c_int,0_c_int)
        if(status/=0) return
        status=fluid_filter_axis(handle,1_c_int)
        if(status/=0) return
        status=exchange(handle,device_q,2_c_int,0_c_int)
        if(status/=0) return
        status=fluid_filter_axis(handle,2_c_int)
        if(status/=0) return
        status=fluid_filter_commit(handle)
        if(status/=0) return
      else
        status=-1
        return
      endif
    endif
    do axis=0,2
      status=exchange(handle,device_q,axis,1_c_int)
      if(status/=0) return
    enddo
    do axis=0,2
      status=exchange(handle,device_q,axis,0_c_int)
      if(status/=0) return
    enddo
    status=fluid_refresh_primitives(handle,c2,c6)
  end function

  integer(c_int) function fluid_advance_stage(handle,stage,dt,c2,c6,reynolds,prandtl,c5,tc,tc1, &
      filtering,viscous,exchange,begin_exchange,end_exchange) result(status)
    type(c_ptr),value :: handle
    integer,intent(in) :: stage
    real(c_double),intent(in) :: dt,c2,c6,reynolds,prandtl,c5,tc,tc1
    logical,intent(in) :: filtering,viscous
    procedure(exchange_axis) :: exchange
    procedure(exchange_axis) :: begin_exchange,end_exchange
    status=fluid_prepare_stage(handle,c2,c6,filtering,exchange)
    if(status/=0) return
    status=fluid_advance_prepared_stage(handle,stage,dt,c2,c6,reynolds,prandtl,c5,tc,tc1, &
      viscous,exchange,begin_exchange,end_exchange)
  end function

  integer(c_int) function fluid_advance_prepared_stage(handle,stage,dt,c2,c6,reynolds,prandtl,c5,tc,tc1, &
      viscous,exchange,begin_exchange,end_exchange) result(status)
    type(c_ptr),value :: handle
    integer,intent(in) :: stage
    real(c_double),intent(in) :: dt,c2,c6,reynolds,prandtl,c5,tc,tc1
    logical,intent(in) :: viscous
    procedure(exchange_axis) :: exchange
    procedure(exchange_axis) :: begin_exchange,end_exchange
    integer(c_int) :: axis
    real(c_double),parameter :: a(3)=[1.d0,.75d0,1.d0/3.d0], &
      b(3)=[0.d0,.25d0,2.d0/3.d0],c(3)=[1.d0,.25d0,2.d0/3.d0]
    if(stage<1.or.stage>3) then
      status=-1
      return
    endif
    if(stage==1) then
      status=fluid_rk_save(handle)
      if(status/=0) return
    endif
    if(viscous) then
      status=fluid_gradient(handle)
      if(status/=0) return
      status=fluid_viscous_flux(handle,reynolds,prandtl,c5,tc,tc1)
      if(status/=0) return
      status=begin_exchange(handle,device_stress,0_c_int,0_c_int)
      if(status/=0) return
    endif
    ! Convection writes RHS only; stress exchange consumes the independent
    ! viscous-flux field. MPI progresses in end while convection runs.
    status=fluid_convection(handle)
    if(status/=0) return
    if(viscous) then
      status=end_exchange(handle,device_stress,0_c_int,0_c_int)
      if(status/=0) return
      status=exchange(handle,device_heat_flux,0_c_int,0_c_int)
      if(status/=0) return
      do axis=1,2
        status=exchange(handle,device_stress,axis,0_c_int)
        if(status/=0) return
        status=exchange(handle,device_heat_flux,axis,0_c_int)
        if(status/=0) return
      enddo
      status=fluid_viscous_divergence(handle)
      if(status/=0) return
    endif
    status=fluid_rk_update(handle,dt,a(stage),b(stage),c(stage))
    if(status/=0) return
    status=fluid_primitives(handle,c2,c6)
    if(status/=0) return
    status=fluid_check(handle)
  end function
end module
