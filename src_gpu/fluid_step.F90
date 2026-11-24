module fluid_step
  use iso_c_binding, only: c_int,c_ptr
  use device_state, only: real_kind=>c_device_real_kind,device_q,device_filter_work,device_stress,device_heat_flux
  use fluid_explicit
  use halo_transport, only: device_viscous_exchange_begin,device_viscous_exchange_end,halo_average_plane
  implicit none
  private
  public :: fluid_advance_stage,fluid_prepare_stage,fluid_advance_prepared_stage
  public :: fluid_filter_explicit
  abstract interface
    integer(c_int) function device_operator(handle)
      import c_int,c_ptr
      type(c_ptr),value :: handle
    end function
    integer(c_int) function recovery_operator(handle)
      import c_int,c_ptr
      type(c_ptr),intent(in) :: handle
    end function
    integer(c_int) function exchange_axis(handle,field,axis,reconcile) bind(C)
      import c_int,c_ptr
      type(c_ptr),value :: handle
      integer(c_int),value :: field,axis,reconcile
    end function
  end interface
contains
  integer(c_int) function fluid_prepare_stage(handle,c2,c6,filtering,exchange,filter_operator, &
      boundary_operator,primitive_operator,filtered_recovery_operator,input_recovery_operator,regional_preparation) result(status)
    type(c_ptr),value :: handle
    real(real_kind),intent(in) :: c2,c6
    logical,intent(in) :: filtering
    procedure(exchange_axis) :: exchange
    procedure(device_operator),optional :: filter_operator
    procedure(device_operator),optional :: boundary_operator,primitive_operator
    procedure(recovery_operator),optional :: filtered_recovery_operator,input_recovery_operator
    procedure(device_operator),optional :: regional_preparation
    integer(c_int) :: axis,average
    if(filtering) then
      if(present(filter_operator)) then
        status=filter_operator(handle)
        if(status/=0) return
      else
        status=fluid_filter_explicit(handle,exchange)
        if(status/=0) return
      endif
      if(present(filtered_recovery_operator)) then
        status=filtered_recovery_operator(handle)
        if(status/=0) return
      endif
    endif
    if(present(boundary_operator)) then
      if(filtering) then
        ! Wall state uses pressure/temperature from this filtered q, not the
        ! preceding RK state's caches. Halo reconstruction follows exchange.
        status=fluid_primitives(handle,c2,c6)
        if(status/=0) return
      endif
      status=boundary_operator(handle)
      if(status/=0) return
      if(present(input_recovery_operator)) then
        status=input_recovery_operator(handle)
        if(status/=0) return
      endif
    endif
    average=1_c_int
    if(present(regional_preparation)) average=halo_average_plane
    do axis=0,2
      status=exchange(handle,device_q,axis,average)
      if(status/=0) return
    enddo
    if(present(regional_preparation)) then
      status=regional_preparation(handle)
      return
    endif
    do axis=0,2
      status=exchange(handle,device_q,axis,0_c_int)
      if(status/=0) return
    enddo
    if(present(input_recovery_operator)) then
      status=input_recovery_operator(handle)
      if(status/=0) return
    endif
    if(present(primitive_operator)) then
      status=primitive_operator(handle)
    else
      status=fluid_refresh_primitives(handle,c2,c6)
    endif
  end function

  integer(c_int) function fluid_filter_explicit(handle,exchange,lower,upper,high_wall) result(status)
    type(c_ptr),value :: handle
    procedure(exchange_axis) :: exchange
    integer(c_int),optional,intent(in) :: lower(3),upper(3)
    ! Optional pair: zero-based wall axis and existing boundary type.
    integer(c_int),optional,intent(in) :: high_wall(2)
    integer(c_int) :: component,workspace_components
    if(present(lower).neqv.present(upper)) then
      status=-1
      return
    endif
    workspace_components=fluid_filter_workspace_components(handle)
    status=exchange(handle,device_q,0_c_int,0_c_int)
    if(status/=0) return
    if(workspace_components==1) then
      do component=1,5
        status=selected_axis(0_c_int,component)
        if(status/=0) return
        status=exchange(handle,device_filter_work,1_c_int,0_c_int)
        if(status/=0) return
        status=selected_axis(1_c_int,component)
        if(status/=0) return
      enddo
      ! Each component is independent; exchange the updated q z-face once.
      status=exchange(handle,device_q,2_c_int,0_c_int)
      if(status/=0) return
      do component=1,5
        status=selected_axis(2_c_int,component)
        if(status/=0) return
        status=fluid_filter_component_commit(handle,component)
        if(status/=0) return
      enddo
    elseif(workspace_components==5) then
      status=selected_axis(0_c_int,0_c_int)
      if(status/=0) return
      status=exchange(handle,device_filter_work,1_c_int,0_c_int)
      if(status/=0) return
      status=selected_axis(1_c_int,0_c_int)
      if(status/=0) return
      status=exchange(handle,device_q,2_c_int,0_c_int)
      if(status/=0) return
      status=selected_axis(2_c_int,0_c_int)
      if(status/=0) return
      status=fluid_filter_commit(handle)
    else
      status=-1
    endif
  contains
    integer(c_int) function selected_axis(axis,component) result(code)
      integer(c_int),intent(in) :: axis,component
      if(present(high_wall).and.present(lower)) then
        if(axis==high_wall(1)) then
          code=fluid_filter_axis_high_wall(handle,axis,component,lower(axis+1),upper(axis+1), &
            int(merge(1,0,high_wall(2)/=41),c_int))
          return
        endif
      endif
      if(present(lower)) then
        if(component==0) then
          code=fluid_filter_axis_basic(handle,axis,lower(axis+1),upper(axis+1))
        else
          code=fluid_filter_component_axis_basic(handle,axis,component,lower(axis+1),upper(axis+1))
        endif
      elseif(component==0) then
        code=fluid_filter_axis(handle,axis)
      else
        code=fluid_filter_component_axis(handle,axis,component)
      endif
    end function
  end function

  integer(c_int) function fluid_advance_stage(handle,stage,dt,c2,c6,reynolds,prandtl,c5,tc,tc1, &
      filtering,viscous,exchange,begin_exchange,end_exchange) result(status)
    type(c_ptr),value :: handle
    integer,intent(in) :: stage
    real(real_kind),intent(in) :: dt,c2,c6,reynolds,prandtl,c5,tc,tc1
    logical,intent(in) :: filtering,viscous
    procedure(exchange_axis) :: exchange
    procedure(exchange_axis) :: begin_exchange,end_exchange
    status=fluid_prepare_stage(handle,c2,c6,filtering,exchange)
    if(status/=0) return
    status=fluid_advance_prepared_stage(handle,stage,dt,c2,c6,reynolds,prandtl,c5,tc,tc1, &
      viscous,exchange,begin_exchange,end_exchange)
  end function

  integer(c_int) function fluid_advance_prepared_stage(handle,stage,dt,c2,c6,reynolds,prandtl,c5,tc,tc1, &
      viscous,exchange,begin_exchange,end_exchange,gradient_operator,convection_operator,divergence_operator, &
      integrated_viscous_operator,source_operator,sensor_operator,post_update_operator,packed_viscous_axis, &
      prepared_viscous_operator) result(status)
    type(c_ptr),value :: handle
    integer,intent(in) :: stage
    real(real_kind),intent(in) :: dt,c2,c6,reynolds,prandtl,c5,tc,tc1
    logical,intent(in) :: viscous
    procedure(exchange_axis) :: exchange
    procedure(exchange_axis) :: begin_exchange,end_exchange
    procedure(device_operator),optional :: gradient_operator,convection_operator,divergence_operator
    procedure(device_operator),optional :: integrated_viscous_operator,prepared_viscous_operator
    procedure(device_operator),optional :: source_operator
    procedure(device_operator),optional :: sensor_operator
    procedure(device_operator),optional :: post_update_operator
    integer(c_int),intent(in),optional :: packed_viscous_axis
    integer(c_int) :: axis,overlap_axis
    real(real_kind),parameter :: a(3)=[1._real_kind,.75_real_kind,1._real_kind/3._real_kind], &
      b(3)=[0._real_kind,.25_real_kind,2._real_kind/3._real_kind], &
      c(3)=[1._real_kind,.25_real_kind,2._real_kind/3._real_kind]
    if(stage<1.or.stage>3) then
      status=-1
      return
    endif
    overlap_axis=-1_c_int
    if(present(packed_viscous_axis)) overlap_axis=packed_viscous_axis
    if(stage==1) then
      status=fluid_rk_save(handle)
      if(status/=0) return
    endif
    if(present(prepared_viscous_operator)) then
      status=prepared_viscous_operator(handle)
    else
      status=assemble_rhs()
    endif
    if(status/=0) return
    if(present(source_operator)) then
      status=source_operator(handle)
      if(status/=0) return
    endif
    status=fluid_rk_update(handle,dt,a(stage),b(stage),c(stage))
    if(status/=0) return
    if(present(post_update_operator)) then
      status=post_update_operator(handle)
      if(status/=0) return
    endif
    status=fluid_primitives(handle,c2,c6)
    if(status/=0) return
    status=fluid_check(handle)
  contains
  integer(c_int) function assemble_rhs() result(status)
    if(viscous) then
      if(present(gradient_operator)) then
        status=gradient_operator(handle)
      else
        status=fluid_gradient(handle)
      endif
      if(status/=0) return
    endif
    ! Sensor halo exchange must finish before the same transport posts stress.
    if(present(sensor_operator)) then
      status=sensor_operator(handle)
      if(status/=0) return
    endif
    if(viscous) then
      if(.not.present(integrated_viscous_operator)) then
        status=fluid_viscous_flux(handle,reynolds,prandtl,c5,tc,tc1)
        if(status/=0) return
        if(overlap_axis>=0) then
          status=device_viscous_exchange_begin(handle,overlap_axis)
        else
          status=begin_exchange(handle,device_stress,0_c_int,0_c_int)
        endif
        if(status/=0) return
      endif
    endif
    ! Convection and the already-packed viscous flux use disjoint fields.
    ! The selected axis finishes before the next exchange reuses the slots.
    if(present(convection_operator)) then
      status=convection_operator(handle)
    else
      status=fluid_convection(handle)
    endif
    if(status/=0) return
    if(viscous) then
      if(present(integrated_viscous_operator)) then
        status=integrated_viscous_operator(handle)
      else
      if(overlap_axis>=0) then
        status=device_viscous_exchange_end(handle,overlap_axis)
        if(status/=0) return
        do axis=0,2
          if(axis==overlap_axis) cycle
          status=device_viscous_exchange_begin(handle,axis)
          if(status/=0) return
          status=device_viscous_exchange_end(handle,axis)
          if(status/=0) return
        enddo
      else
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
      endif
      if(present(divergence_operator)) then
        status=divergence_operator(handle)
      else
        status=fluid_viscous_divergence(handle)
      endif
      endif
      if(status/=0) return
    endif
  end function
  end function
end module
