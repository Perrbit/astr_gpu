module numerical_forms
  implicit none
  character(len=16) :: convection_form='standard'
  character(len=16) :: viscous_form='standard'
  character(len=16) :: filter_scheme=''
contains
  pure logical function inviscid_incident_boundary_layout(boundaries) result(selected)
    integer,intent(in) :: boundaries(:)
    selected=all(boundaries==[11,22,421,52,1,1])
  end function

  pure logical function y_nonreflecting_boundary_layout(boundaries) result(selected)
    integer,intent(in) :: boundaries(:)
    selected=all(boundaries(3:6)==[41,52,1,1]).and. &
      (all(boundaries(1:2)==1).or.all(boundaries(1:2)==[11,21]))
  end function

  pure logical function profile_nscbc_boundary_layout(boundaries) result(selected)
    integer,intent(in) :: boundaries(:)
    selected=all(boundaries==[11,22,41,52,1,1])
  end function

  pure logical function profile_state_boundary_layout(boundaries) result(selected)
    integer,intent(in) :: boundaries(:)
    selected=all(boundaries(1:3)==[11,21,41]).and. &
      (boundaries(4)==50.or.boundaries(4)==51).and.all(boundaries(5:6)==1)
  end function

  pure logical function x_open_boundary_layout(boundaries) result(selected)
    integer,intent(in) :: boundaries(:)
    selected=all(boundaries==[11,21,1,1,1,1]).or.all(boundaries==[12,22,1,1,1,1])
  end function

  pure logical function ldc_boundary_layout(boundaries) result(selected)
    integer,intent(in) :: boundaries(:)
    selected=all(boundaries==[41,41,41,0,1,1])
  end function

  pure integer function simple_physical_boundary_axis(boundaries) result(axis)
    integer,intent(in) :: boundaries(6)
    integer :: candidate,kind
    logical :: physical(3),supported
    do candidate=1,3
      physical(candidate)=any(boundaries(2*candidate-1:2*candidate)/=1)
    enddo
    axis=0
    if(count(physical)/=1) return
    candidate=maxloc(merge(1,0,physical),dim=1)
    kind=boundaries(2*candidate-1)
    if(kind/=boundaries(2*candidate)) return
    supported=.false.
    select case(kind)
    case(41,50,60)
      supported=.true.
    case(42)
      supported=candidate<=2
    case(31,411,421)
      supported=candidate==2
    end select
    if(supported) axis=candidate
  end function

  pure logical function keep_open_boundary_selected(boundaries) result(selected)
    integer,intent(in) :: boundaries(:)
    selected=convection_form=='keep'.and.profile_nscbc_boundary_layout(boundaries)
  end function

  pure function effective_filter_scheme(convection,diffusion,enabled) result(scheme)
    character(*),intent(in) :: convection,diffusion
    logical,intent(in) :: enabled
    character(len=16) :: scheme
    scheme='none'
    if(.not.enabled) return
    scheme=filter_scheme
    ! An omitted selection retains the existing input's deterministic default.
    if(len_trim(scheme)==0) then
      scheme='explicit10'
      if(convection(4:4)=='c'.or.diffusion(4:4)=='c') scheme='compact10'
    endif
  end function

  logical function compact_filter_selected(convection,diffusion) result(selected)
    character(*),intent(in) :: convection,diffusion
    selected=effective_filter_scheme(convection,diffusion,.true.)=='compact10'
  end function

  function keep_configuration_error(scheme,diffusion,species,reacting,boundaries,turbulence) result(message)
    character(*),intent(in) :: scheme,diffusion,turbulence
    integer,intent(in) :: species,boundaries(:)
    logical,intent(in) :: reacting
    character(len=160) :: message
    message=''
    select case(trim(convection_form))
    case('standard')
      return
    case('keep')
      if(scheme/='643e'.and.scheme/='643c') message='KEEP implements sixth-order 643e or 643c convection'
      if(species/=0.or.reacting) message='KEEP currently implements the nonreacting five-variable equations'
      if(turbulence/='none') message='KEEP turbulence-model equations are not implemented'
      if(any(boundaries/=1)) then
        if((diffusion/='643e'.and.diffusion/='643c').or. &
          (simple_physical_boundary_axis(boundaries)==0.and..not.keep_open_boundary_selected(boundaries).and. &
           .not.ldc_boundary_layout(boundaries).and..not.x_open_boundary_layout(boundaries).and. &
           .not.profile_state_boundary_layout(boundaries).and..not.y_nonreflecting_boundary_layout(boundaries).and. &
           .not.inviscid_incident_boundary_layout(boundaries))) &
          message='KEEP physical ends require implemented simple, LDC or profile open boundaries'
        if(y_nonreflecting_boundary_layout(boundaries).and.(scheme/='643e'.or.diffusion/='643e')) &
          message='KEEP wall/nonreflecting-y coupling currently uses 643e convection and nodal gradients'
      endif
    case default
      message='convection_form must be standard or keep'
    end select
  end function

  function me6_configuration_error(species,reacting,boundaries,turbulence,diffusion_enabled,diffusion) result(message)
    integer,intent(in) :: species,boundaries(:)
    logical,intent(in) :: reacting,diffusion_enabled
    character(*),intent(in) :: turbulence,diffusion
    character(len=160) :: message
    message=''
    select case(trim(viscous_form))
    case('standard')
      return
    case('me6_opti')
      if(.not.diffusion_enabled) message='ME6 requires diffterm=t'
      if(species/=0.or.reacting.or.turbulence/='none') message='ME6 implements nonreacting perfect-gas five-variable equations'
      if(any(boundaries/=1)) then
        if((simple_physical_boundary_axis(boundaries)==0.and..not.ldc_boundary_layout(boundaries).and. &
          .not.x_open_boundary_layout(boundaries).and..not.profile_nscbc_boundary_layout(boundaries).and. &
          .not.profile_state_boundary_layout(boundaries).and..not.y_nonreflecting_boundary_layout(boundaries)).or. &
          (diffusion/='643e'.and.diffusion/='643c')) &
          message='ME6 physical ends require implemented simple/LDC/profile boundaries and 643e/643c nodal gradients'
        if(y_nonreflecting_boundary_layout(boundaries).and.diffusion/='643e') &
          message='ME6 wall/nonreflecting-y coupling currently uses 643e nodal gradients'
      endif
    case default
      message='viscous_form must be standard or me6_opti'
    end select
  end function
end module
