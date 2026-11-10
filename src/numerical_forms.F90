module numerical_forms
  implicit none
  character(len=16) :: convection_form='standard'
  character(len=16) :: viscous_form='standard'
  character(len=16) :: filter_scheme=''
contains
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

  function keep_configuration_error(scheme,species,reacting,boundaries,turbulence) result(message)
    character(*),intent(in) :: scheme,turbulence
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
      if(any(boundaries/=1)) message='KEEP physical boundary closures are not implemented yet'
    case default
      message='convection_form must be standard or keep'
    end select
  end function

  function me6_configuration_error(species,reacting,boundaries,turbulence,diffusion_enabled) result(message)
    integer,intent(in) :: species,boundaries(:)
    logical,intent(in) :: reacting,diffusion_enabled
    character(*),intent(in) :: turbulence
    character(len=160) :: message
    message=''
    select case(trim(viscous_form))
    case('standard')
      return
    case('me6_opti')
      if(.not.diffusion_enabled) message='ME6 requires diffterm=t'
      if(species/=0.or.reacting.or.turbulence/='none') message='ME6 implements nonreacting perfect-gas five-variable equations'
      if(any(boundaries/=1)) message='ME6 physical boundary closures are not implemented yet'
    case default
      message='viscous_form must be standard or me6_opti'
    end select
  end function
end module
