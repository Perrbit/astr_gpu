module chemistry_flow_runtime
  use iso_fortran_env, only: real64
  use chemistry_air5_data, only: air5_num_species
  use chemistry_model, only: chemistry_status_ok
  use chemistry_state_layout, only: air5_num_conservative
  use chemistry_flow_state, only: air5_primitive_to_conservative, &
    air5_conservative_to_primitive,configure_air5_species_cache
  use chemistry_source, only: air5_source_mode_coupled, &
    air5_source_mode_chemical,air5_source_mode_vt,air5_source_mode_frozen
  implicit none
  private

  integer, save :: active_source_mode=air5_source_mode_coupled
  logical, save :: source_mode_configured=.false.
  logical, save :: convection_species_budget=.false.
  logical, save :: convection_consistent_species=.false.
  logical, save :: convection_symmetric_species=.false.
  logical, save :: convection_limiter_configured=.false.
  logical, save :: diffusion_layered=.false.,diffusion_limiter_configured=.false.
  logical, save :: integration_control_configured=.false.
  integer, protected, save, public :: air5_tolerance_choice=0
  logical, protected, save, public :: air5_history_enabled=.false.
  real(real64), protected, save, public :: air5_chemistry_rtol=1.0e-9_real64
  real(real64), protected, save, public :: air5_chemistry_atol_factor=1.0e-13_real64
  real(real64), allocatable, save, public :: air5_step_history(:,:,:)

  public :: air5_field_primitive_to_conservative
  public :: air5_field_conservative_to_primitive
  public :: air5_reacting_flowtype
  public :: air5_postshock_flowtype
  public :: air5_normal_shock_flowtype
  public :: air5_open_x_flowtype
  public :: air5_hbl_flowtype
  public :: air5_shock_capturing_enabled
  public :: configure_air5_source_mode
  public :: air5_active_source_mode
  public :: air5_convection_species_budget
  public :: air5_consistent_species_convection
  public :: air5_symmetric_species_convection
  public :: air5_layered_diffusion
  public :: configure_air5_integration_control

contains

  subroutine configure_air5_integration_control(im,jm,km)
    use mpi
    integer, intent(in) :: im,jm,km
    character(32) :: value
    integer :: status,ierr,local(3),lowest(3),highest(3),rank

    if(integration_control_configured) return
    local=0
    call get_environment_variable('ASTR_AIR5_CHEMISTRY_TOLERANCE',value,status=status)
    if(status/=1) then
      if(status/=0) local(3)=1
      select case(trim(value))
      case('baseline'); local(1)=0
      case('loose'); local(1)=1
      case('tight'); local(1)=2
      case default; local(3)=1
      end select
    endif
    call get_environment_variable('ASTR_AIR5_STEP_HISTORY',value,status=status)
    if(status/=1) then
      if(status/=0) local(3)=1
      select case(trim(value))
      case('off'); local(2)=0
      case('previous'); local(2)=1
      case default; local(3)=1
      end select
    endif
    call MPI_Allreduce(local,lowest,3,MPI_INTEGER,MPI_MIN,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,ierr,status)
    call MPI_Allreduce(local,highest,3,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,ierr,status)
    call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)
    if(highest(3)/=0.or.any(lowest/=highest)) then
      if(rank==0) write(*,'(A)') 'invalid or rank-inconsistent AIR5 integration control options'
      call MPI_Abort(MPI_COMM_WORLD,1,ierr)
    endif
    air5_tolerance_choice=local(1)
    select case(air5_tolerance_choice)
    case(1)
      air5_chemistry_rtol=1.0e-8_real64
      air5_chemistry_atol_factor=1.0e-12_real64
    case(2)
      air5_chemistry_rtol=1.0e-10_real64
      air5_chemistry_atol_factor=1.0e-14_real64
    end select
    air5_history_enabled=local(2)==1
    if(air5_history_enabled) then
      allocate(air5_step_history(0:im,0:jm,0:km))
      air5_step_history=0.0_real64
    endif
    integration_control_configured=.true.
    if(rank==0) write(*,'(A,I0,A,L1)') 'AIR5_INTEGRATION_CONTROL tolerance=', &
      air5_tolerance_choice,' history=',air5_history_enabled
  end subroutine configure_air5_integration_control

  pure logical function air5_reacting_flowtype(flowtype)
    character(len=*), intent(in) :: flowtype

    select case(trim(flowtype))
    case('air5reactor','air5postshock','air5normalshock','air5tgv','air5hbl','air5sbli')
      air5_reacting_flowtype=.true.
    case default
      air5_reacting_flowtype=.false.
    end select
  end function air5_reacting_flowtype

  pure logical function air5_postshock_flowtype(flowtype)
    character(len=*), intent(in) :: flowtype

    air5_postshock_flowtype=trim(flowtype)=='air5postshock'
  end function air5_postshock_flowtype

  pure logical function air5_normal_shock_flowtype(flowtype)
    character(len=*), intent(in) :: flowtype

    air5_normal_shock_flowtype=trim(flowtype)=='air5normalshock'
  end function air5_normal_shock_flowtype

  pure logical function air5_open_x_flowtype(flowtype)
    character(len=*), intent(in) :: flowtype

    air5_open_x_flowtype=air5_postshock_flowtype(flowtype) .or. &
      air5_normal_shock_flowtype(flowtype)
  end function air5_open_x_flowtype

  pure logical function air5_hbl_flowtype(flowtype)
    character(len=*), intent(in) :: flowtype

    air5_hbl_flowtype=trim(flowtype)=='air5hbl' .or. trim(flowtype)=='air5sbli'
  end function air5_hbl_flowtype

  logical function air5_shock_capturing_enabled()
    use commvar, only: lcomb,conschm,recon_schem,lchardecomp

    air5_shock_capturing_enabled=lcomb .and. trim(conschm)=='643e' .and. &
      recon_schem==3 .and. .not.lchardecomp
  end function air5_shock_capturing_enabled

  subroutine configure_air5_source_mode()
    use mpi
    character(len=16) :: value
    integer :: choice,lowest,highest,status,ierr,rank

    if(source_mode_configured) return
    value=''
    call get_environment_variable('ASTR_AIR5_SOURCE_MODE',value,status=status)
    choice=-1
    if(status==1 .or. (status==0 .and. len_trim(value)==0)) then
      choice=air5_source_mode_coupled
      value='coupled'
    endif
    if(status==0) then
      select case(trim(adjustl(value)))
      case('coupled')
        choice=air5_source_mode_coupled
      case('chemical')
        choice=air5_source_mode_chemical
      case('vt')
        choice=air5_source_mode_vt
      case('frozen')
        choice=air5_source_mode_frozen
      end select
    endif
    call MPI_Allreduce(choice,lowest,1,MPI_INTEGER,MPI_MIN,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,ierr,status)
    call MPI_Allreduce(choice,highest,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,ierr,status)
    call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,ierr,status)
    if(lowest<0 .or. lowest/=highest) then
      if(rank==0) write(*,'(A)') &
        'Invalid or inconsistent ASTR_AIR5_SOURCE_MODE: expected coupled, chemical, vt, or frozen'
      call MPI_Abort(MPI_COMM_WORLD,1,ierr)
    endif
    active_source_mode=choice
    source_mode_configured=.true.
    if(rank==0) write(*,'(A,A)') 'ASTR_AIR5_SOURCE_MODE=',trim(adjustl(value))
    if(rank==0.and.choice==air5_source_mode_frozen) &
      write(*,'(A)') 'AIR5 frozen diagnostic: chemistry and V-T sources disabled'
    call configure_air5_convection_limiter()
  end subroutine configure_air5_source_mode

  subroutine configure_air5_convection_limiter()
    use mpi
    use commvar, only: use_gpu
    character(len=32) :: value
    integer :: choice,lowest,highest,status,ierr,rank

    if(convection_limiter_configured) return
    value=''
    call get_environment_variable('ASTR_AIR5_CONVECTION_LIMITER',value,status=status)
    choice=-1
    if(status==1 .or. (status==0 .and. len_trim(value)==0)) then
      value='full_state'
      choice=0
    elseif(status==0) then
      select case(trim(adjustl(value)))
      case('full_state'); choice=0
      case('species_budget'); choice=1
      case('consistent_species'); choice=2
      case('symmetric_species'); choice=3
      end select
    endif
    call MPI_Allreduce(choice,lowest,1,MPI_INTEGER,MPI_MIN,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,ierr,status)
    call MPI_Allreduce(choice,highest,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,ierr,status)
    call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,ierr,status)
    if(lowest<0 .or. lowest/=highest) then
      if(rank==0) write(*,'(A)') &
        'Invalid ASTR_AIR5_CONVECTION_LIMITER: full_state/species_budget/consistent_species/symmetric_species required'
      call MPI_Abort(MPI_COMM_WORLD,1,ierr)
    endif
    convection_species_budget=choice==1
    convection_consistent_species=choice==2
    convection_symmetric_species=choice==3
    call configure_air5_species_cache(convection_symmetric_species)
    convection_limiter_configured=.true.
    if(rank==0) write(*,'(A,A)') 'ASTR_AIR5_CONVECTION_LIMITER=',trim(adjustl(value))
  end subroutine configure_air5_convection_limiter

  logical function air5_convection_species_budget()
    if(.not.convection_limiter_configured) call configure_air5_convection_limiter()
    air5_convection_species_budget=convection_species_budget
  end function air5_convection_species_budget

  logical function air5_consistent_species_convection()
    if(.not.convection_limiter_configured) call configure_air5_convection_limiter()
    air5_consistent_species_convection=convection_consistent_species
  end function air5_consistent_species_convection

  logical function air5_symmetric_species_convection()
    if(.not.convection_limiter_configured) call configure_air5_convection_limiter()
    air5_symmetric_species_convection=convection_symmetric_species
  end function air5_symmetric_species_convection

  logical function air5_layered_diffusion()
    use mpi
    character(len=32) :: value
    integer :: choice,lowest,highest,status,ierr,rank

    if(.not.diffusion_limiter_configured) then
      value=''
      call get_environment_variable('ASTR_AIR5_DIFFUSION_LIMITER',value,status=status)
      choice=-1
      if(status==1 .or. (status==0 .and. len_trim(value)==0)) then
        value='full_state'
        choice=0
      elseif(status==0) then
        select case(trim(adjustl(value)))
        case('full_state'); choice=0
        case('layered'); choice=1
        end select
      endif
      call MPI_Allreduce(choice,lowest,1,MPI_INTEGER,MPI_MIN,MPI_COMM_WORLD,ierr)
      if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,ierr,status)
      call MPI_Allreduce(choice,highest,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
      if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,ierr,status)
      call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)
      if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,ierr,status)
      if(lowest<0 .or. lowest/=highest) then
        if(rank==0) write(*,'(A)') &
          'Invalid or inconsistent ASTR_AIR5_DIFFUSION_LIMITER: expected full_state or layered'
        call MPI_Abort(MPI_COMM_WORLD,1,ierr)
      endif
      diffusion_layered=choice==1
      diffusion_limiter_configured=.true.
      if(rank==0) write(*,'(A,A)') 'ASTR_AIR5_DIFFUSION_LIMITER=',trim(adjustl(value))
    endif
    air5_layered_diffusion=diffusion_layered
  end function air5_layered_diffusion

  integer function air5_active_source_mode()
    if(.not.source_mode_configured) &
      error stop 'fixed air5 source mode is not configured'
    air5_active_source_mode=active_source_mode
  end function air5_active_source_mode

  subroutine air5_field_primitive_to_conservative(q,rho,velocity,temperature, &
      mass_fraction,tv,status,failed_index)
    real(real64), intent(inout) :: q(0:,0:,0:,:)
    real(real64), intent(in) :: rho(0:,0:,0:),velocity(0:,0:,0:,:), &
      temperature(0:,0:,0:),mass_fraction(0:,0:,0:,:),tv(0:,0:,0:)
    integer, intent(out) :: status,failed_index(3)
    real(real64) :: local_q(air5_num_conservative)
    integer :: i,j,k

    status=chemistry_status_ok
    failed_index=-1
    if(size(q,4)/=air5_num_conservative .or. &
       size(mass_fraction,4)/=air5_num_species) then
      status=-1
      return
    endif
    do k=0,ubound(q,3)
      do j=0,ubound(q,2)
        do i=0,ubound(q,1)
          call air5_primitive_to_conservative(rho(i,j,k),velocity(i,j,k,:), &
            temperature(i,j,k),mass_fraction(i,j,k,:),tv(i,j,k),local_q,status)
          if(status/=chemistry_status_ok) then
            failed_index=[i,j,k]
            return
          endif
          q(i,j,k,:)=local_q
        enddo
      enddo
    enddo
  end subroutine air5_field_primitive_to_conservative

  subroutine air5_field_conservative_to_primitive(q,rho,velocity,pressure, &
      temperature,mass_fraction,tv,status,failed_index)
    real(real64), intent(in) :: q(0:,0:,0:,:)
    real(real64), intent(inout) :: rho(0:,0:,0:),velocity(0:,0:,0:,:), &
      pressure(0:,0:,0:),temperature(0:,0:,0:), &
      mass_fraction(0:,0:,0:,:),tv(0:,0:,0:)
    integer, intent(out) :: status,failed_index(3)
    integer :: i,j,k

    status=chemistry_status_ok
    failed_index=-1
    if(size(q,4)/=air5_num_conservative .or. &
       size(mass_fraction,4)/=air5_num_species) then
      status=-1
      return
    endif
    do k=0,ubound(q,3)
      do j=0,ubound(q,2)
        do i=0,ubound(q,1)
          call air5_conservative_to_primitive(q(i,j,k,:),rho(i,j,k), &
            velocity(i,j,k,:),temperature(i,j,k),mass_fraction(i,j,k,:), &
            tv(i,j,k),pressure(i,j,k),status)
          if(status/=chemistry_status_ok) then
            failed_index=[i,j,k]
            return
          endif
        enddo
      enddo
    enddo
  end subroutine air5_field_conservative_to_primitive

end module chemistry_flow_runtime
