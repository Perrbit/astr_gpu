module chemistry_flow_runtime
  use iso_fortran_env, only: real64
  use chemistry_air5_data, only: air5_num_species
  use chemistry_model, only: chemistry_status_ok
  use chemistry_state_layout, only: air5_num_conservative
  use chemistry_flow_state, only: air5_primitive_to_conservative, &
    air5_conservative_to_primitive
  use chemistry_source, only: air5_source_mode_coupled, &
    air5_source_mode_chemical,air5_source_mode_vt
  implicit none
  private

  integer, save :: active_source_mode=air5_source_mode_coupled
  logical, save :: source_mode_configured=.false.

  public :: air5_field_primitive_to_conservative
  public :: air5_field_conservative_to_primitive
  public :: air5_reacting_flowtype
  public :: air5_postshock_flowtype
  public :: air5_hbl_flowtype
  public :: air5_shock_capturing_enabled
  public :: configure_air5_source_mode
  public :: air5_active_source_mode

contains

  pure logical function air5_reacting_flowtype(flowtype)
    character(len=*), intent(in) :: flowtype

    select case(trim(flowtype))
    case('air5reactor','air5postshock','air5tgv','air5hbl')
      air5_reacting_flowtype=.true.
    case default
      air5_reacting_flowtype=.false.
    end select
  end function air5_reacting_flowtype

  pure logical function air5_postshock_flowtype(flowtype)
    character(len=*), intent(in) :: flowtype

    air5_postshock_flowtype=trim(flowtype)=='air5postshock'
  end function air5_postshock_flowtype

  pure logical function air5_hbl_flowtype(flowtype)
    character(len=*), intent(in) :: flowtype

    air5_hbl_flowtype=trim(flowtype)=='air5hbl'
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
        'Invalid or inconsistent ASTR_AIR5_SOURCE_MODE: expected coupled, chemical, or vt'
      call MPI_Abort(MPI_COMM_WORLD,1,ierr)
    endif
    active_source_mode=choice
    source_mode_configured=.true.
    if(rank==0) write(*,'(A,A)') 'ASTR_AIR5_SOURCE_MODE=',trim(adjustl(value))
  end subroutine configure_air5_source_mode

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
