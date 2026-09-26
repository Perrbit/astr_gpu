module chemistry_postshock_boundary
  use iso_fortran_env, only: real64
  use chemistry_state_layout, only: air5_num_conservative
  implicit none
  private

  real(real64), save :: left_q(air5_num_conservative)=0.0_real64
  real(real64), save :: right_q(air5_num_conservative)=0.0_real64
  logical, save :: boundary_configured=.false.
  real(real64), save :: outlet_pressure=0.0_real64

  public :: configure_air5_postshock_boundary
  public :: get_air5_postshock_boundary
  public :: apply_air5_postshock_boundary

contains

  subroutine configure_air5_postshock_boundary(left_state,right_state,pressure_target)
    use chemistry_model, only: air5_pressure_is_in_domain
    real(real64), intent(in) :: left_state(air5_num_conservative)
    real(real64), intent(in) :: right_state(air5_num_conservative)
    real(real64), intent(in), optional :: pressure_target

    outlet_pressure=0.0_real64
    if(present(pressure_target)) then
      if(pressure_target/=0.0_real64) then
        if(.not.air5_pressure_is_in_domain(pressure_target)) &
          error stop 'invalid fixed air5 outlet pressure'
        outlet_pressure=pressure_target
      endif
    endif
    left_q=left_state
    right_q=right_state
    boundary_configured=.true.
  end subroutine configure_air5_postshock_boundary

  subroutine get_air5_postshock_boundary(left_state,right_state,pressure_target)
    real(real64), intent(out) :: left_state(air5_num_conservative)
    real(real64), intent(out) :: right_state(air5_num_conservative)
    real(real64), intent(out), optional :: pressure_target

    if(.not.boundary_configured) &
      error stop 'fixed air5 post-shock boundary is not configured'
    left_state=left_q
    right_state=right_q
    if(present(pressure_target)) pressure_target=outlet_pressure
  end subroutine get_air5_postshock_boundary

  subroutine apply_air5_postshock_boundary()
    use mpi
    use chemistry_model, only: chemistry_status_ok
    use chemistry_pressure_outlet_state, only: build_air5_pressure_outlet_state
    use commvar, only: im,jm,km,hm,flowtype
    use commarray, only: q
    use parallel, only: mpileft,mpiright
    integer :: component,i,j,k,status,global_status,ierr
    real(real64) :: boundary(air5_num_conservative)
    logical :: normal_shock_case

    normal_shock_case=trim(flowtype)=='air5normalshock'
    if(trim(flowtype)/='air5postshock' .and. .not.normal_shock_case) return
    if(.not.boundary_configured) &
      error stop 'fixed air5 post-shock boundary is not configured'
    status=chemistry_status_ok
    if(mpileft==MPI_PROC_NULL) then
      do component=1,air5_num_conservative
        q(-hm:0,0:jm,0:km,component)=left_q(component)
      enddo
    endif
    if(mpiright==MPI_PROC_NULL) then
      if(normal_shock_case .and. outlet_pressure>0.0_real64) then
        do k=0,km
          do j=0,jm
            call build_air5_pressure_outlet_state(q(im-1,j,k,:),outlet_pressure,boundary,status)
            if(status/=chemistry_status_ok) exit
            do i=im,im+hm
              q(i,j,k,:)=boundary
            enddo
          enddo
          if(status/=chemistry_status_ok) exit
        enddo
      elseif(normal_shock_case) then
        do component=1,air5_num_conservative
          do i=im,im+hm
            q(i,0:jm,0:km,component)=q(im-1,0:jm,0:km,component)
          enddo
        enddo
      else
        do component=1,air5_num_conservative
          q(im:im+hm,0:jm,0:km,component)=right_q(component)
        enddo
      endif
    endif
    if(normal_shock_case .and. outlet_pressure>0.0_real64) then
      call MPI_Allreduce(status,global_status,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
      if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,ierr,status)
      if(global_status/=chemistry_status_ok) then
        write(*,'(A,I0)') 'fixed air5 pressure outlet failed, status=',global_status
        call MPI_Abort(MPI_COMM_WORLD,global_status,ierr)
      endif
    endif
  end subroutine apply_air5_postshock_boundary

end module chemistry_postshock_boundary

module chemistry_hbl_boundary
  use iso_fortran_env, only: real64
  use chemistry_model, only: chemistry_status_ok
  use chemistry_state_layout, only: air5_num_conservative,air5_idx_species_first, &
    air5_idx_species_last
  use chemistry_flow_state, only: air5_conservative_to_primitive
  use chemistry_hbl_profile, only: air5_hbl_profile_type, &
    air5_hbl_profile_status_ok,sample_air5_hbl_profile,air5_hbl_profile_bounds, &
    air5_hbl_profile_x_origin
  use chemistry_hbl_boundary_state, only: build_air5_hbl_wall_state, &
    build_air5_hbl_outflow_state,build_air5_hbl_similarity_farfield_state
  implicit none
  private

  real(real64), allocatable, save :: inlet_q(:,:)
  real(real64), save :: farfield_q(air5_num_conservative)=0.0_real64
  real(real64), save :: wall_temperature=0.0_real64
  real(real64), save :: hbl_x_origin=0.0_real64
  logical, save :: incident_top=.false.
  real(real64), save :: incident_x=0.0_real64
  real(real64), save :: incident_q(air5_num_conservative,2)=0.0_real64
  logical, save :: boundary_configured=.false.

  public :: configure_air5_hbl_boundary
  public :: get_air5_hbl_boundary
  public :: apply_air5_hbl_boundary
  public :: get_air5_incident_top

contains

  subroutine configure_air5_hbl_boundary(profile)
    use chemistry_air5_data, only: air5_num_species
    use commvar, only: jm,flowtype,ref_len
    use chemistry_incident_shock_state, only: read_air5_incident_shock
    use chemistry_hbl_geometry, only: read_air5_hbl_domain
    use commarray, only: x
    type(air5_hbl_profile_type), intent(in) :: profile
    real(real64) :: y_min,y_max,wall_q(air5_num_conservative)
    real(real64) :: density,velocity(3),temperature,tv,pressure,top_y,normal(3)
    real(real64) :: mass_fraction(air5_num_species)
    real(real64) :: domain_lengths(3)
    integer :: j,status,state_status

    if(allocated(inlet_q)) deallocate(inlet_q)
    allocate(inlet_q(0:jm,air5_num_conservative))
    do j=0,jm
      call sample_air5_hbl_profile(profile,x(0,j,0,2),inlet_q(j,:),status)
      if(status/=air5_hbl_profile_status_ok) &
        error stop 'failed sampling fixed air5 HBL inlet profile'
    enddo
    call air5_hbl_profile_bounds(profile,y_min,y_max,status)
    if(status/=air5_hbl_profile_status_ok) &
      error stop 'fixed air5 HBL profile bounds are unavailable'
    call sample_air5_hbl_profile(profile,y_min,wall_q,status)
    if(status/=air5_hbl_profile_status_ok) &
      error stop 'failed sampling fixed air5 HBL wall state'
    call sample_air5_hbl_profile(profile,y_max,farfield_q,status)
    if(status/=air5_hbl_profile_status_ok) &
      error stop 'failed sampling fixed air5 HBL farfield state'
    call air5_hbl_profile_x_origin(profile,hbl_x_origin,status)
    if(status/=air5_hbl_profile_status_ok) &
      error stop 'fixed air5 HBL x origin is unavailable'
    call air5_conservative_to_primitive(wall_q,density,velocity,temperature, &
      mass_fraction,tv,pressure,state_status)
    if(state_status/=chemistry_status_ok) &
      error stop 'fixed air5 HBL wall state is invalid'
    if(abs(tv-temperature)>2.0e-11_real64*max(abs(temperature),1.0_real64)) &
      error stop 'fixed air5 HBL A0 requires the one-temperature limit at the wall'
    wall_temperature=temperature
    incident_top=trim(flowtype)=='air5sbli'
    incident_q=0.0_real64
    incident_x=0.0_real64
    if(incident_top) then
      call read_air5_incident_shock('datin/air5_incident_shock.dat',incident_x, &
        top_y,normal,incident_q,status)
      if(status/=chemistry_status_ok) error stop 'invalid air5 incident shock data'
      call read_air5_hbl_domain(flowtype,ref_len,domain_lengths)
      if(abs(top_y-domain_lengths(2))>2.0e-11_real64*ref_len .or. &
         incident_x>=domain_lengths(1)) error stop 'air5 incident shock lies outside domain'
      if(any(abs(incident_q(:,1)-farfield_q)>2.0e-11_real64* &
        max(abs(farfield_q),1.0e-280_real64))) &
        error stop 'air5 incident shock upstream does not match inlet profile edge'
    endif
    boundary_configured=.true.
  end subroutine configure_air5_hbl_boundary

  subroutine get_air5_hbl_boundary(inlet_state,farfield_state,wall_temp,x_origin)
    real(real64), allocatable, intent(out) :: inlet_state(:,:)
    real(real64), intent(out) :: farfield_state(air5_num_conservative),wall_temp
    real(real64), intent(out) :: x_origin

    if(.not.boundary_configured) &
      error stop 'fixed air5 HBL boundary is not configured'
    allocate(inlet_state(0:ubound(inlet_q,1),air5_num_conservative))
    inlet_state=inlet_q
    farfield_state=farfield_q
    wall_temp=wall_temperature
    x_origin=hbl_x_origin
  end subroutine get_air5_hbl_boundary

  subroutine get_air5_incident_top(enabled,x_top,states)
    logical, intent(out) :: enabled
    real(real64), intent(out) :: x_top,states(air5_num_conservative,2)
    if(.not.boundary_configured) error stop 'air5 boundary is not configured'
    enabled=incident_top
    x_top=incident_x
    states=incident_q
  end subroutine get_air5_incident_top

  subroutine apply_air5_hbl_boundary()
    use chemistry_compensation, only: air5_compensated,air5_carry
    use mpi
    use commvar, only: im,jm,km,hm,flowtype
    use commarray, only: q,x
    use parallel, only: mpileft,mpiright,mpidown,mpiup
    real(real64) :: boundary_q(air5_num_conservative)
    real(real64) :: candidate_q(air5_num_conservative)
    integer :: i,j,k,component,status,global_status,ierr

    if(trim(flowtype)/='air5hbl' .and. trim(flowtype)/='air5sbli') return
    if(.not.boundary_configured) &
      error stop 'fixed air5 HBL boundary is not configured'
    status=chemistry_status_ok

    if(mpiright==MPI_PROC_NULL) then
      do k=0,km
        do j=0,jm
          call build_air5_hbl_outflow_state(q(im-1,j,k,:),q(im-2,j,k,:), &
            boundary_q,status)
          if(status/=chemistry_status_ok) then
            candidate_q=(4.0_real64*q(im-1,j,k,:)-q(im-2,j,k,:))/3.0_real64
            write(*,'(A,3(I0,1X),A,I0)') &
              'fixed air5 HBL outflow state failed at i/j/k=',im,j,k, &
              'status=',status
            write(*,'(A,ES24.16E3,A,I0,A,ES24.16E3)') &
              'candidate minimum partial density=', &
              minval(candidate_q(air5_idx_species_first:air5_idx_species_last)), &
              ', local species=',minloc(candidate_q(air5_idx_species_first: &
              air5_idx_species_last),dim=1),', species sum residual=', &
              sum(candidate_q(air5_idx_species_first:air5_idx_species_last))- &
              candidate_q(1)
            write(*,'(A,5(ES24.16E3,1X))') 'inner one partial densities=', &
              q(im-1,j,k,air5_idx_species_first:air5_idx_species_last)
            write(*,'(A,5(ES24.16E3,1X))') 'inner two partial densities=', &
              q(im-2,j,k,air5_idx_species_first:air5_idx_species_last)
            exit
          endif
          do component=1,air5_num_conservative
            q(im:im+hm,j,k,component)=boundary_q(component)
          enddo
        enddo
        if(status/=chemistry_status_ok) exit
      enddo
    endif
    if(status==chemistry_status_ok .and. mpidown==MPI_PROC_NULL) then
      do k=0,km
        do i=0,im
          call build_air5_hbl_wall_state(q(i,1,k,:),q(i,2,k,:), &
            wall_temperature,boundary_q,status)
          if(status/=chemistry_status_ok) then
            write(*,'(A,3(I0,1X),A,I0)') &
              'fixed air5 HBL wall state failed at i/j/k=',i,0,k, &
              'status=',status
            exit
          endif
          do component=1,air5_num_conservative
            q(i,-hm:0,k,component)=boundary_q(component)
          enddo
        enddo
        if(status/=chemistry_status_ok) exit
      enddo
    endif
    if(status==chemistry_status_ok .and. mpiup==MPI_PROC_NULL) then
      do k=0,km
        do i=0,im
          if(incident_top) then
            if(x(i,jm,k,1)<incident_x) then
              boundary_q=incident_q(:,1)
            else
              boundary_q=incident_q(:,2)
            endif
          else
            call build_air5_hbl_similarity_farfield_state(farfield_q,hbl_x_origin, &
              x(i,jm,k,1),boundary_q,status)
          endif
          if(status/=chemistry_status_ok) exit
          do component=1,air5_num_conservative
            q(i,jm:jm+hm,k,component)=boundary_q(component)
          enddo
        enddo
        if(status/=chemistry_status_ok) exit
      enddo
    endif
    if(status==chemistry_status_ok .and. mpileft==MPI_PROC_NULL) then
      do k=0,km
        do j=0,jm
          do component=1,air5_num_conservative
            q(-hm:0,j,k,component)=inlet_q(j,component)
          enddo
        enddo
      enddo
    endif

    call MPI_Allreduce(status,global_status,1,MPI_INTEGER,MPI_MAX, &
      MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,ierr,status)
    if(global_status/=chemistry_status_ok) then
      write(*,'(A,I0)') 'fixed air5 HBL boundary failed, status=',global_status
      call MPI_Abort(MPI_COMM_WORLD,global_status,ierr)
    endif
    if(air5_compensated) then
      if(mpileft==MPI_PROC_NULL) air5_carry(0,:,:,:)=0.0_real64
      if(mpiright==MPI_PROC_NULL) air5_carry(im,:,:,:)=0.0_real64
      if(mpidown==MPI_PROC_NULL) air5_carry(:,0,:,:)=0.0_real64
      if(mpiup==MPI_PROC_NULL) air5_carry(:,jm,:,:)=0.0_real64
    endif
  end subroutine apply_air5_hbl_boundary

end module chemistry_hbl_boundary
