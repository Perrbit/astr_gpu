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
  logical, save :: characteristic_top=.false.
  real(real64), save :: top_relaxation_rate=0.0_real64
  real(real64), save :: top_reference_sound=0.0_real64
  real(real64), save :: incident_x=0.0_real64
  real(real64), save :: incident_q(air5_num_conservative,2)=0.0_real64
  logical, save :: boundary_configured=.false.

  public :: configure_air5_hbl_boundary
  public :: get_air5_hbl_boundary
  public :: apply_air5_hbl_boundary
  public :: get_air5_incident_top
  public :: air5_dynamic_top,get_air5_top_target,get_air5_top_rate
  public :: get_air5_top_reference_sound
  public :: write_air5_top_checkpoint

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
    call configure_air5_top_mode()
    call check_air5_top_checkpoint()
    boundary_configured=.true.
  end subroutine configure_air5_hbl_boundary

  subroutine configure_air5_top_mode()
    use chemistry_thermo, only: air5_species_gas_constant,air5_species_cv_tr
    use mpi
    use ieee_arithmetic, only: ieee_is_finite
    use commvar, only: im,jm,km,use_gpu,lrestart,lfilter,rkscheme
    use commarray, only: x
    character(len=64) :: value
    integer :: status,choice,low,high,ierr,axis,i,j,k
    real(real64) :: tau,tau_low,tau_high,spacing(3),expected(3),scale
    real(real64) :: density,velocity(3),temperature,y(5),tv,pressure,gas,cv

    value='prescribed'
    call get_environment_variable('ASTR_AIR5_TOP_MODE',value,status=status)
    if(status==1) value='prescribed'
    choice=-1
    if(status==0.or.status==1) then
      if(trim(value)=='prescribed') choice=0
      if(trim(value)=='characteristic') choice=1
    endif
    call MPI_Allreduce(choice,low,1,MPI_INTEGER,MPI_MIN,MPI_COMM_WORLD,ierr)
    call MPI_Allreduce(choice,high,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
    if(low<0.or.low/=high) call MPI_Abort(MPI_COMM_WORLD,90,ierr)
    characteristic_top=choice==1
    top_relaxation_rate=0.0_real64
    top_reference_sound=0.0_real64
    if(.not.characteristic_top) return
    ! Reject unwired combinations explicitly during staged implementation.
    if(rkscheme/='rk3') &
      error stop 'characteristic AIR5 top currently requires RK3'
    if(lfilter) then
      value=''
      call get_environment_variable('ASTR_AIR5_FILTER_VALIDATION',value,status=status)
      choice=0
      if(status==0.and.trim(value)=='on') choice=1
      call MPI_Allreduce(choice,low,1,MPI_INTEGER,MPI_MIN,MPI_COMM_WORLD,ierr)
      if(low/=1) error stop 'characteristic AIR5 filter requires explicit validation opt-in'
    endif
    if(lrestart) then
      value=''
      call get_environment_variable('ASTR_AIR5_TOP_RESTART_VALIDATION',value,status=status)
      choice=0
      if(status==0.and.trim(value)=='on') choice=1
      call MPI_Allreduce(choice,low,1,MPI_INTEGER,MPI_MIN,MPI_COMM_WORLD,ierr)
      if(low/=1) error stop 'characteristic AIR5 restart requires explicit validation opt-in'
    endif
    if(use_gpu) then
      value=''
      call get_environment_variable('ASTR_AIR5_TOP_GPU_VALIDATION',value,status=status)
      choice=0
      if(status==0.and.trim(value)=='on') choice=1
      call MPI_Allreduce(choice,low,1,MPI_INTEGER,MPI_MIN,MPI_COMM_WORLD,ierr)
      if(low/=1) error stop 'characteristic AIR5 GPU top requires explicit validation opt-in'
    endif
    value=''
    tau=-1.0_real64
    call get_environment_variable('ASTR_AIR5_TOP_TAU',value,status=status)
    if(status==0) read(value,*,iostat=status) tau
    choice=0
    if(status/=0.or..not.ieee_is_finite(tau)) choice=1
    if(tau<=0.0_real64) choice=1
    call MPI_Allreduce(choice,high,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
    if(high/=0) error stop 'characteristic AIR5 top requires explicit positive finite tau in seconds'
    call MPI_Allreduce(tau,tau_low,1,MPI_DOUBLE_PRECISION,MPI_MIN,MPI_COMM_WORLD,ierr)
    call MPI_Allreduce(tau,tau_high,1,MPI_DOUBLE_PRECISION,MPI_MAX,MPI_COMM_WORLD,ierr)
    if(tau_low/=tau_high) error stop 'inconsistent AIR5 top tau across ranks'
    top_relaxation_rate=1.0_real64/tau
    if(.not.ieee_is_finite(top_relaxation_rate)) error stop 'AIR5 top relaxation rate overflow'
    call air5_conservative_to_primitive(farfield_q,density,velocity,temperature,y,tv,pressure,status)
    if(status/=chemistry_status_ok) error stop 'invalid AIR5 top reference state'
    gas=0.0_real64; cv=0.0_real64
    do axis=1,5
      gas=gas+y(axis)*air5_species_gas_constant(axis)
      cv=cv+y(axis)*air5_species_cv_tr(axis)
    enddo
    top_reference_sound=sqrt((1.0_real64+gas/cv)*pressure/density)
    if(.not.ieee_is_finite(top_reference_sound).or.top_reference_sound<=0.0_real64) &
      error stop 'invalid AIR5 top reference sound speed'
    choice=0
    if(min(im,jm,km)<5) choice=1
    spacing=[x(1,0,0,1)-x(0,0,0,1),x(0,1,0,2)-x(0,0,0,2),x(0,0,1,3)-x(0,0,0,3)]
    if(any(spacing<=0.0_real64).or.any(.not.ieee_is_finite(spacing))) choice=1
    do k=0,km; do j=0,jm; do i=0,im
      expected=x(0,0,0,:)+[i,j,k]*spacing
      do axis=1,3
        scale=max(abs(expected(axis)),spacing(axis))
        if(.not.ieee_is_finite(x(i,j,k,axis))) choice=1
        if(abs(x(i,j,k,axis)-expected(axis))>2.0e-11_real64*scale) choice=1
      enddo
    enddo; enddo; enddo
    call MPI_Allreduce(choice,high,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
    if(high/=0) error stop 'characteristic AIR5 top requires uniform Cartesian local grids with >=6 nodes'
  end subroutine configure_air5_top_mode

  function air5_top_contract() result(values)
    real(real64) :: values(38)
    values(1:5)=[top_relaxation_rate,wall_temperature,hbl_x_origin,incident_x, &
      real(merge(1,0,incident_top),real64)]
    values(6:16)=farfield_q
    values(17:38)=reshape(incident_q,[22])
  end function air5_top_contract

  subroutine write_air5_top_checkpoint()
    use hdf5io, only: h5write
    integer :: m
    real(real64) :: values(38)
    character(len=32) :: name
    if(.not.boundary_configured) return
    call h5write(varname='air5_top_version',var=merge(2,1,characteristic_top))
    call h5write(varname='air5_top_mode',var=merge(1,0,characteristic_top))
    if(.not.characteristic_top) return
    values=air5_top_contract()
    do m=1,38
      write(name,'(A,I2.2)') 'air5_top_contract_',m
      call h5write(varname=trim(name),var=values(m))
    enddo
  end subroutine write_air5_top_checkpoint

  subroutine check_air5_top_checkpoint()
    use mpi
    use commvar, only: lrestart
    use hdf5io
    use ieee_arithmetic, only: ieee_is_finite
    integer :: version,mode,bad,global_bad,m,ierr,env_status
    logical :: present_field
    real(real64) :: values(38),saved
    character(len=32) :: name,option
    if(.not.lrestart) return
#ifdef HDF5
    bad=0; version=0; mode=0
    call h5io_init('outdat/flowfield.h5',mode='read')
    call h5lexists_f(h5file_id,'air5_top_version',present_field,ierr)
    if(ierr/=0) bad=1
    if(present_field) then
      call h5read('air5_top_version',version)
      if(version/=merge(2,1,characteristic_top)) bad=1
      call h5lexists_f(h5file_id,'air5_top_mode',present_field,ierr)
      if(ierr/=0.or..not.present_field) bad=1
      if(present_field) call h5read('air5_top_mode',mode)
    endif
    if(mode/=merge(1,0,characteristic_top)) bad=1
    if(characteristic_top.and.version/=2) bad=1
    if(mode==1) then
      option=''
      call get_environment_variable('ASTR_AIR5_COMPENSATION',option,status=env_status)
      if(env_status/=0.or.trim(option)/='on') bad=1
      option='restore'
      call get_environment_variable('ASTR_AIR5_COMPENSATION_RESTART',option,status=env_status)
      if(env_status/=1) then
        if(env_status/=0.or.trim(option)/='restore') bad=1
      endif
      values=air5_top_contract()
      do m=1,38
        write(name,'(A,I2.2)') 'air5_top_contract_',m
        call h5lexists_f(h5file_id,trim(name),present_field,ierr)
        if(ierr/=0.or..not.present_field) bad=1
        if(present_field) then
          call h5read(trim(name),saved)
          if(.not.ieee_is_finite(saved)) bad=1
          if(saved/=values(m)) bad=1
        endif
      enddo
      call h5lexists_f(h5file_id,'air5_compensation_version',present_field,ierr)
      if(ierr/=0.or..not.present_field) bad=1
    endif
    call h5io_end
    call MPI_Allreduce(bad,global_bad,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
    if(global_bad/=0) then
      print *, 'AIR5 top checkpoint mode/target/tau mismatch or missing metadata'
      call MPI_Abort(MPI_COMM_WORLD,91,ierr)
    endif
#else
    if(characteristic_top) error stop 'characteristic AIR5 restart requires HDF5'
#endif
  end subroutine check_air5_top_checkpoint

  logical function air5_dynamic_top()
    air5_dynamic_top=characteristic_top
  end function air5_dynamic_top

  real(real64) function get_air5_top_rate()
    get_air5_top_rate=top_relaxation_rate
  end function get_air5_top_rate

  real(real64) function get_air5_top_reference_sound()
    get_air5_top_reference_sound=top_reference_sound
  end function get_air5_top_reference_sound

  subroutine get_air5_top_target(x_value,state,status)
    real(real64),intent(in) :: x_value
    real(real64),intent(out) :: state(11)
    integer,intent(out) :: status
    status=chemistry_status_ok
    if(incident_top) then
      state=incident_q(:,2)
      if(x_value<incident_x) state=incident_q(:,1)
    else
      call build_air5_hbl_similarity_farfield_state(farfield_q,hbl_x_origin,x_value,state,status)
    endif
  end subroutine get_air5_top_target

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
    real(real64) :: density,velocity(3),temperature,fractions(5),tv,pressure

    if(trim(flowtype)/='air5hbl' .and. trim(flowtype)/='air5sbli') return
    if(.not.boundary_configured) &
      error stop 'fixed air5 HBL boundary is not configured'
    status=chemistry_status_ok

    if(mpiright==MPI_PROC_NULL) then
      do k=0,km
        do j=0,jm
          if(characteristic_top.and.mpiup==MPI_PROC_NULL.and.j==jm) then
            do i=im+1,im+hm
              q(i,j,k,:)=q(im,j,k,:)
            enddo
            cycle
          endif
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
          if(characteristic_top) then
            boundary_q=q(i,jm,k,:)
            call air5_conservative_to_primitive(boundary_q,density,velocity,temperature, &
              fractions,tv,pressure,status)
            if(status/=chemistry_status_ok) exit
          elseif(incident_top) then
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
      if(mpiright==MPI_PROC_NULL) then
        if(characteristic_top.and.mpiup==MPI_PROC_NULL) then
          air5_carry(im,0:jm-1,:,:)=0.0_real64
        else
          air5_carry(im,:,:,:)=0.0_real64
        endif
      endif
      if(mpidown==MPI_PROC_NULL) air5_carry(:,0,:,:)=0.0_real64
      if(mpiup==MPI_PROC_NULL.and..not.characteristic_top) air5_carry(:,jm,:,:)=0.0_real64
    endif
  end subroutine apply_air5_hbl_boundary

end module chemistry_hbl_boundary
