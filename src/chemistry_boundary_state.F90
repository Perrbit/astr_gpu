module chemistry_hbl_profile
  use iso_fortran_env, only: real64
  use chemistry_air5_data, only: air5_num_species,air5_molar_mass,air5_ru
  use chemistry_model, only: chemistry_status_ok
  use chemistry_state_layout, only: air5_num_conservative
  use chemistry_flow_state, only: air5_primitive_to_conservative, &
    air5_conservative_to_primitive
  implicit none
  private

  integer, parameter, public :: air5_hbl_profile_status_ok=0
  integer, parameter, public :: air5_hbl_profile_status_io=1
  integer, parameter, public :: air5_hbl_profile_status_format=2
  integer, parameter, public :: air5_hbl_profile_status_invalid_state=3
  integer, parameter, public :: air5_hbl_profile_status_unconfigured=4
  integer, parameter :: air5_hbl_profile_columns=13

  type, public :: air5_hbl_profile_type
    private
    real(real64), allocatable :: values(:,:)
  end type air5_hbl_profile_type

  public :: read_air5_hbl_profile
  public :: sample_air5_hbl_profile
  public :: air5_hbl_profile_bounds
  public :: air5_hbl_profile_point_count

contains

  subroutine read_air5_hbl_profile(path,profile,status)
    character(len=*), intent(in) :: path
    type(air5_hbl_profile_type), intent(inout) :: profile
    integer, intent(out) :: status
    character(len=2048) :: line
    real(real64) :: row(air5_hbl_profile_columns)
    integer :: unit,ios,row_count,index

    status=air5_hbl_profile_status_ok
    if(allocated(profile%values)) deallocate(profile%values)
    open(newunit=unit,file=path,status='old',action='read',iostat=ios)
    if(ios/=0) then
      status=air5_hbl_profile_status_io
      return
    endif
    row_count=0
    do
      read(unit,'(A)',iostat=ios) line
      if(ios<0) exit
      if(ios>0) then
        status=air5_hbl_profile_status_io
        close(unit)
        return
      endif
      line=adjustl(line)
      if(len_trim(line)==0 .or. line(1:1)=='#') cycle
      read(line,*,iostat=ios) row
      if(ios/=0) then
        status=air5_hbl_profile_status_format
        close(unit)
        return
      endif
      row_count=row_count+1
    enddo
    if(row_count<2) then
      status=air5_hbl_profile_status_format
      close(unit)
      return
    endif

    rewind(unit)
    allocate(profile%values(air5_hbl_profile_columns,row_count))
    index=0
    do
      read(unit,'(A)',iostat=ios) line
      if(ios<0) exit
      if(ios>0) then
        status=air5_hbl_profile_status_io
        exit
      endif
      line=adjustl(line)
      if(len_trim(line)==0 .or. line(1:1)=='#') cycle
      read(line,*,iostat=ios) row
      if(ios/=0) then
        status=air5_hbl_profile_status_format
        exit
      endif
      index=index+1
      profile%values(:,index)=row
      call validate_air5_hbl_profile_row(row,status)
      if(status/=air5_hbl_profile_status_ok) exit
      if(index>1) then
        if(profile%values(1,index)<=profile%values(1,index-1)) then
          status=air5_hbl_profile_status_format
          exit
        endif
      endif
    enddo
    close(unit)
    if(status/=air5_hbl_profile_status_ok) then
      deallocate(profile%values)
      return
    endif
    if(index/=row_count) then
      deallocate(profile%values)
      status=air5_hbl_profile_status_format
    endif
  end subroutine read_air5_hbl_profile

  subroutine validate_air5_hbl_profile_row(row,status)
    real(real64), intent(in) :: row(air5_hbl_profile_columns)
    integer, intent(out) :: status
    real(real64) :: q(air5_num_conservative),velocity(3)
    real(real64) :: mass_fraction(air5_num_species),density,temperature,tv,pressure
    real(real64) :: mixture_gas_constant,expected_density,scale
    integer :: state_status

    status=air5_hbl_profile_status_invalid_state
    if(row(1)<0.0_real64 .or. row(2)<=0.0_real64 .or. row(6)<=0.0_real64 .or. &
       row(7)<=0.0_real64 .or. row(8)<=0.0_real64) return
    mass_fraction=row(9:13)
    if(any(mass_fraction<0.0_real64)) return
    if(abs(sum(mass_fraction)-1.0_real64)>64.0_real64*epsilon(1.0_real64)) return
    mixture_gas_constant=sum(mass_fraction*air5_ru/air5_molar_mass)
    expected_density=row(6)/(mixture_gas_constant*row(7))
    scale=max(abs(expected_density),1.0_real64)
    if(abs(row(2)-expected_density)>2.0e-11_real64*scale) return
    velocity=row(3:5)
    call air5_primitive_to_conservative(row(2),velocity,row(7),mass_fraction, &
      row(8),q,state_status)
    if(state_status/=chemistry_status_ok) return
    call air5_conservative_to_primitive(q,density,velocity,temperature, &
      mass_fraction,tv,pressure,state_status)
    if(state_status/=chemistry_status_ok) return
    scale=max(abs(row(6)),1.0_real64)
    if(abs(pressure-row(6))>2.0e-11_real64*scale) return
    status=air5_hbl_profile_status_ok
  end subroutine validate_air5_hbl_profile_row

  subroutine sample_air5_hbl_profile(profile,y,q,status)
    type(air5_hbl_profile_type), intent(in) :: profile
    real(real64), intent(in) :: y
    real(real64), intent(out) :: q(air5_num_conservative)
    integer, intent(out) :: status
    real(real64) :: row(air5_hbl_profile_columns),weight
    real(real64) :: velocity(3),mass_fraction(air5_num_species)
    real(real64) :: mixture_gas_constant,density
    integer :: lower,upper,middle,state_status,n

    q=0.0_real64
    if(.not.allocated(profile%values)) then
      status=air5_hbl_profile_status_unconfigured
      return
    endif
    n=size(profile%values,2)
    if(y<=profile%values(1,1)) then
      row=profile%values(:,1)
    elseif(y>=profile%values(1,n)) then
      row=profile%values(:,n)
    else
      lower=1
      upper=n
      do while(upper-lower>1)
        middle=(lower+upper)/2
        if(profile%values(1,middle)<=y) then
          lower=middle
        else
          upper=middle
        endif
      enddo
      weight=(y-profile%values(1,lower))/ &
        (profile%values(1,upper)-profile%values(1,lower))
      row=(1.0_real64-weight)*profile%values(:,lower)+ &
        weight*profile%values(:,upper)
    endif

    mass_fraction=row(9:13)
    mixture_gas_constant=sum(mass_fraction*air5_ru/air5_molar_mass)
    density=row(6)/(mixture_gas_constant*row(7))
    velocity=row(3:5)
    call air5_primitive_to_conservative(density,velocity,row(7),mass_fraction, &
      row(8),q,state_status)
    if(state_status/=chemistry_status_ok) then
      status=air5_hbl_profile_status_invalid_state
      q=0.0_real64
      return
    endif
    status=air5_hbl_profile_status_ok
  end subroutine sample_air5_hbl_profile

  subroutine air5_hbl_profile_bounds(profile,y_min,y_max,status)
    type(air5_hbl_profile_type), intent(in) :: profile
    real(real64), intent(out) :: y_min,y_max
    integer, intent(out) :: status

    y_min=0.0_real64
    y_max=0.0_real64
    if(.not.allocated(profile%values)) then
      status=air5_hbl_profile_status_unconfigured
      return
    endif
    y_min=profile%values(1,1)
    y_max=profile%values(1,size(profile%values,2))
    status=air5_hbl_profile_status_ok
  end subroutine air5_hbl_profile_bounds

  integer function air5_hbl_profile_point_count(profile)
    type(air5_hbl_profile_type), intent(in) :: profile

    air5_hbl_profile_point_count=0
    if(allocated(profile%values)) &
      air5_hbl_profile_point_count=size(profile%values,2)
  end function air5_hbl_profile_point_count

end module chemistry_hbl_profile

module chemistry_hbl_initial_field
  use iso_fortran_env, only: real64
  use ieee_arithmetic, only: ieee_is_finite
  use chemistry_air5_data, only: air5_num_species
  use chemistry_model, only: chemistry_status_ok
  use chemistry_state_layout, only: air5_num_conservative
  use chemistry_flow_state, only: air5_conservative_to_primitive
  implicit none
  private

  integer, parameter, public :: air5_hbl_initial_status_ok=0
  integer, parameter, public :: air5_hbl_initial_status_io=1
  integer, parameter, public :: air5_hbl_initial_status_format=2
  integer, parameter, public :: air5_hbl_initial_status_invalid_state=3
  integer, parameter, public :: air5_hbl_initial_columns=13

  type, public :: air5_hbl_initial_field_type
    private
    real(real64), allocatable :: x(:)
    real(real64), allocatable :: y(:)
    real(real64), allocatable :: q(:,:,:)
  end type air5_hbl_initial_field_type

  public :: read_air5_hbl_initial_field
  public :: sample_air5_hbl_initial_field

contains

  subroutine read_next_initial_data_line(unit,line,ios)
    integer, intent(in) :: unit
    character(len=*), intent(out) :: line
    integer, intent(out) :: ios

    do
      read(unit,'(A)',iostat=ios) line
      if(ios/=0) return
      line=adjustl(line)
      if(len_trim(line)>0 .and. line(1:1)/='#') return
    enddo
  end subroutine read_next_initial_data_line

  subroutine clear_air5_hbl_initial_field(field)
    type(air5_hbl_initial_field_type), intent(inout) :: field

    if(allocated(field%x)) deallocate(field%x)
    if(allocated(field%y)) deallocate(field%y)
    if(allocated(field%q)) deallocate(field%q)
  end subroutine clear_air5_hbl_initial_field

  subroutine read_air5_hbl_initial_field(path,field,status)
    character(len=*), intent(in) :: path
    type(air5_hbl_initial_field_type), intent(inout) :: field
    integer, intent(out) :: status
    character(len=2048) :: line
    real(real64) :: row(air5_hbl_initial_columns),scale
    real(real64) :: density,velocity(3),temperature,mass_fraction(air5_num_species)
    real(real64) :: tv,pressure
    integer :: unit,ios,nx,ny,ix,iy,state_status

    call clear_air5_hbl_initial_field(field)
    status=air5_hbl_initial_status_ok
    open(newunit=unit,file=path,status='old',action='read',iostat=ios)
    if(ios/=0) then
      status=air5_hbl_initial_status_io
      return
    endif
    call read_next_initial_data_line(unit,line,ios)
    if(ios/=0) then
      status=air5_hbl_initial_status_format
      close(unit)
      return
    endif
    read(line,*,iostat=ios) nx,ny
    if(ios/=0 .or. min(nx,ny)<2) then
      status=air5_hbl_initial_status_format
      close(unit)
      return
    endif
    allocate(field%x(nx),field%y(ny),field%q(air5_num_conservative,nx,ny))

    do ix=1,nx
      do iy=1,ny
        call read_next_initial_data_line(unit,line,ios)
        if(ios/=0) then
          status=air5_hbl_initial_status_format
          exit
        endif
        read(line,*,iostat=ios) row
        if(ios/=0 .or. .not.all(ieee_is_finite(row))) then
          status=air5_hbl_initial_status_format
          exit
        endif
        if(iy==1) then
          field%x(ix)=row(1)
        else
          scale=max(abs(field%x(ix)),1.0_real64)
          if(abs(row(1)-field%x(ix))>2.0e-12_real64*scale) then
            status=air5_hbl_initial_status_format
            exit
          endif
        endif
        if(ix==1) then
          field%y(iy)=row(2)
        else
          scale=max(abs(field%y(iy)),1.0_real64)
          if(abs(row(2)-field%y(iy))>2.0e-12_real64*scale) then
            status=air5_hbl_initial_status_format
            exit
          endif
        endif
        field%q(:,ix,iy)=row(3:air5_hbl_initial_columns)
        call air5_conservative_to_primitive(field%q(:,ix,iy),density,velocity, &
          temperature,mass_fraction,tv,pressure,state_status)
        if(state_status/=chemistry_status_ok) then
          status=air5_hbl_initial_status_invalid_state
          exit
        endif
      enddo
      if(status/=air5_hbl_initial_status_ok) exit
    enddo
    close(unit)
    if(status==air5_hbl_initial_status_ok) then
      if(any(field%x(2:)<=field%x(:nx-1)) .or. &
         any(field%y(2:)<=field%y(:ny-1))) status=air5_hbl_initial_status_format
    endif
    if(status/=air5_hbl_initial_status_ok) call clear_air5_hbl_initial_field(field)
  end subroutine read_air5_hbl_initial_field

  subroutine coordinate_bracket(coordinate,value,lower,weight)
    real(real64), intent(in) :: coordinate(:),value
    integer, intent(out) :: lower
    real(real64), intent(out) :: weight
    integer :: upper,middle,n

    n=size(coordinate)
    if(value<=coordinate(1)) then
      lower=1
      weight=0.0_real64
      return
    elseif(value>=coordinate(n)) then
      lower=n-1
      weight=1.0_real64
      return
    endif
    lower=1
    upper=n
    do while(upper-lower>1)
      middle=(lower+upper)/2
      if(coordinate(middle)<=value) then
        lower=middle
      else
        upper=middle
      endif
    enddo
    weight=(value-coordinate(lower))/(coordinate(lower+1)-coordinate(lower))
  end subroutine coordinate_bracket

  subroutine sample_air5_hbl_initial_field(field,x,y,q,status)
    type(air5_hbl_initial_field_type), intent(in) :: field
    real(real64), intent(in) :: x,y
    real(real64), intent(out) :: q(air5_num_conservative)
    integer, intent(out) :: status
    real(real64) :: wx,wy,density,velocity(3),temperature
    real(real64) :: mass_fraction(air5_num_species),tv,pressure
    integer :: ix,iy,state_status

    q=0.0_real64
    if(.not.allocated(field%q)) then
      status=air5_hbl_initial_status_format
      return
    endif
    call coordinate_bracket(field%x,x,ix,wx)
    call coordinate_bracket(field%y,y,iy,wy)
    q=(1.0_real64-wx)*(1.0_real64-wy)*field%q(:,ix,iy)+ &
      wx*(1.0_real64-wy)*field%q(:,ix+1,iy)+ &
      (1.0_real64-wx)*wy*field%q(:,ix,iy+1)+ &
      wx*wy*field%q(:,ix+1,iy+1)
    call air5_conservative_to_primitive(q,density,velocity,temperature, &
      mass_fraction,tv,pressure,state_status)
    if(state_status/=chemistry_status_ok) then
      status=air5_hbl_initial_status_invalid_state
      q=0.0_real64
      return
    endif
    status=air5_hbl_initial_status_ok
  end subroutine sample_air5_hbl_initial_field

end module chemistry_hbl_initial_field

module chemistry_hbl_boundary_state
  use iso_fortran_env, only: real64
  use chemistry_air5_data, only: air5_num_species,air5_molar_mass,air5_ru
  use chemistry_model, only: chemistry_status_ok,chemistry_status_invalid_density
  use chemistry_state_layout, only: air5_num_conservative,air5_idx_density, &
    air5_idx_species_first,air5_idx_species_last
  use chemistry_flow_state, only: air5_primitive_to_conservative, &
    air5_conservative_to_primitive
  implicit none
  private

  public :: build_air5_hbl_wall_state
  public :: build_air5_hbl_outflow_state

contains

  pure subroutine build_air5_hbl_wall_state(q_inner_one,q_inner_two, &
      wall_temperature,q_wall,status)
    real(real64), intent(in) :: q_inner_one(air5_num_conservative)
    real(real64), intent(in) :: q_inner_two(air5_num_conservative)
    real(real64), intent(in) :: wall_temperature
    real(real64), intent(out) :: q_wall(air5_num_conservative)
    integer, intent(out) :: status
    real(real64) :: density_one,density_two,temperature_one,temperature_two
    real(real64) :: tv_one,tv_two,pressure_one,pressure_two,pressure_wall
    real(real64) :: velocity_one(3),velocity_two(3),velocity_wall(3)
    real(real64) :: y_one(air5_num_species),y_two(air5_num_species)
    real(real64) :: y_wall(air5_num_species),mixture_gas_constant,density_wall

    q_wall=0.0_real64
    call air5_conservative_to_primitive(q_inner_one,density_one,velocity_one, &
      temperature_one,y_one,tv_one,pressure_one,status)
    if(status/=chemistry_status_ok) return
    call air5_conservative_to_primitive(q_inner_two,density_two,velocity_two, &
      temperature_two,y_two,tv_two,pressure_two,status)
    if(status/=chemistry_status_ok) return
    pressure_wall=(4.0_real64*pressure_one-pressure_two)/3.0_real64
    if(pressure_wall<=0.0_real64) then
      status=chemistry_status_invalid_density
      return
    endif
    y_wall=y_one
    mixture_gas_constant=sum(y_wall*air5_ru/air5_molar_mass)
    density_wall=pressure_wall/(mixture_gas_constant*wall_temperature)
    velocity_wall=0.0_real64
    call air5_primitive_to_conservative(density_wall,velocity_wall, &
      wall_temperature,y_wall,wall_temperature,q_wall,status)
  end subroutine build_air5_hbl_wall_state

  pure subroutine build_air5_hbl_outflow_state(q_inner_one,q_inner_two, &
      q_outflow,status,limiter_theta)
    real(real64), intent(in) :: q_inner_one(air5_num_conservative)
    real(real64), intent(in) :: q_inner_two(air5_num_conservative)
    real(real64), intent(out) :: q_outflow(air5_num_conservative)
    integer, intent(out) :: status
    real(real64), intent(out), optional :: limiter_theta
    real(real64) :: density,temperature,tv,pressure,velocity(3)
    real(real64) :: mass_fraction(air5_num_species)
    real(real64) :: q_high(air5_num_conservative),q_trial(air5_num_conservative)
    real(real64) :: theta_lower,theta_upper,theta_trial,species_theta
    integer :: component,iteration,trial_status
    logical :: species_limited

    q_high=(4.0_real64*q_inner_one-q_inner_two)/3.0_real64
    q_outflow=q_high
    call air5_conservative_to_primitive(q_outflow,density,velocity,temperature, &
      mass_fraction,tv,pressure,status)
    if(status==chemistry_status_ok) then
      if(present(limiter_theta)) limiter_theta=1.0_real64
      return
    endif

    q_outflow=q_inner_one
    call air5_conservative_to_primitive(q_outflow,density,velocity,temperature, &
      mass_fraction,tv,pressure,status)
    if(status/=chemistry_status_ok) then
      q_outflow=0.0_real64
      if(present(limiter_theta)) limiter_theta=0.0_real64
      return
    endif

    q_trial=q_high
    species_theta=1.0_real64
    species_limited=.false.
    do component=air5_idx_species_first+1,air5_idx_species_last
      if(q_high(component)<0.0_real64) then
        species_limited=.true.
        if(q_inner_one(component)<=0.0_real64) then
          theta_trial=0.0_real64
        else
          theta_trial=q_inner_one(component)/ &
            (q_inner_one(component)-q_high(component))
          theta_trial=theta_trial*(1.0_real64-256.0_real64*epsilon(1.0_real64))
        endif
        theta_trial=max(0.0_real64,min(1.0_real64,theta_trial))
        q_trial(component)=q_inner_one(component)+theta_trial* &
          (q_high(component)-q_inner_one(component))
        species_theta=min(species_theta,theta_trial)
      endif
    enddo
    if(species_limited) then
      q_trial(air5_idx_species_first)=q_high(air5_idx_density)- &
        sum(q_trial(air5_idx_species_first+1:air5_idx_species_last))
      call air5_conservative_to_primitive(q_trial,density,velocity,temperature, &
        mass_fraction,tv,pressure,trial_status)
      if(trial_status==chemistry_status_ok) then
        q_outflow=q_trial
        status=chemistry_status_ok
        if(present(limiter_theta)) limiter_theta=species_theta
        return
      endif
    endif

    theta_upper=1.0_real64
    do component=air5_idx_species_first,air5_idx_species_last
      if(q_high(component)<0.0_real64) then
        if(q_inner_one(component)<=0.0_real64) then
          theta_upper=0.0_real64
        else
          theta_upper=min(theta_upper,q_inner_one(component)/ &
            (q_inner_one(component)-q_high(component)))
        endif
      endif
    enddo
    theta_upper=max(0.0_real64,min(1.0_real64,theta_upper))
    if(theta_upper<1.0_real64) &
      theta_upper=theta_upper*(1.0_real64-256.0_real64*epsilon(1.0_real64))
    q_trial=q_inner_one+theta_upper*(q_high-q_inner_one)
    call air5_conservative_to_primitive(q_trial,density,velocity,temperature, &
      mass_fraction,tv,pressure,trial_status)
    if(trial_status==chemistry_status_ok) then
      q_outflow=q_trial
      status=chemistry_status_ok
      if(present(limiter_theta)) limiter_theta=theta_upper
      return
    endif

    theta_lower=0.0_real64
    do iteration=1,64
      theta_trial=0.5_real64*(theta_lower+theta_upper)
      q_trial=q_inner_one+theta_trial*(q_high-q_inner_one)
      call air5_conservative_to_primitive(q_trial,density,velocity,temperature, &
        mass_fraction,tv,pressure,trial_status)
      if(trial_status==chemistry_status_ok) then
        theta_lower=theta_trial
        q_outflow=q_trial
      else
        theta_upper=theta_trial
      endif
    enddo
    status=chemistry_status_ok
    if(present(limiter_theta)) limiter_theta=theta_lower
  end subroutine build_air5_hbl_outflow_state

end module chemistry_hbl_boundary_state
