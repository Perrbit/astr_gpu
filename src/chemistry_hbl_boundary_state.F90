module chemistry_hbl_boundary_state
  use iso_fortran_env, only: real64
  use chemistry_air5_data, only: air5_num_species,air5_molar_mass,air5_ru
  use chemistry_model, only: chemistry_status_ok,chemistry_status_invalid_density
  use chemistry_state_layout, only: air5_num_conservative
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
      q_outflow,status)
    real(real64), intent(in) :: q_inner_one(air5_num_conservative)
    real(real64), intent(in) :: q_inner_two(air5_num_conservative)
    real(real64), intent(out) :: q_outflow(air5_num_conservative)
    integer, intent(out) :: status
    real(real64) :: density,temperature,tv,pressure,velocity(3)
    real(real64) :: mass_fraction(air5_num_species)

    q_outflow=(4.0_real64*q_inner_one-q_inner_two)/3.0_real64
    call air5_conservative_to_primitive(q_outflow,density,velocity,temperature, &
      mass_fraction,tv,pressure,status)
    if(status/=chemistry_status_ok) q_outflow=0.0_real64
  end subroutine build_air5_hbl_outflow_state

end module chemistry_hbl_boundary_state
