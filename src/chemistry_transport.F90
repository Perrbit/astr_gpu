module chemistry_transport
  use iso_fortran_env, only: real64
  use ieee_arithmetic, only: ieee_is_finite
  use chemistry_air5_data, only: air5_num_species,air5_molar_mass, &
    air5_blottner,air5_binary_diffusion,air5_atom_count,air5_formation_energy
  use chemistry_model, only: chemistry_status_ok,chemistry_status_invalid_composition, &
    chemistry_status_nonfinite,chemistry_status_out_of_domain, &
    air5_pressure_is_in_domain
  use chemistry_thermo, only: air5_species_gas_constant,air5_species_cv_tr, &
    air5_species_vibrational_energy,air5_species_vibrational_cv
  implicit none
  private

  public :: air5_transport_properties
  public :: air5_diffusive_flux

contains

  pure subroutine air5_transport_properties(temperature,tv,pressure,mass_fraction, &
      viscosity,conductivity_tr,conductivity_v,species_viscosity,binary_diffusion, &
      mixture_diffusion,status)
    real(real64), intent(in) :: temperature,tv,pressure
    real(real64), intent(in) :: mass_fraction(air5_num_species)
    real(real64), intent(out) :: viscosity,conductivity_tr,conductivity_v
    real(real64), intent(out) :: species_viscosity(air5_num_species)
    real(real64), intent(out) :: binary_diffusion(air5_num_species,air5_num_species)
    real(real64), intent(out) :: mixture_diffusion(air5_num_species)
    integer, intent(out) :: status
    real(real64) :: mole_fraction(air5_num_species),phi(air5_num_species,air5_num_species)
    real(real64) :: species_conductivity(air5_num_species), &
      species_vibrational_conductivity(air5_num_species)
    real(real64) :: log_temperature,mole_sum,denominator,numerator
    integer :: species,partner

    viscosity=0.0_real64
    conductivity_tr=0.0_real64
    conductivity_v=0.0_real64
    species_viscosity=0.0_real64
    binary_diffusion=0.0_real64
    mixture_diffusion=0.0_real64
    if(.not.ieee_is_finite(temperature) .or. .not.ieee_is_finite(tv) .or. &
       .not.ieee_is_finite(pressure) .or. .not.all(ieee_is_finite(mass_fraction))) then
      status=chemistry_status_nonfinite
      return
    endif
    if(temperature<=0.0_real64 .or. tv<=0.0_real64 .or. &
       .not.air5_pressure_is_in_domain(pressure)) then
      status=chemistry_status_out_of_domain
      return
    endif
    if(any(mass_fraction<0.0_real64) .or. &
       abs(sum(mass_fraction)-1.0_real64)>64.0_real64*epsilon(1.0_real64)) then
      status=chemistry_status_invalid_composition
      return
    endif

    mole_sum=sum(mass_fraction/air5_molar_mass)
    if(mole_sum<=0.0_real64) then
      status=chemistry_status_invalid_composition
      return
    endif
    mole_fraction=(mass_fraction/air5_molar_mass)/mole_sum
    log_temperature=log(temperature)
    do species=1,air5_num_species
      species_viscosity(species)=0.1_real64*exp( &
        air5_blottner(1,species)*log_temperature*log_temperature+ &
        air5_blottner(2,species)*log_temperature+air5_blottner(3,species))
      if(sum(air5_atom_count(:,species))==1) then
        species_conductivity(species)=3.75_real64*species_viscosity(species)* &
          air5_species_gas_constant(species)
      else
        species_conductivity(species)=4.75_real64*species_viscosity(species)* &
          air5_species_gas_constant(species)
      endif
      species_vibrational_conductivity(species)=species_viscosity(species)* &
        air5_species_vibrational_cv(species,tv)
    enddo

    do species=1,air5_num_species
      do partner=1,air5_num_species
        phi(species,partner)=(1.0_real64+sqrt(species_viscosity(species)/ &
          species_viscosity(partner))*(air5_molar_mass(partner)/ &
          air5_molar_mass(species))**0.25_real64)**2/ &
          sqrt(8.0_real64*(1.0_real64+air5_molar_mass(species)/ &
          air5_molar_mass(partner)))
      enddo
      denominator=sum(mole_fraction*phi(species,:))
      viscosity=viscosity+mole_fraction(species)*species_viscosity(species)/denominator
      conductivity_tr=conductivity_tr+ &
        mole_fraction(species)*species_conductivity(species)/denominator
      conductivity_v=conductivity_v+ &
        mole_fraction(species)*species_vibrational_conductivity(species)/denominator
    enddo

    do species=1,air5_num_species
      do partner=1,air5_num_species
        binary_diffusion(species,partner)=10.1325_real64/pressure* &
          exp(air5_binary_diffusion(4,species,partner))* &
          temperature**(air5_binary_diffusion(1,species,partner)* &
          log_temperature*log_temperature+air5_binary_diffusion(2,species,partner)* &
          log_temperature+air5_binary_diffusion(3,species,partner))
      enddo
    enddo

    do species=1,air5_num_species
      numerator=sum(mole_fraction,mask=[(partner/=species,partner=1,air5_num_species)])
      denominator=0.0_real64
      do partner=1,air5_num_species
        if(partner/=species) denominator=denominator+ &
          mole_fraction(partner)/binary_diffusion(species,partner)
      enddo
      if(numerator==0.0_real64) then
        mixture_diffusion(species)=0.0_real64
      elseif(denominator>0.0_real64) then
        mixture_diffusion(species)=numerator/denominator
      else
        status=chemistry_status_invalid_composition
        return
      endif
    enddo
    if(.not.ieee_is_finite(viscosity) .or. .not.ieee_is_finite(conductivity_tr) .or. &
       .not.ieee_is_finite(conductivity_v) .or. &
       .not.all(ieee_is_finite(binary_diffusion)) .or. &
       .not.all(ieee_is_finite(mixture_diffusion))) then
      status=chemistry_status_nonfinite
    else
      status=chemistry_status_ok
    endif
  end subroutine air5_transport_properties

  pure subroutine air5_diffusive_flux(rho,velocity,temperature,tv,pressure, &
      mass_fraction,grad_velocity,grad_temperature,grad_tv,grad_mass_fraction, &
      momentum_flux,species_flux,energy_flux,ev_flux,status)
    real(real64), intent(in) :: rho,velocity(3),temperature,tv,pressure
    real(real64), intent(in) :: mass_fraction(air5_num_species)
    real(real64), intent(in) :: grad_velocity(3,3),grad_temperature(3),grad_tv(3)
    real(real64), intent(in) :: grad_mass_fraction(air5_num_species,3)
    real(real64), intent(out) :: momentum_flux(3,3)
    real(real64), intent(out) :: species_flux(air5_num_species,3)
    real(real64), intent(out) :: energy_flux(3),ev_flux(3)
    integer, intent(out) :: status
    real(real64) :: viscosity,conductivity_tr,conductivity_v, &
      species_viscosity(air5_num_species),binary_diffusion(air5_num_species,air5_num_species), &
      mixture_diffusion(air5_num_species),raw_flux(air5_num_species)
    real(real64) :: divergence,trace_correction,enthalpy(air5_num_species), &
      vibrational_energy(air5_num_species)
    integer :: direction,species,i,j

    momentum_flux=0.0_real64
    species_flux=0.0_real64
    energy_flux=0.0_real64
    ev_flux=0.0_real64
    if(.not.ieee_is_finite(rho) .or. rho<=0.0_real64 .or. &
       .not.all(ieee_is_finite(velocity)) .or. &
       .not.all(ieee_is_finite(grad_velocity)) .or. &
       .not.all(ieee_is_finite(grad_temperature)) .or. &
       .not.all(ieee_is_finite(grad_tv)) .or. &
       .not.all(ieee_is_finite(grad_mass_fraction))) then
      status=chemistry_status_nonfinite
      return
    endif
    call air5_transport_properties(temperature,tv,pressure,mass_fraction,viscosity, &
      conductivity_tr,conductivity_v,species_viscosity,binary_diffusion, &
      mixture_diffusion,status)
    if(status/=chemistry_status_ok) return

    divergence=grad_velocity(1,1)+grad_velocity(2,2)+grad_velocity(3,3)
    do j=1,3
      do i=1,3
        momentum_flux(i,j)=viscosity*(grad_velocity(i,j)+grad_velocity(j,i))
        if(i==j) momentum_flux(i,j)=momentum_flux(i,j)- &
          (2.0_real64/3.0_real64)*viscosity*divergence
      enddo
    enddo

    do species=1,air5_num_species
      vibrational_energy(species)=air5_species_vibrational_energy(species,tv)
      enthalpy(species)=air5_species_cv_tr(species)*temperature+ &
        vibrational_energy(species)+air5_formation_energy(species)+ &
        air5_species_gas_constant(species)*temperature
    enddo
    do direction=1,3
      raw_flux=-rho*mixture_diffusion*grad_mass_fraction(:,direction)
      trace_correction=sum(raw_flux)
      species_flux(:,direction)=raw_flux-mass_fraction*trace_correction
      energy_flux(direction)=dot_product(momentum_flux(:,direction),velocity)+ &
        conductivity_tr*grad_temperature(direction)+conductivity_v*grad_tv(direction)- &
        dot_product(enthalpy,species_flux(:,direction))
      ev_flux(direction)=conductivity_v*grad_tv(direction)- &
        dot_product(vibrational_energy,species_flux(:,direction))
    enddo
    if(.not.all(ieee_is_finite(momentum_flux)) .or. &
       .not.all(ieee_is_finite(species_flux)) .or. &
       .not.all(ieee_is_finite(energy_flux)) .or. &
       .not.all(ieee_is_finite(ev_flux))) status=chemistry_status_nonfinite
  end subroutine air5_diffusive_flux

end module chemistry_transport
