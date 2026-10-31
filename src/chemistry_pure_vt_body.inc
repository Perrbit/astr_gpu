    real(real64), intent(in) :: rho_species(air5_num_species),temperature,tv
    real(real64), intent(out) :: source
    integer, intent(out) :: status
    real(real64) :: concentration(air5_num_species),mole_fraction(air5_num_species)
    real(real64) :: pair_rate(air5_num_species),concentration_sum,pressure
    real(real64) :: mw_time,sigma,reduced_mass,mean_speed,collision_coefficient,collision_rate
    real(real64) :: product,denominator,inverse_time,equilibrium_energy,current_energy,energy_difference
    integer :: molecule,partner,local_status

    source=0.0_real64
    call air5_validate_partial_densities(rho_species,status)
    if(status/=chemistry_status_ok) return
    if(.not.air5_is_finite(temperature).or..not.air5_is_finite(tv)) then
      status=chemistry_status_nonfinite
      return
    endif
    if(temperature<air5_temperature_min_k.or.temperature>air5_temperature_max_k.or. &
       tv<air5_temperature_min_k.or.tv>air5_temperature_max_k) then
      status=chemistry_status_out_of_domain
      return
    endif
    call air5_pressure(rho_species,temperature,pressure,local_status)
    if(local_status/=chemistry_status_ok) then
      status=local_status
      return
    endif
    if(.not.air5_pressure_is_in_domain(pressure).or. &
       .not.air5_relaxation_uses_relaxing_species_density) then
      status=chemistry_status_out_of_domain
      return
    endif
    concentration_sum=0.0_real64
    do partner=1,air5_num_species
      concentration(partner)=rho_species(partner)/air5_molar_mass(partner)
      concentration_sum=concentration_sum+concentration(partner)
    enddo
    do partner=1,air5_num_species
      mole_fraction(partner)=concentration(partner)/concentration_sum
    enddo
    do molecule=1,air5_num_species
      if(air5_theta_v(molecule)<=0.0_real64) cycle
      if(rho_species(molecule)<=0.0_real64) cycle
      do partner=1,air5_num_species
        mw_time=pressure_atmosphere_pa/pressure*exp(air5_mw_a(molecule,partner)* &
          (temperature**(-1.0_real64/3.0_real64)-air5_mw_b(molecule,partner))-18.42_real64)
        if(.not.air5_is_finite(mw_time)) then
          status=chemistry_status_nonfinite
          return
        endif
        if(mw_time<=0.0_real64) then
          status=chemistry_status_out_of_domain
          return
        endif
        if(air5_collision_sigma0(molecule,partner)>0.0_real64) then
          sigma=air5_collision_sigma0(molecule,partner)*temperature**air5_collision_sigma_power(molecule,partner)
          reduced_mass=air5_molar_mass(molecule)*air5_molar_mass(partner)/ &
            (air5_molar_mass(molecule)+air5_molar_mass(partner))
          mean_speed=sqrt(8.0_real64*air5_ru*temperature/(pi*reduced_mass))
          collision_coefficient=((avogadro/air5_molar_mass(molecule))*sigma)*mean_speed
          if(.not.air5_is_finite(sigma).or..not.air5_is_finite(mean_speed).or. &
             .not.air5_is_finite(collision_coefficient)) then
            status=chemistry_status_nonfinite
            return
          endif
          collision_rate=collision_coefficient*rho_species(molecule)
          product=mw_time*collision_rate
          denominator=1.0_real64+product
          if(.not.air5_is_finite(collision_rate).or..not.air5_is_finite(product).or. &
             .not.air5_is_finite(denominator)) then
            status=chemistry_status_nonfinite
            return
          endif
          pair_rate(partner)=collision_rate/denominator
        else
          pair_rate(partner)=1.0_real64/mw_time
        endif
        if(.not.air5_is_finite(pair_rate(partner))) then
          status=chemistry_status_nonfinite
          return
        endif
      enddo
      inverse_time=0.0_real64
      do partner=1,air5_num_species
        inverse_time=inverse_time+mole_fraction(partner)*pair_rate(partner)
      enddo
      equilibrium_energy=air5_species_vibrational_energy(molecule,temperature)
      current_energy=air5_species_vibrational_energy(molecule,tv)
      energy_difference=equilibrium_energy-current_energy
      source=source+rho_species(molecule)*energy_difference*inverse_time
    enddo
    status=chemistry_status_ok
    if(.not.air5_is_finite(source)) status=chemistry_status_nonfinite
