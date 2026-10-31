    real(real64), intent(in) :: rho,momentum(3),q5,state(6)
    real(real64), intent(out) :: source(6)
    integer, intent(out) :: status
    integer, intent(in) :: source_mode
#if AIR5_ACTIVITY_DIAGNOSTICS
    real(real64), intent(out) :: production(5),loss(5),energy_parts(2),temperatures(2)
#endif
    real(real64) :: temperature,tv,pressure,concentration(air5_num_species)
    real(real64) :: third_body,forward_product,reverse_product,rate_temperature
    real(real64) :: forward_coefficient,reverse_coefficient,equilibrium_constant
    real(real64) :: forward_rate,reverse_rate,progress_rate,reaction_progress(air5_num_reactions)
    real(real64) :: z10000,ln_kc,mass_factor,energy_per_mass,psi,unused_derivative,vt_source
    integer :: reaction,species,variable,net_stoichiometry,local_status

    source=0.0_real64
    reaction_progress=0.0_real64
#if AIR5_ACTIVITY_DIAGNOSTICS
    production=0.0_real64; loss=0.0_real64; energy_parts=0.0_real64; temperatures=0.0_real64
#endif
    if(source_mode<air5_source_mode_coupled.or.source_mode>air5_source_mode_frozen) then
      status=chemistry_status_invalid_source_mode
      return
    endif
    call air5_validate_partial_densities(state(1:air5_num_species),status)
    if(status/=chemistry_status_ok) return
    do variable=1,6
      if(.not.air5_is_finite(state(variable))) then
        status=chemistry_status_nonfinite
        return
      endif
    enddo
    if(state(6)<0.0_real64) then
      status=chemistry_status_out_of_domain
      return
    endif
    call air5_temperature_from_q5(rho,momentum,state(1:air5_num_species),state(6),q5,temperature,status)
    if(status/=chemistry_status_ok) return
    call air5_tv_from_ev(state(1:air5_num_species),state(6),tv,status)
    if(status/=chemistry_status_ok) return
    call air5_pressure(state(1:air5_num_species),temperature,pressure,status)
    if(status/=chemistry_status_ok) return
    if(.not.air5_pressure_is_in_domain(pressure)) then
      status=chemistry_status_out_of_domain
      return
    endif
#if AIR5_ACTIVITY_DIAGNOSTICS
    temperatures=[temperature,tv]
#endif
    if(source_mode==air5_source_mode_frozen) return
    if(source_mode==air5_source_mode_coupled.or.source_mode==air5_source_mode_chemical) then
      do species=1,air5_num_species
        concentration(species)=state(species)/air5_molar_mass(species)
      enddo
      do reaction=1,air5_num_reactions
        third_body=1.0_real64
        if(air5_has_third_body(reaction)) then
          third_body=0.0_real64
          do species=1,air5_num_species
            third_body=third_body+air5_third_body_efficiency(species,reaction)*concentration(species)
          enddo
        endif
        forward_product=air5_mass_action(concentration,air5_nu_reactant(:,reaction))
        reverse_product=air5_mass_action(concentration,air5_nu_product(:,reaction))
        if(air5_rate_uses_tv(reaction)) then
          rate_temperature=sqrt(temperature*tv)
        else
          rate_temperature=temperature
        endif
        forward_coefficient=air5_arrhenius_a(reaction)*rate_temperature**air5_arrhenius_b(reaction)* &
          exp(-air5_activation_temperature(reaction)/rate_temperature)
        z10000=10000.0_real64/temperature
        ln_kc=air5_ln_kc_coefficient(1,reaction)/z10000+air5_ln_kc_coefficient(2,reaction)+ &
          air5_ln_kc_coefficient(3,reaction)*log(z10000)+air5_ln_kc_coefficient(4,reaction)*z10000+ &
          air5_ln_kc_coefficient(5,reaction)*z10000*z10000
        equilibrium_constant=exp(ln_kc)
        reverse_coefficient=air5_arrhenius_a(reaction)*temperature**air5_arrhenius_b(reaction)* &
          exp(-air5_activation_temperature(reaction)/temperature)/equilibrium_constant
        forward_rate=forward_coefficient*third_body*forward_product
        reverse_rate=reverse_coefficient*third_body*reverse_product
        progress_rate=forward_rate-reverse_rate
        reaction_progress(reaction)=progress_rate
        do species=1,air5_num_species
#if AIR5_ACTIVITY_DIAGNOSTICS
          production(species)=production(species)+air5_molar_mass(species)*( &
            real(air5_nu_product(species,reaction),real64)*forward_rate+ &
            real(air5_nu_reactant(species,reaction),real64)*reverse_rate)
          loss(species)=loss(species)+air5_molar_mass(species)*( &
            real(air5_nu_reactant(species,reaction),real64)*forward_rate+ &
            real(air5_nu_product(species,reaction),real64)*reverse_rate)
#endif
          net_stoichiometry=air5_nu_product(species,reaction)-air5_nu_reactant(species,reaction)
          if(net_stoichiometry==0) cycle
          mass_factor=air5_molar_mass(species)*real(net_stoichiometry,real64)
          source(species)=source(species)+mass_factor*progress_rate
          if(air5_psi_bounds(2,species)<=0.0_real64) cycle
          if(air5_rate_uses_tv(reaction)) then
            call air5_preferential_factor(species,temperature,psi,unused_derivative)
            energy_per_mass=psi*air5_species_gas_constant(species)*air5_activation_temperature(reaction)
          else
            energy_per_mass=air5_species_vibrational_energy(species,tv)
          endif
          source(6)=source(6)+mass_factor*progress_rate*energy_per_mass
        enddo
      enddo
    endif
#if AIR5_ACTIVITY_DIAGNOSTICS
    energy_parts(1)=source(6)
#endif
    if(source_mode==air5_source_mode_coupled.or.source_mode==air5_source_mode_vt) then
      call air5_vt_relaxation_source(state(1:air5_num_species),temperature,tv,vt_source,local_status)
      if(local_status/=chemistry_status_ok) then
        status=local_status
        return
      endif
#if AIR5_ACTIVITY_DIAGNOSTICS
      energy_parts(2)=vt_source
#endif
      source(6)=source(6)+vt_source
    endif
    do variable=1,6
      if(.not.air5_is_finite(source(variable))) then
        status=chemistry_status_nonfinite
        return
      endif
    enddo
    do reaction=1,air5_num_reactions
      if(.not.air5_is_finite(reaction_progress(reaction))) then
        status=chemistry_status_nonfinite
        return
      endif
    enddo
    status=chemistry_status_ok
