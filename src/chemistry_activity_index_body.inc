    real(real64), intent(in) :: rho,state(6),duration,production(5),loss(5),energy_parts(2),temperatures(2)
    real(real64), intent(out) :: activity,components(9)
    real(real64) :: temperature,tv,ev_specific,cv_tr,cv_v,thermal_bound,vibrational_bound
    real(real64) :: species_scale,energy_scale,temperature_scale,tv_scale,exchange
    integer :: species,item

    activity=huge(1.0_real64)
    components=huge(1.0_real64)
    if(.not.air5_is_finite(rho).or..not.air5_is_finite(duration)) return
    if(rho<=0.0_real64.or.duration<=0.0_real64) return
    do item=1,6
      if(.not.air5_is_finite(state(item))) return
    enddo
    do item=1,2
      if(.not.air5_is_finite(energy_parts(item)).or..not.air5_is_finite(temperatures(item))) return
    enddo
    temperature=temperatures(1); tv=temperatures(2)
    if(temperature<=0.0_real64.or.tv<=0.0_real64) return
    cv_tr=0.0_real64; cv_v=0.0_real64
    thermal_bound=0.0_real64; vibrational_bound=0.0_real64
    do species=1,5
      if(.not.air5_is_finite(production(species)).or..not.air5_is_finite(loss(species))) return
      if(production(species)<0.0_real64.or.loss(species)<0.0_real64) return
      species_scale=1.0e-12_real64+1.0e-6_real64*abs(state(species)/rho)
      components(species)=((duration*max(production(species),loss(species)))/rho)/species_scale
      cv_tr=cv_tr+state(species)*air5_species_cv_tr(species)
      cv_v=cv_v+state(species)*air5_species_vibrational_cv(species,tv)
      exchange=production(species)+loss(species)
      thermal_bound=thermal_bound+abs(air5_formation_energy(species)+ &
        air5_species_cv_tr(species)*temperature)*exchange
      vibrational_bound=vibrational_bound+abs(air5_species_vibrational_energy(species,tv))*exchange
    enddo
    if(.not.air5_is_finite(cv_tr).or..not.air5_is_finite(cv_v)) return
    if(cv_tr<=0.0_real64.or.cv_v<=0.0_real64) return
    ev_specific=state(6)/rho
    energy_scale=1.0e-10_real64*max(abs(ev_specific),1.0_real64)+1.0e-7_real64*abs(ev_specific)
    temperature_scale=1.0e-5_real64+1.0e-7_real64*abs(temperature)
    tv_scale=1.0e-5_real64+1.0e-7_real64*abs(tv)
    components(6:7)=((duration*abs(energy_parts))/rho)/energy_scale
    thermal_bound=thermal_bound+abs(energy_parts(1))+abs(energy_parts(2))
    vibrational_bound=vibrational_bound+abs(energy_parts(1))+abs(energy_parts(2))
    components(8)=((duration*thermal_bound)/cv_tr)/temperature_scale
    components(9)=((duration*vibrational_bound)/cv_v)/tv_scale
    do item=1,9
      if(.not.air5_is_finite(components(item))) return
    enddo
    activity=maxval(components)
