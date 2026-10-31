    if(present(activity_threshold)) then
      if(.not.air5_is_finite(activity_threshold)) then
        status=chemistry_status_invalid_tolerance
        return
      endif
      if(activity_threshold<0.0_real64) then
        status=chemistry_status_invalid_tolerance
        return
      endif
      if(activity_threshold>0.0_real64) then
        call air5_activity_source(rho,momentum,q5,state,screen_source,status,AIR5_SCREEN_MODE, &
          production,loss,energy_parts,temperatures)
        rhs_evaluations=1
        if(status/=chemistry_status_ok) return
        call air5_activity_index(rho,state,duration,production,loss,energy_parts,temperatures, &
          activity,activity_components)
        if(activity<huge(1.0_real64).and.activity<=activity_threshold) then
          if(present(skipped)) skipped=.true.
          suggested_step=duration
          return
        endif
      endif
    endif
