    real(real64), value :: duration
    integer, value :: source_mode,is,ie,js,je,ks,ke
    integer, parameter :: max_attempts=200000
    integer :: id,nx,ny,ntot,i,j,k,status
    integer :: accepted,rejected,rhs_evaluations,jacobian_evaluations
    integer :: ignored
    logical :: skipped
    real(real64) :: density,momentum(3),state(6),final_state(6),atol(6),suggested_step
    real(real64) :: local_q(air5_num_conservative),velocity(3)
    real(real64) :: mass_fraction(air5_num_species),temperature,tv,pressure
    real(real64) :: trial_carry(6),trial_history

    id=(blockIdx%x-1)*blockDim%x+threadIdx%x-1
    nx=ie-is+1
    ny=je-js+1
    ntot=nx*ny*(ke-ks+1)
    if(id>=ntot) return
    i=is+mod(id,nx)
    j=js+mod(id/nx,ny)
    k=ks+id/(nx*ny)
    density=q_d(i,j,k,air5_idx_density)
    momentum=q_d(i,j,k,air5_idx_momentum_first:air5_idx_momentum_last)
    state=q_d(i,j,k,air5_idx_species_first:air5_idx_ev)
    atol(1:air5_num_species)=chemistry_atol_factor_d*density
    atol(6)=chemistry_atol_factor_d*max(abs(state(6)),1.0_real64)
    if(history_enabled_d) trial_history=air5_step_history_d(i,j,k)
    if(air5_compensated_d) then
      trial_carry=air5_carry_d(i,j,k,air5_idx_species_first:air5_idx_ev)
      if(history_enabled_d) then
        call air5_ros2_advance_gpu(density,momentum,q_d(i,j,k,air5_idx_total_energy), &
          state,duration,duration,chemistry_rtol_d,atol,max_attempts,final_state,suggested_step, &
          accepted,rejected,rhs_evaluations,jacobian_evaluations,status,source_mode,trial_carry,trial_history, &
          activity_threshold_d,skipped)
      else
      call air5_ros2_advance_gpu(density,momentum,q_d(i,j,k,air5_idx_total_energy), &
        state,duration,duration,chemistry_rtol_d,atol,max_attempts,final_state,suggested_step, &
        accepted,rejected,rhs_evaluations,jacobian_evaluations,status,source_mode,trial_carry, &
        activity_threshold=activity_threshold_d,skipped=skipped)
      endif
    else
    if(history_enabled_d) then
      call air5_ros2_advance_gpu(density,momentum,q_d(i,j,k,air5_idx_total_energy), &
        state,duration,duration,chemistry_rtol_d,atol,max_attempts,final_state,suggested_step, &
        accepted,rejected,rhs_evaluations,jacobian_evaluations,status,source_mode,step_history=trial_history, &
        activity_threshold=activity_threshold_d,skipped=skipped)
    else
    call air5_ros2_advance_gpu(density,momentum,q_d(i,j,k,air5_idx_total_energy), &
      state,duration,duration,chemistry_rtol_d,atol,max_attempts,final_state,suggested_step, &
      accepted,rejected,rhs_evaluations,jacobian_evaluations,status,source_mode, &
      activity_threshold=activity_threshold_d,skipped=skipped)
    endif
    endif
    if(status/=chemistry_status_ok) then
      call record_air5_chemistry_failure(i,j,k,status,1)
      call record_air5_chemistry_status(status)
      return
    endif
    local_q=q_d(i,j,k,1:air5_num_conservative)
    local_q(air5_idx_species_first:air5_idx_ev)=final_state
    call air5_conservative_to_primitive_gpu(local_q,density,velocity,temperature, &
      mass_fraction,tv,pressure,status)
    if(status/=chemistry_status_ok) then
      call record_air5_chemistry_failure(i,j,k,status,2)
      call record_air5_chemistry_status(status)
      return
    endif
    q_d(i,j,k,air5_idx_species_first:air5_idx_ev)=final_state
    if(air5_compensated_d) air5_carry_d(i,j,k,air5_idx_species_first:air5_idx_ev)=trial_carry
    if(history_enabled_d) air5_step_history_d(i,j,k)=trial_history
    rho_d(i,j,k)=density
    vel_d(i,j,k,:)=velocity
    prs_d(i,j,k)=pressure
    tmp_d(i,j,k)=temperature
    spc_d(i,j,k,:)=mass_fraction
    tve_d(i,j,k)=tv
    if(collect_diagnostics) then
      call record_air5_chemistry_diagnostics(final_state,temperature,tv,accepted, &
        rejected,rhs_evaluations,jacobian_evaluations)
      if(skipped) ignored=atomicAdd(air5_chemistry_skipped_d,1)
    endif
