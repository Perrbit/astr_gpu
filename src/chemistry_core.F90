module chemistry_model
  use iso_fortran_env, only: real64
  use ieee_arithmetic, only: ieee_is_finite
  use chemistry_air5_data, only: air5_mechanism_id, air5_num_species, &
    air5_pressure_min_pa, air5_pressure_max_pa
  implicit none
  private

  integer, parameter, public :: chemistry_status_ok = 0
  integer, parameter, public :: chemistry_status_invalid_mechanism = 1
  integer, parameter, public :: chemistry_status_invalid_density = 2
  integer, parameter, public :: chemistry_status_invalid_composition = 3
  integer, parameter, public :: chemistry_status_out_of_domain = 4
  integer, parameter, public :: chemistry_status_no_vibrational_capacity = 5
  integer, parameter, public :: chemistry_status_nonconverged = 6
  integer, parameter, public :: chemistry_status_nonfinite = 7
  integer, parameter, public :: chemistry_status_linear_failure = 8
  integer, parameter, public :: chemistry_status_invalid_timestep = 9
  integer, parameter, public :: chemistry_status_invalid_tolerance = 10
  integer, parameter, public :: chemistry_status_step_limit = 11
  integer, parameter, public :: chemistry_status_invalid_source_mode = 12
  integer, parameter, public :: chemistry_status_infeasible_flux = 13
  integer, parameter, public :: air5_ratio_bisection_iterations = digits(1.0_real64)
  ! Upper operation budget for six-face blending, budget evaluation and RK3
  ! recombination (computed fluxes are inputs). Not a physical species floor.
  integer, parameter :: air5_transport_roundoff_operations=128
  real(real64), parameter, public :: air5_transport_roundoff_gamma= &
    (air5_transport_roundoff_operations*(0.5_real64*epsilon(1.0_real64)))/ &
    (1.0_real64-air5_transport_roundoff_operations*(0.5_real64*epsilon(1.0_real64)))

  public :: air5_validate_mechanism_id
  public :: air5_validate_partial_densities
  public :: air5_validate_physical_species_state
  public :: air5_pressure_is_in_domain
  public :: air5_interior_ratio
  public :: air5_transport_species_ratio
  public :: air5_limit_filter_species
  public :: air5_close_species_flux
  public :: air5_close_species_flux_moment

contains

  pure subroutine air5_close_species_flux_moment(candidate,lower,upper,mass_flux, &
      weights,moment_flux,flux,status)
    ! Fixed-size active-set enumeration for a box and two affine constraints.
    ! Bounds and energy admissibility belong to the calling RK face algorithm.
    real(real64), intent(in) :: candidate(5),lower(5),upper(5),weights(5)
    real(real64), intent(in) :: mass_flux,moment_flux
    real(real64), intent(out) :: flux(5)
    integer, intent(out) :: status
    real(real64) :: scale_f,scale_w,c(5),lo(5),hi(5),g(5),trial(5),best(5)
    real(real64) :: target_mass,target_moment,r1,r2,mean_g,variance,cost,best_cost
    real(real64) :: tolerance,coefficient
    real(real64) :: mass_tolerance,moment_tolerance
    real(real64) :: dlo(5),dhi(5),delta_mass,delta_moment
    integer :: active_set,digits_left,s,kind,nfree,refinement
    logical :: free(5),found

    flux=candidate
    status=chemistry_status_nonfinite
    if(.not.all(ieee_is_finite(candidate)) .or. .not.all(ieee_is_finite(lower)) .or. &
      .not.all(ieee_is_finite(upper)) .or. .not.all(ieee_is_finite(weights)) .or. &
      .not.ieee_is_finite(mass_flux) .or. .not.ieee_is_finite(moment_flux)) return
    status=chemistry_status_infeasible_flux
    if(any(lower>upper)) return
    scale_w=maxval(abs(weights))
    if(scale_w==0.0_real64) then
      if(moment_flux/=0.0_real64) return
      call air5_close_species_flux(candidate,lower,upper,mass_flux,flux,status)
      return
    endif
    scale_f=max(maxval(abs(candidate)),maxval(abs(lower)),maxval(abs(upper)),abs(mass_flux))
    if(scale_f==0.0_real64) then
      if(moment_flux==0.0_real64) status=chemistry_status_ok
      return
    endif
    c=candidate/scale_f; lo=lower/scale_f; hi=upper/scale_f; g=weights/scale_w
    target_mass=mass_flux/scale_f; target_moment=(moment_flux/scale_w)/scale_f
    status=chemistry_status_nonconverged
    if(.not.ieee_is_finite(target_moment)) return
    if(any((lo==0.0_real64) .and. (lower/=0.0_real64)) .or. &
      any((hi==0.0_real64) .and. (upper/=0.0_real64))) return
    tolerance=64.0_real64*epsilon(1.0_real64)* &
      max(abs(target_mass),abs(target_moment),sum(abs(c)))
    if(all(candidate>=lower) .and. all(candidate<=upper)) then
      if(abs(sum(c)-target_mass)<=tolerance .and. &
        abs(dot_product(g,c)-target_moment)<=tolerance) then
        status=chemistry_status_ok
        return
      endif
    endif
    ! Solve for increments, not absolute bulk fluxes: subtracting two bulk
    ! sums after activating a trace bound loses the actual trace correction.
    dlo=(lower-candidate)/scale_f; dhi=(upper-candidate)/scale_f
    delta_mass=(mass_flux-sum(candidate))/scale_f
    delta_moment=((moment_flux-dot_product(weights,candidate))/scale_w)/scale_f
    ! The unchanged-candidate test above already accepts these backward errors.
    ! Do not inject them into trace species merely because a bound is active.
    ! Final output is still checked against the ORIGINAL two target fluxes.
    if(abs(delta_mass)<=tolerance) delta_mass=0.0_real64
    if(abs(delta_moment)<=tolerance) delta_moment=0.0_real64
    found=.false.; best_cost=huge(1.0_real64)
    active_sets: do active_set=0,3**5-1
      digits_left=active_set; trial=0.0_real64; free=.false.
      do s=1,5
        kind=mod(digits_left,3); digits_left=digits_left/3
        select case(kind)
        case(0); free(s)=.true.
        case(1); trial(s)=dlo(s)
        case(2); trial(s)=dhi(s)
        end select
      enddo
      r1=delta_mass-sum(trial); r2=delta_moment-dot_product(g,trial)
      nfree=count(free)
      if(nfree>0) then
        mean_g=sum(g,mask=free)/real(nfree,real64)
        variance=sum((g-mean_g)**2,mask=free)
        ! Refine both affine residuals after cancellation in nearly zero fluxes.
        ! Bound arithmetic error using the actual operands, not the net flux.
        do refinement=1,3
          r1=delta_mass-sum(trial); r2=delta_moment-dot_product(g,trial)
          mass_tolerance=max(tolerance,64*epsilon(tolerance)*(abs(delta_mass)+sum(abs(trial))))
          moment_tolerance=max(tolerance,64*epsilon(tolerance)*(abs(delta_moment)+sum(abs(g*trial))))
          coefficient=0.0_real64
          if(variance>0.0_real64) then
            coefficient=(r2-mean_g*r1)/variance
          elseif(abs(r2-mean_g*r1)>moment_tolerance+abs(mean_g)*mass_tolerance) then
            cycle active_sets
          endif
          where(free) trial=trial+r1/real(nfree,real64)+coefficient*(g-mean_g)
        enddo
      endif
      if(any(trial<dlo) .or. any(trial>dhi)) cycle
      mass_tolerance=max(tolerance,64*epsilon(tolerance)*(abs(delta_mass)+sum(abs(trial))))
      moment_tolerance=max(tolerance,64*epsilon(tolerance)*(abs(delta_moment)+sum(abs(g*trial))))
      if(abs(sum(trial)-delta_mass)>mass_tolerance .or. &
        abs(dot_product(g,trial)-delta_moment)>moment_tolerance) cycle
      cost=sum(trial**2)
      if(.not.found .or. cost<best_cost) then
        best=trial; best_cost=cost; found=.true.
      endif
    enddo active_sets
    status=chemistry_status_infeasible_flux
    if(.not.found) return
    best=max(lower,min(upper,candidate+best*scale_f))
    status=chemistry_status_nonconverged
    mass_tolerance=tolerance+64*epsilon(tolerance)*sum(abs(best/scale_f))
    moment_tolerance=tolerance+64*epsilon(tolerance)*sum(abs(g*(best/scale_f)))
    if(abs(sum(best/scale_f)-target_mass)>mass_tolerance .or. &
      abs(dot_product(g,best/scale_f)-target_moment)>moment_tolerance) return
    flux=best
    status=chemistry_status_ok
  end subroutine air5_close_species_flux_moment

  pure subroutine air5_close_species_flux(candidate,lower,upper,mass_flux,flux,status)
    ! Euclidean projection of a face flux onto supplied bounds and mass closure.
    ! This helper does not construct cell positivity budgets or energy constraints.
    real(real64), intent(in) :: candidate(air5_num_species),lower(air5_num_species)
    real(real64), intent(in) :: upper(air5_num_species),mass_flux
    real(real64), intent(out) :: flux(air5_num_species)
    integer, intent(out) :: status
    real(real64) :: magnitude,c(air5_num_species),lo(air5_num_species),hi(air5_num_species)
    real(real64) :: trial(air5_num_species),target,left,right,middle,residual,tolerance
    integer :: iteration

    flux=candidate
    status=chemistry_status_nonfinite
    if(.not.all(ieee_is_finite(candidate)) .or. .not.all(ieee_is_finite(lower)) .or. &
       .not.all(ieee_is_finite(upper)) .or. .not.ieee_is_finite(mass_flux)) return
    status=chemistry_status_infeasible_flux
    if(any(lower>upper)) return
    magnitude=max(abs(mass_flux),maxval(abs(candidate)),maxval(abs(lower)),maxval(abs(upper)))
    if(magnitude==0.0_real64) then
      status=chemistry_status_ok
      return
    endif
    c=candidate/magnitude; lo=lower/magnitude; hi=upper/magnitude
    target=mass_flux/magnitude
    status=chemistry_status_nonconverged
    if(any((lo==0.0_real64) .and. (lower/=0.0_real64)) .or. &
       any((hi==0.0_real64) .and. (upper/=0.0_real64)) .or. &
       (target==0.0_real64 .and. mass_flux/=0.0_real64)) return
    status=chemistry_status_infeasible_flux
    if(target<sum(lo) .or. target>sum(hi)) return
    tolerance=32.0_real64*epsilon(1.0_real64)*max(abs(target),sum(abs(c)))
    if(all(candidate>=lower) .and. all(candidate<=upper)) then
      if(abs(sum(c)-target)<=tolerance) then
        status=chemistry_status_ok
        return
      endif
    endif

    left=minval(lo-c); right=maxval(hi-c)
    do iteration=1,2*digits(1.0_real64)
      middle=0.5_real64*left+0.5_real64*right
      trial=max(lo,min(hi,c+middle))
      residual=sum(trial)-target
      if(residual==0.0_real64 .or. middle==left .or. middle==right) exit
      if(residual<0.0_real64) then
        left=middle
      else
        right=middle
      endif
    enddo
    status=chemistry_status_nonconverged
    if(abs(residual)>tolerance) return
    ! Clamp only roundoff in the rescaling of a bounded flux, never a cell state.
    trial=max(lower,min(upper,trial*magnitude))
    if(abs(sum(trial/magnitude)-target)>tolerance) return
    flux=trial
    status=chemistry_status_ok
  end subroutine air5_close_species_flux

  pure subroutine air5_validate_mechanism_id(identifier, status)
    character(len=*), intent(in) :: identifier
    integer, intent(out) :: status

    status = chemistry_status_ok
    if (trim(identifier) /= air5_mechanism_id) status = chemistry_status_invalid_mechanism
  end subroutine air5_validate_mechanism_id

  pure subroutine air5_validate_partial_densities(rho_species, status)
    real(real64), intent(in) :: rho_species(air5_num_species)
    integer, intent(out) :: status

    status = chemistry_status_ok
    if (.not. all(ieee_is_finite(rho_species))) then
      status = chemistry_status_nonfinite
    else if (any(rho_species < 0.0_real64) .or. &
             sum(rho_species) <= tiny(1.0_real64)) then
      status = chemistry_status_invalid_composition
      end if
  end subroutine air5_validate_partial_densities

  pure subroutine air5_validate_physical_species_state(rho, rho_species, status)
    real(real64), intent(in) :: rho
    real(real64), intent(in) :: rho_species(air5_num_species)
    integer, intent(out) :: status
    real(real64) :: tolerance

    status = chemistry_status_ok
    if (.not. ieee_is_finite(rho)) then
      status = chemistry_status_nonfinite
      return
    end if
    if (rho <= 0.0_real64) then
      status = chemistry_status_invalid_density
      return
    end if
    call air5_validate_partial_densities(rho_species, status)
    if (status /= chemistry_status_ok) return
    tolerance = 1.0e-10_real64*max(rho, 1.0_real64)
    if (abs(sum(rho_species)-rho) > tolerance) status = chemistry_status_invalid_composition
  end subroutine air5_validate_physical_species_state

  pure elemental real(real64) function air5_interior_ratio(value)
    real(real64), intent(in) :: value

    air5_interior_ratio=max(0.0_real64,min(1.0_real64,value))
    if(air5_interior_ratio>0.0_real64 .and. air5_interior_ratio<1.0_real64) &
      air5_interior_ratio=nearest(air5_interior_ratio,-1.0_real64)
  end function air5_interior_ratio

  pure real(real64) function air5_transport_species_ratio(base,negative_budget,scale)
    real(real64), intent(in) :: base,negative_budget,scale
    real(real64) :: reserve,available,denominator

    air5_transport_species_ratio=1.0_real64
    if(negative_budget>=0.0_real64) return
    reserve=0.0_real64
    if(scale>0.0_real64) reserve=nearest(air5_transport_roundoff_gamma*scale,1.0_real64)
    available=max(0.0_real64,base-reserve)
    denominator=-negative_budget+reserve
    air5_transport_species_ratio=air5_interior_ratio(available/denominator)
  end function air5_transport_species_ratio

  pure subroutine air5_limit_filter_species(rho_base,base_species,rho_filtered, &
      filtered_species,limited,theta,status)
    real(real64), intent(in) :: rho_base,rho_filtered
    real(real64), intent(in) :: base_species(air5_num_species)
    real(real64), intent(inout) :: filtered_species(air5_num_species)
    logical, intent(out) :: limited
    real(real64), intent(out) :: theta
    integer, intent(out) :: status
    real(real64) :: candidate(air5_num_species),low_species(air5_num_species)
    real(real64) :: base_sum,direction,other_sum
    integer :: species,closure_species

    limited=.false.
    theta=1.0_real64
    status=chemistry_status_ok
    call air5_validate_physical_species_state(rho_base,base_species,status)
    if(status/=chemistry_status_ok) return
    if(.not.ieee_is_finite(rho_filtered)) then
      status=chemistry_status_nonfinite
      return
    endif
    if(rho_filtered<=0.0_real64) then
      status=chemistry_status_invalid_density
      return
    endif
    if(.not.all(ieee_is_finite(filtered_species))) then
      status=chemistry_status_nonfinite
      return
    endif

    base_sum=sum(base_species)
    closure_species=maxloc(base_species,dim=1)
    other_sum=0.0_real64
    do species=1,air5_num_species
      if(species/=closure_species) other_sum=other_sum+filtered_species(species)
    enddo
    filtered_species(closure_species)=rho_filtered-other_sum
    low_species=rho_filtered*base_species/base_sum
    if(any(filtered_species<0.0_real64)) then
      do species=1,air5_num_species
        direction=filtered_species(species)-low_species(species)
        if(direction<0.0_real64) theta=min(theta, &
          low_species(species)/(low_species(species)-filtered_species(species)))
      enddo
      theta=air5_interior_ratio(theta)
      limited=.true.
    endif
    candidate=low_species+theta*(filtered_species-low_species)
    other_sum=0.0_real64
    do species=1,air5_num_species
      if(species/=closure_species) other_sum=other_sum+candidate(species)
    enddo
    candidate(closure_species)=rho_filtered-other_sum
    if(any(candidate<0.0_real64)) then
      status=chemistry_status_invalid_composition
      return
    endif
    filtered_species=candidate
  end subroutine air5_limit_filter_species

  pure function air5_pressure_is_in_domain(pressure) result(is_in_domain)
    real(real64), intent(in) :: pressure
    logical :: is_in_domain
    real(real64) :: bound_tolerance

    bound_tolerance = 64.0_real64*epsilon(1.0_real64)* &
      max(abs(air5_pressure_max_pa),1.0_real64)
    is_in_domain = ieee_is_finite(pressure) .and. &
      pressure >= air5_pressure_min_pa-bound_tolerance .and. &
      pressure <= air5_pressure_max_pa+bound_tolerance
  end function air5_pressure_is_in_domain

end module chemistry_model

module chemistry_state_layout
  use chemistry_air5_data, only: air5_num_species
  implicit none
  private

  integer, parameter, public :: air5_idx_density = 1
  integer, parameter, public :: air5_idx_momentum_first = 2
  integer, parameter, public :: air5_idx_momentum_last = 4
  integer, parameter, public :: air5_idx_total_energy = 5
  integer, parameter, public :: air5_idx_species_first = 6
  integer, parameter, public :: air5_idx_species_last = &
    air5_idx_species_first + air5_num_species - 1
  integer, parameter, public :: air5_idx_ev = air5_idx_species_last + 1
  integer, parameter, public :: air5_num_mode_equations = 1
  integer, parameter, public :: air5_num_conservative = air5_idx_ev

  integer, parameter, public :: air5_layout_status_ok = 0
  integer, parameter, public :: air5_layout_status_invalid_species = 1
  integer, parameter, public :: air5_layout_status_turbulence = 2
  integer, parameter, public :: air5_layout_status_filter = 3
  integer, parameter, public :: air5_layout_status_nondimensional = 4

  public :: air5_configure_runtime_layout

contains

  pure subroutine air5_configure_runtime_layout(enabled,nondimen,lfilter,turbmode, &
      num_species,num_modequ,numq,status)
    logical, intent(in) :: enabled, nondimen, lfilter
    character(len=*), intent(in) :: turbmode
    integer, intent(in) :: num_species
    integer, intent(inout) :: num_modequ, numq
    integer, intent(out) :: status

    status = air5_layout_status_ok
    if(.not.enabled) return

    if(nondimen) then
      status = air5_layout_status_nondimensional
      return
    endif
    if(num_species /= air5_num_species) then
      status = air5_layout_status_invalid_species
      return
    endif
    if(trim(turbmode) /= 'none') then
      status = air5_layout_status_turbulence
      return
    endif
    num_modequ = air5_num_mode_equations
    numq = air5_num_conservative
  end subroutine air5_configure_runtime_layout

end module chemistry_state_layout
