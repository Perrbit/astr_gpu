module chemistry_flow_solver
  use iso_fortran_env, only: real64
  use constdef, only: num1d60,num1d12,num2d3
  use chemistry_air5_data, only: air5_num_species,air5_species_name, &
    air5_temperature_min_k,air5_formation_energy
  use chemistry_model, only: chemistry_status_ok,air5_ratio_bisection_iterations, &
    air5_interior_ratio,air5_limit_filter_species
  use chemistry_state_layout, only: air5_num_conservative,air5_idx_density, &
    air5_idx_momentum_first,air5_idx_total_energy,air5_idx_species_first, &
    air5_idx_species_last,air5_idx_ev
  use chemistry_flow_state, only: air5_conservative_to_primitive
  use chemistry_thermo, only: air5_species_gas_constant,air5_species_cv_tr, &
    air5_species_vibrational_energy
  use chemistry_ros2, only: air5_ros2_advance
  use chemistry_flow_runtime, only: configure_air5_source_mode,air5_active_source_mode
  use chemistry_transport, only: air5_diffusive_flux
  implicit none
  private

  real(real64), allocatable, save :: momentum_flux(:,:,:,:)
  real(real64), allocatable, save :: energy_flux(:,:,:,:)
  real(real64), allocatable, save :: species_flux(:,:,:,:,:)
  real(real64), allocatable, save :: vibrational_flux(:,:,:,:)
  real(real64), allocatable, save :: diffusion_ratio(:,:,:)
  real(real64), allocatable, save :: transport_origin_species(:,:,:,:)
  real(real64), allocatable, save :: transport_origin_ev(:,:,:)
  real(real64), allocatable, save :: transport_origin_state(:,:,:,:)
  real(real64), save :: vibrational_floor_energy(air5_num_species)=0.0_real64
  logical, save :: vibrational_floor_configured=.false.
  integer, parameter :: air5_limiter_constraint_species=1
  integer, parameter :: air5_limiter_constraint_vibrational=2
  integer, parameter :: air5_limiter_constraint_density=3
  integer, parameter :: air5_limiter_constraint_translational=4
  integer, parameter :: air5_limiter_constraint_count=4

  public :: air5_diffusion_rhs
  public :: air5_convection_rhs
  public :: air5_limit_full_state_convection
  public :: air5_save_filter_species_base
  public :: air5_limit_filtered_state
  public :: air5_chemistry_half_step

contains

  subroutine configure_air5_vibrational_floor()
    integer :: species

    if(vibrational_floor_configured) return
    do species=1,air5_num_species
      vibrational_floor_energy(species)= &
        air5_species_vibrational_energy(species,air5_temperature_min_k)
    enddo
    vibrational_floor_configured=.true.
  end subroutine configure_air5_vibrational_floor

  real(real64) function air5_vibrational_excess(ev,rho_species)
    real(real64), intent(in) :: ev,rho_species(air5_num_species)

    air5_vibrational_excess=ev-dot_product(vibrational_floor_energy,rho_species)
  end function air5_vibrational_excess

  pure real(real64) function air5_translational_margin(state)
    real(real64), intent(in) :: state(air5_num_conservative)
    real(real64) :: thermal_energy
    integer :: species,component

    if(state(air5_idx_density)<=0.0_real64) then
      air5_translational_margin=-huge(1.0_real64)
      return
    endif
    thermal_energy=state(air5_idx_total_energy)-state(air5_idx_ev)
    do species=1,air5_num_species
      component=air5_idx_species_first+species-1
      thermal_energy=thermal_energy-state(component)*(air5_formation_energy(species)+ &
        air5_temperature_min_k*air5_species_cv_tr(species))
    enddo
    air5_translational_margin=thermal_energy- &
      sum(state(air5_idx_momentum_first:air5_idx_momentum_first+2)**2)/ &
      (2.0_real64*state(air5_idx_density))
  end function air5_translational_margin

  logical function air5_state_is_admissible(state)
    real(real64), intent(in) :: state(air5_num_conservative)
    real(real64) :: rho_species(air5_num_species)

    rho_species=state(air5_idx_species_first:air5_idx_species_last)
    air5_state_is_admissible=state(air5_idx_density)>0.0_real64 .and. &
      minval(rho_species)>=0.0_real64 .and. &
      air5_vibrational_excess(state(air5_idx_ev),rho_species)>=0.0_real64 .and. &
      air5_translational_margin(state)>=0.0_real64
  end function air5_state_is_admissible

  integer function air5_state_constraint_mask(state,species_mask)
    real(real64), intent(in) :: state(air5_num_conservative)
    real(real64) :: rho_species(air5_num_species)
    integer, intent(out) :: species_mask
    integer :: species

    rho_species=state(air5_idx_species_first:air5_idx_species_last)
    air5_state_constraint_mask=0
    species_mask=0
    do species=1,air5_num_species
      if(rho_species(species)<0.0_real64) species_mask=ibset(species_mask,species-1)
    enddo
    if(species_mask/=0) air5_state_constraint_mask=ibset( &
      air5_state_constraint_mask,air5_limiter_constraint_species-1)
    if(air5_vibrational_excess(state(air5_idx_ev),rho_species)<0.0_real64) &
      air5_state_constraint_mask=ibset(air5_state_constraint_mask, &
        air5_limiter_constraint_vibrational-1)
    if(state(air5_idx_density)<=0.0_real64) air5_state_constraint_mask= &
      ibset(air5_state_constraint_mask,air5_limiter_constraint_density-1)
    if(air5_translational_margin(state)<0.0_real64) air5_state_constraint_mask= &
      ibset(air5_state_constraint_mask,air5_limiter_constraint_translational-1)
  end function air5_state_constraint_mask

  real(real64) function air5_admissible_face_ratio(low_state,face_correction, &
      collect_diagnostics,constraint_mask,species_mask)
    real(real64), intent(in) :: low_state(air5_num_conservative)
    real(real64), intent(in) :: face_correction(air5_num_conservative)
    logical, intent(in) :: collect_diagnostics
    integer, intent(out) :: constraint_mask,species_mask
    real(real64) :: lower,upper,middle
    real(real64) :: trial_state(air5_num_conservative)
    integer :: iteration

    trial_state=low_state+6.0_real64*face_correction
    if(air5_state_is_admissible(trial_state)) then
      air5_admissible_face_ratio=1.0_real64
      constraint_mask=0
      species_mask=0
      return
    endif
    lower=0.0_real64
    upper=1.0_real64
    do iteration=1,air5_ratio_bisection_iterations
      middle=0.5_real64*(lower+upper)
      trial_state=low_state+6.0_real64*middle*face_correction
      if(air5_state_is_admissible(trial_state)) then
        lower=middle
      else
        upper=middle
      endif
    enddo
    air5_admissible_face_ratio=air5_interior_ratio(lower)
    constraint_mask=0
    species_mask=0
    if(collect_diagnostics) then
      trial_state=low_state+6.0_real64*upper*face_correction
      constraint_mask=air5_state_constraint_mask(trial_state,species_mask)
    endif
  end function air5_admissible_face_ratio

  subroutine air5_save_filter_species_base()
    use commvar, only: im,jm,km
    use commarray, only: q,qrhs

    qrhs(0:im,0:jm,0:km,air5_idx_density)= &
      q(0:im,0:jm,0:km,air5_idx_density)
    qrhs(0:im,0:jm,0:km,air5_idx_species_first:air5_idx_species_last)= &
      q(0:im,0:jm,0:km,air5_idx_species_first:air5_idx_species_last)
  end subroutine air5_save_filter_species_base

  subroutine air5_limit_filtered_state()
    use commvar, only: im,jm,km
    use commarray, only: q,qrhs
    real(real64) :: state(air5_num_conservative)
    real(real64) :: base_species(air5_num_species)
    real(real64) :: rho_species(air5_num_species)
    real(real64) :: theta
    logical :: limited
    integer :: i,j,k,status

    call configure_air5_vibrational_floor()
    do k=0,km
      do j=0,jm
        do i=0,im
          base_species=qrhs(i,j,k,air5_idx_species_first:air5_idx_species_last)
          rho_species=q(i,j,k,air5_idx_species_first:air5_idx_species_last)
          call air5_limit_filter_species(qrhs(i,j,k,air5_idx_density),base_species, &
            q(i,j,k,air5_idx_density),rho_species,limited,theta,status)
          if(status/=chemistry_status_ok) then
            write(*,'(a,3(1x,i0),3(1x,es24.16))') &
              'air5 explicit-filter species repair failed at',i,j,k, &
              minval(rho_species),sum(rho_species),q(i,j,k,air5_idx_density)
            error stop 'air5 parameter-free explicit-filter species limiter failed'
          endif
          q(i,j,k,air5_idx_species_first:air5_idx_species_last)=rho_species
          state=q(i,j,k,1:air5_num_conservative)
          if(.not.air5_state_is_admissible(state)) then
            write(*,'(a,3(1x,i0))') &
              'air5 explicit-filter thermodynamic state failed at',i,j,k
            error stop 'air5 explicit-filter state is not admissible'
          endif
        enddo
      enddo
    enddo
  end subroutine air5_limit_filtered_state

  pure real(real64) function air5_minmod2(first,second)
    real(real64), intent(in) :: first,second

    if(first>0.0_real64 .and. second>0.0_real64) then
      air5_minmod2=min(abs(first),abs(second))
    elseif(first<0.0_real64 .and. second<0.0_real64) then
      air5_minmod2=-min(abs(first),abs(second))
    else
      air5_minmod2=0.0_real64
    endif
  end function air5_minmod2

  pure real(real64) function air5_minmod4(first,second,third,fourth)
    real(real64), intent(in) :: first,second,third,fourth

    if(first>0.0_real64 .and. second>0.0_real64 .and. &
       third>0.0_real64 .and. fourth>0.0_real64) then
      air5_minmod4=min(abs(first),abs(second),abs(third),abs(fourth))
    elseif(first<0.0_real64 .and. second<0.0_real64 .and. &
           third<0.0_real64 .and. fourth<0.0_real64) then
      air5_minmod4=-min(abs(first),abs(second),abs(third),abs(fourth))
    else
      air5_minmod4=0.0_real64
    endif
  end function air5_minmod4

  pure real(real64) function air5_suw3(values)
    real(real64), intent(in) :: values(3)

    air5_suw3=(-values(1)+5.0_real64*values(2)+2.0_real64*values(3))/6.0_real64
  end function air5_suw3

  pure real(real64) function air5_mp5(values)
    real(real64), intent(in) :: values(5)
    real(real64) :: linear,mp,upper_limit,average,median,large_curvature
    real(real64) :: lower,upper,dm1,d0,d1,dhm1,dh0

    linear=(2.0_real64*values(1)-13.0_real64*values(2)+ &
      47.0_real64*values(3)+27.0_real64*values(4)-3.0_real64*values(5))/60.0_real64
    mp=values(3)+air5_minmod2(values(4)-values(3), &
      4.0_real64*(values(3)-values(2)))
    if((linear-values(3))*(linear-mp)<1.0e-10_real64) then
      air5_mp5=linear
      return
    endif
    dm1=values(1)-2.0_real64*values(2)+values(3)
    d0=values(2)-2.0_real64*values(3)+values(4)
    d1=values(3)-2.0_real64*values(4)+values(5)
    dhm1=air5_minmod4(4.0_real64*dm1-d0,4.0_real64*d0-dm1,dm1,d0)
    dh0=air5_minmod4(4.0_real64*d0-d1,4.0_real64*d1-d0,d0,d1)
    upper_limit=values(3)+4.0_real64*(values(3)-values(2))
    average=0.5_real64*(values(3)+values(4))
    median=average-0.5_real64*dh0
    large_curvature=values(3)+0.5_real64*(values(3)-values(2))+ &
      (4.0_real64/3.0_real64)*dhm1
    lower=max(min(values(3),values(4),median), &
      min(values(3),upper_limit,large_curvature))
    upper=min(max(values(3),values(4),median), &
      max(values(3),upper_limit,large_curvature))
    air5_mp5=linear+air5_minmod2(lower-linear,upper-linear)
  end function air5_mp5

  pure real(real64) function air5_mp7(values)
    real(real64), intent(in) :: values(7)
    real(real64) :: linear,mp,upper_limit,average,median,large_curvature
    real(real64) :: lower,upper,dm1,d0,d1,dhm1,dh0

    linear=(-3.0_real64*values(1)+25.0_real64*values(2)- &
      101.0_real64*values(3)+319.0_real64*values(4)+ &
      214.0_real64*values(5)-38.0_real64*values(6)+ &
      4.0_real64*values(7))/420.0_real64
    mp=values(4)+air5_minmod2(values(5)-values(4), &
      4.0_real64*(values(4)-values(3)))
    if((linear-values(4))*(linear-mp)<1.0e-10_real64) then
      air5_mp7=linear
      return
    endif
    dm1=values(2)-2.0_real64*values(3)+values(4)
    d0=values(3)-2.0_real64*values(4)+values(5)
    d1=values(4)-2.0_real64*values(5)+values(6)
    dhm1=air5_minmod4(4.0_real64*dm1-d0,4.0_real64*d0-dm1,dm1,d0)
    dh0=air5_minmod4(4.0_real64*d0-d1,4.0_real64*d1-d0,d0,d1)
    upper_limit=values(4)+4.0_real64*(values(4)-values(3))
    average=0.5_real64*(values(4)+values(5))
    median=average-0.5_real64*dh0
    large_curvature=values(4)+0.5_real64*(values(4)-values(3))+ &
      (4.0_real64/3.0_real64)*dhm1
    lower=max(min(values(4),values(5),median), &
      min(values(4),upper_limit,large_curvature))
    upper=min(max(values(4),values(5),median), &
      max(values(4),upper_limit,large_curvature))
    air5_mp7=linear+air5_minmod2(lower-linear,upper-linear)
  end function air5_mp7

  real(real64) function air5_frozen_spectral_radius(i,j,k,direction,index)
    use commarray, only: rho,vel,prs,spc,dxi
    integer, intent(in) :: i,j,k,direction,index
    integer :: ii,jj,kk,species
    real(real64) :: mixture_r,mixture_cv,gamma_tr,metric_norm,normal_velocity

    ii=i; jj=j; kk=k
    select case(direction)
    case(1); ii=index
    case(2); jj=index
    case(3); kk=index
    case default; error stop 'invalid air5 spectral-radius direction'
    end select
    if(rho(ii,jj,kk)<=0.0_real64 .or. prs(ii,jj,kk)<=0.0_real64) &
      error stop 'air5 LLF spectral radius requires positive density and pressure'
    mixture_r=0.0_real64
    mixture_cv=0.0_real64
    do species=1,air5_num_species
      mixture_r=mixture_r+spc(ii,jj,kk,species)*air5_species_gas_constant(species)
      mixture_cv=mixture_cv+spc(ii,jj,kk,species)*air5_species_cv_tr(species)
    enddo
    if(mixture_r<=0.0_real64 .or. mixture_cv<=0.0_real64) &
      error stop 'air5 LLF spectral radius requires a valid frozen mixture'
    gamma_tr=1.0_real64+mixture_r/mixture_cv
    metric_norm=sqrt(sum(dxi(ii,jj,kk,direction,:)**2))
    normal_velocity=sum(dxi(ii,jj,kk,direction,:)*vel(ii,jj,kk,:))
    air5_frozen_spectral_radius=abs(normal_velocity)+ &
      sqrt(gamma_tr*prs(ii,jj,kk)/rho(ii,jj,kk))*metric_norm
  end function air5_frozen_spectral_radius

  subroutine air5_llf_split_flux(i,j,k,direction,index,alpha,fplus,fminus)
    use commarray, only: q,jacob
    integer, intent(in) :: i,j,k,direction,index
    real(real64), intent(in) :: alpha
    real(real64), intent(out) :: fplus(air5_num_conservative)
    real(real64), intent(out) :: fminus(air5_num_conservative)
    real(real64) :: flux(air5_num_conservative),state(air5_num_conservative)
    real(real64) :: metric_jacobian
    integer :: ii,jj,kk

    ii=i; jj=j; kk=k
    select case(direction)
    case(1); ii=index
    case(2); jj=index
    case(3); kk=index
    end select
    call air5_projected_convective_flux(ii,jj,kk,direction,flux)
    state=q(ii,jj,kk,1:air5_num_conservative)
    metric_jacobian=jacob(ii,jj,kk)
    fplus=0.5_real64*(flux+alpha*metric_jacobian*state)
    fminus=0.5_real64*(flux-alpha*metric_jacobian*state)
  end subroutine air5_llf_split_flux

  logical function air5_shock_interface_active(i,j,k,direction,face,dim,ntype)
    use commarray, only: lshock
    integer, intent(in) :: i,j,k,direction,face,dim,ntype

    air5_shock_interface_active=.false.
    if(.not.allocated(lshock)) return
    select case(direction)
    case(1)
      if(face<0 .or. (face==0 .and. (ntype==2 .or. ntype==3))) then
        air5_shock_interface_active=lshock(0,j,k)
      elseif(face>=dim .or. (face==dim-1 .and. (ntype==1 .or. ntype==3))) then
        air5_shock_interface_active=lshock(dim,j,k)
      else
        air5_shock_interface_active=lshock(face,j,k) .or. lshock(face+1,j,k)
      endif
    case(2)
      if(face<0 .or. (face==0 .and. (ntype==2 .or. ntype==3))) then
        air5_shock_interface_active=lshock(i,0,k)
      elseif(face>=dim .or. (face==dim-1 .and. (ntype==1 .or. ntype==3))) then
        air5_shock_interface_active=lshock(i,dim,k)
      else
        air5_shock_interface_active=lshock(i,face,k) .or. lshock(i,face+1,k)
      endif
    case(3)
      if(face<0 .or. (face==0 .and. (ntype==2 .or. ntype==3))) then
        air5_shock_interface_active=lshock(i,j,0)
      elseif(face>=dim .or. (face==dim-1 .and. (ntype==1 .or. ntype==3))) then
        air5_shock_interface_active=lshock(i,j,dim)
      else
        air5_shock_interface_active=lshock(i,j,face) .or. lshock(i,j,face+1)
      endif
    end select
  end function air5_shock_interface_active

  subroutine air5_projected_convective_flux(i,j,k,direction,f)
    use commarray, only: q,vel,prs,dxi,jacob
    integer, intent(in) :: i,j,k,direction
    real(real64), intent(out) :: f(air5_num_conservative)
    real(real64) :: normal_velocity,metric_jacobian
    integer :: component

    normal_velocity=sum(dxi(i,j,k,direction,:)*vel(i,j,k,:))
    metric_jacobian=jacob(i,j,k)
    f(air5_idx_density)=metric_jacobian*q(i,j,k,air5_idx_density)*normal_velocity
    do component=air5_idx_momentum_first,air5_idx_momentum_first+2
      f(component)=metric_jacobian*(q(i,j,k,component)*normal_velocity+ &
        dxi(i,j,k,direction,component-air5_idx_momentum_first+1)*prs(i,j,k))
    enddo
    f(air5_idx_total_energy)=metric_jacobian* &
      (q(i,j,k,air5_idx_total_energy)+prs(i,j,k))*normal_velocity
    do component=air5_idx_species_first,air5_idx_species_last
      f(component)=metric_jacobian*q(i,j,k,component)*normal_velocity
    enddo
    f(air5_idx_ev)=metric_jacobian*q(i,j,k,air5_idx_ev)*normal_velocity
  end subroutine air5_projected_convective_flux

  subroutine air5_projected_convective_line_flux(i,j,k,direction,index,f)
    integer, intent(in) :: i,j,k,direction,index
    real(real64), intent(out) :: f(air5_num_conservative)

    select case(direction)
    case(1)
      call air5_projected_convective_flux(index,j,k,direction,f)
    case(2)
      call air5_projected_convective_flux(i,index,k,direction,f)
    case(3)
      call air5_projected_convective_flux(i,j,index,direction,f)
    case default
      error stop 'invalid air5 convection direction'
    end select
  end subroutine air5_projected_convective_line_flux

  subroutine air5_centered_convective_face_flux(i,j,k,direction,face,f)
    integer, intent(in) :: i,j,k,direction,face
    real(real64), intent(out) :: f(air5_num_conservative)
    real(real64) :: fm2(air5_num_conservative),fm1(air5_num_conservative)
    real(real64) :: f0(air5_num_conservative),fp1(air5_num_conservative)
    real(real64) :: fp2(air5_num_conservative),fp3(air5_num_conservative)

    call air5_projected_convective_line_flux(i,j,k,direction,face-2,fm2)
    call air5_projected_convective_line_flux(i,j,k,direction,face-1,fm1)
    call air5_projected_convective_line_flux(i,j,k,direction,face,f0)
    call air5_projected_convective_line_flux(i,j,k,direction,face+1,fp1)
    call air5_projected_convective_line_flux(i,j,k,direction,face+2,fp2)
    call air5_projected_convective_line_flux(i,j,k,direction,face+3,fp3)
    f=(37.0_real64*num1d60)*(f0+fp1)-(2.0_real64/15.0_real64)*(fm1+fp2)+ &
      num1d60*(fm2+fp3)
  end subroutine air5_centered_convective_face_flux

  subroutine air5_shock_convective_face_flux(i,j,k,direction,face,dim,ntype,f)
    integer, intent(in) :: i,j,k,direction,face,dim,ntype
    real(real64), intent(out) :: f(air5_num_conservative)
    real(real64) :: fplus(air5_num_conservative,7)
    real(real64) :: fminus(air5_num_conservative,7)
    real(real64) :: unused(air5_num_conservative)
    real(real64) :: alpha
    integer :: order,first_node,last_node,node,n,component

    order=7
    if((ntype==1 .or. ntype==4) .and. face==0) then
      order=1
    elseif((ntype==1 .or. ntype==4) .and. face==1) then
      order=3
    elseif((ntype==1 .or. ntype==4) .and. face==2) then
      order=5
    elseif((ntype==2 .or. ntype==4) .and. face==dim-1) then
      order=1
    elseif((ntype==2 .or. ntype==4) .and. face==dim-2) then
      order=3
    elseif((ntype==2 .or. ntype==4) .and. face==dim-3) then
      order=5
    endif

    select case(order)
    case(1); first_node=face;   last_node=face+1
    case(3); first_node=face-1; last_node=face+2
    case(5); first_node=face-2; last_node=face+3
    case default; first_node=face-3; last_node=face+4
    end select
    alpha=0.0_real64
    do node=first_node,last_node
      alpha=max(alpha,air5_frozen_spectral_radius(i,j,k,direction,node))
    enddo
    if(alpha<=0.0_real64) error stop 'air5 LLF interface has non-positive spectral radius'

    if(order==1) then
      call air5_llf_split_flux(i,j,k,direction,face,alpha,fplus(:,1),unused)
      call air5_llf_split_flux(i,j,k,direction,face+1,alpha,unused,fminus(:,1))
      f=fplus(:,1)+fminus(:,1)
    else
      do n=1,order
        select case(order)
        case(3)
          call air5_llf_split_flux(i,j,k,direction,face+n-2,alpha, &
            fplus(:,n),unused)
          call air5_llf_split_flux(i,j,k,direction,face+3-n,alpha, &
            unused,fminus(:,n))
        case(5)
          call air5_llf_split_flux(i,j,k,direction,face+n-3,alpha, &
            fplus(:,n),unused)
          call air5_llf_split_flux(i,j,k,direction,face+4-n,alpha, &
            unused,fminus(:,n))
        case default
          call air5_llf_split_flux(i,j,k,direction,face+n-4,alpha, &
            fplus(:,n),unused)
          call air5_llf_split_flux(i,j,k,direction,face+5-n,alpha, &
            unused,fminus(:,n))
        end select
      enddo
      do component=1,air5_num_conservative
        select case(order)
        case(3)
          f(component)=air5_suw3(fplus(component,1:3))+ &
            air5_suw3(fminus(component,1:3))
        case(5)
          f(component)=air5_mp5(fplus(component,1:5))+ &
            air5_mp5(fminus(component,1:5))
        case default
          f(component)=air5_mp7(fplus(component,1:7))+ &
            air5_mp7(fminus(component,1:7))
        end select
      enddo
    endif
    f(air5_idx_species_first)=f(air5_idx_density)- &
      sum(f(air5_idx_species_first+1:air5_idx_species_last))
  end subroutine air5_shock_convective_face_flux

  subroutine air5_high_order_convective_face_flux(i,j,k,direction,face,dim,ntype,f)
    integer, intent(in) :: i,j,k,direction,face,dim,ntype
    real(real64), intent(out) :: f(air5_num_conservative)
    real(real64) :: fn(0:5,air5_num_conservative)
    real(real64) :: anchor(air5_num_conservative),d0(air5_num_conservative)
    real(real64) :: d1(air5_num_conservative),d2(air5_num_conservative)
    integer :: offset

    if(dim<5) error stop 'air5 convection face reconstruction requires at least six points'
    if((ntype==1 .or. ntype==4) .and. face<=1) then
      do offset=0,5
        call air5_projected_convective_line_flux(i,j,k,direction,offset,fn(offset,:))
      enddo
      anchor=(37.0_real64*num1d60)*(fn(2,:)+fn(3,:))- &
        (2.0_real64/15.0_real64)*(fn(1,:)+fn(4,:))+num1d60*(fn(0,:)+fn(5,:))
      d0=-0.5_real64*fn(2,:)+2.0_real64*fn(1,:)-1.5_real64*fn(0,:)
      d1=0.5_real64*(fn(2,:)-fn(0,:))
      d2=num2d3*(fn(3,:)-fn(1,:))-num1d12*(fn(4,:)-fn(0,:))
      select case(face)
      case(1);  f=anchor-d2
      case(0);  f=anchor-d2-d1
      case(-1); f=anchor-d2-d1-d0
      case default; error stop 'invalid lower air5 convection face'
      end select
    elseif((ntype==2 .or. ntype==4) .and. face>=dim-2) then
      do offset=0,5
        call air5_projected_convective_line_flux(i,j,k,direction,dim-5+offset,fn(offset,:))
      enddo
      anchor=(37.0_real64*num1d60)*(fn(2,:)+fn(3,:))- &
        (2.0_real64/15.0_real64)*(fn(1,:)+fn(4,:))+num1d60*(fn(0,:)+fn(5,:))
      d0=num2d3*(fn(4,:)-fn(2,:))-num1d12*(fn(5,:)-fn(1,:))
      d1=0.5_real64*(fn(5,:)-fn(3,:))
      d2=0.5_real64*fn(3,:)-2.0_real64*fn(4,:)+1.5_real64*fn(5,:)
      select case(face-(dim-3))
      case(1); f=anchor+d0
      case(2); f=anchor+d0+d1
      case(3); f=anchor+d0+d1+d2
      case default; error stop 'invalid upper air5 convection face'
      end select
    else
      call air5_centered_convective_face_flux(i,j,k,direction,face,f)
    endif
    f(air5_idx_species_first)=f(air5_idx_density)- &
      sum(f(air5_idx_species_first+1:air5_idx_species_last))
  end subroutine air5_high_order_convective_face_flux

  subroutine air5_selective_convective_face_flux(i,j,k,direction,face,dim,ntype,f)
    use chemistry_flow_runtime, only: air5_shock_capturing_enabled
    integer, intent(in) :: i,j,k,direction,face,dim,ntype
    real(real64), intent(out) :: f(air5_num_conservative)

    if(air5_shock_capturing_enabled() .and. &
       air5_shock_interface_active(i,j,k,direction,face,dim,ntype)) then
      call air5_shock_convective_face_flux(i,j,k,direction,face,dim,ntype,f)
    else
      call air5_high_order_convective_face_flux(i,j,k,direction,face,dim,ntype,f)
    endif
  end subroutine air5_selective_convective_face_flux

  subroutine air5_convection_rhs()
    use commvar, only: im,jm,km,hm,npdci,npdcj,npdck,is,ie,js,je,ks,ke
    use commarray, only: qrhs
    use chemistry_flow_runtime, only: air5_shock_capturing_enabled
    real(real64) :: left_flux(air5_num_conservative)
    real(real64) :: right_flux(air5_num_conservative)
    integer :: dims(3),ntypes(3),i,j,k,direction,index

    if(.not.air5_shock_capturing_enabled()) &
      error stop 'air5 selective convection called while shock capturing is disabled'
    if(hm<4) error stop 'air5 shock capturing requires hm>=4'
    dims=[im,jm,km]
    ntypes=[npdci,npdcj,npdck]
    do k=ks,ke
      do j=js,je
        do i=is,ie
          do direction=1,3
            select case(direction)
            case(1); index=i
            case(2); index=j
            case(3); index=k
            end select
            call air5_selective_convective_face_flux(i,j,k,direction,index-1, &
              dims(direction),ntypes(direction),left_flux)
            call air5_selective_convective_face_flux(i,j,k,direction,index, &
              dims(direction),ntypes(direction),right_flux)
            qrhs(i,j,k,1:air5_num_conservative)= &
              qrhs(i,j,k,1:air5_num_conservative)+right_flux-left_flux
          enddo
        enddo
      enddo
    enddo
  end subroutine air5_convection_rhs

  subroutine air5_low_order_convective_face_flux(i,j,k,direction,face,f)
    integer, intent(in) :: i,j,k,direction,face
    real(real64), intent(out) :: f(air5_num_conservative)
    real(real64) :: fplus(air5_num_conservative),fminus(air5_num_conservative)
    real(real64) :: unused(air5_num_conservative),alpha

    alpha=max(air5_frozen_spectral_radius(i,j,k,direction,face), &
      air5_frozen_spectral_radius(i,j,k,direction,face+1))
    if(alpha<=0.0_real64) error stop 'air5 low-order LLF face has non-positive spectral radius'
    call air5_llf_split_flux(i,j,k,direction,face,alpha,fplus,unused)
    call air5_llf_split_flux(i,j,k,direction,face+1,alpha,unused,fminus)
    f=fplus+fminus
    f(air5_idx_species_first)=f(air5_idx_density)- &
      sum(f(air5_idx_species_first+1:air5_idx_species_last))
  end subroutine air5_low_order_convective_face_flux

  subroutine air5_full_state_node_face_fluxes(i,j,k,dims,ntypes,high_left, &
      high_right,low_left,low_right)
    integer, intent(in) :: i,j,k,dims(3),ntypes(3)
    real(real64), intent(out) :: high_left(3,air5_num_conservative)
    real(real64), intent(out) :: high_right(3,air5_num_conservative)
    real(real64), intent(out) :: low_left(3,air5_num_conservative)
    real(real64), intent(out) :: low_right(3,air5_num_conservative)
    integer :: direction,index

    do direction=1,3
      select case(direction)
      case(1); index=i
      case(2); index=j
      case(3); index=k
      end select
      call air5_selective_convective_face_flux(i,j,k,direction,index-1, &
        dims(direction),ntypes(direction),high_left(direction,:))
      call air5_selective_convective_face_flux(i,j,k,direction,index, &
        dims(direction),ntypes(direction),high_right(direction,:))
      call air5_low_order_convective_face_flux(i,j,k,direction,index-1, &
        low_left(direction,:))
      call air5_low_order_convective_face_flux(i,j,k,direction,index, &
        low_right(direction,:))
    enddo
  end subroutine air5_full_state_node_face_fluxes

  subroutine air5_limit_full_state_convection()
    use mpi
    use commvar, only: im,jm,km,hm,npdci,npdcj,npdck,is,ie,js,je,ks,ke, &
      deltat,rkstep,rkscheme,nstep,feqchkpt
    use commarray, only: q,qrhs,jacob
    use parallel, only: dataswap,lio
    real(real64) :: high_left(3,air5_num_conservative)
    real(real64) :: high_right(3,air5_num_conservative)
    real(real64) :: low_left(3,air5_num_conservative)
    real(real64) :: low_right(3,air5_num_conservative)
    real(real64) :: low_rhs(air5_num_conservative),pminus(air5_num_species+1)
    real(real64) :: correction_left,correction_right,base,ratio,candidate_ratio,cdt
    real(real64) :: face_correction(air5_num_conservative)
    real(real64) :: low_state(air5_num_conservative)
    real(real64) :: limited_rhs(air5_num_conservative)
    real(real64) :: theta_left,theta_right,rk_a,rk_b,rk_c
    real(real64) :: local_min_ratio,global_min_ratio
    real(real64) :: constraint_min_ratio(air5_limiter_constraint_count)
    real(real64) :: global_constraint_min_ratio(air5_limiter_constraint_count)
    real(real64) :: species_constraint_min_ratio(air5_num_species)
    real(real64) :: global_species_constraint_min_ratio(air5_num_species)
    integer :: dims(3),ntypes(3),i,j,k,species,component,direction,constraint
    integer :: constraint_mask,face_species_mask
    integer :: local_limited,global_limited,local_invalid,global_invalid,ierr
    integer :: constraint_active(air5_limiter_constraint_count)
    integer :: local_constraint_active(air5_limiter_constraint_count)
    integer :: global_constraint_active(air5_limiter_constraint_count)
    integer :: species_constraint_active(air5_num_species)
    integer :: local_species_constraint_active(air5_num_species)
    integer :: global_species_constraint_active(air5_num_species)
    logical :: report_diagnostics

    if(trim(rkscheme)/='rk3') error stop 'air5 convection limiter requires SSPRK3'
    select case(rkstep)
    case(1); rk_a=1.0_real64; rk_b=0.0_real64; rk_c=1.0_real64
    case(2); rk_a=0.75_real64; rk_b=0.25_real64; rk_c=0.25_real64
    case(3); rk_a=1.0_real64/3.0_real64; rk_b=2.0_real64/3.0_real64; rk_c=2.0_real64/3.0_real64
    case default; error stop 'air5 convection limiter received an invalid RK stage'
    end select
    call allocate_air5_flux_workspace(im,jm,km,hm)
    dims=[im,jm,km]; ntypes=[npdci,npdcj,npdck]; cdt=rk_c*deltat
    report_diagnostics=nstep==0 .or. (feqchkpt>0 .and. mod(nstep,feqchkpt)==0)
    diffusion_ratio=1.0_real64
    if(rkstep==1) then
      do component=1,air5_num_conservative
        transport_origin_state(:,:,:,component)=q(0:im,0:jm,0:km,component)* &
          jacob(0:im,0:jm,0:km)
      enddo
      do species=1,air5_num_species
        component=air5_idx_species_first+species-1
        transport_origin_species(:,:,:,species)=q(0:im,0:jm,0:km,component)* &
          jacob(0:im,0:jm,0:km)
      enddo
      transport_origin_ev=q(0:im,0:jm,0:km,air5_idx_ev)* &
        jacob(0:im,0:jm,0:km)
    endif
    local_limited=0; local_invalid=0; local_min_ratio=1.0_real64
    local_constraint_active=0
    global_constraint_active=0
    global_constraint_min_ratio=1.0_real64
    local_species_constraint_active=0
    global_species_constraint_active=0
    global_species_constraint_min_ratio=1.0_real64

    do k=ks,ke; do j=js,je; do i=is,ie
      call air5_full_state_node_face_fluxes(i,j,k,dims,ntypes,high_left,high_right, &
        low_left,low_right)
      low_rhs=0.0_real64; pminus=0.0_real64
      do component=1,air5_num_conservative
        do direction=1,3
          low_rhs(component)=low_rhs(component)+low_left(direction,component)- &
            low_right(direction,component)
        enddo
      enddo
      low_state=rk_a*transport_origin_state(i,j,k,:)+ &
        rk_b*q(i,j,k,1:air5_num_conservative)*jacob(i,j,k)+cdt*low_rhs
      ratio=1.0_real64
      constraint_active=0
      constraint_min_ratio=1.0_real64
      species_constraint_active=0
      species_constraint_min_ratio=1.0_real64
      if(.not.air5_state_is_admissible(low_state)) local_invalid=1
      do species=1,air5_num_species
        component=air5_idx_species_first+species-1
        do direction=1,3
          correction_left=cdt*(high_left(direction,component)- &
            low_left(direction,component))
          correction_right=-cdt*(high_right(direction,component)- &
            low_right(direction,component))
          pminus(species)=pminus(species)+min(0.0_real64,correction_left)+ &
            min(0.0_real64,correction_right)
        enddo
        base=low_state(component)
        if(base<0.0_real64) then
          local_invalid=1
        elseif(pminus(species)<0.0_real64) then
          candidate_ratio=air5_interior_ratio(base/(-pminus(species)))
          ratio=min(ratio,candidate_ratio)
          if(report_diagnostics .and. candidate_ratio<1.0_real64) then
            constraint_active(air5_limiter_constraint_species)=1
            constraint_min_ratio(air5_limiter_constraint_species)=min( &
              constraint_min_ratio(air5_limiter_constraint_species),candidate_ratio)
            species_constraint_active(species)=1
            species_constraint_min_ratio(species)=min( &
              species_constraint_min_ratio(species),candidate_ratio)
          endif
        endif
      enddo
      do direction=1,3
        correction_left=cdt*((high_left(direction,air5_idx_ev)- &
          dot_product(vibrational_floor_energy, &
          high_left(direction,air5_idx_species_first:air5_idx_species_last)))- &
          (low_left(direction,air5_idx_ev)-dot_product(vibrational_floor_energy, &
          low_left(direction,air5_idx_species_first:air5_idx_species_last))))
        correction_right=-cdt*((high_right(direction,air5_idx_ev)- &
          dot_product(vibrational_floor_energy, &
          high_right(direction,air5_idx_species_first:air5_idx_species_last)))- &
          (low_right(direction,air5_idx_ev)-dot_product(vibrational_floor_energy, &
          low_right(direction,air5_idx_species_first:air5_idx_species_last))))
        pminus(air5_num_species+1)=pminus(air5_num_species+1)+ &
          min(0.0_real64,correction_left)+min(0.0_real64,correction_right)
      enddo
      base=air5_vibrational_excess(low_state(air5_idx_ev), &
        low_state(air5_idx_species_first:air5_idx_species_last))
      if(base<0.0_real64) then
        local_invalid=1
      elseif(pminus(air5_num_species+1)<0.0_real64) then
        candidate_ratio=air5_interior_ratio(base/(-pminus(air5_num_species+1)))
        ratio=min(ratio,candidate_ratio)
        if(report_diagnostics .and. candidate_ratio<1.0_real64) then
          constraint_active(air5_limiter_constraint_vibrational)=1
          constraint_min_ratio(air5_limiter_constraint_vibrational)=min( &
            constraint_min_ratio(air5_limiter_constraint_vibrational),candidate_ratio)
        endif
      endif
      do direction=1,3
        face_correction=cdt*(high_left(direction,:)-low_left(direction,:))
        candidate_ratio=air5_admissible_face_ratio(low_state,face_correction, &
          report_diagnostics,constraint_mask,face_species_mask)
        ratio=min(ratio,candidate_ratio)
        if(report_diagnostics .and. candidate_ratio<1.0_real64) then
          do constraint=1,air5_limiter_constraint_count
            if(btest(constraint_mask,constraint-1)) then
              constraint_active(constraint)=1
              constraint_min_ratio(constraint)=min( &
                constraint_min_ratio(constraint),candidate_ratio)
            endif
          enddo
          do species=1,air5_num_species
            if(btest(face_species_mask,species-1)) then
              species_constraint_active(species)=1
              species_constraint_min_ratio(species)=min( &
                species_constraint_min_ratio(species),candidate_ratio)
            endif
          enddo
        endif
        face_correction=-cdt*(high_right(direction,:)-low_right(direction,:))
        candidate_ratio=air5_admissible_face_ratio(low_state,face_correction, &
          report_diagnostics,constraint_mask,face_species_mask)
        ratio=min(ratio,candidate_ratio)
        if(report_diagnostics .and. candidate_ratio<1.0_real64) then
          do constraint=1,air5_limiter_constraint_count
            if(btest(constraint_mask,constraint-1)) then
              constraint_active(constraint)=1
              constraint_min_ratio(constraint)=min( &
                constraint_min_ratio(constraint),candidate_ratio)
            endif
          enddo
          do species=1,air5_num_species
            if(btest(face_species_mask,species-1)) then
              species_constraint_active(species)=1
              species_constraint_min_ratio(species)=min( &
                species_constraint_min_ratio(species),candidate_ratio)
            endif
          enddo
        endif
      enddo
      ratio=max(0.0_real64,min(1.0_real64,ratio))
      diffusion_ratio(i,j,k)=ratio
      local_min_ratio=min(local_min_ratio,ratio)
      if(ratio<1.0_real64-1.0e-14_real64) local_limited=local_limited+1
      if(report_diagnostics) then
        local_constraint_active=local_constraint_active+constraint_active
        do constraint=1,air5_limiter_constraint_count
          if(constraint_active(constraint)/=0) &
            global_constraint_min_ratio(constraint)=min( &
              global_constraint_min_ratio(constraint),constraint_min_ratio(constraint))
        enddo
        local_species_constraint_active=local_species_constraint_active+ &
          species_constraint_active
        do species=1,air5_num_species
          if(species_constraint_active(species)/=0) &
            global_species_constraint_min_ratio(species)=min( &
              global_species_constraint_min_ratio(species), &
              species_constraint_min_ratio(species))
        enddo
      endif
    enddo; enddo; enddo

    call MPI_Allreduce(local_invalid,global_invalid,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,ierr,local_invalid)
    if(global_invalid/=0) then
      if(local_invalid/=0) write(*,'(A,I0)') &
        'air5 low-order full-state convection baseline failed at RK stage ',rkstep
      call MPI_Abort(MPI_COMM_WORLD,3,ierr)
    endif
    call dataswap(diffusion_ratio)

    do k=ks,ke; do j=js,je; do i=is,ie
      call air5_full_state_node_face_fluxes(i,j,k,dims,ntypes,high_left,high_right, &
        low_left,low_right)
      limited_rhs=0.0_real64
      do component=1,air5_num_conservative
        do direction=1,3
          select case(direction)
          case(1)
            theta_left=min(diffusion_ratio(i,j,k),diffusion_ratio(i-1,j,k))
            theta_right=min(diffusion_ratio(i,j,k),diffusion_ratio(i+1,j,k))
          case(2)
            theta_left=min(diffusion_ratio(i,j,k),diffusion_ratio(i,j-1,k))
            theta_right=min(diffusion_ratio(i,j,k),diffusion_ratio(i,j+1,k))
          case(3)
            theta_left=min(diffusion_ratio(i,j,k),diffusion_ratio(i,j,k-1))
            theta_right=min(diffusion_ratio(i,j,k),diffusion_ratio(i,j,k+1))
          end select
          limited_rhs(component)=limited_rhs(component)+low_left(direction,component)- &
            low_right(direction,component)+theta_left*(high_left(direction,component)- &
            low_left(direction,component))-theta_right*(high_right(direction,component)- &
            low_right(direction,component))
        enddo
      enddo
      qrhs(i,j,k,1:air5_num_conservative)=limited_rhs
    enddo; enddo; enddo

    if(report_diagnostics) then
      call MPI_Allreduce(local_limited,global_limited,1,MPI_INTEGER,MPI_SUM,MPI_COMM_WORLD,ierr)
      if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,ierr,local_limited)
      call MPI_Allreduce(local_min_ratio,global_min_ratio,1,MPI_DOUBLE_PRECISION,MPI_MIN, &
        MPI_COMM_WORLD,ierr)
      if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,ierr,local_limited)
      call MPI_Allreduce(local_constraint_active,global_constraint_active, &
        air5_limiter_constraint_count,MPI_INTEGER,MPI_SUM,MPI_COMM_WORLD,ierr)
      if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,ierr,local_limited)
      constraint_min_ratio=global_constraint_min_ratio
      call MPI_Allreduce(constraint_min_ratio,global_constraint_min_ratio, &
        air5_limiter_constraint_count,MPI_DOUBLE_PRECISION,MPI_MIN,MPI_COMM_WORLD,ierr)
      if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,ierr,local_limited)
      call MPI_Allreduce(local_species_constraint_active, &
        global_species_constraint_active,air5_num_species,MPI_INTEGER,MPI_SUM, &
        MPI_COMM_WORLD,ierr)
      if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,ierr,local_limited)
      species_constraint_min_ratio=global_species_constraint_min_ratio
      call MPI_Allreduce(species_constraint_min_ratio, &
        global_species_constraint_min_ratio,air5_num_species, &
        MPI_DOUBLE_PRECISION,MPI_MIN,MPI_COMM_WORLD,ierr)
      if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,ierr,local_limited)
      if(lio .and. global_limited>0) write(*,'(A,I0,A,I0,A,ES12.4)') &
        'AIR5_FULL_STATE_CONVECTION_LIMITER stage=',rkstep, &
        ' limited_points=',global_limited, &
        ' min_ratio=',global_min_ratio
      if(lio) then
        do constraint=1,air5_limiter_constraint_count
          if(global_constraint_active(constraint)<=0) cycle
          select case(constraint)
          case(air5_limiter_constraint_species)
            write(*,'(A,I0,A,A,A,I0,A,ES12.4)') &
              'AIR5_FULL_STATE_CONVECTION_CONSTRAINT stage=',rkstep, &
              ' constraint=','species',' active_points=', &
              global_constraint_active(constraint),' min_ratio=', &
              global_constraint_min_ratio(constraint)
          case(air5_limiter_constraint_vibrational)
            write(*,'(A,I0,A,A,A,I0,A,ES12.4)') &
              'AIR5_FULL_STATE_CONVECTION_CONSTRAINT stage=',rkstep, &
              ' constraint=','vibrational',' active_points=', &
              global_constraint_active(constraint),' min_ratio=', &
              global_constraint_min_ratio(constraint)
          case(air5_limiter_constraint_density)
            write(*,'(A,I0,A,A,A,I0,A,ES12.4)') &
              'AIR5_FULL_STATE_CONVECTION_CONSTRAINT stage=',rkstep, &
              ' constraint=','density',' active_points=', &
              global_constraint_active(constraint),' min_ratio=', &
              global_constraint_min_ratio(constraint)
          case(air5_limiter_constraint_translational)
            write(*,'(A,I0,A,A,A,I0,A,ES12.4)') &
              'AIR5_FULL_STATE_CONVECTION_CONSTRAINT stage=',rkstep, &
              ' constraint=','translational',' active_points=', &
              global_constraint_active(constraint),' min_ratio=', &
              global_constraint_min_ratio(constraint)
          end select
        enddo
        do species=1,air5_num_species
          if(global_species_constraint_active(species)<=0) cycle
          write(*,'(A,I0,A,A,A,I0,A,ES12.4)') &
            'AIR5_FULL_STATE_CONVECTION_SPECIES_CONSTRAINT stage=',rkstep, &
            ' species=',trim(air5_species_name(species)),' active_points=', &
            global_species_constraint_active(species),' min_ratio=', &
            global_species_constraint_min_ratio(species)
        enddo
      endif
    endif
  end subroutine air5_limit_full_state_convection

  subroutine air5_chemistry_half_step(duration,half_index)
    use mpi
    use commvar, only: is,ie,js,je,ks,ke,nstep,feqchkpt
    use commarray, only: q
    use parallel, only: lio
    real(real64), intent(in) :: duration
    integer, intent(in) :: half_index
    real(real64), parameter :: rtol=1.0e-9_real64
    real(real64), parameter :: atol_factor=1.0e-13_real64
    integer, parameter :: max_attempts=200000
    real(real64) :: state(6),final_state(6),atol(6),suggested_step
    real(real64) :: local_q(air5_num_conservative),velocity(3),mass_fraction(air5_num_species)
    real(real64) :: density,temperature,tv,pressure
    real(real64) :: local_minimums(3),global_minimums(3)
    real(real64) :: local_maximums(2),global_maximums(2)
    integer :: i,j,k,status,global_status,ierr,source_mode
    integer :: accepted,rejected,rhs_evaluations,jacobian_evaluations
    integer :: local_counts(4),global_counts(4),failed_index(3)

    if(duration<=0.0_real64) error stop 'air5 chemistry half-step requires positive duration'
    if(half_index<1 .or. half_index>2) &
      error stop 'air5 chemistry half-step index must be one or two'
    call configure_air5_source_mode()
    source_mode=air5_active_source_mode()
    status=chemistry_status_ok
    failed_index=-1
    local_minimums=huge(1.0_real64)
    local_maximums=-huge(1.0_real64)
    local_counts=0

    do k=ks,ke
      do j=js,je
        do i=is,ie
          density=q(i,j,k,air5_idx_density)
          state=q(i,j,k,air5_idx_species_first:air5_idx_ev)
          atol(1:air5_num_species)=atol_factor*density
          atol(6)=atol_factor*max(abs(state(6)),1.0_real64)
          call air5_ros2_advance(density, &
            q(i,j,k,air5_idx_momentum_first:air5_idx_momentum_first+2), &
            q(i,j,k,air5_idx_total_energy),state,duration,duration,rtol,atol, &
            max_attempts,final_state,suggested_step,accepted,rejected, &
            rhs_evaluations,jacobian_evaluations,status,source_mode)
          if(status/=chemistry_status_ok) then
            failed_index=[i,j,k]
            exit
          endif
          local_q=q(i,j,k,1:air5_num_conservative)
          local_q(air5_idx_species_first:air5_idx_ev)=final_state
          call air5_conservative_to_primitive(local_q,density,velocity,temperature, &
            mass_fraction,tv,pressure,status)
          if(status/=chemistry_status_ok) then
            failed_index=[i,j,k]
            exit
          endif
          q(i,j,k,air5_idx_species_first:air5_idx_ev)=final_state
          local_minimums=min(local_minimums, &
            [minval(final_state(1:air5_num_species)),temperature,tv])
          local_maximums=max(local_maximums,[temperature,tv])
          local_counts=max(local_counts, &
            [accepted,rejected,rhs_evaluations,jacobian_evaluations])
        enddo
        if(status/=chemistry_status_ok) exit
      enddo
      if(status/=chemistry_status_ok) exit
    enddo

    call MPI_Allreduce(status,global_status,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,ierr,status)
    if(global_status/=chemistry_status_ok) then
      write(*,'(A,I0,A,I0,A,3(I0,1X))') 'air5 CPU chemistry half ',half_index, &
        ' failed, status=',global_status,', local i/j/k=',failed_index
      call MPI_Abort(MPI_COMM_WORLD,global_status,ierr)
    endif
    call MPI_Allreduce(local_minimums,global_minimums,3,MPI_DOUBLE_PRECISION, &
      MPI_MIN,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,ierr,status)
    call MPI_Allreduce(local_maximums,global_maximums,2,MPI_DOUBLE_PRECISION, &
      MPI_MAX,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,ierr,status)
    call MPI_Allreduce(local_counts,global_counts,4,MPI_INTEGER,MPI_MAX, &
      MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,ierr,status)
    if(lio .and. (nstep==0 .or. (feqchkpt>0 .and. mod(nstep,feqchkpt)==0))) then
      write(*,'(A,I0,A,5(ES12.4,1X),A,4(I0,1X))') 'AIR5_CHEMISTRY half=', &
        half_index,' min_rhos/min_T/max_T/min_Tv/max_Tv=',global_minimums(1), &
        global_minimums(2),global_maximums(1),global_minimums(3), &
        global_maximums(2),' max_acc/rej/rhs/jac=',global_counts
    endif
  end subroutine air5_chemistry_half_step

  subroutine air5_diffusion_rhs()
    use mpi
    use commvar, only: im,jm,km,hm,difschm,npdci,npdcj,npdck,flowtype, &
      nstep,rkstep,rkscheme,deltat,is,ie,js,je,ks,ke
    use commarray, only: rho,vel,tmp,tve,prs,spc,dvel,dtmp,dspc,dtve, &
      dxi,jacob,qrhs,q
    use comsolver, only: grad
    use parallel, only: dataswap,mpidown
    integer :: i,j,k,component,status,global_status,ierr
    real(real64) :: local_momentum_flux(3,3)
    real(real64) :: local_species_flux(air5_num_species,3)
    real(real64) :: local_energy_flux(3),local_vibrational_flux(3)
    real(real64) :: rk_a,rk_b,rk_c

    if(trim(difschm)/='643e') &
      error stop 'fixed air5 CPU diffusion requires explicit 643e'
    if(any([npdci,npdcj,npdck]<1) .or. any([npdci,npdcj,npdck]>4)) &
      error stop 'fixed air5 CPU diffusion has an invalid boundary closure'
    call allocate_air5_flux_workspace(im,jm,km,hm)
    momentum_flux=0.0_real64
    energy_flux=0.0_real64
    species_flux=0.0_real64
    vibrational_flux=0.0_real64
    dtve=grad(tve)
    if(trim(flowtype)=='air5hbl' .and. mpidown==MPI_PROC_NULL) &
      dspc(:,0,:,:,2)=0.0_real64

    status=chemistry_status_ok
    do k=0,km
      do j=0,jm
        do i=0,im
          call air5_diffusive_flux(rho(i,j,k),vel(i,j,k,:),tmp(i,j,k), &
            tve(i,j,k),prs(i,j,k),spc(i,j,k,:),dvel(i,j,k,:,:), &
            dtmp(i,j,k,:),dtve(i,j,k,:),dspc(i,j,k,:,:), &
            local_momentum_flux,local_species_flux,local_energy_flux, &
            local_vibrational_flux,status)
          if(status/=chemistry_status_ok) exit
          momentum_flux(i,j,k,1)=local_momentum_flux(1,1)
          momentum_flux(i,j,k,2)=local_momentum_flux(1,2)
          momentum_flux(i,j,k,3)=local_momentum_flux(1,3)
          momentum_flux(i,j,k,4)=local_momentum_flux(2,2)
          momentum_flux(i,j,k,5)=local_momentum_flux(2,3)
          momentum_flux(i,j,k,6)=local_momentum_flux(3,3)
          energy_flux(i,j,k,:)=local_energy_flux
          species_flux(i,j,k,:,:)=local_species_flux
          vibrational_flux(i,j,k,:)=local_vibrational_flux
        enddo
        if(status/=chemistry_status_ok) exit
      enddo
      if(status/=chemistry_status_ok) exit
    enddo
    call MPI_Allreduce(status,global_status,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,ierr,status)
    if(global_status/=chemistry_status_ok) then
      write(*,'(A,I0,A,3(I0,1X))') 'fixed air5 CPU diffusion failed, status=', &
        global_status,', local i/j/k=',i,j,k
      call MPI_Abort(MPI_COMM_WORLD,global_status,ierr)
    endif

    call dataswap(momentum_flux)
    call dataswap(energy_flux)
    call dataswap(species_flux)
    call dataswap(vibrational_flux)

    if(trim(rkscheme)/='rk3') &
      error stop 'fixed air5 diffusion limiter requires SSPRK3'
    select case(rkstep)
    case(1)
      rk_a=1.0_real64
      rk_b=0.0_real64
      rk_c=1.0_real64
    case(2)
      rk_a=0.75_real64
      rk_b=0.25_real64
      rk_c=0.25_real64
    case(3)
      rk_a=1.0_real64/3.0_real64
      rk_b=2.0_real64/3.0_real64
      rk_c=2.0_real64/3.0_real64
    case default
      error stop 'fixed air5 diffusion limiter received an invalid RK stage'
    end select
    call limit_air5_diffusive_fluxes(q,qrhs,jacob,deltat,rk_a,rk_b,rk_c, &
      rkstep,im,jm,km,hm,npdci,npdcj,npdck,is,ie,js,je,ks,ke)
  end subroutine air5_diffusion_rhs

  subroutine allocate_air5_flux_workspace(im,jm,km,hm)
    integer, intent(in) :: im,jm,km,hm

    call configure_air5_vibrational_floor()
    if(.not.allocated(momentum_flux)) &
      allocate(momentum_flux(-hm:im+hm,-hm:jm+hm,-hm:km+hm,6))
    if(.not.allocated(energy_flux)) &
      allocate(energy_flux(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3))
    if(.not.allocated(species_flux)) &
      allocate(species_flux(-hm:im+hm,-hm:jm+hm,-hm:km+hm,air5_num_species,3))
    if(.not.allocated(vibrational_flux)) &
      allocate(vibrational_flux(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3))
    if(.not.allocated(diffusion_ratio)) &
      allocate(diffusion_ratio(-hm:im+hm,-hm:jm+hm,-hm:km+hm))
    if(.not.allocated(transport_origin_species)) &
      allocate(transport_origin_species(0:im,0:jm,0:km,air5_num_species))
    if(.not.allocated(transport_origin_ev)) &
      allocate(transport_origin_ev(0:im,0:jm,0:km))
    if(.not.allocated(transport_origin_state)) &
      allocate(transport_origin_state(0:im,0:jm,0:km,air5_num_conservative))
  end subroutine allocate_air5_flux_workspace

  subroutine projected_air5_diffusive_flux(i,j,k,direction,f)
    use commarray, only: dxi,jacob
    integer, intent(in) :: i,j,k,direction
    real(real64), intent(out) :: f(air5_num_conservative)
    real(real64) :: metric(3),metric_jacobian
    integer :: species

    metric=dxi(i,j,k,direction,:)
    metric_jacobian=jacob(i,j,k)
    f=0.0_real64
    f(air5_idx_momentum_first)=metric_jacobian* &
      (momentum_flux(i,j,k,1)*metric(1)+momentum_flux(i,j,k,2)*metric(2)+ &
       momentum_flux(i,j,k,3)*metric(3))
    f(air5_idx_momentum_first+1)=metric_jacobian* &
      (momentum_flux(i,j,k,2)*metric(1)+momentum_flux(i,j,k,4)*metric(2)+ &
       momentum_flux(i,j,k,5)*metric(3))
    f(air5_idx_momentum_first+2)=metric_jacobian* &
      (momentum_flux(i,j,k,3)*metric(1)+momentum_flux(i,j,k,5)*metric(2)+ &
       momentum_flux(i,j,k,6)*metric(3))
    f(air5_idx_total_energy)=metric_jacobian*sum(energy_flux(i,j,k,:)*metric)
    do species=1,air5_num_species
      f(air5_idx_species_first+species-1)=metric_jacobian* &
        sum(species_flux(i,j,k,species,:)*metric)
    enddo
    f(air5_idx_ev)=metric_jacobian*sum(vibrational_flux(i,j,k,:)*metric)
  end subroutine projected_air5_diffusive_flux

  subroutine projected_air5_line_flux(i,j,k,direction,index,f)
    integer, intent(in) :: i,j,k,direction,index
    real(real64), intent(out) :: f(air5_num_conservative)

    select case(direction)
    case(1)
      call projected_air5_diffusive_flux(index,j,k,direction,f)
    case(2)
      call projected_air5_diffusive_flux(i,index,k,direction,f)
    case(3)
      call projected_air5_diffusive_flux(i,j,index,direction,f)
    case default
      error stop 'invalid air5 diffusion direction'
    end select
  end subroutine projected_air5_line_flux

  subroutine projected_air5_centered_face_flux(i,j,k,direction,face,f)
    integer, intent(in) :: i,j,k,direction,face
    real(real64), intent(out) :: f(air5_num_conservative)
    real(real64) :: fm2(air5_num_conservative),fm1(air5_num_conservative)
    real(real64) :: f0(air5_num_conservative),fp1(air5_num_conservative)
    real(real64) :: fp2(air5_num_conservative),fp3(air5_num_conservative)

    call projected_air5_line_flux(i,j,k,direction,face-2,fm2)
    call projected_air5_line_flux(i,j,k,direction,face-1,fm1)
    call projected_air5_line_flux(i,j,k,direction,face,f0)
    call projected_air5_line_flux(i,j,k,direction,face+1,fp1)
    call projected_air5_line_flux(i,j,k,direction,face+2,fp2)
    call projected_air5_line_flux(i,j,k,direction,face+3,fp3)
    f=(37.0_real64*num1d60)*(f0+fp1)-(2.0_real64/15.0_real64)*(fm1+fp2)+ &
      num1d60*(fm2+fp3)
  end subroutine projected_air5_centered_face_flux

  subroutine projected_air5_face_flux(i,j,k,direction,face,dim,ntype,f)
    integer, intent(in) :: i,j,k,direction,face,dim,ntype
    real(real64), intent(out) :: f(air5_num_conservative)
    real(real64) :: fn(0:5,air5_num_conservative)
    real(real64) :: anchor(air5_num_conservative),d0(air5_num_conservative)
    real(real64) :: d1(air5_num_conservative),d2(air5_num_conservative)
    integer :: offset

    if(dim<5) error stop 'air5 diffusion face reconstruction requires at least six points'
    if((ntype==1 .or. ntype==4) .and. face<=1) then
      do offset=0,5
        call projected_air5_line_flux(i,j,k,direction,offset,fn(offset,:))
      enddo
      anchor=(37.0_real64*num1d60)*(fn(2,:)+fn(3,:))- &
        (2.0_real64/15.0_real64)*(fn(1,:)+fn(4,:))+ &
        num1d60*(fn(0,:)+fn(5,:))
      d0=-0.5_real64*fn(2,:)+2.0_real64*fn(1,:)-1.5_real64*fn(0,:)
      d1=0.5_real64*(fn(2,:)-fn(0,:))
      d2=num2d3*(fn(3,:)-fn(1,:))-num1d12*(fn(4,:)-fn(0,:))
      select case(face)
      case(1)
        f=anchor-d2
      case(0)
        f=anchor-d2-d1
      case(-1)
        f=anchor-d2-d1-d0
      case default
        error stop 'invalid lower air5 diffusion face'
      end select
    elseif((ntype==2 .or. ntype==4) .and. face>=dim-2) then
      do offset=0,5
        call projected_air5_line_flux(i,j,k,direction,dim-5+offset,fn(offset,:))
      enddo
      anchor=(37.0_real64*num1d60)*(fn(2,:)+fn(3,:))- &
        (2.0_real64/15.0_real64)*(fn(1,:)+fn(4,:))+ &
        num1d60*(fn(0,:)+fn(5,:))
      d0=num2d3*(fn(4,:)-fn(2,:))-num1d12*(fn(5,:)-fn(1,:))
      d1=0.5_real64*(fn(5,:)-fn(3,:))
      d2=0.5_real64*fn(3,:)-2.0_real64*fn(4,:)+1.5_real64*fn(5,:)
      select case(face-(dim-3))
      case(1)
        f=anchor+d0
      case(2)
        f=anchor+d0+d1
      case(3)
        f=anchor+d0+d1+d2
      case default
        error stop 'invalid upper air5 diffusion face'
      end select
    else
      call projected_air5_centered_face_flux(i,j,k,direction,face,f)
    endif
  end subroutine projected_air5_face_flux

  subroutine air5_node_face_fluxes(i,j,k,dims,ntypes,left_flux,right_flux)
    integer, intent(in) :: i,j,k,dims(3),ntypes(3)
    real(real64), intent(out) :: left_flux(3,air5_num_conservative)
    real(real64), intent(out) :: right_flux(3,air5_num_conservative)
    integer :: direction,index

    do direction=1,3
      select case(direction)
      case(1)
        index=i
      case(2)
        index=j
      case(3)
        index=k
      end select
      call projected_air5_face_flux(i,j,k,direction,index-1,dims(direction), &
        ntypes(direction),left_flux(direction,:))
      call projected_air5_face_flux(i,j,k,direction,index,dims(direction), &
        ntypes(direction),right_flux(direction,:))
    enddo
  end subroutine air5_node_face_fluxes

  subroutine limit_air5_diffusive_fluxes(q,qrhs,jacob,deltat,rk_a,rk_b,rk_c, &
      rkstep,im,jm,km,hm,npdci,npdcj,npdck,is,ie,js,je,ks,ke)
    use mpi
    use commvar, only: nstep,feqchkpt
    use parallel, only: dataswap,lio
    integer, intent(in) :: rkstep,im,jm,km,hm,npdci,npdcj,npdck
    integer, intent(in) :: is,ie,js,je,ks,ke
    real(real64), intent(in) :: q(-hm:,-hm:,-hm:,:),jacob(-hm:,-hm:,-hm:)
    real(real64), intent(inout) :: qrhs(0:,0:,0:,:)
    real(real64), intent(in) :: deltat,rk_a,rk_b,rk_c
    real(real64) :: left_flux(3,air5_num_conservative)
    real(real64) :: right_flux(3,air5_num_conservative)
    real(real64) :: pminus(air5_num_species+1),base,ratio,cdt
    real(real64) :: theta_left,theta_right,df
    real(real64) :: base_state(air5_num_conservative)
    real(real64) :: face_correction(air5_num_conservative),rhs_sign
    real(real64) :: left_excess_flux,right_excess_flux
    real(real64) :: origin_excess,current_excess,rhs_excess
    real(real64) :: local_min_ratio,global_min_ratio
    integer :: dims(3),ntypes(3),i,j,k,species,component,direction,constraint_mask
    integer :: species_mask
    integer :: local_limited,global_limited,local_invalid,global_invalid,ierr
    logical :: report_diagnostics

    dims=[im,jm,km]
    ntypes=[npdci,npdcj,npdck]
    cdt=rk_c*deltat
    report_diagnostics=nstep==0 .or. (feqchkpt>0 .and. mod(nstep,feqchkpt)==0)
    diffusion_ratio=1.0_real64
    if(rkstep==1) then
      do component=1,air5_num_conservative
        transport_origin_state(:,:,:,component)=q(0:im,0:jm,0:km,component)* &
          jacob(0:im,0:jm,0:km)
      enddo
      do species=1,air5_num_species
        transport_origin_species(:,:,:,species)= &
          q(0:im,0:jm,0:km,air5_idx_species_first+species-1)* &
          jacob(0:im,0:jm,0:km)
      enddo
      transport_origin_ev=q(0:im,0:jm,0:km,air5_idx_ev)* &
        jacob(0:im,0:jm,0:km)
    endif

    local_limited=0
    local_invalid=0
    local_min_ratio=1.0_real64
    do k=ks,ke
      do j=js,je
        do i=is,ie
          call air5_node_face_fluxes(i,j,k,dims,ntypes,left_flux,right_flux)
          do component=1,air5_num_conservative
            base_state(component)=rk_a*transport_origin_state(i,j,k,component)+ &
              rk_b*q(i,j,k,component)*jacob(i,j,k)+cdt*qrhs(i,j,k,component)
          enddo
          pminus=0.0_real64
          do species=1,air5_num_species
            component=air5_idx_species_first+species-1
            do direction=1,3
              pminus(species)=pminus(species)+ &
                min(0.0_real64,cdt*left_flux(direction,component))+ &
                min(0.0_real64,-cdt*right_flux(direction,component))
            enddo
          enddo
          do direction=1,3
            left_excess_flux=left_flux(direction,air5_idx_ev)+ &
              dot_product(vibrational_floor_energy, &
                left_flux(direction,air5_idx_species_first:air5_idx_species_last))
            right_excess_flux=right_flux(direction,air5_idx_ev)+ &
              dot_product(vibrational_floor_energy, &
                right_flux(direction,air5_idx_species_first:air5_idx_species_last))
            pminus(air5_num_species+1)=pminus(air5_num_species+1)+ &
              min(0.0_real64,-cdt*left_excess_flux)+ &
              min(0.0_real64,cdt*right_excess_flux)
          enddo
          ratio=1.0_real64
          if(.not.air5_state_is_admissible(base_state)) local_invalid=1
          do species=1,air5_num_species
            component=air5_idx_species_first+species-1
            base=rk_a*transport_origin_species(i,j,k,species)+ &
              rk_b*q(i,j,k,component)*jacob(i,j,k)+cdt*qrhs(i,j,k,component)
            if(base<0.0_real64) then
              local_invalid=1
            elseif(pminus(species)<0.0_real64) then
              ratio=min(ratio,air5_interior_ratio(base/(-pminus(species))))
            endif
          enddo
          origin_excess=transport_origin_ev(i,j,k)- &
            dot_product(vibrational_floor_energy,transport_origin_species(i,j,k,:))
          current_excess=air5_vibrational_excess(q(i,j,k,air5_idx_ev), &
            q(i,j,k,air5_idx_species_first:air5_idx_species_last))*jacob(i,j,k)
          rhs_excess=qrhs(i,j,k,air5_idx_ev)- &
            dot_product(vibrational_floor_energy, &
              qrhs(i,j,k,air5_idx_species_first:air5_idx_species_last))
          base=rk_a*origin_excess+rk_b*current_excess+cdt*rhs_excess
          if(base<0.0_real64) then
            local_invalid=1
          elseif(pminus(air5_num_species+1)<0.0_real64) then
            ratio=min(ratio,air5_interior_ratio(base/(-pminus(air5_num_species+1))))
          endif
          do direction=1,3
            face_correction=0.0_real64
            do component=2,air5_num_conservative
              rhs_sign=1.0_real64
              if(component>=air5_idx_species_first .and. &
                 component<=air5_idx_species_last) rhs_sign=-1.0_real64
              face_correction(component)=-cdt*rhs_sign*left_flux(direction,component)
            enddo
            ratio=min(ratio,air5_admissible_face_ratio(base_state,face_correction, &
              .false.,constraint_mask,species_mask))
            face_correction=0.0_real64
            do component=2,air5_num_conservative
              rhs_sign=1.0_real64
              if(component>=air5_idx_species_first .and. &
                 component<=air5_idx_species_last) rhs_sign=-1.0_real64
              face_correction(component)=cdt*rhs_sign*right_flux(direction,component)
            enddo
            ratio=min(ratio,air5_admissible_face_ratio(base_state,face_correction, &
              .false.,constraint_mask,species_mask))
          enddo
          ratio=max(0.0_real64,min(1.0_real64,ratio))
          diffusion_ratio(i,j,k)=ratio
          local_min_ratio=min(local_min_ratio,ratio)
          if(ratio<1.0_real64-1.0e-14_real64) local_limited=local_limited+1
        enddo
      enddo
    enddo
    call MPI_Allreduce(local_invalid,global_invalid,1,MPI_INTEGER,MPI_MAX, &
      MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,ierr,local_invalid)
    if(global_invalid/=0) then
      if(local_invalid/=0) write(*,'(A,I0)') &
        'air5 transport baseline is outside the admissible domain at RK stage ',rkstep
      call MPI_Abort(MPI_COMM_WORLD,3,ierr)
    endif
    call dataswap(diffusion_ratio)

    do k=ks,ke
      do j=js,je
        do i=is,ie
          call air5_node_face_fluxes(i,j,k,dims,ntypes,left_flux,right_flux)
          do direction=1,3
            select case(direction)
            case(1)
              theta_left=min(diffusion_ratio(i,j,k),diffusion_ratio(i-1,j,k))
              theta_right=min(diffusion_ratio(i,j,k),diffusion_ratio(i+1,j,k))
            case(2)
              theta_left=min(diffusion_ratio(i,j,k),diffusion_ratio(i,j-1,k))
              theta_right=min(diffusion_ratio(i,j,k),diffusion_ratio(i,j+1,k))
            case(3)
              theta_left=min(diffusion_ratio(i,j,k),diffusion_ratio(i,j,k-1))
              theta_right=min(diffusion_ratio(i,j,k),diffusion_ratio(i,j,k+1))
            end select
            do component=2,air5_num_conservative
              df=theta_right*right_flux(direction,component)- &
                theta_left*left_flux(direction,component)
              if(component>=air5_idx_species_first .and. &
                 component<=air5_idx_species_last) df=-df
              qrhs(i,j,k,component)=qrhs(i,j,k,component)+df
            enddo
          enddo
        enddo
      enddo
    enddo

    if(report_diagnostics) then
      call MPI_Allreduce(local_limited,global_limited,1,MPI_INTEGER,MPI_SUM, &
        MPI_COMM_WORLD,ierr)
      if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,ierr,local_limited)
      call MPI_Allreduce(local_min_ratio,global_min_ratio,1,MPI_DOUBLE_PRECISION, &
        MPI_MIN,MPI_COMM_WORLD,ierr)
      if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,ierr,local_limited)
      if(lio .and. global_limited>0) write(*,'(A,I0,A,I0,A,ES12.4)') &
        'AIR5_DIFFUSION_LIMITER stage=',rkstep,' limited_points=',global_limited, &
        ' min_ratio=',global_min_ratio
    endif
  end subroutine limit_air5_diffusive_fluxes

  pure subroutine differentiate_air5_flux(f,df,dim,hm,ntype)
    integer, intent(in) :: dim,hm,ntype
    real(real64), intent(in) :: f(-hm:dim+hm,air5_num_conservative)
    real(real64), intent(out) :: df(0:dim,air5_num_conservative)
    integer :: i,component

    df=0.0_real64
    do component=2,air5_num_conservative
      select case(ntype)
      case(1)
        df(0,component)=-0.5_real64*f(2,component)+ &
          2.0_real64*f(1,component)-1.5_real64*f(0,component)
        df(1,component)=0.5_real64*(f(2,component)-f(0,component))
        df(2,component)=num2d3*(f(3,component)-f(1,component))- &
          num1d12*(f(4,component)-f(0,component))
        do i=3,dim
          df(i,component)=0.75_real64*(f(i+1,component)-f(i-1,component))- &
            0.15_real64*(f(i+2,component)-f(i-2,component))+ &
            num1d60*(f(i+3,component)-f(i-3,component))
        enddo
      case(2)
        do i=0,dim-3
          df(i,component)=0.75_real64*(f(i+1,component)-f(i-1,component))- &
            0.15_real64*(f(i+2,component)-f(i-2,component))+ &
            num1d60*(f(i+3,component)-f(i-3,component))
        enddo
        df(dim-2,component)=num2d3*(f(dim-1,component)-f(dim-3,component))- &
          num1d12*(f(dim,component)-f(dim-4,component))
        df(dim-1,component)=0.5_real64*(f(dim,component)-f(dim-2,component))
        df(dim,component)=0.5_real64*f(dim-2,component)- &
          2.0_real64*f(dim-1,component)+1.5_real64*f(dim,component)
      case(3)
        do i=0,dim
          df(i,component)=0.75_real64*(f(i+1,component)-f(i-1,component))- &
            0.15_real64*(f(i+2,component)-f(i-2,component))+ &
            num1d60*(f(i+3,component)-f(i-3,component))
        enddo
      case(4)
        df(0,component)=-0.5_real64*f(2,component)+ &
          2.0_real64*f(1,component)-1.5_real64*f(0,component)
        df(1,component)=0.5_real64*(f(2,component)-f(0,component))
        df(2,component)=num2d3*(f(3,component)-f(1,component))- &
          num1d12*(f(4,component)-f(0,component))
        do i=3,dim-3
          df(i,component)=0.75_real64*(f(i+1,component)-f(i-1,component))- &
            0.15_real64*(f(i+2,component)-f(i-2,component))+ &
            num1d60*(f(i+3,component)-f(i-3,component))
        enddo
        df(dim-2,component)=num2d3*(f(dim-1,component)-f(dim-3,component))- &
          num1d12*(f(dim,component)-f(dim-4,component))
        df(dim-1,component)=0.5_real64*(f(dim,component)-f(dim-2,component))
        df(dim,component)=0.5_real64*f(dim-2,component)- &
          2.0_real64*f(dim-1,component)+1.5_real64*f(dim,component)
      end select
    enddo
  end subroutine differentiate_air5_flux

  pure subroutine add_air5_diffusive_divergence(qrhs,df)
    real(real64), intent(inout) :: qrhs(:,:)
    real(real64), intent(in) :: df(:,:)

    ! air5_diffusive_flux returns Js=-rho*Ds*grad(Ys), so species RHS uses -div(Js).
    qrhs(:,air5_idx_momentum_first:air5_idx_total_energy)= &
      qrhs(:,air5_idx_momentum_first:air5_idx_total_energy)+ &
      df(:,air5_idx_momentum_first:air5_idx_total_energy)
    qrhs(:,air5_idx_species_first:air5_idx_species_last)= &
      qrhs(:,air5_idx_species_first:air5_idx_species_last)- &
      df(:,air5_idx_species_first:air5_idx_species_last)
    qrhs(:,air5_idx_ev)=qrhs(:,air5_idx_ev)+df(:,air5_idx_ev)
  end subroutine add_air5_diffusive_divergence

end module chemistry_flow_solver
