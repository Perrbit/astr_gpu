module chemistry_flow_solver
  use iso_fortran_env, only: real64
  use constdef, only: num1d60,num1d12,num2d3
  use chemistry_air5_data, only: air5_num_species
  use chemistry_model, only: chemistry_status_ok,air5_flux_limiter_safety
  use chemistry_state_layout, only: air5_num_conservative,air5_idx_density, &
    air5_idx_momentum_first,air5_idx_total_energy,air5_idx_species_first, &
    air5_idx_species_last,air5_idx_ev
  use chemistry_flow_state, only: air5_conservative_to_primitive
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

  public :: air5_diffusion_rhs
  public :: air5_limit_species_convection
  public :: air5_chemistry_half_step

contains

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

  subroutine air5_low_order_species_face_flux(i,j,k,direction,face,density_flux, &
      species_face_flux)
    use commarray, only: q
    integer, intent(in) :: i,j,k,direction,face
    real(real64), intent(in) :: density_flux
    real(real64), intent(out) :: species_face_flux(air5_num_species)
    integer :: donor_i,donor_j,donor_k,species,component
    real(real64) :: density

    donor_i=i; donor_j=j; donor_k=k
    select case(direction)
    case(1)
      donor_i=merge(face,face+1,density_flux>=0.0_real64)
    case(2)
      donor_j=merge(face,face+1,density_flux>=0.0_real64)
    case(3)
      donor_k=merge(face,face+1,density_flux>=0.0_real64)
    end select
    density=q(donor_i,donor_j,donor_k,air5_idx_density)
    if(density<=0.0_real64) error stop 'air5 low-order species flux has non-positive density'
    do species=2,air5_num_species
      component=air5_idx_species_first+species-1
      species_face_flux(species)=density_flux* &
        q(donor_i,donor_j,donor_k,component)/density
    enddo
    species_face_flux(1)=density_flux-sum(species_face_flux(2:air5_num_species))
  end subroutine air5_low_order_species_face_flux

  subroutine air5_species_node_face_fluxes(i,j,k,dims,ntypes,high_left,high_right, &
      low_left,low_right)
    integer, intent(in) :: i,j,k,dims(3),ntypes(3)
    real(real64), intent(out) :: high_left(3,air5_num_species)
    real(real64), intent(out) :: high_right(3,air5_num_species)
    real(real64), intent(out) :: low_left(3,air5_num_species)
    real(real64), intent(out) :: low_right(3,air5_num_species)
    real(real64) :: full_left(air5_num_conservative),full_right(air5_num_conservative)
    integer :: direction,index

    do direction=1,3
      select case(direction)
      case(1); index=i
      case(2); index=j
      case(3); index=k
      end select
      call air5_high_order_convective_face_flux(i,j,k,direction,index-1, &
        dims(direction),ntypes(direction),full_left)
      call air5_high_order_convective_face_flux(i,j,k,direction,index, &
        dims(direction),ntypes(direction),full_right)
      high_left(direction,:)=full_left(air5_idx_species_first:air5_idx_species_last)
      high_right(direction,:)=full_right(air5_idx_species_first:air5_idx_species_last)
      call air5_low_order_species_face_flux(i,j,k,direction,index-1, &
        full_left(air5_idx_density),low_left(direction,:))
      call air5_low_order_species_face_flux(i,j,k,direction,index, &
        full_right(air5_idx_density),low_right(direction,:))
    enddo
  end subroutine air5_species_node_face_fluxes

  subroutine air5_limit_species_convection()
    use mpi
    use commvar, only: im,jm,km,hm,npdci,npdcj,npdck,is,ie,js,je,ks,ke, &
      deltat,rkstep,rkscheme,nstep,feqchkpt
    use commarray, only: q,qrhs,jacob
    use parallel, only: dataswap,lio
    real(real64) :: high_left(3,air5_num_species),high_right(3,air5_num_species)
    real(real64) :: low_left(3,air5_num_species),low_right(3,air5_num_species)
    real(real64) :: low_rhs(air5_num_species),pminus(air5_num_species)
    real(real64) :: correction_left,correction_right,base,ratio,cdt
    real(real64) :: theta_left,theta_right,rk_a,rk_b,rk_c
    real(real64) :: local_min_ratio,global_min_ratio
    integer :: dims(3),ntypes(3),i,j,k,species,component,direction
    integer :: local_limited,global_limited,local_invalid,global_invalid,ierr
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
      do species=1,air5_num_species
        component=air5_idx_species_first+species-1
        transport_origin_species(:,:,:,species)=q(0:im,0:jm,0:km,component)* &
          jacob(0:im,0:jm,0:km)
      enddo
    endif
    local_limited=0; local_invalid=0; local_min_ratio=1.0_real64

    do k=ks,ke; do j=js,je; do i=is,ie
      call air5_species_node_face_fluxes(i,j,k,dims,ntypes,high_left,high_right, &
        low_left,low_right)
      low_rhs=0.0_real64; pminus=0.0_real64
      do species=1,air5_num_species
        do direction=1,3
          low_rhs(species)=low_rhs(species)+low_left(direction,species)- &
            low_right(direction,species)
          correction_left=cdt*(high_left(direction,species)-low_left(direction,species))
          correction_right=-cdt*(high_right(direction,species)-low_right(direction,species))
          pminus(species)=pminus(species)+min(0.0_real64,correction_left)+ &
            min(0.0_real64,correction_right)
        enddo
      enddo
      ratio=1.0_real64
      do species=1,air5_num_species
        component=air5_idx_species_first+species-1
        base=rk_a*transport_origin_species(i,j,k,species)+ &
          rk_b*q(i,j,k,component)*jacob(i,j,k)+ &
          cdt*low_rhs(species)
        if(base<0.0_real64) then
          local_invalid=1
        elseif(pminus(species)<0.0_real64) then
          ratio=min(ratio,air5_flux_limiter_safety*base/(-pminus(species)))
        endif
      enddo
      ratio=max(0.0_real64,min(1.0_real64,ratio))
      diffusion_ratio(i,j,k)=ratio
      local_min_ratio=min(local_min_ratio,ratio)
      if(ratio<1.0_real64-1.0e-14_real64) local_limited=local_limited+1
    enddo; enddo; enddo

    call MPI_Allreduce(local_invalid,global_invalid,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,ierr,local_invalid)
    if(global_invalid/=0) then
      if(local_invalid/=0) write(*,'(A,I0)') &
        'air5 low-order species convection baseline failed at RK stage ',rkstep
      call MPI_Abort(MPI_COMM_WORLD,3,ierr)
    endif
    call dataswap(diffusion_ratio)

    do k=ks,ke; do j=js,je; do i=is,ie
      call air5_species_node_face_fluxes(i,j,k,dims,ntypes,high_left,high_right, &
        low_left,low_right)
      do species=1,air5_num_species
        low_rhs(species)=0.0_real64
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
          low_rhs(species)=low_rhs(species)+low_left(direction,species)- &
            low_right(direction,species)+theta_left*(high_left(direction,species)- &
            low_left(direction,species))-theta_right*(high_right(direction,species)- &
            low_right(direction,species))
        enddo
        qrhs(i,j,k,air5_idx_species_first+species-1)=low_rhs(species)
      enddo
    enddo; enddo; enddo

    if(report_diagnostics) then
      call MPI_Allreduce(local_limited,global_limited,1,MPI_INTEGER,MPI_SUM,MPI_COMM_WORLD,ierr)
      if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,ierr,local_limited)
      call MPI_Allreduce(local_min_ratio,global_min_ratio,1,MPI_DOUBLE_PRECISION,MPI_MIN, &
        MPI_COMM_WORLD,ierr)
      if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,ierr,local_limited)
      if(lio .and. global_limited>0) write(*,'(A,I0,A,I0,A,ES12.4)') &
        'AIR5_CONVECTION_LIMITER stage=',rkstep,' limited_points=',global_limited, &
        ' min_ratio=',global_min_ratio
    endif
  end subroutine air5_limit_species_convection

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
    real(real64) :: pminus(air5_num_species),base,ratio,cdt
    real(real64) :: theta_left,theta_right,df
    real(real64) :: local_min_ratio,global_min_ratio
    integer :: dims(3),ntypes(3),i,j,k,species,component,direction
    integer :: local_limited,global_limited,local_invalid,global_invalid,ierr
    logical :: report_diagnostics

    dims=[im,jm,km]
    ntypes=[npdci,npdcj,npdck]
    cdt=rk_c*deltat
    report_diagnostics=nstep==0 .or. (feqchkpt>0 .and. mod(nstep,feqchkpt)==0)
    diffusion_ratio=1.0_real64
    if(rkstep==1) then
      do species=1,air5_num_species
        transport_origin_species(:,:,:,species)= &
          q(0:im,0:jm,0:km,air5_idx_species_first+species-1)* &
          jacob(0:im,0:jm,0:km)
      enddo
    endif

    local_limited=0
    local_invalid=0
    local_min_ratio=1.0_real64
    do k=ks,ke
      do j=js,je
        do i=is,ie
          call air5_node_face_fluxes(i,j,k,dims,ntypes,left_flux,right_flux)
          pminus=0.0_real64
          do species=1,air5_num_species
            component=air5_idx_species_first+species-1
            do direction=1,3
              pminus(species)=pminus(species)+ &
                min(0.0_real64,cdt*left_flux(direction,component))+ &
                min(0.0_real64,-cdt*right_flux(direction,component))
            enddo
          enddo
          ratio=1.0_real64
          do species=1,air5_num_species
            component=air5_idx_species_first+species-1
            base=rk_a*transport_origin_species(i,j,k,species)+ &
              rk_b*q(i,j,k,component)*jacob(i,j,k)+cdt*qrhs(i,j,k,component)
            if(base<0.0_real64) then
              local_invalid=1
            elseif(pminus(species)<0.0_real64) then
              ratio=min(ratio,air5_flux_limiter_safety*base/(-pminus(species)))
            endif
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
        'air5 transport baseline is negative before diffusion at RK stage ',rkstep
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
