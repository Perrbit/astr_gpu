module chemistry_flow_solver
  use iso_fortran_env, only: real64
  use constdef, only: num1d60,num1d12,num2d3
  use chemistry_air5_data, only: air5_num_species
  use chemistry_model, only: chemistry_status_ok
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

  public :: air5_diffusion_rhs
  public :: air5_chemistry_half_step

contains

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
    use commvar, only: im,jm,km,hm,difschm,npdci,npdcj,npdck,flowtype
    use commarray, only: rho,vel,tmp,tve,prs,spc,dvel,dtmp,dspc,dtve, &
      dxi,jacob,qrhs
    use comsolver, only: grad
    use parallel, only: dataswap,mpidown
    integer :: i,j,k,component,status,global_status,ierr
    real(real64) :: local_momentum_flux(3,3)
    real(real64) :: local_species_flux(air5_num_species,3)
    real(real64) :: local_energy_flux(3),local_vibrational_flux(3)
    real(real64), allocatable :: f(:,:),df(:,:)

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

    allocate(f(-hm:im+hm,air5_num_conservative), &
      df(0:im,air5_num_conservative))
    do k=0,km
      do j=0,jm
        do i=-hm,im+hm
          call projected_air5_diffusive_flux(i,j,k,1,f(i,:))
        enddo
        call differentiate_air5_flux(f,df,im,hm,npdci)
        call add_air5_diffusive_divergence(qrhs(:,j,k,:),df)
      enddo
    enddo
    deallocate(f,df)

    allocate(f(-hm:jm+hm,air5_num_conservative), &
      df(0:jm,air5_num_conservative))
    do k=0,km
      do i=0,im
        do j=-hm,jm+hm
          call projected_air5_diffusive_flux(i,j,k,2,f(j,:))
        enddo
        call differentiate_air5_flux(f,df,jm,hm,npdcj)
        call add_air5_diffusive_divergence(qrhs(i,:,k,:),df)
      enddo
    enddo
    deallocate(f,df)

    allocate(f(-hm:km+hm,air5_num_conservative), &
      df(0:km,air5_num_conservative))
    do j=0,jm
      do i=0,im
        do k=-hm,km+hm
          call projected_air5_diffusive_flux(i,j,k,3,f(k,:))
        enddo
        call differentiate_air5_flux(f,df,km,hm,npdck)
        call add_air5_diffusive_divergence(qrhs(i,j,:,:),df)
      enddo
    enddo
    deallocate(f,df)
  end subroutine air5_diffusion_rhs

  subroutine allocate_air5_flux_workspace(im,jm,km,hm)
    integer, intent(in) :: im,jm,km,hm

    if(allocated(momentum_flux)) return
    allocate(momentum_flux(-hm:im+hm,-hm:jm+hm,-hm:km+hm,6))
    allocate(energy_flux(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3))
    allocate(species_flux(-hm:im+hm,-hm:jm+hm,-hm:km+hm,air5_num_species,3))
    allocate(vibrational_flux(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3))
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
