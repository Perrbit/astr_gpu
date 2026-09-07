module conservative_boundary_runtime
  use conservative_boundary_config, only: conservative_boundary_settings,read_conservative_boundary_config
  use parallel, only: mpirank,bcast
  implicit none
  private
  type(conservative_boundary_settings),save,public,protected :: conservative_boundary
  public :: load_conservative_boundary_environment
  public :: initialize_conservative_boundary_state,apply_conservative_boundary_stage
  public :: validate_conservative_initial_profile,validate_conservative_restart_state
  public :: validate_conservative_sbli_mode
  logical,save :: state_ready=.false.
contains
  subroutine load_conservative_boundary_environment()
    character(len=:),allocatable :: filename
    integer :: status,length,env_status
    conservative_boundary=conservative_boundary_settings()
    state_ready=.false.
    status=0
    if(mpirank==0) then
      call get_environment_variable('ASTR_CONSERVATIVE_BOUNDARY_FILE',length=length,status=env_status)
      if(env_status/=0.and.env_status/=1) then
        status=1
      elseif(env_status==0.and.length>0) then
        allocate(character(len=length) :: filename)
        call get_environment_variable('ASTR_CONSERVATIVE_BOUNDARY_FILE',value=filename,status=env_status)
        if(env_status/=0.or.len_trim(filename)==0) then
          status=1
        else
          call read_conservative_boundary_config(filename,conservative_boundary,status)
        endif
      endif
    endif
    ! All ranks reach the status broadcast before any error exit.
    call bcast(status)
    if(status/=0) then
      conservative_boundary=conservative_boundary_settings()
      error stop 'Invalid ASTR_CONSERVATIVE_BOUNDARY_FILE configuration'
    endif
    call bcast(conservative_boundary%enabled)
    call bcast(conservative_boundary%split_x)
    call bcast(conservative_boundary%q_left)
    call bcast(conservative_boundary%q_right)
  end subroutine load_conservative_boundary_environment

  subroutine validate_conservative_sbli_mode(tw)
    use commvar, only: numq,num_species,num_modequ,ndims,nondimen,turbmode, &
      rkscheme,flowtype,conschm,difschm,recon_schem,lchardecomp,lfilter, &
      diffterm,lihomo,ljhomo,lkhomo,lfftk,ninit,prandtl,gamma, &
      mach,reynolds,ref_tem,tempconst,hm
    use bc, only: bctype
    use sponge_layer, only: spg_i0,spg_im,spg_j0,spg_jm,spg_k0,spg_km
    use parallel, only: por,mpisize,isize,jsize,ksize
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    real(8),intent(in) :: tw
    real(8),parameter :: target_left(5)=[1.00000596004d0,1.00000268202d0, &
      0.00565001630205d0,0.d0,0.94644428042d0]
    real(8),parameter :: target_right(5)=[1.129734572d0,1.0921171d0, &
      -0.058866065d0,0.d0,1.0590824d0]
    logical :: topology_ok
    if(.not.conservative_boundary%enabled) return
    if(por(trim(flowtype)/='bl')) error stop 'OpenSBLI conservative mode requires flowtype=bl'
    if(por(numq/=5.or.num_species/=0.or.num_modequ/=0.or.ndims/=3.or. &
      .not.nondimen.or.trim(turbmode)/='none')) &
      error stop 'OpenSBLI conservative mode requires 3D perfect-gas DNS layout'
    if(por(trim(rkscheme)/='rk3'.or.trim(conschm)/='543e'.or. &
      trim(difschm)/='643e'.or.recon_schem/=3.or..not.lchardecomp)) &
      error stop 'OpenSBLI conservative mode requires conschm=543e difschm=643e RK3 MP7 Roe'
    if(por(lfilter)) error stop 'OpenSBLI conservative mode requires lfilter=false'
    if(por(.not.diffterm)) error stop 'OpenSBLI conservative mode requires diffusion'
    if(por(lihomo.or.ljhomo.or..not.lkhomo.or.lfftk)) &
      error stop 'OpenSBLI conservative mode requires x/y physical and z periodic topology'
    if(por(ninit/=3)) error stop 'OpenSBLI conservative mode requires ninit=3'
    if(por(hm<5)) error stop 'OpenSBLI conservative mode requires at least five halo layers'
    if(por(any(bctype/=[11,21,41,52,1,1]))) &
      error stop 'OpenSBLI conservative mode boundary topology contract is 11/21/41/52/1/1'
    if(por(any([spg_i0,spg_im,spg_j0,spg_jm,spg_k0,spg_km]/=0))) &
      error stop 'OpenSBLI conservative mode requires sponge widths zero'
    topology_ok=(mpisize==1.and.isize==1.and.jsize==1.and.ksize==1).or. &
      (mpisize==2.and.((isize==2.and.jsize==1.and.ksize==1).or. &
                       (isize==1.and.jsize==2.and.ksize==1).or. &
                       (isize==1.and.jsize==1.and.ksize==2))).or. &
      (mpisize==4.and.((isize==2.and.jsize==2.and.ksize==1).or. &
                       (isize==2.and.jsize==1.and.ksize==2).or. &
                       (isize==1.and.jsize==2.and.ksize==2)))
    if(por(.not.topology_ok)) &
      error stop 'OpenSBLI conservative mode requires a supported NP=1/2/4 topology'
    if(por(.not.ieee_is_finite(prandtl).or. &
      (abs(prandtl-0.71d0)>1.d-12.and.abs(prandtl-0.72d0)>1.d-12))) &
      error stop 'OpenSBLI conservative mode Prandtl number must be explicitly 0.71 or 0.72'
    if(por(.not.ieee_is_finite(tw).or.tw<=0.d0)) &
      error stop 'OpenSBLI conservative mode wall temperature must be finite and positive'
    if(por(abs(tw-1.676194d0)>1.d-12)) &
      error stop 'OpenSBLI conservative mode wall temperature must equal 1.676194'
    if(por(abs(gamma-1.4d0)>1.d-12.or.abs(mach-2.d0)>1.d-12.or. &
      abs(reynolds-950.d0)>1.d-12.or.abs(ref_tem-288.d0)>1.d-12.or. &
      abs(tempconst-110.4d0/288.d0)>1.d-12)) &
      error stop 'OpenSBLI conservative mode reference normalization mismatch'
    if(por(abs(conservative_boundary%split_x-40.d0)>1.d-12.or. &
      any(abs(conservative_boundary%q_left-target_left)>1.d-12).or. &
      any(abs(conservative_boundary%q_right-target_right)>1.d-12))) &
      error stop 'OpenSBLI conservative mode top-state contract mismatch'
  end subroutine validate_conservative_sbli_mode

  subroutine initialize_conservative_boundary_state(tw)
    use commvar, only: im,jm,km,hm,numq,ndims,nondimen,npdci,npdcj,npdck,lrestart
    use commarray, only: q,dxi,jacob
    use parallel, only: por
    use rectilinear_metric_halo, only: fill_rectilinear_metric_halo
    real(8),intent(in) :: tw
    logical :: owned(6)
    integer :: h,status
    if(.not.conservative_boundary%enabled) error stop 'Conservative boundary configuration is disabled'
    if(por(state_ready)) error stop 'Conservative boundary state cannot be reseeded'
    if(por(numq/=5.or.ndims/=3.or..not.nondimen.or.npdck/=3.or.min(im,jm,km)<hm)) &
      error stop 'Unsupported conservative boundary state layout'
    if(lrestart) then
      call validate_conservative_restart_state()
    else
      call validate_conservative_initial_profile()
    endif
    owned=[npdci==1.or.npdci==4,npdci==2.or.npdci==4, &
           npdcj==1.or.npdcj==4,npdcj==2.or.npdcj==4,.false.,.false.]
    call fill_rectilinear_metric_halo(im,jm,km,hm,owned,jacob,dxi,status)
    if(por(status/=0)) error stop 'Invalid conservative boundary rectilinear metric'
    if(.not.lrestart) then
      ! Only a fresh x-uniform profile needs initial x-face halo seeding.
      do h=1,hm
        if(owned(1)) q(-h,0:jm,0:km,:)=q(0,0:jm,0:km,:)
        if(owned(2)) q(im+h,0:jm,0:km,:)=q(im,0:jm,0:km,:)
      enddo
    else
      call seed_conservative_restart_inlet(owned(1))
    endif
    state_ready=.true.
    call apply_conservative_boundary_stage(tw)
  end subroutine initialize_conservative_boundary_state

  subroutine validate_conservative_initial_profile()
    use commvar, only: im,jm,km
    use commarray, only: q
    use parallel, only: por,irk,jrk,krk,jsize
    use mpi
    real(8) :: reference(0:jm,0:km,5)
    integer :: i,j,k,line_comm,ierr,count_min,count_max,n
    logical :: bad
    call validate_conservative_restart_state()
    ! One initialization-only communicator per transverse MPI block. No new
    ! halo tags or production transport buffers are introduced.
    call MPI_Comm_split(MPI_COMM_WORLD,jrk+jsize*krk,irk,line_comm,ierr)
    if(ierr/=MPI_SUCCESS) error stop 'Initial profile communicator failed'
    n=size(reference)
    call MPI_Allreduce(n,count_min,1,MPI_INTEGER,MPI_MIN,line_comm,ierr)
    if(ierr/=MPI_SUCCESS) error stop 'Initial profile size reduction failed'
    call MPI_Allreduce(n,count_max,1,MPI_INTEGER,MPI_MAX,line_comm,ierr)
    if(ierr/=MPI_SUCCESS) error stop 'Initial profile size reduction failed'
    if(por(count_min/=count_max)) error stop 'Inconsistent transverse initial profile layout'
    reference=q(0,0:jm,0:km,:)
    call MPI_Bcast(reference,n,MPI_DOUBLE_PRECISION,0,line_comm,ierr)
    if(ierr/=MPI_SUCCESS) error stop 'Initial profile broadcast failed'
    bad=.false.
    do i=0,im
      if(any(abs(q(i,0:jm,0:km,:)-reference)>1.d-12*(1.d0+abs(reference)))) bad=.true.
    enddo
    call MPI_Comm_free(line_comm,ierr)
    if(ierr/=MPI_SUCCESS) error stop 'Initial profile communicator release failed'
    if(por(bad)) error stop 'Conservative initial profile must be x-uniform'
  end subroutine validate_conservative_initial_profile

  subroutine validate_conservative_restart_state()
    use commvar, only: im,jm,km
    use commarray, only: q
    use parallel, only: por
    use perfect_gas_boundary, only: admissible
    integer :: i,j,k
    logical :: bad
    bad=.false.
    do k=0,km
    do j=0,jm
    do i=0,im
      if(.not.admissible(q(i,j,k,:))) bad=.true.
    enddo
    enddo
    enddo
    if(por(bad)) error stop 'Invalid conservative physical state'
  end subroutine validate_conservative_restart_state

  subroutine seed_conservative_restart_inlet(owns_inlet)
    use commvar, only: jm,km,hm,gamma
    use commarray, only: q
    use bc, only: rho_prof,vel_prof,prs_prof
    use parallel, only: por
    use perfect_gas_boundary, only: admissible
    logical,intent(in) :: owns_inlet
    real(8) :: inlet(5)
    integer :: h,j,k
    logical :: bad
    bad=.false.
    if(owns_inlet) then
      bad=.not.allocated(rho_prof).or..not.allocated(vel_prof).or..not.allocated(prs_prof)
      if(.not.bad) then
        bad=lbound(rho_prof,1)>0.or.ubound(rho_prof,1)<jm.or. &
            lbound(vel_prof,1)>0.or.ubound(vel_prof,1)<jm.or. &
            lbound(vel_prof,2)>1.or.ubound(vel_prof,2)<3.or. &
            lbound(prs_prof,1)>0.or.ubound(prs_prof,1)<jm
      endif
    endif
    if(por(bad)) error stop 'Missing conservative restart inlet profile'
    if(owns_inlet) then
      do k=0,km
      do j=0,jm
        inlet(1)=rho_prof(j)
        inlet(2:4)=rho_prof(j)*vel_prof(j,1:3)
        inlet(5)=prs_prof(j)/(gamma-1.d0)+0.5d0*rho_prof(j)*sum(vel_prof(j,1:3)**2)
        if(.not.admissible(inlet)) bad=.true.
        do h=1,hm
          q(-h,j,k,:)=inlet
        enddo
      enddo
      enddo
    endif
    if(por(bad)) error stop 'Invalid conservative restart inlet profile'
  end subroutine seed_conservative_restart_inlet

  subroutine apply_conservative_boundary_stage(tw)
    use commvar, only: im,jm,km,hm,gamma,mach,npdci,npdcj
    use commarray, only: q,x,rho,vel,prs,tmp,spc
    use parallel, only: qswap,por
    use fludyna, only: updatefvar,q2fvar
    use conservative_boundary_faces, only: apply_conservative_face_column
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    real(8),intent(in) :: tw
    real(8) :: xline(-hm:im+hm)
    logical :: owned(4),bad
    integer :: side,s,k,lo,hi,status,h
    if(por(.not.state_ready)) error stop 'Conservative boundary state is not initialized'
    owned=[npdci==1.or.npdci==4,npdci==2.or.npdci==4, &
           npdcj==1.or.npdcj==4,npdcj==2.or.npdcj==4]
    xline=x(:,0,0,1)
    do h=1,hm
      if(owned(1)) xline(-h)=xline(0)-h*(xline(1)-xline(0))
      if(owned(2)) xline(im+h)=xline(im)+h*(xline(im)-xline(im-1))
    enddo
    if(por(.not.all(ieee_is_finite(xline)))) error stop 'Invalid boundary x coordinates'
    call updatefvar()
    call qswap()
    do side=1,4
      bad=.false.
      if(owned(side)) then
        lo=0; hi=jm
        if(side>=3) then
          lo=-hm; hi=im+hm
        endif
        do k=0,km
        do s=lo,hi
          call apply_conservative_face_column(side,s,k,im,jm,km,hm,q,xline,tw,gamma,mach, &
            conservative_boundary%split_x,conservative_boundary%q_left,conservative_boundary%q_right,status)
          bad=bad.or.status/=0
        enddo
        enddo
      endif
      if(por(bad)) error stop 'Invalid conservative boundary column'
      ! Wall columns on x-halo indices require the updated x-face/MPI states.
      if(side==2) call qswap()
    enddo
    call qswap()
    call updatefvar()
    ! qswap refreshes communicated faces, not physical boundary ghosts.
    call q2fvar(q=q(-hm:im+hm,0:jm,0:km,:),density=rho(-hm:im+hm,0:jm,0:km), &
      velocity=vel(-hm:im+hm,0:jm,0:km,:),pressure=prs(-hm:im+hm,0:jm,0:km), &
      temperature=tmp(-hm:im+hm,0:jm,0:km),species=spc(-hm:im+hm,0:jm,0:km,:))
    call q2fvar(q=q(0:im,-hm:jm+hm,0:km,:),density=rho(0:im,-hm:jm+hm,0:km), &
      velocity=vel(0:im,-hm:jm+hm,0:km,:),pressure=prs(0:im,-hm:jm+hm,0:km), &
      temperature=tmp(0:im,-hm:jm+hm,0:km),species=spc(0:im,-hm:jm+hm,0:km,:))
  end subroutine apply_conservative_boundary_stage
end module conservative_boundary_runtime
