module boundary_rhs_manufactured
  implicit none
contains
  subroutine check_conservative_admission()
    use commvar
    use bc, only: bctype,twall
    use sponge_layer, only: spg_i0,spg_im,spg_j0,spg_jm,spg_k0,spg_km
    use parallel, only: mpisize,isize,jsize,ksize
    use conservative_boundary_runtime, only: load_conservative_boundary_environment, &
      validate_conservative_sbli_mode
    character(len=32) :: mode
    integer :: env_status
    flowtype='bl'; numq=5; num_species=0; num_modequ=0; ndims=3
    nondimen=.true.; turbmode='none'; rkscheme='rk3'; conschm='543e'; difschm='643e'
    recon_schem=3; lchardecomp=.true.; lfilter=.false.; diffterm=.true.
    lihomo=.false.; ljhomo=.false.; lkhomo=.true.; lfftk=.false.
    ninit=3; lrestart=.false.; prandtl=.72d0; gamma=1.4d0; mach=2.d0
    reynolds=950.d0; ref_tem=288.d0; tempconst=110.4d0/288.d0
    bctype=[11,21,41,52,1,1]; twall=1.d0; twall(3)=1.676194d0
    spg_i0=0; spg_im=0; spg_j0=0; spg_jm=0; spg_k0=0; spg_km=0
    isize=1; jsize=1; ksize=1
    if(mpisize/=1) error stop 'Conservative admission probe requires NP1'
    call get_environment_variable('ASTR_CONSERVATIVE_ADMISSION_TEST',mode,status=env_status)
    if(env_status/=0) error stop 'Missing conservative admission test mode'
    select case(trim(mode))
    case('valid')
      continue
    case('wrong_flow')
      flowtype='tgv'
    case('wrong_scheme')
      conschm='643e'
    case('filter')
      lfilter=.true.
    case('bad_pr')
      prandtl=.715d0
    case('bad_tw')
      twall(3)=-1.d0
    case('wrong_ninit')
      ninit=2
    case('restart')
      lrestart=.true.
    case('wrong_bctype')
      bctype(4)=51
    case('wrong_topology')
      isize=2
    case default
      error stop 'Unknown conservative admission test mode'
    end select
    call load_conservative_boundary_environment()
    call validate_conservative_sbli_mode(twall(3))
    print*,'CONSERVATIVE_ADMISSION_PASS'
  end subroutine check_conservative_admission

  subroutine check_initial_profile_mpi()
    use commvar, only: im,jm,km,hm
    use commarray, only: q
    use parallel, only: mpirank,mpisize,irk,jrk,krk,jsize
    use conservative_boundary_runtime, only: validate_conservative_initial_profile, &
      validate_conservative_restart_state
    integer :: j,k,env_status
    character(len=32) :: mode
    logical :: restart_mode
    im=8; jm=8; km=8
    irk=mpirank; jrk=0; krk=0; jsize=1
    allocate(q(-hm:im+hm,-hm:jm+hm,-hm:km+hm,5))
    q=0.d0
    do k=0,km
    do j=0,jm
      q(:,j,k,1)=1.d0+0.01d0*j
      q(:,j,k,2)=0.2d0
      q(:,j,k,5)=2.5d0+0.01d0*j
    enddo
    enddo
    call get_environment_variable('ASTR_BOUNDARY_STAGE_INITIAL_TEST',mode,status=env_status)
    if(env_status/=0) error stop 'Missing initial profile test mode'
    restart_mode=.false.
    select case(trim(mode))
    case('valid')
      continue
    case('nonuniform')
      if(mpirank==mpisize-1) q(3,3,3,5)=q(3,3,3,5)+0.01d0
    case('rank_offset')
      q(:,0:jm,0:km,5)=q(:,0:jm,0:km,5)+0.01d0*mpirank
    case('negative_density')
      if(mpirank==mpisize-1) q(3,3,3,1)=-1.d0
    case('restart_nonuniform')
      restart_mode=.true.
      if(mpirank==mpisize-1) q(3,3,3,5)=q(3,3,3,5)+0.01d0
    case('restart_rank_offset')
      restart_mode=.true.
      q(:,0:jm,0:km,5)=q(:,0:jm,0:km,5)+0.01d0*mpirank
    case('restart_negative_density')
      restart_mode=.true.
      if(mpirank==mpisize-1) q(3,3,3,1)=-1.d0
    case default
      error stop 'Unknown initial profile test mode'
    end select
    if(restart_mode) then
      call validate_conservative_restart_state()
    else
      call validate_conservative_initial_profile()
    endif
    if(mpirank==0) print*,'BOUNDARY_INITIAL_MPI_PASS'
  end subroutine check_initial_profile_mpi

  subroutine check_boundary_stage()
    use commvar
    use commarray, only: q,rho,vel,prs,tmp,spc,x,jacob,dxi
    use parallel, only: mpisize,isize,jsize,ksize,mpileft,mpiright,mpidown,mpiup,mpiback,mpifront,irk,jrk,krk
    use mpi, only: MPI_PROC_NULL
    use conservative_boundary_runtime
    use bc, only: rho_prof,vel_prof,tmp_prof,prs_prof
#ifdef _CUDA
    use commarray_gpu, only: q_d,rho_d,vel_d,prs_d,tmp_d
    use conservative_boundary_stage_gpu
#endif
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    integer :: i,j,k,d
    real(8) :: energy,error
    character(len=32) :: initial_test
    integer :: env_status
#ifdef _CUDA
    real(8),allocatable :: gpu_q(:,:,:,:),gpu_scalar(:,:,:),gpu_vel(:,:,:,:)
#endif
    if(mpisize/=1) error stop 'Boundary stage probe requires NP1'
    im=8; jm=8; km=8; ndims=3; numq=5; num_species=0; num_modequ=0
    gamma=1.4d0; mach=2.d0; const2=gamma*mach**2; const6=1.d0/(gamma-1.d0)
    nondimen=.true.; turbmode='none'; ltimrpt=.false.; lreport=.false.; lrestart=.false.
    lihomo=.false.; ljhomo=.false.; lkhomo=.true.; lfftk=.false.; ka=km
    npdci=4; npdcj=4; npdck=3; isize=1; jsize=1; ksize=1
    irk=0; jrk=0; krk=0
    mpileft=MPI_PROC_NULL; mpiright=MPI_PROC_NULL; mpidown=MPI_PROC_NULL; mpiup=MPI_PROC_NULL
    mpiback=0; mpifront=0
    allocate(q(-hm:im+hm,-hm:jm+hm,-hm:km+hm,5),x(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3))
    allocate(rho(-hm:im+hm,-hm:jm+hm,-hm:km+hm),prs(-hm:im+hm,-hm:jm+hm,-hm:km+hm))
    allocate(tmp(-hm:im+hm,-hm:jm+hm,-hm:km+hm),vel(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3))
    allocate(spc(-hm:im+hm,-hm:jm+hm,-hm:km+hm,0),jacob(-hm:im+hm,-hm:jm+hm,-hm:km+hm))
    allocate(dxi(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3,3))
    jacob=1.d0; dxi=0.d0
    do d=1,3
      dxi(:,:,:,d,d)=1.d0
    enddo
    energy=const6/const2+0.02d0
    do k=-hm,km+hm
    do j=-hm,jm+hm
    do i=-hm,im+hm
      q(i,j,k,:)=[1.d0,0.2d0,0.d0,0.d0,energy]
      x(i,j,k,:)=real([i,j,k],8)*10.d0
    enddo
    enddo
    enddo
    allocate(rho_prof(0:jm),vel_prof(0:jm,3),tmp_prof(0:jm),prs_prof(0:jm))
    rho_prof=1.d0
    vel_prof=0.d0
    vel_prof(:,1)=0.2d0
    tmp_prof=1.d0
    prs_prof=1.d0/const2
    call load_conservative_boundary_environment()
    call get_environment_variable('ASTR_BOUNDARY_STAGE_INITIAL_TEST',initial_test,status=env_status)
    if(env_status==0) then
      select case(trim(initial_test))
      case('nonuniform')
        q(3,3,3,5)=q(3,3,3,5)+0.01d0
      case('negative_density')
        q(3,3,3,1)=-1.d0
      case('restart_nonuniform')
        lrestart=.true.
        q(3,3,3,5)=q(3,3,3,5)+0.01d0
        q(-hm:-1,:,:,:)=0.d0
        q(im+1:im+hm,:,:,:)=0.d0
        q(:,-hm:-1,:,:)=0.d0
        q(:,jm+1:jm+hm,:,:)=0.d0
        q(:,:,-hm:-1,:)=0.d0
        q(:,:,km+1:km+hm,:)=0.d0
      case default
        error stop 'Unknown boundary initial profile test'
      end select
    endif
    call initialize_conservative_boundary_state(1.d0)
    q(3,0,:,1)=1.01d0
    q(0,3,:,5)=energy+0.001d0
#ifdef _CUDA
    allocate(q_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm,5))
    allocate(rho_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm),prs_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm))
    allocate(tmp_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm),vel_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3))
    q_d=q
    call initialize_conservative_stage_gpu()
    call apply_conservative_stage_gpu(1.d0)
#endif
    call apply_conservative_boundary_stage(1.d0)
    error=maxval(abs(q(3,0,:,1)-1.01d0))
    error=max(error,maxval(abs(vel(0:im,0,0:km,:))),maxval(abs(tmp(0:im,0,0:km)-1.d0)))
    error=max(error,maxval(abs(q(-hm:-1,3,0:km,5)-(energy+0.001d0))))
    error=max(error,maxval(abs(prs(-hm:-1,3,0:km)-prs(0,3,0))))
    do k=0,km
    do i=0,im
      if(x(i,jm,k,1)<=conservative_boundary%split_x) then
        error=max(error,maxval(abs(q(i,jm,k,:)-conservative_boundary%q_left)))
      else
        error=max(error,maxval(abs(q(i,jm,k,:)-conservative_boundary%q_right)))
      endif
    enddo
    enddo
    if(.not.all(ieee_is_finite(q))) error stop 'Nonfinite staged boundary state'
    if(error>1.d-12) error stop 'Boundary staging or primitive refresh failed'
    print*,'BOUNDARY_STAGE_PASS max_abs=',error
#ifdef _CUDA
    allocate(gpu_q(-hm:im+hm,-hm:jm+hm,-hm:km+hm,5))
    allocate(gpu_scalar(-hm:im+hm,-hm:jm+hm,-hm:km+hm))
    allocate(gpu_vel(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3))
    gpu_q=q_d
    if(.not.all(ieee_is_finite(gpu_q))) error stop 'Nonfinite GPU boundary state'
    error=maxval(abs(gpu_q-q))
    gpu_scalar=rho_d
    error=max(error,maxval(abs(gpu_scalar(-hm:im+hm,0:jm,0:km)-rho(-hm:im+hm,0:jm,0:km))))
    gpu_scalar=prs_d
    error=max(error,maxval(abs(gpu_scalar(-hm:im+hm,0:jm,0:km)-prs(-hm:im+hm,0:jm,0:km))))
    gpu_scalar=tmp_d
    error=max(error,maxval(abs(gpu_scalar(0:im,-hm:jm+hm,0:km)-tmp(0:im,-hm:jm+hm,0:km))))
    gpu_vel=vel_d
    error=max(error,maxval(abs(gpu_vel(0:im,-hm:jm+hm,0:km,:)-vel(0:im,-hm:jm+hm,0:km,:))))
    if(error>1.d-12) error stop 'GPU staged boundary mismatch'
    print*,'BOUNDARY_STAGE_GPU_PASS max_abs=',error
#endif
  end subroutine check_boundary_stage

  subroutine check_nscbc_characteristic_policy()
    use bc, only: nscbc_farfield_balance_incoming_lodi,              &
                  nscbc_remove_incoming_source
    implicit none
    real(8) :: metric(3),normal(3),lodi0(5),source(5),pinv(5,5)
    integer :: m

    metric=[0.3d0,0.9d0,-0.2d0]
    normal=metric/sqrt(sum(metric*metric))
    lodi0=[1.d0,2.d0,3.d0,4.d0,5.d0]
    source=[0.25d0,-0.5d0,0.75d0,-1.d0,1.25d0]
    pinv=0.d0
    do m=1,5
      pinv(m,m)=1.d0
    enddo
    call run_case('subsonic_outflow',metric, 0.2d0*normal,1.d0,lodi0,source,pinv,[0,0,0,0,1])
    call run_case('subsonic_inflow', metric,-0.2d0*normal,1.d0,lodi0,source,pinv,[1,1,1,0,1])
    call run_case('supersonic_outflow',metric,2.d0*normal,1.d0,lodi0,source,pinv,[0,0,0,0,0])
    call run_case('supersonic_inflow', metric,-2.d0*normal,1.d0,lodi0,source,pinv,[1,1,1,1,1])
    call run_case('roundoff_outflow',metric,1.d-16*normal,1.d0,lodi0,source,pinv,[0,0,0,0,1])
    call run_case('roundoff_inflow',metric,-1.d-16*normal,1.d0,lodi0,source,pinv,[0,0,0,0,1])
    print*,'NSCBC_CPU_POLICY_PASS'
    call check_viscous_source_projection(metric,normal,source)

  contains

    subroutine run_case(name,case_metric,case_velocity,css,lodi_initial,source,pinv,expected_mask)
      character(len=*),intent(in) :: name
      real(8),intent(in) :: case_metric(3),case_velocity(3),css,lodi_initial(5)
      real(8),intent(in) :: source(5),pinv(5,5)
      integer,intent(in) :: expected_mask(5)
      real(8) :: lodi(5),lambda(5),source_characteristic(5),total_characteristic(5)
      logical :: incoming(5)
      integer :: mask(5),m

      lodi=lodi_initial
      call nscbc_farfield_balance_incoming_lodi(lodi,source,pinv,2.d0,       &
                                                case_metric,case_velocity,   &
                                                css,lambda,incoming)
      mask=merge(1,0,incoming)
      if(any(mask/=expected_mask)) error stop 'NSCBC CPU incoming mask mismatch'
      source_characteristic=matmul(pinv,source)/2.d0
      total_characteristic=lodi+source_characteristic
      do m=1,5
        if(incoming(m)) then
          if(abs(total_characteristic(m))>1.d-15) &
            error stop 'NSCBC CPU total incoming characteristic was not zeroed'
        elseif(lodi(m)/=lodi_initial(m)) then
          error stop 'NSCBC CPU outgoing wave was modified'
        endif
      enddo
      write(*,'(A,1X,A,1X,5(ES25.16,1X),5(I1,1X),5(ES25.16,1X))') &
        'NSCBC_CPU_POLICY',trim(name),lambda,mask,lodi
    end subroutine run_case

    subroutine check_viscous_source_projection(metric,normal,source)
      real(8),intent(in) :: metric(3),normal(3),source(5)
      real(8) :: pnor(5,5),pinv(5,5),rhs0(5),velocity(3)
      integer :: icase,m
      character(len=20) :: labels(6)

      pnor=0.d0
      pinv=0.d0
      do m=1,5
        pnor(m,m)=1.d0
        pinv(m,m)=1.d0
      enddo
      pnor(1,2)=0.2d0
      pinv(1,2)=-0.2d0
      pnor(4,5)=0.3d0
      pinv(4,5)=-0.3d0
      rhs0=[3.d0,-2.d0,1.d0,0.5d0,-0.25d0]
      labels=[character(len=20) :: 'subsonic_outflow','subsonic_inflow', &
                                   'supersonic_outflow','supersonic_inflow', &
                                   'roundoff_outflow','roundoff_inflow']
      do icase=1,6
        select case(icase)
        case(1)
          velocity=0.2d0*normal
        case(2)
          velocity=-0.2d0*normal
        case(3)
          velocity=2.d0*normal
        case(4)
          velocity=-2.d0*normal
        case(5)
          velocity=1.d-16*normal
        case default
          velocity=-1.d-16*normal
        end select
        call run_viscous_case(trim(labels(icase)),metric,velocity,source, &
                              pnor,pinv,rhs0)
      enddo
      print*,'NSCBC_CPU_VISCOUS_PASS'
    end subroutine check_viscous_source_projection

    subroutine run_viscous_case(name,metric,velocity,source,pnor,pinv,rhs_initial)
      character(len=*),intent(in) :: name
      real(8),intent(in) :: metric(3),velocity(3),source(5),pnor(5,5), &
                            pinv(5,5),rhs_initial(5)
      real(8) :: rhs(5),expected(5),lambda(5),source_characteristic(5)
      logical :: incoming(5)
      integer :: mask(5),m

      rhs=rhs_initial
      call nscbc_remove_incoming_source(rhs,source,pnor,pinv,2.d0,metric, &
                                        velocity,1.d0,lambda,incoming)
      source_characteristic=matmul(pinv,source)/2.d0
      do m=1,5
        if(.not.incoming(m)) source_characteristic(m)=0.d0
      enddo
      expected=rhs_initial-2.d0*matmul(pnor,source_characteristic)
      if(maxval(abs(rhs-expected))>1.d-15) &
        error stop 'NSCBC CPU viscous source sign or projection mismatch'
      mask=merge(1,0,incoming)
      write(*,'(A,1X,A,1X,5(ES25.16,1X),5(I1,1X),5(ES25.16,1X))') &
        'NSCBC_CPU_VISCOUS',trim(name),lambda,mask,rhs
    end subroutine run_viscous_case
  end subroutine check_nscbc_characteristic_policy

  subroutine check_boundary_rhs
    use commvar
    use commarray, only: q,qrhs,rho,vel,prs,tmp,spc,dxi,jacob,crinod
    use solver, only: convrsduwd
    use parallel, only: mpisize,mpirank
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    integer :: i,j,k,d,m
    real(8) :: slope(3),speed(3),pressure_factor,energy_factor,expected(5),flux_factor(5),error
    if(mpisize/=1) error stop 'Boundary RHS manufactured probe requires NP=1'
    im=8; jm=8; km=8
    numq=5; num_species=0; num_modequ=0; ndims=3
    nondimen=.true.; lchardecomp=.false.; recon_schem=3
    conschm='543e'; gamma=1.4d0; mach=2.d0; bfacmpld=0.3d0
    ltimrpt=.false.; lreport=.false.
    npdci=4; npdcj=4; npdck=3
    is=1; ie=im-1; js=1; je=jm-1; ks=0; ke=km
    allocate(q(-hm:im+hm,-hm:jm+hm,-hm:km+hm,5),qrhs(0:im,0:jm,0:km,5))
    allocate(rho(-hm:im+hm,-hm:jm+hm,-hm:km+hm),prs(-hm:im+hm,-hm:jm+hm,-hm:km+hm))
    allocate(tmp(-hm:im+hm,-hm:jm+hm,-hm:km+hm),vel(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3))
    allocate(spc(-hm:im+hm,-hm:jm+hm,-hm:km+hm,0),crinod(-hm:im+hm,-hm:jm+hm,-hm:km+hm))
    allocate(dxi(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3,3),jacob(-hm:im+hm,-hm:jm+hm,-hm:km+hm))
    slope=[0.001d0,0.002d0,0.003d0]
    speed=[0.2d0,0.1d0,0.05d0]
    pressure_factor=1.d0/(gamma*mach**2)
    energy_factor=pressure_factor/(gamma-1.d0)+0.5d0*sum(speed**2)
    expected=0.d0
    do d=1,3
      flux_factor=[speed(d),speed(d)*speed,speed(d)*(energy_factor+pressure_factor)]
      flux_factor(d+1)=flux_factor(d+1)+pressure_factor
      expected=expected+slope(d)*flux_factor
    enddo
    do k=-hm,km+hm
    do j=-hm,jm+hm
    do i=-hm,im+hm
      rho(i,j,k)=1.d0+slope(1)*i+slope(2)*j+slope(3)*k
      vel(i,j,k,:)=speed
      prs(i,j,k)=pressure_factor*rho(i,j,k)
      tmp(i,j,k)=1.d0
      q(i,j,k,:)=[rho(i,j,k),rho(i,j,k)*speed,rho(i,j,k)*energy_factor]
    enddo
    enddo
    enddo
    jacob=1.d0; dxi=0.d0; crinod=.false.
    do d=1,3
      dxi(:,:,:,d,d)=1.d0
    enddo
    qrhs=0.d0
    call convrsduwd()
    if(.not.all(ieee_is_finite(qrhs))) error stop 'Nonfinite legacy RHS'
    if(any(qrhs(0,:,:,:)/=0.d0).or.any(qrhs(:,0,:,:)/=0.d0)) &
      error stop 'Legacy physical boundary RHS changed'
    qrhs=0.d0
    call convrsduwd(physical_halo_rhs=.true.)
    if(.not.all(ieee_is_finite(qrhs))) error stop 'Nonfinite physical-space RHS'
    error=0.d0
    do m=1,5
      error=max(error,maxval(abs(qrhs(:,:,:,m)-expected(m))))
    enddo
    if(error>1.d-10) error stop 'Full-halo boundary RHS manufactured solution failed'
    if(is/=1.or.ie/=im-1.or.js/=1.or.je/=jm-1.or.npdci/=4.or.npdcj/=4) &
      error stop 'Boundary RHS changed global range or geometry policy'
    if(mpirank==0) print*,'BOUNDARY_RHS_MANUFACTURED_PASS max_abs=',error
    lchardecomp=.true.
    qrhs=0.d0
    call convrsduwd(physical_halo_rhs=.true.)
    if(.not.all(ieee_is_finite(qrhs))) error stop 'Nonfinite characteristic RHS'
    error=0.d0
    do m=1,5
      error=max(error,maxval(abs(qrhs(:,:,:,m)-expected(m))))
    enddo
    if(error>1.d-10) error stop 'Characteristic full-halo boundary RHS failed'
    print*,'BOUNDARY_RHS_ROE_PASS max_abs=',error
#ifdef _CUDA
    call check_boundary_rhs_gpu(expected)
#endif
    call check_boundary_diffusion_rhs()
    call check_stretched_metric_rhs()
    deallocate(q,qrhs,rho,vel,prs,tmp,spc,dxi,jacob,crinod)
  end subroutine check_boundary_rhs

  subroutine check_stretched_metric_rhs()
    use commvar, only: im,jm,km,hm,gamma,mach,lchardecomp,diffterm,flowtype,limmbou
    use commarray, only: q,qrhs,rho,vel,prs,tmp,dxi,jacob
    use solver, only: convrsduwd,rhscal
    use rectilinear_metric_halo, only: fill_rectilinear_metric_halo
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    integer :: i,j,k,d,status,mode
    real(8) :: spacing(3),speed(3),pressure,error
    real(8) :: fixed_rhs(0:im,0:jm,0:km,5),fixed_error(2)
    logical :: owned(6),failed
    owned=[.true.,.true.,.true.,.true.,.false.,.false.]
    speed=[0.2d0,0.1d0,0.05d0]; pressure=1.d0/(gamma*mach**2)
    rho=1.d0; prs=pressure; tmp=1.d0
    do d=1,3
      vel(:,:,:,d)=speed(d)
      q(:,:,:,d+1)=speed(d)
    enddo
    q(:,:,:,1)=1.d0
    q(:,:,:,5)=pressure/(gamma-1.d0)+0.5d0*sum(speed**2)
    jacob=0.d0; dxi=0.d0
    ! z-periodic halos are initialized here; owned x/y halos are filled below.
    do k=-hm,km+hm
    do j=0,jm
    do i=0,im
      spacing=[1.d0+0.01d0*i,1.d0+0.02d0*j,1.d0]
      jacob(i,j,k)=product(spacing)
      do d=1,3
        dxi(i,j,k,d,d)=1.d0/spacing(d)
      enddo
    enddo
    enddo
    enddo
    call fill_rectilinear_metric_halo(im,jm,km,hm,owned,jacob,dxi,status)
    if(status/=0) error stop 'Stretched metric halo preparation failed'
    failed=.false.
    do mode=0,1
      lchardecomp=mode==1
      qrhs=0.d0
      call convrsduwd(physical_halo_rhs=.true.)
      if(.not.all(ieee_is_finite(qrhs))) error stop 'Nonfinite fixed-epsilon control'
      fixed_rhs=qrhs
      fixed_error(mode+1)=maxval(abs(qrhs))
      if(fixed_error(mode+1)<1.d-8) error stop 'Fixed-epsilon failing control unexpectedly changed'
      qrhs=0.d0
      call convrsduwd(physical_halo_rhs=.true.,metric_consistent_eps=.false.)
      if(any(qrhs/=fixed_rhs)) error stop 'Explicitly disabled epsilon differs from default'
      qrhs=0.d0
      call convrsduwd(physical_halo_rhs=.true.,metric_consistent_eps=.true.)
      if(.not.all(ieee_is_finite(qrhs))) error stop 'Nonfinite stretched-metric free stream RHS'
      error=maxval(abs(qrhs))
      if(error>1.d-10) then
        print*,'STRETCHED_RHS_FAILURE mode/max_abs/location=',mode,error,maxloc(abs(qrhs))
        failed=.true.
      else
        if(mode==0) print*,'BOUNDARY_RHS_STRETCHED_PHYSICAL_PASS max_abs=',error
        if(mode==1) print*,'BOUNDARY_RHS_STRETCHED_ROE_PASS max_abs=',error
      endif
    enddo
    if(failed) error stop 'Metric halos destroy free-stream preservation'
    print*,'BOUNDARY_RHS_FIXED_EPS_CONTROL_PASS max_abs=',0.d0
    print*,'BOUNDARY_RHS_FIXED_EPS_RESIDUALS=',fixed_error
    lchardecomp=.false.; diffterm=.false.; flowtype='bl'; limmbou=.false.
    qrhs=0.d0
    call rhscal(physical_halo_rhs=.true.,metric_consistent_eps=.true.)
    if(.not.all(ieee_is_finite(qrhs))) error stop 'Nonfinite staged rhscal free stream'
    error=maxval(abs(qrhs))
    if(error>1.d-10) error stop 'rhscal did not forward metric epsilon'
    print*,'BOUNDARY_RHSCAL_METRIC_PASS max_abs=',error
#ifdef _CUDA
    call check_boundary_rhs_gpu([0.d0,0.d0,0.d0,0.d0,0.d0],stretched=.true.)
    call check_metric_nonuniform_rhs()
#endif
  end subroutine check_stretched_metric_rhs

  subroutine check_boundary_diffusion_rhs()
    use commvar
    use commarray, only: qrhs,vel,tmp,dvel,dtmp,vor
    use parallel, only: isize,jsize,ksize,mpileft,mpiright,mpidown,mpiup,mpiback,mpifront
    use mpi, only: MPI_PROC_NULL
    use derivative, only: fd_scheme_initiate,fds,fds_compact_i,fds_compact_j,fds_compact_k,explicit_central
    use solver, only: diffrsdcal6,convrsduwd,rhscal
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    real(8) :: a(3),coord(3),du(3),stress(3),expected(5),error,mu,continuous_error,cubic
    integer :: i,j,k,d,index(3),dims(3)
    integer :: mode
    real(8) :: combined(0:im,0:jm,0:km,5),default_rhs(0:im,0:jm,0:km,5)
    isize=1; jsize=1; ksize=1
    mpileft=MPI_PROC_NULL; mpiright=MPI_PROC_NULL
    mpidown=MPI_PROC_NULL; mpiup=MPI_PROC_NULL
    mpiback=MPI_PROC_NULL; mpifront=MPI_PROC_NULL
    lihomo=.false.; ljhomo=.false.; lkhomo=.false.
    npdci=4; npdcj=4; npdck=4
    is=1; ie=im-1; js=1; je=jm-1; ks=1; ke=km-1
    difschm='643e'; turbmode='none'; reynolds=950.d0; prandtl=0.72d0
    const5=(gamma-1.d0)*mach**2
    tempconst=110.4d0/288.d0; tempconst1=1.d0+tempconst
    call fd_scheme_initiate(fds_compact_i,difschm,npdci,im,1)
    call fd_scheme_initiate(fds_compact_j,difschm,npdcj,jm,2)
    call fd_scheme_initiate(fds_compact_k,difschm,npdck,km,3)
    allocate(explicit_central :: fds)
    allocate(dvel(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3,3))
    allocate(dtmp(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3),vor(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3))
    a=[0.001d0,0.002d0,0.003d0]; mu=1.d0/reynolds
    tmp=1.d0; dtmp=0.d0; dvel=0.d0
    do k=-hm,km+hm
    do j=-hm,jm+hm
    do i=-hm,im+hm
      coord=real([i,j,k],8)
      vel(i,j,k,:)=0.1d0+a*coord**2
      do d=1,3
        dvel(i,j,k,d,d)=2.d0*a(d)*coord(d)
      enddo
    enddo
    enddo
    enddo
    qrhs=0.d0
    call diffrsdcal6()
    if(.not.all(ieee_is_finite(qrhs))) error stop 'Nonfinite legacy diffusion RHS'
    if(any(qrhs(0,0,0,:)/=0.d0)) error stop 'Legacy diffusion corner changed'
    qrhs=0.d0
    call diffrsdcal6(physical_boundary_rhs=.true.)
    if(.not.all(ieee_is_finite(qrhs))) error stop 'Nonfinite full-face diffusion RHS'
    error=0.d0; continuous_error=0.d0
    dims=[im,jm,km]
    do k=0,km
    do j=0,jm
    do i=0,im
      coord=real([i,j,k],8)
      du=2.d0*a*coord
      stress=2.d0*mu*(du-sum(du)/3.d0)
      expected(1)=0.d0
      expected(2:4)=8.d0*mu*a/3.d0
      expected(5)=sum(stress*du+vel(i,j,k,:)*expected(2:4))
      continuous_error=max(continuous_error,maxval(abs(qrhs(i,j,k,:)-expected)))
      ! The second-order end/adjacent stencils have exact cubic errors -2c/+c.
      index=[i,j,k]
      do d=1,3
        cubic=8.d0*mu*a(d)**2/3.d0
        if(index(d)==0.or.index(d)==dims(d)) expected(5)=expected(5)-2.d0*cubic
        if(index(d)==1.or.index(d)==dims(d)-1) expected(5)=expected(5)+cubic
      enddo
      error=max(error,maxval(abs(qrhs(i,j,k,:)-expected)))
    enddo
    enddo
    enddo
    if(error>1.d-10) error stop 'Full-face diffusion manufactured solution failed'
    if(is/=1.or.js/=1.or.ks/=1.or.npdci/=4.or.npdcj/=4.or.npdck/=4) &
      error stop 'Diffusion RHS changed global closure policy'
    print*,'BOUNDARY_RHS_DIFFUSION_PASS max_abs=',error
    print*,'BOUNDARY_RHS_DIFFUSION_CONTINUOUS_ERROR=',continuous_error
    ! This checks orchestration against the same direct operators, not a new
    ! physical manufactured solution (the stored q and primitive probes differ).
    flowtype='bl'; limmbou=.false.; lchardecomp=.false.; diffterm=.true.
    error=0.d0
    do mode=0,1
      qrhs=0.d0
      call convrsduwd(physical_halo_rhs=mode==1,metric_consistent_eps=mode==1)
      qrhs=-qrhs
      call diffrsdcal6(physical_boundary_rhs=mode==1)
      combined=qrhs
      qrhs=0.d0
      call rhscal(physical_halo_rhs=mode==1,metric_consistent_eps=mode==1)
      if(.not.all(ieee_is_finite(qrhs))) error stop 'Nonfinite combined rhscal'
      error=max(error,maxval(abs(qrhs-combined)))
      if(mode==0) default_rhs=qrhs
    enddo
    qrhs=0.d0
    call rhscal()
    if(.not.all(ieee_is_finite(qrhs))) error stop 'Nonfinite default rhscal'
    error=max(error,maxval(abs(qrhs-default_rhs)))
    if(error>1.d-10) error stop 'rhscal operator composition mismatch'
    print*,'BOUNDARY_RHSCAL_COMPOSITION_PASS max_abs=',error
#ifdef _CUDA
    call check_boundary_diffusion_gpu()
#endif
    deallocate(dvel,dtmp,vor,fds)
  end subroutine check_boundary_diffusion_rhs
#ifdef _CUDA
  subroutine check_metric_nonuniform_rhs()
    use commvar, only: im,jm,km,hm,gamma,mach,lchardecomp
    use commarray, only: q,qrhs,rho,vel,prs,tmp
    use solver, only: convrsduwd
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    real(8) :: reference(0:im,0:jm,0:km,5,2),speed(3)
    integer :: i,j,k,mode,trial
    character(len=6),parameter :: names(3)=[character(len=6)::'SMOOTH','JUMP_X','JUMP_Y']
    do trial=1,3
      do k=-hm,km+hm
      do j=-hm,jm+hm
      do i=-hm,im+hm
        if(trial==1) then
          rho(i,j,k)=1.d0+0.1d0*sin(0.3d0*i)*cos(0.2d0*j)
          prs(i,j,k)=0.2d0+0.02d0*cos(0.2d0*i+0.3d0*j)
          speed=[0.2d0+0.03d0*sin(0.2d0*j),0.1d0+0.01d0*cos(0.3d0*i),0.05d0]
        else
          rho(i,j,k)=1.d0; prs(i,j,k)=1.d0; speed=0.d0
          if((trial==2.and.i>=im/2).or.(trial==3.and.j>=jm/2)) then
            rho(i,j,k)=0.125d0; prs(i,j,k)=0.1d0
          endif
        endif
        vel(i,j,k,:)=speed
        tmp(i,j,k)=gamma*mach**2*prs(i,j,k)/rho(i,j,k)
        q(i,j,k,:)=[rho(i,j,k),rho(i,j,k)*speed, &
          prs(i,j,k)/(gamma-1.d0)+0.5d0*rho(i,j,k)*sum(speed**2)]
      enddo
      enddo
      enddo
      do mode=1,2
        lchardecomp=mode==2; qrhs=0.d0
        call convrsduwd(physical_halo_rhs=.true.,metric_consistent_eps=.true.)
        if(.not.all(ieee_is_finite(qrhs))) error stop 'Nonfinite nonuniform CPU RHS'
        reference(:,:,:,:,mode)=qrhs
      enddo
      if(maxval(abs(reference))<1.d-6) error stop 'Nonuniform RHS probe is vacuous'
      call check_boundary_rhs_gpu([0.d0,0.d0,0.d0,0.d0,0.d0],stretched=.true., &
        field_reference=reference,case_name=names(trial))
    enddo
  end subroutine check_metric_nonuniform_rhs

  subroutine check_boundary_diffusion_gpu()
    use cudafor
    use commvar, only: im,jm,km,hm,reynolds,prandtl,const5,tempconst,tempconst1
    use commarray, only: vel,tmp,qrhs
    use commarray_gpu, only: vel_d,tmp_d,qrhs_d,sigma_d,qflux_d
    use solver_gpu, only: diffusion_flux_xyphysical_global_kernel, &
         diffusion_rhs_x_xyphysical_stored_global_kernel,diffusion_rhs_y_xyphysical_stored_global_kernel, &
         diffusion_rhs_z_xyphysical_stored_global_kernel
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    type(dim3) :: bx,by,bz,gx,gy,gz
    real(8) :: a(3),coord(3),du(3),stress(3),expected(5),mu,cubic,error
    real(8),allocatable :: shalo(:,:,:,:),qhalo(:,:,:,:)
    integer :: i,j,k,d,index(3),dims(3)
    ! x/y quadratic velocities, z-uniform state on the periodic thin extrusion.
    vel(:,:,:,3)=0.1d0
    vel_d=vel; tmp_d=tmp; qrhs_d=0.d0
    bx=dim3(512,1,1); by=dim3(32,16,1); bz=dim3(64,1,8)
    gx=dim3((im+512)/512,jm+1,km+1)
    gy=dim3((im+32)/32,(jm+16)/16,km+1)
    gz=dim3((im+64)/64,jm+1,(km+8)/8)
    call diffusion_flux_xyphysical_global_kernel<<<gx,bx>>>(im,jm,km,reynolds,prandtl,const5, &
         tempconst,tempconst1,1,1,1,1)
    call checked_sync()
    ! The stored-flux kernels require real halo storage even for NP1.
    allocate(shalo(0:im,0:jm,1:hm,6),qhalo(0:im,0:jm,1:hm,3))
    shalo=sigma_d(0:im,0:jm,km-hm:km-1,:)
    sigma_d(0:im,0:jm,-hm:-1,:)=shalo
    shalo=sigma_d(0:im,0:jm,1:hm,:)
    sigma_d(0:im,0:jm,km+1:km+hm,:)=shalo
    qhalo=qflux_d(0:im,0:jm,km-hm:km-1,:)
    qflux_d(0:im,0:jm,-hm:-1,:)=qhalo
    qhalo=qflux_d(0:im,0:jm,1:hm,:)
    qflux_d(0:im,0:jm,km+1:km+hm,:)=qhalo
    deallocate(shalo,qhalo)
    call checked_sync()
    call diffusion_rhs_x_xyphysical_stored_global_kernel<<<gx,bx>>>(im,jm,km,0,im,0,jm,0,km,1,1)
    call checked_sync()
    call diffusion_rhs_y_xyphysical_stored_global_kernel<<<gy,by>>>(im,jm,km,0,im,0,jm,0,km,1,1)
    call checked_sync()
    call diffusion_rhs_z_xyphysical_stored_global_kernel<<<gz,bz>>>(im,jm,km,0,im,0,jm,0,km)
    call checked_sync()
    qrhs=qrhs_d
    if(.not.all(ieee_is_finite(qrhs))) error stop 'Nonfinite GPU viscous RHS'
    a=[0.001d0,0.002d0,0.d0]; dims=[im,jm,km]; mu=1.d0/reynolds; error=0.d0
    do k=0,km
    do j=0,jm
    do i=0,im
      coord=real([i,j,k],8); index=[i,j,k]; du=2.d0*a*coord
      stress=2.d0*mu*(du-sum(du)/3.d0)
      expected(1)=0.d0; expected(2:4)=8.d0*mu*a/3.d0
      expected(5)=sum(stress*du+vel(i,j,k,:)*expected(2:4))
      do d=1,2
        cubic=8.d0*mu*a(d)**2/3.d0
        if(index(d)==0.or.index(d)==dims(d)) expected(5)=expected(5)-2.d0*cubic
        if(index(d)==1.or.index(d)==dims(d)-1) expected(5)=expected(5)+cubic
      enddo
      error=max(error,maxval(abs(qrhs(i,j,k,:)-expected)))
    enddo
    enddo
    enddo
    if(error>1.d-10) error stop 'GPU full-face diffusion manufactured solution failed'
    print*,'BOUNDARY_RHS_GPU_DIFFUSION_PASS max_abs=',error
  end subroutine check_boundary_diffusion_gpu

  subroutine check_boundary_rhs_gpu(expected,stretched,field_reference,case_name)
    use cudafor
    use commvar, only: im,jm,km,hm,gamma
    use commarray, only: q,qrhs,rho,vel,prs,tmp,dxi,jacob
    use commarray_gpu
    use commvar_gpu, only: copy_commvar_to_gpu,metric_consistent_eps_d
    use solver_gpu
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    real(8),intent(in) :: expected(5)
    logical,intent(in),optional :: stretched
    real(8),intent(in),optional :: field_reference(0:im,0:jm,0:km,5,2)
    character(len=*),intent(in),optional :: case_name
    logical :: metric_test
    type(dim3) :: bx,by,bz,gx,gy,gz,gxi,gyi,gzi
    integer :: ivar,sgn,m,ierr
    real(8) :: error
    metric_test=.false.
    if(present(stretched)) metric_test=stretched
    ierr=cudaSetDevice(0)
    if(ierr/=cudaSuccess) error stop 'Manufactured RHS cannot select GPU'
    call alloc_gpu_arrays()
    call copy_commvar_to_gpu()
    metric_consistent_eps_d=metric_test
    q_d=q; rho_d=rho; vel_d=vel; prs_d=prs; tmp_d=tmp; dxi_d=dxi; jacob_d=jacob
    bx=dim3(512,1,1); by=dim3(32,16,1); bz=dim3(64,1,8)
    gx=dim3((im+512)/512,jm+1,km+1)
    gy=dim3((im+32)/32,(jm+16)/16,km+1)
    gz=dim3((im+64)/64,jm+1,(km+8)/8)
    gxi=dim3((im+513)/512,jm+1,km+1)
    gyi=dim3((im+32)/32,(jm+17)/16,km+1)
    gzi=dim3((im+64)/64,jm+1,(km+9)/8)
    qrhs_d=0.d0
    do ivar=1,5
      do sgn=1,-1,-2
        call explicit_upwind_flux_x_xyphysical_global_kernel<<<gxi,bx>>>( &
             im,jm,km,hm,0,im,0,jm,0,km,3,ivar,sgn,3,gamma)
        call checked_sync()
      enddo
      call explicit_upwind_rhs_x_xyphysical_global_kernel<<<gx,bx>>>(im,jm,km,0,im,0,jm,0,km,ivar)
      call checked_sync()
      do sgn=1,-1,-2
        call explicit_upwind_flux_y_physical_global_kernel<<<gyi,by>>>( &
             im,jm,km,hm,0,im,0,jm,0,km,3,ivar,sgn,3,gamma)
        call checked_sync()
      enddo
      call explicit_upwind_rhs_y_physical_global_kernel<<<gy,by>>>(im,jm,km,0,im,0,jm,0,km,ivar)
      call checked_sync()
      do sgn=1,-1,-2
        call explicit_upwind_flux_z_xyphysical_global_kernel<<<gzi,bz>>>( &
             im,jm,km,0,im,0,jm,0,km,ivar,sgn,3,gamma)
        call checked_sync()
      enddo
      call explicit_upwind_rhs_z_xyphysical_global_kernel<<<gz,bz>>>(im,jm,km,0,im,0,jm,0,km,ivar)
      call checked_sync()
    enddo
    qrhs=qrhs_d
    if(.not.all(ieee_is_finite(qrhs))) error stop 'Nonfinite GPU physical-space RHS'
    error=0.d0
    do m=1,5
      error=max(error,maxval(abs(qrhs(:,:,:,m)+expected(m))))
    enddo
    if(present(field_reference)) error=maxval(abs(qrhs+field_reference(:,:,:,:,1)))
    if(error>1.d-10) error stop 'GPU physical-space full-halo RHS failed'
    if(present(field_reference)) then
      if(.not.present(case_name)) error stop 'Missing nonuniform probe case name'
      print '(A,ES25.16)','BOUNDARY_RHS_'//trim(case_name)//'_PHYSICAL_PASS max_abs=',error
    elseif(metric_test) then
      print*,'BOUNDARY_RHS_GPU_STRETCHED_PHYSICAL_PASS max_abs=',error
    else
      print*,'BOUNDARY_RHS_GPU_PHYSICAL_PASS max_abs=',error
    endif
    qrhs_d=0.d0
    shock_mask_d=1_1
    call characteristic_upwind_flux_x_xyphysical_global_kernel<<<gxi,bx>>>( &
         im,jm,km,hm,0,im,0,jm,0,km,3,3,gamma)
    call checked_sync()
    call characteristic_upwind_rhs_x_xyphysical_global_kernel<<<gx,bx>>>(im,jm,km,0,im,0,jm,0,km)
    call checked_sync()
    call characteristic_upwind_flux_y_xyphysical_global_kernel<<<gyi,by>>>( &
         im,jm,km,hm,0,im,0,jm,0,km,3,3,gamma)
    call checked_sync()
    call characteristic_upwind_rhs_y_xyphysical_global_kernel<<<gy,by>>>(im,jm,km,0,im,0,jm,0,km)
    call checked_sync()
    call characteristic_upwind_flux_z_xyphysical_global_kernel<<<gzi,bz>>>( &
         im,jm,km,0,im,0,jm,0,km,3,gamma)
    call checked_sync()
    call characteristic_upwind_rhs_z_xyphysical_global_kernel<<<gz,bz>>>(im,jm,km,0,im,0,jm,0,km)
    call checked_sync()
    qrhs=qrhs_d
    if(.not.all(ieee_is_finite(qrhs))) error stop 'Nonfinite GPU characteristic RHS'
    error=0.d0
    do m=1,5
      error=max(error,maxval(abs(qrhs(:,:,:,m)+expected(m))))
    enddo
    if(present(field_reference)) error=maxval(abs(qrhs+field_reference(:,:,:,:,2)))
    if(error>1.d-10) error stop 'GPU characteristic full-halo RHS failed'
    if(present(field_reference)) then
      print '(A,ES25.16)','BOUNDARY_RHS_'//trim(case_name)//'_ROE_PASS max_abs=',error
    elseif(metric_test) then
      print*,'BOUNDARY_RHS_GPU_STRETCHED_ROE_PASS max_abs=',error
    else
      print*,'BOUNDARY_RHS_GPU_ROE_PASS max_abs=',error
    endif
    metric_consistent_eps_d=.false.
  end subroutine check_boundary_rhs_gpu

  subroutine checked_sync()
    use cudafor
    integer :: ierr
    ierr=cudaDeviceSynchronize()
    if(ierr/=cudaSuccess) error stop 'Boundary RHS CUDA kernel failed'
  end subroutine checked_sync
#endif
end module boundary_rhs_manufactured
