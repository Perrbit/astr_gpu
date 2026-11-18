program nscbc_top_halo_phase_probe
  use iso_c_binding, only: c_int64_t
  use mpi
  use commvar
  use commarray, only: q,rho,vel,prs,tmp,spc,jacob,dxi,x,qrhs
  use bc, only: bctype,farfield_nscbc,nscbc_farfield_configured,nscbc_farfield_policy, &
    nscbc_farfield_rho_target,nscbc_farfield_u_target,nscbc_farfield_v_target, &
    nscbc_farfield_w_target,nscbc_farfield_t_target,nscbc_farfield_sbli_shock
  use parallel, only: mpirank,mpisize,lio,irk,irkm,jrk,jrkm,krk,krkm,isize,jsize,ksize, &
    mpileft,mpiright,mpidown,mpiup,mpiback,mpifront,mpitag,qswap
  use filter, only: filter_coefficient_explicit
  use fludyna, only: q2fvar
  implicit none
  integer :: ierr,i,j,k,c,unit,changed
  real(8) :: xx,zz,rr,tt,uu,vv,ww,pp,theta,cache_error
  real(8),allocatable :: original_rhs(:,:,:,:),refreshed_rhs(:,:,:,:),before(:,:,:,:),cache(:,:,:,:)
  character(1024) :: directory,path
  character(64) :: suffix
  call MPI_Init(ierr);call MPI_Comm_rank(MPI_COMM_WORLD,mpirank,ierr)
  call MPI_Comm_size(MPI_COMM_WORLD,mpisize,ierr)
  if(mpisize/=1.and.mpisize/=2) error stop 'top halo phase probe uses NP=1/2 z'
  call get_command_argument(1,directory)
  ia=16;ja=16;ka=16;im=ia;jm=ja;km=ka/mpisize;ndims=3
  irk=0;irkm=0;jrk=0;jrkm=0;krk=mpirank;krkm=mpisize-1
  isize=1;jsize=1;ksize=mpisize;lio=mpirank==0;mpitag=100
  is=0;ie=im;js=1;je=jm-1;ks=0;ke=km
  lihomo=.true.;ljhomo=.false.;lkhomo=.true.;npdci=3;npdcj=4;npdck=3
  mpileft=MPI_PROC_NULL;mpiright=MPI_PROC_NULL;mpidown=MPI_PROC_NULL;mpiup=MPI_PROC_NULL
  mpiback=MPI_PROC_NULL;mpifront=MPI_PROC_NULL
  if(mpisize>1) then
    mpiback=modulo(mpirank-1,mpisize);mpifront=modulo(mpirank+1,mpisize)
  endif
  numq=5;num_species=0;num_modequ=0;flowtype='tgv';turbmode='none';nondimen=.true.;lcomb=.false.
  conschm='643e';difschm='643e';gamma=1.4d0;mach=1;const2=gamma;const6=1.d0/(gamma-1)
  roinf=1;uinf=.4d0;vinf=.02d0;winf=.01d0;pinf=1.d0/const2;ymin=0;ymax=16
  bctype=[1,1,50,52,1,1];lfilter=.false.;diffterm=.false.
  nscbc_farfield_configured=.true.;nscbc_farfield_policy=0;nscbc_farfield_sbli_shock=.false.
  nscbc_farfield_rho_target=roinf;nscbc_farfield_u_target=uinf;nscbc_farfield_v_target=vinf
  nscbc_farfield_w_target=winf;nscbc_farfield_t_target=1
  allocate(q(-hm:im+hm,-hm:jm+hm,-hm:km+hm,5),rho(-hm:im+hm,-hm:jm+hm,-hm:km+hm), &
    vel(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3),prs(-hm:im+hm,-hm:jm+hm,-hm:km+hm), &
    tmp(-hm:im+hm,-hm:jm+hm,-hm:km+hm),spc(-hm:im+hm,-hm:jm+hm,-hm:km+hm,0), &
    jacob(-hm:im+hm,-hm:jm+hm,-hm:km+hm),dxi(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3,3), &
    x(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3),qrhs(0:im,0:jm,0:km,5), &
    original_rhs(0:im,0:jm,0:km,5),refreshed_rhs(0:im,0:jm,0:km,5),before(0:im,0:jm,0:km,5), &
    cache(0:im,0:jm,0:km,6))
  jacob=1;dxi=0
  do c=1,3
    dxi(:,:,:,c,c)=1
  enddo
  theta=2*acos(-1.d0)
  do k=-hm,km+hm
  do j=-hm,jm+hm
  do i=-hm,im+hm
    xx=real(modulo(i,ia),8)/ia;zz=real(modulo(k+mpirank*km,ka),8)/ka
    rr=1+.03d0*cos(3*theta*xx)*cos(5*theta*zz)
    tt=1+.005d0*cos(2*theta*zz);uu=.4d0+.01d0*sin(2*theta*xx)
    vv=.03d0+.01d0*cos(4*theta*zz);ww=.02d0*sin(2*theta*zz);pp=rr*tt/const2
    q(i,j,k,:)=[rr,rr*uu,rr*vv,rr*ww,pp*const6+.5d0*rr*(uu*uu+vv*vv+ww*ww)]
    x(i,j,k,:)=[real(i,8),real(j,8),real(k+mpirank*km,8)]
  enddo
  enddo
  enddo
  call q2fvar(q=q,density=rho,velocity=vel,pressure=prs,temperature=tmp,species=spc)
  call filter_coefficient_explicit()
  call qswap()
  qrhs=0
  call farfield_nscbc(4)
  original_rhs=qrhs;before=q(0:im,0:jm,0:km,:)
  cache(:,:,:,1)=rho(0:im,0:jm,0:km);cache(:,:,:,2:4)=vel(0:im,0:jm,0:km,:)
  cache(:,:,:,5)=prs(0:im,0:jm,0:km);cache(:,:,:,6)=tmp(0:im,0:jm,0:km)
  ! This is the same qswap the main loop performs after boucon, followed by
  ! a diagnostic RHS reevaluation with both transverse filters disabled.
  call qswap()
  changed=count(transfer(before,[0_c_int64_t],size(before))/= &
    transfer(q(0:im,0:jm,0:km,:),[0_c_int64_t],size(before)))
  cache_error=max(maxval(abs(cache(:,:,:,1)-rho(0:im,0:jm,0:km))), &
    maxval(abs(cache(:,:,:,2:4)-vel(0:im,0:jm,0:km,:))), &
    maxval(abs(cache(:,:,:,5)-prs(0:im,0:jm,0:km))),maxval(abs(cache(:,:,:,6)-tmp(0:im,0:jm,0:km))))
  qrhs=0;lfilter=.true.
  call farfield_nscbc(4)
  refreshed_rhs=qrhs
  write(suffix,'(a,i8.8,a)') '.rank',mpirank,'.bin'
  path=trim(directory)//'/phase'//trim(suffix)
  open(newunit=unit,file=trim(path),access='stream',form='unformatted',status='new')
  write(unit) [im,jm,km,mpirank*km,changed]
  write(unit) original_rhs,refreshed_rhs,before,q(0:im,0:jm,0:km,:)
  write(unit) cache_error
  close(unit)
  print *, 'TOP_HALO_PHASE ',mpirank,changed,maxval(abs(original_rhs-refreshed_rhs)),cache_error
  call MPI_Finalize(ierr)
end program
