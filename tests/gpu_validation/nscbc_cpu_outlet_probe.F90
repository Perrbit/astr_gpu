program nscbc_cpu_outlet_probe
  use astr_precision, only: real_kind
  use commvar
  use commarray, only: q,rho,vel,prs,tmp,spc,jacob,dxi,qrhs
  use bc, only: bctype,pout,outflow_nscbc
  use filter, only: filter_coefficient_explicit
  use parallel, only: mpirank,mpisize,lio,irk,irkm,jrk,jrkm,krk,krkm,isize,jsize,ksize
  use mpi
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
#ifdef TEST_CUDA
  use cudafor
  use commarray_gpu, only: q_d,rho_d,vel_d,prs_d,tmp_d,jacob_d,dxi_d,qrhs_d
  use boundary_gpu, only: nscbc_outflow_x_rhs_kernel
#endif
  implicit none
  real(real_kind) :: u,v,w,p,target,css,enthalpy,rate,back,plus_rate,gradient,local_p
  real(real_kind) :: expected(5),delta(5),plus(5),target_q(5),dp,du,errors(3),global_errors(3)
  real(real_kind),allocatable :: actual(:,:,:,:)
  integer :: ierr,i,j,k,c
#ifdef TEST_CUDA
  integer :: status,devices
  type(dim3) :: block_size,grid
#endif
  character(32) :: mode
  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD,mpirank,ierr)
  call MPI_Comm_size(MPI_COMM_WORLD,mpisize,ierr)
  call get_command_argument(1,mode)
  im=8;jm=8;km=8;ia=im*mpisize;ja=jm;ka=km;ndims=3;numq=5;num_species=0;num_modequ=0
  irk=mpirank;irkm=mpisize-1;jrk=0;jrkm=0;krk=0;krkm=0;lio=mpirank==0
  isize=mpisize;jsize=1;ksize=1;is=0;ie=im;js=0;je=jm;ks=0;ke=km
  npdci=3;npdcj=3;npdck=3;lihomo=.false.;ljhomo=.true.;lkhomo=.true.
  flowtype='tgv';conschm='643e';difschm='643e';turbmode='none';nondimen=.true.;lcomb=.false.;lfilter=.false.
  bctype=[11,22,1,1,1,1];gamma=1.4e0_real_kind;mach=1.e0_real_kind
  const2=gamma*mach**2;const6=1.e0_real_kind/(gamma-1.e0_real_kind)
  xmin=0.e0_real_kind;xmax=real(ia,real_kind);p=1.e0_real_kind/gamma
  u=0.2e0_real_kind;v=0.e0_real_kind;w=0.e0_real_kind;gradient=0.e0_real_kind
  pinf=p-0.01e0_real_kind;pout=1.e0_real_kind
  select case(trim(mode))
  case('out_sub')
    continue
  case('out_sup');u=2.e0_real_kind
  case('out_tangent');v=10.e0_real_kind
  case('out_grazing');u=0.e0_real_kind
  case('out_uniform');pinf=p
  case('out_sup_gradient');u=2.e0_real_kind;gradient=0.001e0_real_kind
  case('out_backflow');u=-0.2e0_real_kind;v=0.1e0_real_kind;w=-0.05e0_real_kind
  case('out_backflow_sup');u=-2.e0_real_kind
  case('out_backflow_pout');u=-0.2e0_real_kind;pout=0.8e0_real_kind
  case default
    error stop 'Unknown CPU outlet probe mode'
  end select
  roinf=1.e0_real_kind;uinf=u;vinf=v;winf=w
  if(index(mode,'out_backflow')==1) then
    uinf=0.4e0_real_kind;vinf=0.03e0_real_kind;winf=0.02e0_real_kind
  endif
  target=pinf*pout
  call filter_coefficient_explicit()
  allocate(q(-hm:im+hm,-hm:jm+hm,-hm:km+hm,5),rho(-hm:im+hm,-hm:jm+hm,-hm:km+hm), &
    vel(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3),prs(-hm:im+hm,-hm:jm+hm,-hm:km+hm), &
    tmp(-hm:im+hm,-hm:jm+hm,-hm:km+hm),spc(-hm:im+hm,-hm:jm+hm,-hm:km+hm,0), &
    jacob(-hm:im+hm,-hm:jm+hm,-hm:km+hm),dxi(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3,3), &
    qrhs(0:im,0:jm,0:km,5),actual(0:im,0:jm,0:km,5))
  rho=1.e0_real_kind;jacob=1.e0_real_kind;dxi=0.e0_real_kind;qrhs=0.e0_real_kind
  do c=1,3
    dxi(:,:,:,c,c)=1.e0_real_kind
  enddo
  do k=-hm,km+hm
  do j=-hm,jm+hm
  do i=-hm,im+hm
    prs(i,j,k)=p+gradient*real(i+mpirank*im,real_kind)
    tmp(i,j,k)=gamma*prs(i,j,k);vel(i,j,k,:)=[u,v,w]
    q(i,j,k,:)=[1.e0_real_kind,u,v,w,prs(i,j,k)*const6+0.5e0_real_kind*(u*u+v*v+w*w)]
  enddo
  enddo
  enddo
#ifdef TEST_CUDA
  status=cudaGetDeviceCount(devices)
  if(status/=cudaSuccess.or.devices<=0) error stop 'Outlet probe has no GPU'
  status=cudaSetDevice(mod(mpirank,devices))
  if(status/=cudaSuccess) error stop 'Outlet probe cannot select GPU'
  allocate(q_d,mold=q);allocate(rho_d,mold=rho);allocate(vel_d,mold=vel)
  allocate(prs_d,mold=prs);allocate(tmp_d,mold=tmp);allocate(jacob_d,mold=jacob)
  allocate(dxi_d,mold=dxi);allocate(qrhs_d,mold=qrhs)
  q_d=q;rho_d=rho;vel_d=vel;prs_d=prs;tmp_d=tmp;jacob_d=jacob;dxi_d=dxi;qrhs_d=0.e0_real_kind
  block_size=dim3(256,1,1);grid=dim3(((jm+1)*(km+1)+255)/256,1,1)
  call nscbc_outflow_x_rhs_kernel<<<grid,block_size>>>(im,jm,km,hm,npdcj,npdck,gamma,mach,xmin,xmax, &
    roinf,uinf,vinf,winf,pinf,pout,100.e0_real_kind,merge(1,0,mpirank==mpisize-1),1)
  status=cudaGetLastError()
  if(status/=cudaSuccess) error stop 'Outlet probe kernel launch failed'
  status=cudaDeviceSynchronize()
  if(status/=cudaSuccess) error stop 'Outlet probe kernel failed'
  actual=qrhs_d
#endif
  call outflow_nscbc(2)
  expected=0.e0_real_kind;errors=0.e0_real_kind
  if(mpirank==mpisize-1) then
    local_p=p+gradient*real(ia,real_kind);css=sqrt(gamma*local_p)
    enthalpy=css**2*const6+0.5e0_real_kind*(u*u+v*v+w*w)
    if(u<css) then
      rate=3.75e0_real_kind*max(0.e0_real_kind,1.e0_real_kind-(u/css)**2)*css/(xmax-xmin)
      expected=-rate*(local_p-target)/css**2*[1.e0_real_kind,u-css,v,w,enthalpy-css*u]
    endif
    if(u<0.e0_real_kind) then
      target_q=[roinf*pout,roinf*pout*uinf,roinf*pout*vinf,roinf*pout*winf, &
        target*const6+0.5e0_real_kind*roinf*pout*(uinf*uinf+vinf*vinf+winf*winf)]
      delta=q(im,jm/2,km/2,:)-target_q
      dp=(gamma-1.e0_real_kind)*(delta(5)-u*delta(2)-v*delta(3)-w*delta(4)+ &
        0.5e0_real_kind*(u*u+v*v+w*w)*delta(1))
      du=delta(2)-u*delta(1)
      plus=(dp+css*du)/(2*css**2)*[1.e0_real_kind,u+css,v,w,enthalpy+css*u]
      back=0.25e0_real_kind*(-u)/(xmax-xmin)
      plus_rate=0.25e0_real_kind*max(-u-css,0.e0_real_kind)/(xmax-xmin)
      expected=expected-back*(delta-plus)-plus_rate*plus
    endif
    if(gradient/=0.e0_real_kind) expected=-gradient*[0.e0_real_kind,1.e0_real_kind,0.e0_real_kind, &
      0.e0_real_kind,gamma*u*const6]
    do c=1,5
      errors(1)=max(errors(1),maxval(abs(qrhs(im,:,:,c)+expected(c))))
#ifdef TEST_CUDA
      errors(2)=max(errors(2),maxval(abs(actual(im,:,:,c)-expected(c))))
      errors(3)=max(errors(3),maxval(abs(actual(im,:,:,c)+qrhs(im,:,:,c))))
#endif
    enddo
  else
    errors(1)=maxval(abs(qrhs))
#ifdef TEST_CUDA
    errors(2)=maxval(abs(actual));errors(3)=maxval(abs(actual+qrhs))
#endif
  endif
  if(.not.all(ieee_is_finite(qrhs))) error stop 'Nonfinite CPU outlet RHS'
  call MPI_Allreduce(errors,global_errors,3,MPI_DOUBLE_PRECISION,MPI_MAX,MPI_COMM_WORLD,ierr)
  if(lio) write(*,'(A,A,I4,3ES25.16)') 'NSCBC_CPU_OUTLET ',trim(mode),mpisize,global_errors
  call MPI_Finalize(ierr)
  if(maxval(global_errors)>2.e-10_real_kind) error stop 'CPU outlet violates classified characteristic reference'
end program
