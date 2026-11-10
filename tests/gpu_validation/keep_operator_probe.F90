program keep_operator_probe
  use astr_precision, only: real_kind
  use keep_operator, only: keep_two_point_flux5,keep_explicit_line
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
#ifdef TEST_CUDA
  use cudafor
  use keep_operator_gpu, only: keep_explicit_line_kernel
#endif
  implicit none
  real(real_kind),parameter :: gamma=1.4e0_real_kind,pi=acos(-1.e0_real_kind)
  real(real_kind) :: flux(5),reverse(5),expected(5),v(3),m(3),energy
  real(real_kind) :: errors(5,3),orders(5),budget,scale
  integer :: level,n
  v=[0.3e0_real_kind,-0.2e0_real_kind,0.1e0_real_kind]
  m=[0.7e0_real_kind,0.2e0_real_kind,-0.1e0_real_kind]
  call keep_two_point_flux5(1.2e0_real_kind,v,0.9e0_real_kind,m, &
    0.8e0_real_kind,2.e0_real_kind*v,1.1e0_real_kind,2.e0_real_kind*m,gamma,flux)
  if(.not.all(ieee_is_finite(flux))) error stop 'KEEP finite pair flux'
  call keep_two_point_flux5(0.8e0_real_kind,2.e0_real_kind*v,1.1e0_real_kind, &
    2.e0_real_kind*m,1.2e0_real_kind,v,0.9e0_real_kind,m,gamma,reverse)
  if(any(flux/=reverse)) error stop 'KEEP two-point symmetry'
  call keep_two_point_flux5(1.2e0_real_kind,v,0.9e0_real_kind,m, &
    1.2e0_real_kind,v,0.9e0_real_kind,m,gamma,flux)
  energy=0.9e0_real_kind/(gamma-1.e0_real_kind)+0.6e0_real_kind*sum(v*v)
  expected(1)=1.2e0_real_kind*dot_product(v,m)
  expected(2:4)=expected(1)*v+0.9e0_real_kind*m
  expected(5)=(energy+0.9e0_real_kind)*dot_product(v,m)
  if(maxval(abs(flux-expected))>2.e-14_real_kind) error stop 'KEEP consistent Euler flux'
  do level=1,3
    n=16*2**level
    call periodic_check(n,errors(:,level),budget,scale)
    write(*,'(A,I0,A,5ES16.7)') 'KEEP n=',n,' component_errors=',errors(:,level)
    write(*,'(A,ES16.7)') 'KEEP kinetic_pressure_balance=',budget
    if(abs(budget)>2.e-12_real_kind*scale) error stop 'KEEP kinetic pressure exchange'
  enddo
  do level=1,2
    orders=log(errors(:,level)/errors(:,level+1))/log(2.e0_real_kind)
    write(*,'(A,5F12.6)') 'KEEP component_orders=',orders
    if(any(orders<5.5e0_real_kind)) error stop 'KEEP sixth-order convergence'
  enddo
  write(*,'(A)') 'KEEP_CPU_COMPONENT_PASS symmetry consistency convergence conservation kinetic_exchange'
contains
  subroutine periodic_check(n,error,balance,balance_scale)
    integer,intent(in) :: n
    real(real_kind),intent(out) :: error(5),balance,balance_scale
    real(real_kind) :: rho(-3:n+2),vel(-3:n+2,3),p(-3:n+2),metric(-3:n+2,3)
    real(real_kind) :: rhs(0:n-1,5),exact(5),x,h,dr,dp,dv(3),enthalpy,dh,dp_discrete,kinetic_rate
    real(real_kind),parameter :: a(3)=[0.75e0_real_kind,-0.15e0_real_kind,1.e0_real_kind/60.e0_real_kind]
    integer :: i,l
    h=2.e0_real_kind*pi/real(n,real_kind)
    do i=-3,n+2
      x=real(modulo(i,n),real_kind)*h
      rho(i)=1.e0_real_kind+0.1e0_real_kind*sin(x)
      p(i)=1.e0_real_kind+0.1e0_real_kind*cos(2.e0_real_kind*x)
      vel(i,:)=[0.2e0_real_kind+0.1e0_real_kind*sin(2.e0_real_kind*x), &
        0.1e0_real_kind*cos(x),0.05e0_real_kind*sin(3.e0_real_kind*x)]
      metric(i,:)=[1.e0_real_kind,0.e0_real_kind,0.e0_real_kind]
    enddo
    call keep_explicit_line(n-1,h,rho,vel,p,metric,gamma,rhs)
    if(.not.all(ieee_is_finite(rhs))) error stop 'KEEP finite CPU divergence'
#ifdef TEST_CUDA
    block
      real(real_kind),device,allocatable :: rd(:),vd(:,:),pd(:),md(:,:),result_d(:,:)
      real(real_kind) :: gpu(0:n-1,5),scaled_difference
      integer :: status
      allocate(rd(-3:n+2),vd(-3:n+2,3),pd(-3:n+2),md(-3:n+2,3),result_d(0:n-1,5))
      rd=rho
      vd=vel
      pd=p
      md=metric
      call keep_explicit_line_kernel<<<(n+511)/512,512>>>(n-1,h,rd,vd,pd,md,gamma,result_d)
      status=cudaDeviceSynchronize()
      if(status/=cudaSuccess) error stop 'KEEP GPU kernel execution'
      gpu=result_d
      if(.not.all(ieee_is_finite(gpu))) error stop 'KEEP finite GPU divergence'
      scaled_difference=maxval(abs(gpu-rhs)/max(1.e0_real_kind,abs(gpu),abs(rhs)))
      write(*,'(A,ES16.7)') 'KEEP CPU_GPU_scaled_difference=',scaled_difference
      if(scaled_difference>2.e-10_real_kind) error stop 'KEEP CPU/GPU frozen-state parity'
      deallocate(rd,vd,pd,md,result_d)
    end block
#endif
    if(maxval(abs(sum(rhs,dim=1)))>2.e-12_real_kind*max(1.e0_real_kind,sum(abs(rhs)))) &
      error stop 'KEEP periodic conservative sum'
    error=0.e0_real_kind
    balance=0.e0_real_kind
    balance_scale=1.e0_real_kind
    do i=0,n-1
      x=real(i,real_kind)*h
      dr=0.1e0_real_kind*cos(x)
      dp=-0.2e0_real_kind*sin(2.e0_real_kind*x)
      dv=[0.2e0_real_kind*cos(2.e0_real_kind*x),-0.1e0_real_kind*sin(x), &
        0.15e0_real_kind*cos(3.e0_real_kind*x)]
      exact(1)=dr*vel(i,1)+rho(i)*dv(1)
      exact(2:4)=exact(1)*vel(i,:)+rho(i)*vel(i,1)*dv
      exact(2)=exact(2)+dp
      enthalpy=gamma*p(i)/(gamma-1.e0_real_kind)+0.5e0_real_kind*rho(i)*sum(vel(i,:)**2)
      dh=gamma*dp/(gamma-1.e0_real_kind)+0.5e0_real_kind*dr*sum(vel(i,:)**2) &
        +rho(i)*dot_product(vel(i,:),dv)
      exact(5)=dh*vel(i,1)+enthalpy*dv(1)
      error=max(error,abs(rhs(i,:)-exact))
      dp_discrete=0.e0_real_kind
      do l=1,3
        dp_discrete=dp_discrete+a(l)*(p(i+l)-p(i-l))/h
      enddo
      kinetic_rate=dot_product(vel(i,:),rhs(i,2:4))-0.5e0_real_kind*sum(vel(i,:)**2)*rhs(i,1)
      balance=balance+h*(kinetic_rate-vel(i,1)*dp_discrete)
      balance_scale=balance_scale+h*(abs(kinetic_rate)+abs(vel(i,1)*dp_discrete))
    enddo
  end subroutine
end program
