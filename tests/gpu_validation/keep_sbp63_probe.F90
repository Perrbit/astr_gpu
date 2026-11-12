program keep_sbp63_probe
  use astr_precision, only: real_kind
  use keep_sbp63
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_value,ieee_quiet_nan
#ifdef TEST_CUDA
  use cudafor
  use keep_sbp63_gpu, only: sbp63_convection_line_kernel,sbp63_flux_line_kernel,sbp63_divergence_line_kernel
#endif
  implicit none
  integer :: level,n
  real(real_kind) :: errors(2,3),orders(2)
  character(1024) :: destination
  call get_command_argument(1,destination)
  do level=1,3
    n=16*2**(level-1)
    call check(n,trim(destination),errors(:,level))
    write(*,'(A,I0,A,2ES24.16)') 'SBP63 intervals=',n,' boundary/interior error=',errors(:,level)
  enddo
  do level=1,2
    orders=log(errors(:,level)/errors(:,level+1))/log(2.e0_real_kind)
    write(*,'(A,2F14.7)') 'SBP63 orders=',orders
  enddo
  if(orders(1)<2.5e0_real_kind.or.orders(2)<5.5e0_real_kind) error stop 'SBP63 derivative order differs'
  print '(A)', 'SBP63_COMPONENT_PASS derivative_moments sbp convective_wall viscous_work_heat'
contains
  subroutine check(n,destination,error)
    integer,intent(in) :: n
    character(*),intent(in) :: destination
    real(real_kind),intent(out) :: error(2)
    integer,parameter :: hm=5
    real(real_kind) :: rho(-hm:n+hm),v(-hm:n+hm,3),p(-hm:n+hm),t(-hm:n+hm)
    real(real_kind) :: conv(0:n,5),visc(0:n,5),flux(0:n,4),face(0:n,5),grad(0:n,12)
    real(real_kind) :: mu(0:n),kappa(0:n),d(0:n,0:n),h(0:n),sbp(0:n,0:n),rhs(0:n,5)
    real(real_kind) :: x,spacing,expected,moment,error_sbp
    real(real_kind) :: mass_rate,energy_rate,kinetic,work_pressure,dissipation,penalty,stress(3,3),eflux(3),gu(3,3)
    real(real_kind) :: nanvalue,suth,pi,gamma,scaled_difference
    integer :: i,j,degree,unit,category
    character(1024) :: path
    spacing=1.e0_real_kind/real(n,real_kind);gamma=1.4e0_real_kind;pi=acos(-1.e0_real_kind)
    nanvalue=ieee_value(0.e0_real_kind,ieee_quiet_nan)
    rho=nanvalue;v=nanvalue;p=nanvalue;t=nanvalue
    suth=110.3e0_real_kind/273.15e0_real_kind
    do i=0,n
      x=real(i,real_kind)*spacing;h(i)=spacing*sbp63_norm(i,n)
      rho(i)=1.e0_real_kind+x*x/20.e0_real_kind;t(i)=(1.e0_real_kind+x/20.e0_real_kind)**2
      v(i,:)=[0.1e0_real_kind+x/20.e0_real_kind,0.025e0_real_kind-x/30.e0_real_kind,x*x/50.e0_real_kind]
      p(i)=rho(i)*t(i)/(gamma*0.09e0_real_kind)
      mu(i)=t(i)*sqrt(t(i))*(1.e0_real_kind+suth)/(1000.e0_real_kind*(t(i)+suth))
      kappa(i)=mu(i)/0.72e0_real_kind/0.036e0_real_kind
      do j=0,n
        d(i,j)=sbp63_weight(i,j-i,n)/spacing
      enddo
    enddo
    sbp=0.e0_real_kind
    do i=0,n
      do j=0,n
        sbp(i,j)=h(i)*d(i,j)+h(j)*d(j,i)
      enddo
    enddo
    sbp(0,0)=sbp(0,0)+1.e0_real_kind;sbp(n,n)=sbp(n,n)-1.e0_real_kind
    error_sbp=maxval(abs(sbp))
    if(error_sbp>2.e-10_real_kind) error stop 'SBP63 SBP identity differs'
    error=0.e0_real_kind
    do i=0,n
      category=2
      if(min(i,n-i)<6) category=1
      x=real(i,real_kind)*spacing
      error(category)=max(error(category),abs(sum(d(i,:)*cos(2*pi*[(real(j,real_kind)*spacing,j=0,n)]))+ &
        2*pi*sin(2*pi*x)))
      do degree=0,merge(3,6,category==1)
        moment=0.e0_real_kind
        do j=0,n
          moment=moment+d(i,j)*(real(j,real_kind)*spacing)**degree
        enddo
        expected=0.e0_real_kind
        if(degree>0) expected=real(degree,real_kind)*x**(degree-1)
        if(abs(moment-expected)>2.e-10_real_kind) error stop 'SBP63 polynomial moment differs'
      enddo
    enddo
    call sbp63_convection_line(n,n,0,hm,spacing,rho,v,p,gamma,conv)
    call sbp63_viscous_line(n,hm,spacing,v,t,mu,kappa,1.e0_real_kind,visc,flux,face,grad)
    if(.not.all(ieee_is_finite(conv)).or..not.all(ieee_is_finite(visc))) error stop 'SBP63 physical RHS nonfinite'
#ifdef TEST_CUDA
    block
      real(real_kind),allocatable,device :: rd(:),vd(:,:),pd(:),td(:),mud(:),kd(:),cd(:,:),dd(:,:)
      real(real_kind),allocatable,device :: fd(:,:),bd(:,:),gd(:,:)
      real(real_kind) :: compared(0:n,5),bulk(0:n,4),walls(0:n,5),gradients(0:n,12)
      integer :: status
      allocate(rd(-hm:n+hm),vd(-hm:n+hm,3),pd(-hm:n+hm),td(-hm:n+hm),mud(0:n),kd(0:n))
      allocate(cd(0:n,5),dd(0:n,5),fd(0:n,4),bd(0:n,5),gd(0:n,12))
      rd=rho;vd=v;pd=p;td=t;mud=mu;kd=kappa
      call sbp63_convection_line_kernel<<<(n+128)/128,128>>>(n,n,0,hm,spacing,rd,vd,pd,gamma,cd)
      status=cudaDeviceSynchronize();if(status/=cudaSuccess) error stop 'SBP63 convection kernel failed'
      compared=cd
      scaled_difference=maxval(abs(compared-conv)/max(1.e0_real_kind,max(abs(compared),abs(conv))))
      call sbp63_flux_line_kernel<<<(n+128)/128,128>>>(n,hm,spacing,vd,td,mud,kd,1.e0_real_kind,fd,bd,gd)
      status=cudaDeviceSynchronize();if(status/=cudaSuccess) error stop 'SBP63 flux kernel failed'
      call sbp63_divergence_line_kernel<<<(n+128)/128,128>>>(n,spacing,fd,bd,dd)
      status=cudaDeviceSynchronize();if(status/=cudaSuccess) error stop 'SBP63 viscosity kernel failed'
      compared=dd;bulk=fd;walls=bd;gradients=gd
      scaled_difference=max(scaled_difference,maxval(abs(compared-visc)/max(1.e0_real_kind,max(abs(compared),abs(visc)))), &
        maxval(abs(bulk-flux)/max(1.e0_real_kind,max(abs(bulk),abs(flux)))), &
        maxval(abs(walls-face)/max(1.e0_real_kind,max(abs(walls),abs(face)))), &
        maxval(abs(gradients-grad)/max(1.e0_real_kind,max(abs(gradients),abs(grad)))))
      write(*,'(A,I0,1X,ES24.16)') 'SBP63 CPU_GPU max_scaled n=',n,scaled_difference
      if(.not.ieee_is_finite(scaled_difference).or.scaled_difference>2.e-10_real_kind) &
        error stop 'SBP63 CPU/GPU component differs'
      conv=cd;visc=dd;flux=fd;face=bd;grad=gd
    end block
#endif
    rhs=conv
    mass_rate=sum(h*rhs(:,1));energy_rate=sum(h*rhs(:,5))
    kinetic=0.e0_real_kind;work_pressure=0.e0_real_kind;dissipation=0.e0_real_kind;penalty=0.e0_real_kind
    do i=0,n
      kinetic=kinetic+h(i)*(dot_product(v(i,:),rhs(i,2:4))-0.5e0_real_kind*sum(v(i,:)**2)*rhs(i,1))
      work_pressure=work_pressure-h(i)*v(i,2)*sum(d(i,:)*p(0:n))
      gu=reshape(grad(i,1:9),[3,3])
      call sbp63_material_flux(mu(i),kappa(i),v(i,:),gu,grad(i,10:12),stress,eflux)
      dissipation=dissipation+h(i)*sum(stress*gu)
      if(i==0.or.i==n) penalty=penalty+mu(i)/h(i)*(sum(v(i,:)**2)+v(i,2)**2/3.e0_real_kind)
    enddo
    if(max(abs(mass_rate),abs(energy_rate),abs(kinetic-work_pressure))>2.e-10_real_kind) &
      error stop 'SBP63 weak inviscid mass/energy/pressure work differs'
    kinetic=sum(h*(sum(v(0:n,:)*visc(:,2:4),dim=2)))
    if(abs(kinetic+dissipation+penalty)>2.e-10_real_kind.or. &
       abs(sum(h*visc(:,5))-face(0,5)-face(n,5))>2.e-10_real_kind) &
      error stop 'SBP63 weak viscous work/heat differs'
    write(*,'(A,I0,A,4ES24.16)') 'SBP63 n=',n,' SBP/mass/energy/viscous_work=', &
      error_sbp,mass_rate,energy_rate,kinetic+dissipation+penalty
    write(path,'(A,"/component_n",I0,".txt")') trim(destination),n
    open(newunit=unit,file=trim(path),status='new',action='write')
    do i=0,n
      write(unit,'(I6,39ES26.17)') i,real(i,real_kind)*spacing,h(i),rho(i),t(i),p(i),v(i,:), &
        conv(i,:),visc(i,:),grad(i,:),flux(i,:),face(i,:)
    enddo
    close(unit)
    write(path,'(A,"/matrix_n",I0,".txt")') trim(destination),n
    open(newunit=unit,file=trim(path),status='new',action='write')
    do i=0,n
      do j=0,n
        write(unit,'(2I6,ES26.17)') i,j,d(i,j)
      enddo
    enddo
    close(unit)
  end subroutine
end program
