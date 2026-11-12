program keep_wall_probe
  use astr_precision, only: real_kind
  use keep_operator, only: keep_explicit_volume
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_value,ieee_quiet_nan
#ifdef TEST_CUDA
  use cudafor
  use keep_operator_gpu, only: keep_explicit_volume_kernel
#endif
  implicit none
  real(real_kind),parameter :: gamma=1.4e0_real_kind,pi=acos(-1.e0_real_kind),amplitude=0.1e0_real_kind
  real(real_kind) :: errors(3,3),orders(3)
  integer :: level
  do level=1,3
    call check_wall(16*2**(level-1),errors(:,level))
    write(*,'(A,I0,A,3ES24.16)') 'KEEP_WALL intervals=',16*2**(level-1),' row1/row2/interior errors=',errors(:,level)
  enddo
  do level=1,2
    orders=log(errors(:,level)/errors(:,level+1))/log(2.e0_real_kind)
    write(*,'(A,3F14.7)') 'KEEP_WALL orders=',orders
  enddo
  if(any(orders<[1.5e0_real_kind,3.5e0_real_kind,5.5e0_real_kind])) &
    error stop 'KEEP wall does not exhibit second/fourth/sixth-order local consistency'
  print '(A)', 'KEEP_WALL_COMPONENT_PASS polynomial_rows wall_compatible_manufactured'
contains
  subroutine check_wall(n,error)
    integer,intent(in) :: n
    real(real_kind),intent(out) :: error(3)
    integer,parameter :: hm=5
    real(real_kind) :: rho(-hm:n+hm,-hm:n+hm,-hm:hm),p(-hm:n+hm,-hm:n+hm,-hm:hm)
    real(real_kind) :: v(-hm:n+hm,-hm:n+hm,-hm:hm,3),q(-hm:n+hm,-hm:n+hm,-hm:hm,5)
    real(real_kind) :: jac(-hm:n+hm,-hm:n+hm,-hm:hm),metric(-hm:n+hm,-hm:n+hm,-hm:hm,3,3)
    real(real_kind) :: rhs(0:n,0:n,0:0,5),hx,hy,x,y,uvel,vvel,ux,uy,vx,vy,acc(2),exact(5),e,nanval
    real(real_kind) :: polynomial_error,expected,leading(3),predicted(3)
    integer :: i,j,k,d,degree,category
    hx=2*pi/n;hy=pi/n;nanval=ieee_value(0.e0_real_kind,ieee_quiet_nan)
    rho=nanval;p=nanval;v=nanval;q=nanval
    jac=hx*hy;metric=0
    metric(:,:,:,1,1)=1/hx;metric(:,:,:,2,2)=1/hy;metric(:,:,:,3,3)=1
    do k=-hm,hm
    do j=0,n
    do i=-hm,n+hm
      x=i*hx;y=j*hy
      uvel=amplitude*sin(x)*sin(2*y)
      vvel=-amplitude*cos(x)*sin(y)**2
      rho(i,j,k)=1;p(i,j,k)=1
      v(i,j,k,:)=[uvel,vvel,0.e0_real_kind]
      q(i,j,k,1)=1;q(i,j,k,2:4)=v(i,j,k,:)
      q(i,j,k,5)=1/(gamma-1)+0.5e0_real_kind*(uvel*uvel+vvel*vvel)
    enddo
    enddo
    enddo
    call evaluate(n,hm,q,rho,v,p,jac,metric,rhs)
    error=0
    do j=1,n-1
    do i=0,n
      x=i*hx;y=j*hy
      uvel=v(i,j,0,1);vvel=v(i,j,0,2)
      ux=amplitude*cos(x)*sin(2*y);uy=2*amplitude*sin(x)*cos(2*y)
      vx=amplitude*sin(x)*sin(y)**2;vy=-amplitude*cos(x)*sin(2*y)
      acc=[uvel*ux+vvel*uy,uvel*vx+vvel*vy]
      exact=[0.e0_real_kind,acc,0.e0_real_kind,uvel*acc(1)+vvel*acc(2)]
      category=3
      if(min(j,n-j)==1) category=1
      if(min(j,n-j)==2) category=2
      e=maxval(abs(rhs(i,j,0,:)/jac(i,j,0)-exact))
      error(category)=max(error(category),e)
    enddo
    enddo
    if(n/=16) return
    polynomial_error=0;leading=0
    predicted=[0.01e0_real_kind/(pi*n**2),-0.04e0_real_kind/(pi*real(n,real_kind)**4), &
      0.36e0_real_kind/(pi*real(n,real_kind)**6)]
    do degree=0,7
      do k=-hm,hm
      do j=0,n
      do i=-hm,n+hm
        y=real(j,real_kind)/n
        rho(i,j,k)=1;p(i,j,k)=1+0.01e0_real_kind*y**degree
        v(i,j,k,:)=0;q(i,j,k,:)=0
        q(i,j,k,1)=1;q(i,j,k,5)=p(i,j,k)/(gamma-1)
      enddo
      enddo
      enddo
      call evaluate(n,hm,q,rho,v,p,jac,metric,rhs)
      select case(degree)
      case(3)
        leading(1)=rhs(0,1,0,3)/jac(0,1,0)-0.03e0_real_kind*(1.e0_real_kind/n)**2/pi
      case(5)
        leading(2)=rhs(0,2,0,3)/jac(0,2,0)-0.05e0_real_kind*(2.e0_real_kind/n)**4/pi
      case(7)
        leading(3)=rhs(0,n/2,0,3)/jac(0,n/2,0)-0.07e0_real_kind*(0.5e0_real_kind)**6/pi
      end select
      do j=1,n-1
        d=6
        if(min(j,n-j)==1) d=2
        if(min(j,n-j)==2) d=4
        if(degree>d) cycle
        expected=0
        if(degree>0) expected=0.01e0_real_kind*degree*(real(j,real_kind)/n)**(degree-1)/pi
        polynomial_error=max(polynomial_error,maxval(abs(rhs(:,j,0,3)/jac(0,j,0)-expected)))
      enddo
    enddo
    write(*,'(A,ES24.16)') 'KEEP_WALL polynomial error=',polynomial_error
    if(polynomial_error>2.e-10_real_kind) error stop 'KEEP wall polynomial consistency'
    write(*,'(A,3ES24.16)') 'KEEP_WALL leading truncation residual=',leading
    write(*,'(A,3ES24.16)') 'KEEP_WALL Taylor prediction=',predicted
    if(maxval(abs(leading-predicted))>2.e-10_real_kind) error stop 'KEEP wall leading truncation coefficient'
  end subroutine

    subroutine evaluate(n,hm,q,rho,v,p,jac,metric,rhs)
      integer,intent(in) :: n,hm
      real(real_kind),intent(in) :: q(-hm:n+hm,-hm:n+hm,-hm:hm,5),rho(-hm:n+hm,-hm:n+hm,-hm:hm)
      real(real_kind),intent(in) :: v(-hm:n+hm,-hm:n+hm,-hm:hm,3),p(-hm:n+hm,-hm:n+hm,-hm:hm)
      real(real_kind),intent(in) :: jac(-hm:n+hm,-hm:n+hm,-hm:hm),metric(-hm:n+hm,-hm:n+hm,-hm:hm,3,3)
      real(real_kind),intent(out) :: rhs(0:n,0:n,0:0,5)
      real(real_kind) :: accumulated(0:n,0:n,0:0,5),difference
      rhs=0.e0_real_kind
      call keep_explicit_volume(n,n,0,hm,[3,4,3],[0,1,0],[n,n-1,0],q,rho,v,p,jac,metric,gamma,rhs)
      if(.not.all(ieee_is_finite(rhs))) error stop 'KEEP wall read undefined physical-boundary halo'
      if(any(rhs(:,0,0,:)/=0).or.any(rhs(:,n,0,:)/=0)) error stop 'KEEP wall advanced fixed wall nodes'
      if(n==16) then
        accumulated=0.125e0_real_kind
        call keep_explicit_volume(n,n,0,hm,[3,4,3],[0,1,0],[n,n-1,0],q,rho,v,p,jac,metric,gamma,accumulated)
        if(any(accumulated(:,0,0,:)/=0.125e0_real_kind).or. &
          any(accumulated(:,n,0,:)/=0.125e0_real_kind)) error stop 'KEEP overwrote boundary RHS'
        difference=maxval(abs(accumulated-(rhs+0.125e0_real_kind)))
        if(difference>2.e-10_real_kind) error stop 'KEEP did not add to pre-existing RHS'
        write(*,'(A,ES24.16)') 'KEEP_WALL_RHS_ACCUMULATION max_abs=',difference
      endif
#ifdef TEST_CUDA
      block
        real(real_kind),device,allocatable :: qd(:,:,:,:),rd(:,:,:),vd(:,:,:,:),pd(:,:,:),jd(:,:,:),gd(:,:,:,:,:),fd(:,:,:,:)
        real(real_kind) :: gpu(0:n,0:n,0:0,5)
        integer :: a,status,types(3)
        type(dim3) :: block_size,grid
        allocate(qd(-hm:n+hm,-hm:n+hm,-hm:hm,5),rd(-hm:n+hm,-hm:n+hm,-hm:hm), &
          vd(-hm:n+hm,-hm:n+hm,-hm:hm,3),pd(-hm:n+hm,-hm:n+hm,-hm:hm), &
          jd(-hm:n+hm,-hm:n+hm,-hm:hm),gd(-hm:n+hm,-hm:n+hm,-hm:hm,3,3),fd(0:n,0:n,0:0,5))
        qd=q;rd=rho;vd=v;pd=p;jd=jac;gd=metric;fd=0.125e0_real_kind
        types=[3,4,3]
        do a=1,3
          select case(a)
          case(1);block_size=dim3(512,1,1)
          case(2);block_size=dim3(32,16,1)
          case(3);block_size=dim3(64,1,8)
          end select
          grid=dim3((n+block_size%x)/block_size%x,(n+block_size%y)/block_size%y,1)
          call keep_explicit_volume_kernel<<<grid,block_size>>>(n,n,0,hm,a,types(a),0,n,1,n-1,0,0,gamma, &
            qd,rd,vd,pd,jd,gd,fd)
          status=cudaGetLastError()
          if(status/=cudaSuccess) error stop 'KEEP wall kernel launch failed'
          status=cudaDeviceSynchronize()
          if(status/=cudaSuccess) error stop 'KEEP wall kernel failed'
        enddo
        gpu=fd
        if(.not.all(ieee_is_finite(gpu))) error stop 'KEEP wall GPU read undefined halo'
        if(any(gpu(:,0,0,:)/=0.125e0_real_kind).or.any(gpu(:,n,0,:)/=0.125e0_real_kind)) &
          error stop 'KEEP GPU overwrote boundary RHS'
        difference=maxval(abs(gpu-(0.125e0_real_kind-rhs))/max(1.e0_real_kind,abs(gpu),abs(rhs)))
        write(*,'(A,I0,A,ES24.16)') 'KEEP_WALL_GPU n=',n,' scaled=',difference
        if(difference>2.e-10_real_kind) error stop 'KEEP wall CPU/GPU RHS differs'
        deallocate(qd,rd,vd,pd,jd,gd,fd)
      end block
#endif
    end subroutine
end program
