program me6opti_candidate_probe
  use astr_precision, only: real_kind
  use me6opti_operator
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  real(real_kind),parameter :: pi=acos(-1.e0_real_kind)
  real(real_kind) :: errors(13,3),orders(13),roundoff1(13),roundoff2(13)
  integer :: level
  character(len=32) :: mode
  call get_command_argument(1,mode)
  if(trim(mode)=='--mapped-symbol') then
    call check_mapped_symbol()
    call check_mapped_cartesian()
    print '(A)', 'ME6_G2_MAPPED_SYMBOL_PASS'
    stop
  endif
  call check_coefficients()
  call check_full_symbol()
  call check_manufactured(256,roundoff1,1.e0_real_kind)
  call check_manufactured(256,roundoff2,0.2e0_real_kind)
  write(*,'(A,2ES24.16)') 'ME6 n256 zz error, T offsets 1/0.2=',roundoff1(9),roundoff2(9)
  do level=1,3
    call check_manufactured(16*2**level,errors(:,level))
    write(*,'(A,I0,A,13ES14.5)') 'ME6 candidate n=',16*2**level,' errors=',errors(:,level)
  enddo
  do level=1,2
    orders=log(errors(:,level)/errors(:,level+1))/log(2.e0_real_kind)
    write(*,'(A,13F10.5)') 'ME6 candidate orders=',orders
  enddo
  if(any(orders<5.5e0_real_kind)) error stop 'ME6 candidate lacks sixth-order asymptotic convergence'
  call record_periodic_budget()
  print '(A)', 'ME6_CANDIDATE_PASS coefficients full_viscous_symbol variable_property_manufactured'
contains
  subroutine check_mapped_cartesian()
    integer,parameter :: hm=5
    real(real_kind) :: v(-hm:hm,-hm:hm,-hm:hm,3),t(-hm:hm,-hm:hm,-hm:hm)
    real(real_kind) :: mu(-hm:hm,-hm:hm,-hm:hm),kappa(-hm:hm,-hm:hm,-hm:hm)
    real(real_kind) :: g(-hm:hm,-hm:hm,-hm:hm,3,3),jac(-hm:hm,-hm:hm,-hm:hm)
    real(real_kind) :: spacing(3),theta(3),phase,original(4),mapped(4),difference,largest
    integer :: shape,sample,i,j,k,d,axis
    largest=0
    do shape=1,4
      select case(shape)
      case(1);spacing=[1.e0_real_kind,1.e0_real_kind,1.e0_real_kind]
      case(2);spacing=[0.5e0_real_kind,1.e0_real_kind,2.e0_real_kind]
      case(3);spacing=[1.e0_real_kind,0.25e0_real_kind,0.75e0_real_kind]
      case(4);spacing=[1.e0_real_kind,0.01e0_real_kind,2.e0_real_kind]
      end select
      g=0;jac=product(spacing)
      do d=1,3
        g(:,:,:,d,d)=1.e0_real_kind/spacing(d)
      enddo
      do sample=1,4
        theta=[0.13e0_real_kind,0.21e0_real_kind,-0.17e0_real_kind]*sample
        do k=-hm,hm
        do j=-hm,hm
        do i=-hm,hm
          phase=0.37e0_real_kind+dot_product(theta,real([i,j,k],real_kind))
          v(i,j,k,:)=[0.2e0_real_kind*sin(phase),0.13e0_real_kind*cos(phase),-0.17e0_real_kind*sin(2*phase)]
          t(i,j,k)=1.e0_real_kind+0.1e0_real_kind*cos(phase+0.2e0_real_kind)
          mu(i,j,k)=1.e0_real_kind+0.1e0_real_kind*cos(phase+0.3e0_real_kind)
          kappa(i,j,k)=0.7e0_real_kind+0.07e0_real_kind*sin(phase+0.4e0_real_kind)
        enddo
        enddo
        enddo
        do axis=1,3
          call me6_direction(axis,[0,0,0],hm,v,t,mu,kappa,spacing,original)
          call me6_mapped_direction(axis,[0,0,0],hm,v,t,mu,kappa,g,jac,mapped)
          mapped=mapped/jac(0,0,0)
          difference=maxval(abs(mapped-original)/max(1.e0_real_kind,abs(mapped),abs(original)))
          largest=max(largest,difference)
        enddo
      enddo
    enddo
    write(*,'(A,ES24.16)') 'G2 Cartesian component scaled difference=',largest
    if(largest>2.e-10_real_kind) error stop 'G2 Cartesian component changed existing ME6'
  end subroutine

  subroutine check_mapped_symbol()
    real(real_kind),parameter :: shears(3)=[0.1e0_real_kind,0.3e0_real_kind,1.e0_real_kind]
    complex(real_kind) :: momentum(3,3),thermal,reference(3,3),reference_t
    real(real_kind) :: theta(3),spacing(3),h(3,3),g(3,3),rotation(3,3),ry(3,3),rz(3,3)
    real(real_kind) :: positive(3,3),scale,minor,determinant,minimum,imaginary,heat,baseline_error,sigma
    integer :: shape,geometry,a,b,c,i,j,modes
    rotation=0.e0_real_kind;ry=0.e0_real_kind;rz=0.e0_real_kind
    rz(1,:)=[cos(pi/7),-sin(pi/7),0.e0_real_kind]
    rz(2,:)=[sin(pi/7),cos(pi/7),0.e0_real_kind];rz(3,3)=1.e0_real_kind
    ry(1,:)=[cos(pi/9),0.e0_real_kind,sin(pi/9)]
    ry(2,2)=1.e0_real_kind;ry(3,:)=[-sin(pi/9),0.e0_real_kind,cos(pi/9)]
    rotation=matmul(ry,rz)
    modes=0;baseline_error=0;minimum=0;imaginary=0;heat=0
    do geometry=1,3
      do shape=1,4
        if(geometry==3.and.shape==4) cycle
        select case(shape)
        case(1);spacing=[1.e0_real_kind,1.e0_real_kind,1.e0_real_kind]
        case(2);spacing=[0.5e0_real_kind,1.e0_real_kind,2.e0_real_kind]
        case(3);spacing=[1.e0_real_kind,0.25e0_real_kind,0.75e0_real_kind]
        case(4);spacing=[1.e0_real_kind,0.01e0_real_kind,2.e0_real_kind]
        end select
        h=0.e0_real_kind
        do i=1,3
          h(i,i)=spacing(i)
        enddo
        if(geometry==2) h=matmul(rotation,h)
        if(geometry==3) then
          spacing=[1.e0_real_kind,0.25e0_real_kind,0.75e0_real_kind]
          sigma=shears(shape)
          h=0.e0_real_kind
          do i=1,3
            h(i,i)=spacing(i)
          enddo
          h(1,2)=sigma*spacing(2)
        endif
        call inverse_mapping(h,g)
        call check_mapped_symbol_component(g,abs(dot_product(h(1,:), &
          [h(2,2)*h(3,3)-h(2,3)*h(3,2),h(2,3)*h(3,1)-h(2,1)*h(3,3), &
          h(2,1)*h(3,2)-h(2,2)*h(3,1)])))
        do c=-12,12
        do b=-12,12
        do a=-12,12
          theta=real([a,b,c],real_kind)*pi/12.e0_real_kind
          call mapped_viscous_symbol(theta,g,momentum,thermal)
          scale=max(1.e0_real_kind,maxval(abs(momentum)),abs(thermal))
          if(geometry==1) then
            call me6_viscous_symbol(theta,spacing,reference,reference_t)
            baseline_error=max(baseline_error,maxval(abs(momentum-reference))/scale,abs(thermal-reference_t)/scale)
            if(baseline_error>2.e-10_real_kind) error stop 'G2 Cartesian symbol differs from existing ME6'
          endif
          if(.not.all(ieee_is_finite(real(momentum,real_kind))).or. &
            .not.all(ieee_is_finite(aimag(momentum))).or..not.ieee_is_finite(abs(thermal))) &
            error stop 'G2 nonfinite mapped symbol'
          imaginary=max(imaginary,maxval(abs(aimag(momentum)))/scale,abs(aimag(thermal))/scale)
          heat=max(heat,real(thermal,real_kind)/scale)
          positive=-real(momentum+conjg(transpose(momentum)),real_kind)/(2.e0_real_kind*scale)
          do i=1,3
            minimum=min(minimum,positive(i,i))
            do j=i+1,3
              minor=positive(i,i)*positive(j,j)-positive(i,j)**2
              minimum=min(minimum,minor)
            enddo
          enddo
          determinant=positive(1,1)*(positive(2,2)*positive(3,3)-positive(2,3)*positive(3,2)) &
            -positive(1,2)*(positive(2,1)*positive(3,3)-positive(2,3)*positive(3,1)) &
            +positive(1,3)*(positive(2,1)*positive(3,2)-positive(2,2)*positive(3,1))
          minimum=min(minimum,determinant);modes=modes+1
          if(minimum< -2.e-10_real_kind.or.heat>2.e-10_real_kind.or.imaginary>2.e-10_real_kind) then
            write(*,'(A,2I4,A,3F16.10)') 'G2 geometry/shape=',geometry,shape,' theta=',theta
            write(*,'(A,9ES24.16)') 'G2 inverse metric rows=',transpose(g)
            write(*,'(A,3ES24.16)') 'G2 principal/thermal/imaginary=',minimum,heat,imaginary
            write(*,'(A,9ES24.16)') 'G2 real momentum rows=',transpose(real(momentum,real_kind))
            error stop 'G2 mapped candidate can grow perturbation energy'
          endif
        enddo
        enddo
        enddo
        write(*,'(A,2I4,A,I0,A,4ES17.8)') 'G2 geometry/shape=',geometry,shape,' modes=',modes, &
          ' baseline/principal/thermal/imaginary=',baseline_error,minimum,heat,imaginary
      enddo
    enddo
  end subroutine

  subroutine check_mapped_symbol_component(metric,determinant)
    integer,parameter :: hm=5
    real(real_kind),intent(in) :: metric(3,3),determinant
    real(real_kind) :: g(-hm:hm,-hm:hm,-hm:hm,3,3),jac(-hm:hm,-hm:hm,-hm:hm)
    real(real_kind) :: v(-hm:hm,-hm:hm,-hm:hm,3),t(-hm:hm,-hm:hm,-hm:hm)
    real(real_kind) :: coefficient(-hm:hm,-hm:hm,-hm:hm),field(-hm:hm,-hm:hm,-hm:hm)
    real(real_kind) :: theta(3),phase,actual(4),part(4),expected(4),error
    complex(real_kind) :: momentum(3,3),thermal
    integer :: c,d,basis,quadrature,i,j,k,axis
    do d=1,3
    do c=1,3
      g(:,:,:,d,c)=metric(d,c)
    enddo
    enddo
    jac=determinant;coefficient=1
    theta=[pi/3,-pi/2,pi/4]
    call mapped_viscous_symbol(theta,metric,momentum,thermal)
    do quadrature=1,2
      do k=-hm,hm
      do j=-hm,hm
      do i=-hm,hm
        phase=dot_product(theta,real([i,j,k],real_kind))
        field(i,j,k)=cos(phase)
        if(quadrature==2) field(i,j,k)=sin(phase)
      enddo
      enddo
      enddo
      do basis=1,4
        v=0;t=0;expected=0
        if(basis<=3) then
          v(:,:,:,basis)=field
          expected(1:3)=real(momentum(:,basis),real_kind)
          if(quadrature==2) expected(1:3)=aimag(momentum(:,basis))
        else
          t=field
          expected(4)=real(thermal,real_kind)
          if(quadrature==2) expected(4)=aimag(thermal)
        endif
        actual=0
        do axis=1,3
          call me6_mapped_direction(axis,[0,0,0],hm,v,t,coefficient,coefficient,g,jac,part)
          actual=actual+part/determinant
        enddo
        if(basis<=3) then
          error=maxval(abs(actual(1:3)-expected(1:3)))/max(1.e0_real_kind,maxval(abs(expected(1:3))))
        else
          error=abs(actual(4)-expected(4))/max(1.e0_real_kind,abs(expected(4)))
        endif
        if(.not.ieee_is_finite(error).or.error>2.e-10_real_kind) &
          error stop 'G2 actual constant-metric component disagrees with Fourier symbol'
      enddo
    enddo
  end subroutine

  subroutine inverse_mapping(h,g)
    real(real_kind),intent(in) :: h(3,3)
    real(real_kind),intent(out) :: g(3,3)
    real(real_kind) :: determinant
    g(1,:)=[h(2,2)*h(3,3)-h(2,3)*h(3,2),h(1,3)*h(3,2)-h(1,2)*h(3,3), &
      h(1,2)*h(2,3)-h(1,3)*h(2,2)]
    g(2,:)=[h(2,3)*h(3,1)-h(2,1)*h(3,3),h(1,1)*h(3,3)-h(1,3)*h(3,1), &
      h(1,3)*h(2,1)-h(1,1)*h(2,3)]
    g(3,:)=[h(2,1)*h(3,2)-h(2,2)*h(3,1),h(1,2)*h(3,1)-h(1,1)*h(3,2), &
      h(1,1)*h(2,2)-h(1,2)*h(2,1)]
    determinant=dot_product(h(1,:),g(:,1))
    if(determinant<=0.e0_real_kind) error stop 'G2 test mapping has nonpositive determinant'
    g=g/determinant
  end subroutine

  subroutine mapped_viscous_symbol(theta,g,momentum,thermal)
    real(real_kind),intent(in) :: theta(3),g(3,3)
    complex(real_kind),intent(out) :: momentum(3,3),thermal
    complex(real_kind) :: central(3),gradient(3),physical(3),phase,factor
    real(real_kind),parameter :: central_weights(3)=[0.75e0_real_kind,-0.15e0_real_kind,1.e0_real_kind/60]
    real(real_kind) :: norms(3),ci,cd,cp
    integer :: axis,d,i,j,l,p,h,side,half
    do d=1,3
      central(d)=cmplx(0.e0_real_kind,0.e0_real_kind,real_kind)
      do l=1,3
        central(d)=central(d)+cmplx(0.e0_real_kind,2*central_weights(l)*sin(l*theta(d)),real_kind)
      enddo
      norms(d)=sqrt(sum(g(d,:)**2))
    enddo
    momentum=0;thermal=0
    do axis=1,3
    do h=1,3
    do side=1,2
      half=(2*side-3)*(2*h-1);gradient=0
      do p=-4,4
        call me6_coefficients(half,p,ci,cd,cp)
        phase=exp(cmplx(0.e0_real_kind,p*theta(axis),real_kind))
        do d=1,3
          if(d==axis) then
            gradient(d)=gradient(d)+cd*phase
          else
            gradient(d)=gradient(d)+(ci*central(d)+norms(axis)/norms(d)*cp)*phase
          endif
        enddo
      enddo
      physical=matmul(transpose(g),gradient)
      factor=real(2*side-3,real_kind)*me6_outer(h)
      thermal=thermal+factor*sum(g(axis,:)*physical)
      do j=1,3
      do i=1,3
        momentum(i,j)=momentum(i,j)+factor*(g(axis,j)*physical(i) &
          -2.e0_real_kind*g(axis,i)*physical(j)/3.e0_real_kind)
        if(i==j) momentum(i,j)=momentum(i,j)+factor*sum(g(axis,:)*physical)
      enddo
      enddo
    enddo
    enddo
    enddo
  end subroutine

  subroutine record_periodic_budget()
    integer,parameter :: hm=5
    real(real_kind) :: phi(-hm:hm,-hm:hm,-hm:hm),mu(-hm:hm,-hm:hm,-hm:hm)
    real(real_kind) :: spacing(3),phase_values(-hm:255+hm),mean_variable,mean_constant,max_residual,predicted,ci,cd,cp
    complex(real_kind) :: interpolation(2),gradient(2),phase
    integer :: level,n,i,p,j,k,h,side,half
    max_residual=0.e0_real_kind
    do level=1,3
      n=16*2**level
      spacing=2.e0_real_kind*pi/real(n,real_kind)
      do i=-hm,n-1+hm
        phase_values(i)=cos(real(modulo(i,n),real_kind)*spacing(1))
      enddo
      mean_variable=0.e0_real_kind;mean_constant=0.e0_real_kind
      do i=0,n-1
        do k=-hm,hm
        do j=-hm,hm
        do p=-hm,hm
          phi(p,j,k)=1.e0_real_kind+0.1e0_real_kind*phase_values(i+p)
          mu(p,j,k)=1.e0_real_kind+0.2e0_real_kind*phase_values(i+p)
        enddo
        enddo
        enddo
        mean_variable=mean_variable+me6_scalar_direction(1,1,[0,0,0],hm,phi,mu,spacing)
        mu=1.e0_real_kind
        mean_constant=mean_constant+me6_scalar_direction(1,1,[0,0,0],hm,phi,mu,spacing)
      enddo
      mean_variable=mean_variable/real(n,real_kind);mean_constant=mean_constant/real(n,real_kind)
      write(*,'(A,I0,A,2ES24.16)') 'ME6 periodic n=',n,' variable/constant mean_RHS=',mean_variable,mean_constant
      predicted=0.e0_real_kind
      do h=1,3
        interpolation=0.e0_real_kind;gradient=0.e0_real_kind
        do side=1,2
          half=(2*side-3)*(2*h-1)
          do p=-4,4
            phase=exp(cmplx(0.e0_real_kind,p*spacing(1),real_kind))
            call me6_coefficients(half,p,ci,cd,cp)
            interpolation(side)=interpolation(side)+ci*phase
            gradient(side)=gradient(side)+cd*phase/spacing(1)
          enddo
        enddo
        predicted=predicted+0.01e0_real_kind*me6_outer(h)* &
          real(interpolation(2)*conjg(gradient(2))-interpolation(1)*conjg(gradient(1)),real_kind)/spacing(1)
      enddo
      write(*,'(A,ES24.16)') 'ME6 Fourier zero-mode predicted mean_RHS=',predicted
      max_residual=max(max_residual,abs(mean_variable))
    enddo
    ! User decision: retain this property as a diagnostic, not a rejection gate.
    write(*,'(A,ES24.16)') 'ME6_NONCONSERVATIVE_PROPERTY_RECORDED max_mean_RHS=',max_residual
  end subroutine

  subroutine check_coefficients()
    real(real_kind) :: ci,cd,cp,ci_sum,cd_sum,cp_sum,moment
    integer :: h,side,half,p
    do h=1,3
      do side=1,2
        half=(2*side-3)*(2*h-1)
        ci_sum=0;cd_sum=0;cp_sum=0;moment=0
        do p=-4,4
          call me6_coefficients(half,p,ci,cd,cp)
          ci_sum=ci_sum+ci;cd_sum=cd_sum+cd;cp_sum=cp_sum+cp;moment=moment+p*cd
        enddo
        if(max(abs(ci_sum-1.e0_real_kind),abs(cd_sum),abs(cp_sum),abs(moment-1.e0_real_kind)) &
          >2.e-10_real_kind) error stop 'ME6 coefficient consistency'
      enddo
    enddo
  end subroutine

  subroutine check_full_symbol()
    complex(real_kind) :: momentum(3,3),thermal
    real(real_kind) :: theta(3),spacing(3),positive(3,3),scale,determinant,minor
    real(real_kind) :: maximum_imaginary,minimum_principal,maximum_diagonal
    integer :: a,b,c,shape,i,j,modes
    maximum_imaginary=0;minimum_principal=0;maximum_diagonal=0;modes=0
    do shape=1,4
      select case(shape)
      case(1);spacing=[1.e0_real_kind,1.e0_real_kind,1.e0_real_kind]
      case(2);spacing=[0.5e0_real_kind,1.e0_real_kind,2.e0_real_kind]
      case(3);spacing=[1.e0_real_kind,0.25e0_real_kind,0.75e0_real_kind]
      case(4);spacing=[1.e0_real_kind,0.01e0_real_kind,2.e0_real_kind]
      end select
      do c=-12,12
      do b=-12,12
      do a=-12,12
        theta=real([a,b,c],real_kind)*pi/12.e0_real_kind
        call me6_viscous_symbol(theta,spacing,momentum,thermal)
        scale=max(1.e0_real_kind,maxval(abs(momentum)),abs(thermal))
        maximum_imaginary=max(maximum_imaginary,maxval(abs(aimag(momentum)))/scale,abs(aimag(thermal))/scale)
        maximum_diagonal=max(maximum_diagonal,real(thermal,real_kind)/scale)
        positive=-real(momentum+conjg(transpose(momentum)),real_kind)/(2.e0_real_kind*scale)
        do i=1,3
          minimum_principal=min(minimum_principal,positive(i,i))
          do j=i+1,3
            minor=positive(i,i)*positive(j,j)-positive(i,j)**2
            minimum_principal=min(minimum_principal,minor)
          enddo
        enddo
        determinant=positive(1,1)*(positive(2,2)*positive(3,3)-positive(2,3)*positive(3,2)) &
          -positive(1,2)*(positive(2,1)*positive(3,3)-positive(2,3)*positive(3,1)) &
          +positive(1,3)*(positive(2,1)*positive(3,2)-positive(2,2)*positive(3,1))
        minimum_principal=min(minimum_principal,determinant)
        modes=modes+1
        if(minimum_principal< -2.e-10_real_kind.or.maximum_diagonal>2.e-10_real_kind.or. &
          maximum_imaginary>2.e-10_real_kind) then
          write(*,'(A,3F14.8,A,3ES15.6)') 'ME6 symbol failed theta=',theta,' spacing=',spacing
          write(*,'(A,3ES24.16)') 'ME6 principal/thermal/imaginary=', &
            minimum_principal,maximum_diagonal,maximum_imaginary
          error stop 'ME6 candidate viscous symbol can grow perturbation energy'
        endif
      enddo
      enddo
      enddo
    enddo
    write(*,'(A,I0,A,3ES15.6)') 'ME6 symbol modes=',modes,' principal/thermal/imaginary=', &
      minimum_principal,maximum_diagonal,maximum_imaginary
  end subroutine

  subroutine check_manufactured(n,error,temperature_offset)
    integer,intent(in) :: n
    real(real_kind),intent(out) :: error(13)
    real(real_kind),intent(in),optional :: temperature_offset
    integer,parameter :: hm=5
    real(real_kind) :: velocity(-hm:hm,-hm:hm,-hm:hm,3),temperature(-hm:hm,-hm:hm,-hm:hm)
    real(real_kind) :: mu(-hm:hm,-hm:hm,-hm:hm),kappa(-hm:hm,-hm:hm,-hm:hm)
    real(real_kind),parameter :: ua(3)=[0.2e0_real_kind,0.13e0_real_kind,-0.17e0_real_kind]
    real(real_kind),parameter :: ub(3)=[0.07e0_real_kind,-0.03e0_real_kind,0.04e0_real_kind]
    real(real_kind) :: spacing(3),wave(3),phase,theta,u(3),du(3),ddu(3),g(3,3),stress(3,3)
    real(real_kind) :: mut,mup,kt,kp,td,tdd,w2,trace,momentum(3),energy,actual(4),part(4),scalar,exact,offset
    integer :: orientation,sample,i,j,k,a,b,entry
    spacing=2.e0_real_kind*pi/real(n,real_kind)*[1.e0_real_kind,0.7e0_real_kind,1.3e0_real_kind]
    error=0.e0_real_kind
    offset=1.e0_real_kind
    if(present(temperature_offset)) offset=temperature_offset
    do orientation=1,2
      wave=[1.e0_real_kind,2.e0_real_kind,-1.e0_real_kind]
      if(orientation==2) wave=[-2.e0_real_kind,1.e0_real_kind,1.e0_real_kind]
      w2=sum(wave*wave)
      do sample=0,15
        phase=0.13e0_real_kind+2.e0_real_kind*pi*real(sample,real_kind)/16.e0_real_kind
        do k=-hm,hm
        do j=-hm,hm
        do i=-hm,hm
          theta=phase+dot_product(wave,real([i,j,k],real_kind)*spacing)
          velocity(i,j,k,:)=ua*sin(theta)+ub*cos(2.e0_real_kind*theta)
          temperature(i,j,k)=offset+0.1e0_real_kind*cos(theta+0.2e0_real_kind)
          mu(i,j,k)=1.e0_real_kind+0.1e0_real_kind*cos(theta+0.3e0_real_kind)
          kappa(i,j,k)=0.7e0_real_kind+0.07e0_real_kind*sin(theta+0.4e0_real_kind)
        enddo
        enddo
        enddo
        u=ua*sin(phase)+ub*cos(2.e0_real_kind*phase)
        du=ua*cos(phase)-2.e0_real_kind*ub*sin(2.e0_real_kind*phase)
        ddu=-ua*sin(phase)-4.e0_real_kind*ub*cos(2.e0_real_kind*phase)
        mut=mu(0,0,0);mup=-0.1e0_real_kind*sin(phase+0.3e0_real_kind)
        kt=kappa(0,0,0);kp=0.07e0_real_kind*cos(phase+0.4e0_real_kind)
        td=-0.1e0_real_kind*sin(phase+0.2e0_real_kind)
        tdd=-0.1e0_real_kind*cos(phase+0.2e0_real_kind)
        entry=0
        do b=1,3
          do a=1,3
            entry=entry+1
            scalar=me6_scalar_direction(a,b,[0,0,0],hm,temperature,mu,spacing)
            if(.not.ieee_is_finite(scalar)) error stop 'nonfinite ME6 manufactured scalar'
            exact=wave(a)*wave(b)*(mut*tdd+mup*td)
            error(entry)=max(error(entry),abs(scalar-exact))
          enddo
        enddo
        do b=1,3
          g(:,b)=du*wave(b)
        enddo
        trace=sum(du*wave)
        stress=mut*(g+transpose(g))
        do a=1,3
          stress(a,a)=stress(a,a)-2.e0_real_kind*mut*trace/3.e0_real_kind
        enddo
        momentum=mut*(w2*ddu+wave*dot_product(wave,ddu)/3.e0_real_kind) &
          +mup*(w2*du+wave*dot_product(wave,du)/3.e0_real_kind)
        energy=dot_product(u,momentum)+sum(stress*g)+w2*(kt*tdd+kp*td)
        actual=0.e0_real_kind
        do a=1,3
          call me6_direction(a,[0,0,0],hm,velocity,temperature,mu,kappa,spacing,part)
          actual=actual+part
        enddo
        if(.not.all(ieee_is_finite(actual))) error stop 'nonfinite ME6 manufactured result'
        error(10:12)=max(error(10:12),abs(actual(1:3)-momentum))
        error(13)=max(error(13),abs(actual(4)-energy))
      enddo
    enddo
  end subroutine
end program
