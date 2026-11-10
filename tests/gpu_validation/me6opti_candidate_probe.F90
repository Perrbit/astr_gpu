program me6opti_candidate_probe
  use astr_precision, only: real_kind
  use me6opti_operator
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  real(real_kind),parameter :: pi=acos(-1.e0_real_kind)
  real(real_kind) :: errors(13,3),orders(13),roundoff1(13),roundoff2(13)
  integer :: level
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
