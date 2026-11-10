module me6opti_operator
  use astr_precision, only: real_kind
  implicit none
  private
  public :: me6_coefficients,me6_half_gradient,me6_scalar_direction,me6_direction
  public :: me6_scalar_symbol,me6_viscous_symbol,me6_outer
  real(real_kind),parameter :: me6opti_straight_table(9,3)=reshape((/          &
    -3.e0_real_kind/1250.e0_real_kind,89141.e0_real_kind/4480000.e0_real_kind, &
    -49133.e0_real_kind/640000.e0_real_kind,            &
    411173.e0_real_kind/1920000.e0_real_kind,-174629.e0_real_kind/128000.e0_real_kind, &
    851641.e0_real_kind/640000.e0_real_kind,    &
    -282149.e0_real_kind/1920000.e0_real_kind,18413.e0_real_kind/640000.e0_real_kind, &
    -13877.e0_real_kind/4480000.e0_real_kind,   &
    459.e0_real_kind/4480000.e0_real_kind,-547.e0_real_kind/4480000.e0_real_kind, &
    -1289.e0_real_kind/640000.e0_real_kind,          &
    2703.e0_real_kind/640000.e0_real_kind,18379.e0_real_kind/384000.e0_real_kind, &
    -738047.e0_real_kind/640000.e0_real_kind,        &
    742461.e0_real_kind/640000.e0_real_kind,-820391.e0_real_kind/13440000.e0_real_kind, &
    9167.e0_real_kind/2240000.e0_real_kind,   &
    -3377.e0_real_kind/2240000.e0_real_kind,36157.e0_real_kind/4480000.e0_real_kind, &
    -6141.e0_real_kind/640000.e0_real_kind,      &
    -20593.e0_real_kind/640000.e0_real_kind,16367.e0_real_kind/128000.e0_real_kind, &
    -296029.e0_real_kind/1920000.e0_real_kind,    &
    -618391.e0_real_kind/640000.e0_real_kind,4737907.e0_real_kind/4480000.e0_real_kind,                      &
    -400637.e0_real_kind/13440000.e0_real_kind /),(/9,3/))
  real(real_kind),parameter :: me6opti_interp_table(9,3)=reshape((/           &
    -661.e0_real_kind/819200.e0_real_kind,263.e0_real_kind/512000.e0_real_kind, &
    31573.e0_real_kind/1024000.e0_real_kind,           &
    -91107.e0_real_kind/512000.e0_real_kind,302761.e0_real_kind/409600.e0_real_kind, &
    43093.e0_real_kind/102400.e0_real_kind,       &
    6429.e0_real_kind/1024000.e0_real_kind,-12349.e0_real_kind/512000.e0_real_kind, &
    21511.e0_real_kind/4096000.e0_real_kind,      &
    -7673.e0_real_kind/4096000.e0_real_kind,9179.e0_real_kind/512000.e0_real_kind, &
    -15959.e0_real_kind/204800.e0_real_kind,       &
    106337.e0_real_kind/512000.e0_real_kind,-165879.e0_real_kind/409600.e0_real_kind, &
    456421.e0_real_kind/512000.e0_real_kind,    &
    408037.e0_real_kind/1024000.e0_real_kind,-3357.e0_real_kind/102400.e0_real_kind, &
    8279.e0_real_kind/4096000.e0_real_kind,      &
    8279.e0_real_kind/4096000.e0_real_kind,-10273.e0_real_kind/512000.e0_real_kind, &
    92869.e0_real_kind/1024000.e0_real_kind,      &
    -126827.e0_real_kind/512000.e0_real_kind,37877.e0_real_kind/81920.e0_real_kind, &
    -337743.e0_real_kind/512000.e0_real_kind,     &
    1086701.e0_real_kind/1024000.e0_real_kind,166763.e0_real_kind/512000.e0_real_kind,                       &
    -59769.e0_real_kind/4096000.e0_real_kind /),(/9,3/))
  real(real_kind),parameter :: me6opti_penalty_table(9,3)=reshape((/          &
    -13.e0_real_kind/1000.e0_real_kind,13.e0_real_kind/125.e0_real_kind, &
    -91.e0_real_kind/250.e0_real_kind,91.e0_real_kind/125.e0_real_kind,          &
    -91.e0_real_kind/100.e0_real_kind,91.e0_real_kind/125.e0_real_kind, &
    -91.e0_real_kind/250.e0_real_kind,13.e0_real_kind/125.e0_real_kind,           &
    -13.e0_real_kind/1000.e0_real_kind,-1.e0_real_kind/2000.e0_real_kind, &
    1.e0_real_kind/250.e0_real_kind,-7.e0_real_kind/500.e0_real_kind,           &
    7.e0_real_kind/250.e0_real_kind,-7.e0_real_kind/200.e0_real_kind,7.e0_real_kind/250.e0_real_kind, &
    -7.e0_real_kind/500.e0_real_kind,               &
    1.e0_real_kind/250.e0_real_kind,-1.e0_real_kind/2000.e0_real_kind, &
    -1.e0_real_kind/2000.e0_real_kind,1.e0_real_kind/250.e0_real_kind,             &
    -7.e0_real_kind/500.e0_real_kind,7.e0_real_kind/250.e0_real_kind, &
    -7.e0_real_kind/200.e0_real_kind,7.e0_real_kind/250.e0_real_kind,               &
    -7.e0_real_kind/500.e0_real_kind,1.e0_real_kind/250.e0_real_kind,-1.e0_real_kind/2000.e0_real_kind /),(/9,3/))


  real(real_kind),parameter :: central6(3)=[0.75e0_real_kind,-0.15e0_real_kind,1.e0_real_kind/60.e0_real_kind]
  real(real_kind),parameter :: me6_outer(3)=[75.e0_real_kind/64.e0_real_kind, &
    -25.e0_real_kind/384.e0_real_kind,3.e0_real_kind/640.e0_real_kind]
contains
  pure subroutine me6_coefficients(half,p,interpolation,straight,penalty)
    integer,intent(in) :: half,p
    real(real_kind),intent(out) :: interpolation,straight,penalty
    integer :: column,row,orientation
    column=(abs(half)+1)/2
    orientation=sign(1,half)
    row=5+orientation*p
    interpolation=me6opti_interp_table(row,column)
    straight=real(orientation,real_kind)*me6opti_straight_table(row,column)
    ! Approved correction: the negative-half penalty is odd under reflection.
    penalty=real(orientation,real_kind)*me6opti_penalty_table(row,column)
  end subroutine

  pure function me6_half_gradient(axis,direction,node,half,hm,field,spacing) result(gradient)
    integer,intent(in) :: axis,direction,node(3),half,hm
    real(real_kind),intent(in) :: field(-hm:,-hm:,-hm:),spacing(3)
    real(real_kind) :: gradient,ci,cd,cp,transverse
    integer :: p,l,point(3),plus(3),minus(3)
    gradient=0.e0_real_kind
    do p=-4,4
      point=node;point(axis)=node(axis)+p
      call me6_coefficients(half,p,ci,cd,cp)
      if(axis==direction) then
        gradient=gradient+cd*field(point(1),point(2),point(3))/spacing(axis)
      else
        transverse=0.e0_real_kind
        do l=1,3
          plus=point;minus=point
          plus(direction)=point(direction)+l;minus(direction)=point(direction)-l
          transverse=transverse+central6(l)*(field(plus(1),plus(2),plus(3))- &
            field(minus(1),minus(2),minus(3)))/spacing(direction)
        enddo
        gradient=gradient+ci*transverse+cp*field(point(1),point(2),point(3))/spacing(axis)
      endif
    enddo
  end function

  pure function me6_scalar_direction(axis,direction,node,hm,field,coefficient,spacing) result(divergence)
    integer,intent(in) :: axis,direction,node(3),hm
    real(real_kind),intent(in) :: field(-hm:,-hm:,-hm:),coefficient(-hm:,-hm:,-hm:),spacing(3)
    real(real_kind) :: divergence,ci,cd,cp,midpoint,flux(2),gradient
    integer :: h,side,half,p,point(3)
    divergence=0.e0_real_kind
    do h=1,3
      do side=1,2
        half=(2*side-3)*(2*h-1)
        midpoint=0.e0_real_kind
        do p=-4,4
          point=node;point(axis)=node(axis)+p
          call me6_coefficients(half,p,ci,cd,cp)
          midpoint=midpoint+ci*coefficient(point(1),point(2),point(3))
        enddo
        gradient=me6_half_gradient(axis,direction,node,half,hm,field,spacing)
        flux(side)=midpoint*gradient
      enddo
      divergence=divergence+me6_outer(h)*(flux(2)-flux(1))/spacing(axis)
    enddo
  end function

  pure subroutine me6_direction(axis,node,hm,velocity,temperature,viscosity,conductivity,spacing,divergence)
    integer,intent(in) :: axis,node(3),hm
    real(real_kind),intent(in) :: velocity(-hm:,-hm:,-hm:,:),temperature(-hm:,-hm:,-hm:)
    real(real_kind),intent(in) :: viscosity(-hm:,-hm:,-hm:),conductivity(-hm:,-hm:,-hm:),spacing(3)
    real(real_kind),intent(out) :: divergence(4)
    real(real_kind) :: ci,cd,cp,mu,kappa,mid_velocity(3),gradient(3,3),stress(3),flux(4,2),trace
    integer :: h,side,half,p,point(3),c,d
    divergence=0.e0_real_kind
    do h=1,3
      do side=1,2
        half=(2*side-3)*(2*h-1)
        mu=0.e0_real_kind;kappa=0.e0_real_kind;mid_velocity=0.e0_real_kind
        do p=-4,4
          point=node;point(axis)=node(axis)+p
          call me6_coefficients(half,p,ci,cd,cp)
          mu=mu+ci*viscosity(point(1),point(2),point(3))
          kappa=kappa+ci*conductivity(point(1),point(2),point(3))
          mid_velocity=mid_velocity+ci*velocity(point(1),point(2),point(3),:)
        enddo
        do d=1,3
          do c=1,3
            gradient(c,d)=me6_half_gradient(axis,d,node,half,hm,velocity(:,:,:,c),spacing)
          enddo
        enddo
        trace=gradient(1,1)+gradient(2,2)+gradient(3,3)
        stress=mu*(gradient(:,axis)+gradient(axis,:))
        stress(axis)=stress(axis)-2.e0_real_kind*mu*trace/3.e0_real_kind
        flux(1:3,side)=stress
        flux(4,side)=kappa*me6_half_gradient(axis,axis,node,half,hm,temperature,spacing) &
          +dot_product(stress,mid_velocity)
      enddo
      divergence=divergence+me6_outer(h)*(flux(:,2)-flux(:,1))/spacing(axis)
    enddo
  end subroutine

  pure function me6_scalar_symbol(axis,direction,theta,spacing) result(symbol)
    integer,intent(in) :: axis,direction
    real(real_kind),intent(in) :: theta(3),spacing(3)
    complex(real_kind) :: symbol,gradient(2),phase,dy
    real(real_kind) :: ci,cd,cp
    integer :: h,side,p,half,l
    dy=cmplx(0.e0_real_kind,0.e0_real_kind,real_kind)
    do l=1,3
      dy=dy+cmplx(0.e0_real_kind,2.e0_real_kind*central6(l)*sin(l*theta(direction))/spacing(direction),real_kind)
    enddo
    symbol=cmplx(0.e0_real_kind,0.e0_real_kind,real_kind)
    do h=1,3
      gradient=cmplx(0.e0_real_kind,0.e0_real_kind,real_kind)
      do side=1,2
        half=(2*side-3)*(2*h-1)
        do p=-4,4
          phase=exp(cmplx(0.e0_real_kind,p*theta(axis),real_kind))
          call me6_coefficients(half,p,ci,cd,cp)
          if(axis==direction) then
            gradient(side)=gradient(side)+cd*phase/spacing(axis)
          else
            gradient(side)=gradient(side)+(ci*dy+cp/spacing(axis))*phase
          endif
        enddo
      enddo
      symbol=symbol+me6_outer(h)*(gradient(2)-gradient(1))/spacing(axis)
    enddo
  end function

  pure subroutine me6_viscous_symbol(theta,spacing,momentum,thermal)
    real(real_kind),intent(in) :: theta(3),spacing(3)
    complex(real_kind),intent(out) :: momentum(3,3),thermal
    complex(real_kind) :: derivatives(3,3)
    integer :: a,b
    do a=1,3
      do b=1,3
        derivatives(a,b)=me6_scalar_symbol(a,b,theta,spacing)
      enddo
    enddo
    thermal=derivatives(1,1)+derivatives(2,2)+derivatives(3,3)
    do a=1,3
      do b=1,3
        if(a==b) then
          momentum(a,b)=thermal+derivatives(a,a)/3.e0_real_kind
        else
          momentum(a,b)=derivatives(b,a)-2.e0_real_kind*derivatives(a,b)/3.e0_real_kind
        endif
      enddo
    enddo
  end subroutine
end module
