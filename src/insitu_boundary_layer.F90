module insitu_boundary_layer
  use astr_precision, only: real_kind
  use insitu_time_integral, only: clipped_trapezoid
  use ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: wall_transport,wall_stencil_transport,wall_reference,wall_reference_result
  public :: nonreacting_wall_material
  public :: boundary_layer_thickness,thickness_result
  public :: boundary_layer_edge,thickness_reynolds
  public :: resolve_recovery_temperature
  public :: wall_grid_spacings,wall_distance_plus
  public :: wall_scaled_quantity
  public :: cartesian_block_axes,cartesian_axis_direction,cartesian_point
  type :: wall_reference_result
    logical :: scale_available=.false.,stanton_available=.false.,cf_available=.false.,cq_available=.false.
    real(real_kind) :: shear_magnitude=0,u_tau=0,nu=0,length=0,cf=0,cq=0,stanton=0
  end type
  type :: thickness_result
    logical :: delta_available=.false.,explicit_available=.false.,multiple_crossings=.false.
    logical :: shape99_available=.false.,shape_explicit_available=.false.
    integer :: threshold_components=0
    real(real_kind) :: delta99=0,displacement99=0,momentum99=0,shape99=0
    real(real_kind) :: upper=0,displacement_explicit=0,momentum_explicit=0,shape_explicit=0
  end type
contains
#ifdef _CUDA
  attributes(host,device) logical function cartesian_point(point,reference) result(ok)
#else
  logical function cartesian_point(point,reference) result(ok)
#endif
    real(real_kind),intent(in) :: point(3),reference(3)
    ok=all(abs(point)<=huge(point)).and.all(point==reference)
  end function

  subroutine cartesian_block_axes(coordinates,cells,halo,axes,ok)
    integer,intent(in) :: cells(3),halo
    real(real_kind),intent(in) :: coordinates(-halo:,-halo:,-halo:,:)
    real(real_kind),intent(out) :: axes(sum(cells)+3)
    logical,intent(out) :: ok
    integer :: i,j,k,nx,ny
    real(real_kind) :: point(3),reference(3)
    nx=cells(1)+1;ny=cells(2)+1
    axes=[coordinates(0:cells(1),0,0,1),coordinates(0,0:cells(2),0,2),coordinates(0,0,0:cells(3),3)]
    ok=.true.
    do k=0,cells(3)
    do j=0,cells(2)
    do i=0,cells(1)
      point=coordinates(i,j,k,:);reference=[axes(i+1),axes(nx+j+1),axes(nx+ny+k+1)]
      if(.not.cartesian_point(point,reference)) then
        ok=.false.;return
      endif
    enddo
    enddo
    enddo
  end subroutine

  integer function cartesian_axis_direction(axis) result(direction)
    real(real_kind),intent(in) :: axis(:)
    integer :: n
    direction=0;n=size(axis)
    if(n<2.or..not.all(ieee_is_finite(axis))) return
    if(all(axis(2:)>axis(:n-1))) direction=1
    if(all(axis(2:)<axis(:n-1))) direction=-1
  end function

#ifdef _CUDA
  attributes(host,device) subroutine nonreacting_wall_material(temperature,nondimen,tempconst,tempconst1, &
    reynolds,prandtl,const5,cp,need_conductivity,mu,kappa,ok)
#else
  subroutine nonreacting_wall_material(temperature,nondimen,tempconst,tempconst1, &
    reynolds,prandtl,const5,cp,need_conductivity,mu,kappa,ok)
#endif
    real(real_kind),intent(in) :: temperature,tempconst,tempconst1,reynolds,prandtl,const5,cp
    logical,intent(in) :: nondimen,need_conductivity
    real(real_kind),intent(out) :: mu,kappa
    logical,intent(out) :: ok
    real(real_kind) :: tnondim
    ok=.false.;mu=0;kappa=0
    if(.not.(temperature>0.e0_real_kind.and.temperature<=huge(temperature))) return
    if(nondimen) then
      if(.not.all(abs([tempconst,tempconst1,reynolds])<=huge(temperature))) return
      if(temperature+tempconst<=0.e0_real_kind.or.tempconst1<=0.or.reynolds<=0) return
      mu=temperature*sqrt(temperature)*tempconst1/(temperature+tempconst)/reynolds
    else
      tnondim=temperature/273.15e0_real_kind
      mu=1.716e-5_real_kind*tnondim*sqrt(tnondim)*(273.15e0_real_kind+110.4e0_real_kind)/(temperature+110.4e0_real_kind)
    endif
    if(.not.(mu>0.e0_real_kind.and.mu<=huge(mu))) return
    if(need_conductivity) then
      if(.not.(prandtl>0.e0_real_kind.and.prandtl<=huge(mu))) return
      if(nondimen) then
        if(.not.(const5>0.e0_real_kind.and.const5<=huge(mu))) return
        kappa=(mu/prandtl)/const5
      else
        if(.not.(cp>0.e0_real_kind.and.cp<=huge(mu))) return
        kappa=cp*mu/prandtl
      endif
      if(.not.(kappa>0.e0_real_kind.and.kappa<=huge(mu))) return
    endif
    ok=.true.
  end subroutine

#ifdef _CUDA
  attributes(host,device) subroutine wall_stencil_transport(stencil,offsets,weights,metrics,mu,kappa,normal,shear,heat,ok)
#else
  subroutine wall_stencil_transport(stencil,offsets,weights,metrics,mu,kappa,normal,shear,heat,ok)
#endif
    real(real_kind),intent(in) :: stencil(4,7,3),weights(7,0:6),metrics(3,3),mu,kappa,normal(3)
    integer,intent(in) :: offsets(3)
    real(real_kind),intent(out) :: shear(3),heat
    logical,intent(out) :: ok
    real(real_kind) :: directional(4,3),gradient(3,3),thermal(3)
    integer :: c,d
    ok=.false.;shear=0;heat=0
    if(any(offsets<0).or.any(offsets>6)) return
    do d=1,3
      do c=1,4
        directional(c,d)=sum(weights(:,offsets(d))*stencil(c,:,d))
      enddo
    enddo
    do d=1,3
      do c=1,3
        gradient(c,d)=sum(directional(c,:)*metrics(:,d))
      enddo
      thermal(d)=sum(directional(4,:)*metrics(:,d))
    enddo
    call wall_transport(mu,kappa,gradient,thermal,normal,shear,heat,ok)
  end subroutine

#ifdef _CUDA
  attributes(host,device) subroutine wall_transport(mu,kappa,gradient,thermal,normal,shear,heat,ok)
#else
  subroutine wall_transport(mu,kappa,gradient,thermal,normal,shear,heat,ok)
#endif
    real(real_kind),intent(in) :: mu,kappa,gradient(3,3),thermal(3),normal(3)
    real(real_kind),intent(out) :: shear(3),heat
    logical,intent(out) :: ok
    real(real_kind) :: unit(3),stress(3,3),traction(3),scale,trace
    integer :: d
    ok=.false.;shear=0;heat=0
    if(.not.all(abs(gradient)<=huge(mu)).or..not.all(abs(thermal)<=huge(mu)).or. &
       .not.all(abs(normal)<=huge(mu))) return
    if(.not.(mu>=0.e0_real_kind.and.mu<=huge(mu).and.kappa>=0.e0_real_kind.and.kappa<=huge(mu))) return
    scale=maxval(abs(normal))
    if(scale==0.e0_real_kind) return
    unit=normal/scale;unit=unit/sqrt(sum(unit*unit))
    trace=gradient(1,1)+gradient(2,2)+gradient(3,3)
    stress=mu*(gradient+transpose(gradient))
    do d=1,3
      stress(d,d)=stress(d,d)-(2.e0_real_kind/3.e0_real_kind)*mu*trace
    enddo
    do d=1,3
      traction(d)=sum(stress(d,:)*unit)
    enddo
    shear=traction-unit*sum(unit*traction)
    heat=kappa*sum(thermal*unit)
    ok=all(abs(shear)<=huge(mu)).and.abs(heat)<=huge(mu)
  end subroutine

  ! Flat-plate frame: +x is streamwise. Inputs share the same time/span weights.
  subroutine wall_reference(shear,rho,mu,tw,heat,rho_inf,u_inf,result,ok,cp_inf,recovery_temperature)
    real(real_kind),intent(in) :: shear(3),rho,mu,tw,heat,rho_inf,u_inf
    real(real_kind),optional,intent(in) :: cp_inf,recovery_temperature
    type(wall_reference_result),intent(out) :: result
    logical,intent(out) :: ok
    real(real_kind) :: scale,magnitude,utau,nu,length,denominator,stanton
    logical :: reference_valid,valid
    result=wall_reference_result();ok=.false.
    if(.not.all(ieee_is_finite([shear,rho,mu,tw,heat]))) return
    if(min(rho,mu,tw)<=0.e0_real_kind) return
    if(present(cp_inf).neqv.present(recovery_temperature)) return
    reference_valid=all(ieee_is_finite([rho_inf,u_inf])).and.min(rho_inf,u_inf)>0
    if(reference_valid) then
      call product_ratio(shear(1),[2.e0_real_kind],[rho_inf,u_inf,u_inf],result%cf,result%cf_available)
      call product_ratio(heat,[real(real_kind)::],[rho_inf,u_inf,u_inf,u_inf],result%cq,result%cq_available)
    endif
    scale=maxval(abs(shear))
    if(scale>0.e0_real_kind) then
      magnitude=scale*sqrt(sum((shear/scale)**2))
      utau=sqrt(magnitude)/sqrt(rho);nu=mu/rho
      if(all(ieee_is_finite([magnitude,utau,nu])).and.utau>0.e0_real_kind.and.nu>0.e0_real_kind) then
        length=nu/utau
        if(ieee_is_finite(length).and.length>0.e0_real_kind) then
          result%scale_available=.true.;result%shear_magnitude=magnitude
          result%u_tau=utau;result%nu=nu;result%length=length
        endif
      endif
    endif
    if(present(cp_inf).and.present(recovery_temperature)) then
      if(reference_valid.and.all(ieee_is_finite([cp_inf,recovery_temperature])).and. &
        min(cp_inf,recovery_temperature)>0.e0_real_kind) then
        denominator=recovery_temperature-tw
        if(denominator/=0.e0_real_kind.and.ieee_is_finite(denominator)) then
          call product_ratio(sign(1.e0_real_kind,denominator)*heat,[real(real_kind)::], &
            [rho_inf,u_inf,cp_inf,abs(denominator)],stanton,valid)
          if(valid) then
            result%stanton_available=.true.;result%stanton=stanton
          endif
        endif
      endif
    endif
    ok=.true.
  end subroutine

  subroutine resolve_recovery_temperature(mode,value,tinf,gamma,mach,temperature,ok)
    integer,intent(in) :: mode
    real(real_kind),intent(in) :: value,tinf,gamma,mach
    real(real_kind),intent(out) :: temperature
    logical,intent(out) :: ok
    real(real_kind) :: increment
    temperature=0;ok=.false.
    if(.not.ieee_is_finite(value)) return
    if(mode==1) then
      if(value<=0) return
      temperature=value;ok=.true.;return
    endif
    if(mode/=2.or..not.all(ieee_is_finite([tinf,gamma,mach]))) return
    if(value<0.or.tinf<=0.or.gamma<=1.or.mach<0) return
    call product_ratio(tinf,[.5e0_real_kind,value,gamma-1.e0_real_kind,mach,mach], &
      [real(real_kind)::],increment,ok)
    if(.not.ok) return
    if(increment>huge(tinf)-tinf) then
      ok=.false.;return
    endif
    temperature=tinf+increment
    ok=ieee_is_finite(temperature).and.temperature>0
  end subroutine

  ! Report actual adjacent cells, not a nominal spacing on a stretched grid.
  subroutine wall_grid_spacings(x,y,z,bounds,periodic,distances,available,ok)
    real(real_kind),intent(in) :: x(0:),y(0:),z(0:)
    integer,intent(in) :: bounds(2,2)
    logical,intent(in) :: periodic(2)
    real(real_kind),intent(out) :: distances(5,bounds(2,1)-bounds(1,1)+1)
    logical,intent(out) :: available(5,bounds(2,1)-bounds(1,1)+1),ok
    real(real_kind) :: minimum,maximum,delta
    integer :: nx,nz,i,j,a,b,k,node
    distances=0;available=.false.;ok=.false.
    nx=size(x)-1;nz=size(z)-1
    if(min(nx,nz)<1.or.size(y)<2.or.any(bounds(1,:)<0)) return
    if(any(bounds(2,:)<bounds(1,:)).or.bounds(2,1)>nx.or.bounds(2,2)>nz) return
    if(.not.all(ieee_is_finite(x)).or..not.all(ieee_is_finite(y)).or..not.all(ieee_is_finite(z))) return
    if(.not.all(x(1:)>x(:nx-1)).and..not.all(x(1:)<x(:nx-1))) return
    if(.not.all(z(1:)>z(:nz-1)).and..not.all(z(1:)<z(:nz-1))) return
    if(y(1)==y(0)) return
    a=bounds(1,2);b=bounds(2,2)
    if(a==b) then
      ! A single selected node uses its incident cells, including a periodic seam.
      if(periodic(2)) then
        a=modulo(a,nz)
        k=modulo(a-1,nz)
        minimum=abs(z(k+1)-z(k));maximum=minimum
        delta=abs(z(a+1)-z(a));minimum=min(minimum,delta);maximum=max(maximum,delta)
      else
        a=max(0,a-1);b=min(nz,b+1)
        minimum=minval(abs(z(a+1:b)-z(a:b-1)));maximum=maxval(abs(z(a+1:b)-z(a:b-1)))
      endif
    else
      minimum=minval(abs(z(a+1:b)-z(a:b-1)));maximum=maxval(abs(z(a+1:b)-z(a:b-1)))
    endif
    do i=bounds(1,1),bounds(2,1)
      j=i-bounds(1,1)+1
      distances(1,j)=abs(y(1)-y(0));distances(4:5,j)=[minimum,maximum]
      available([1,4,5],j)=.true.;node=i
      if(periodic(1)) node=modulo(node,nx)
      if(node>0.or.periodic(1)) then
        k=node-1
        if(periodic(1)) k=modulo(k,nx)
        distances(2,j)=abs(x(k+1)-x(k));available(2,j)=.true.
      endif
      if(node<nx) then
        distances(3,j)=abs(x(node+1)-x(node));available(3,j)=.true.
      endif
    enddo
    ok=all(ieee_is_finite(distances))
  end subroutine

  subroutine wall_distance_plus(distance,length,value,ok)
    real(real_kind),intent(in) :: distance,length
    real(real_kind),intent(out) :: value
    logical,intent(out) :: ok
    call product_ratio(distance,[real(real_kind)::],[length],value,ok)
    ok=ok.and.distance>=0
  end subroutine

  subroutine wall_scaled_quantity(quantity,denominators,value,ok)
    real(real_kind),intent(in) :: quantity,denominators(:)
    real(real_kind),intent(out) :: value
    logical,intent(out) :: ok
    call product_ratio(quantity,[real(real_kind)::],denominators,value,ok)
  end subroutine

  ! Accumulate binary exponents so intermediate products cannot overflow a finite ratio.
  subroutine product_ratio(numerator,factors,denominators,value,ok)
    real(real_kind),intent(in) :: numerator,factors(:),denominators(:)
    real(real_kind),intent(out) :: value
    logical,intent(out) :: ok
    real(real_kind) :: mantissa
    integer :: power,shift,i
    value=0;ok=.false.
    if(.not.ieee_is_finite(numerator).or..not.all(ieee_is_finite(factors)).or. &
      .not.all(ieee_is_finite(denominators))) return
    if(any(factors<0).or.any(denominators<=0)) return
    if(numerator==0.or.any(factors==0)) then
      ok=.true.;return
    endif
    mantissa=fraction(abs(numerator));power=exponent(abs(numerator))
    do i=1,size(factors)
      mantissa=mantissa*fraction(factors(i));power=power+exponent(factors(i))
      shift=exponent(mantissa);mantissa=fraction(mantissa);power=power+shift
    enddo
    do i=1,size(denominators)
      mantissa=mantissa/fraction(denominators(i));power=power-exponent(denominators(i))
      shift=exponent(mantissa);mantissa=fraction(mantissa);power=power+shift
    enddo
    if(power>maxexponent(value)) return
    if(power==maxexponent(value).and.mantissa>fraction(huge(value))) return
    if(power<minexponent(value)-digits(value)) then
      ok=.true.;return
    endif
    ! Two representable factors avoid constructing 2**maxexponent first.
    shift=power/2
    value=sign(scale(scale(mantissa,shift),power-shift),numerator)
    ok=ieee_is_finite(value)
  end subroutine

  subroutine boundary_layer_edge(y,rho,velocity,temperature,pressure,band,edge,ok)
    real(real_kind),intent(in) :: y(:),rho(:),velocity(:),temperature(:),pressure(:)
    integer,intent(in) :: band(2)
    real(real_kind),intent(out) :: edge(4)
    logical,intent(out) :: ok
    real(real_kind) :: weights(band(2)-band(1)+1),mass(band(2)-band(1)+1),width,density_scale
    integer :: first,last,j,n
    ok=.false.;edge=0;first=band(1);last=band(2);n=size(y)
    if(min(size(rho),size(velocity),size(temperature),size(pressure))/=n) return
    if(max(size(rho),size(velocity),size(temperature),size(pressure))/=n) return
    if(first<1.or.last<first.or.last>n) return
    if(.not.all(ieee_is_finite(y(first:last))).or..not.all(ieee_is_finite(rho(first:last))).or. &
      .not.all(ieee_is_finite(velocity(first:last))).or..not.all(ieee_is_finite(temperature(first:last))).or. &
      .not.all(ieee_is_finite(pressure(first:last)))) return
    if(any(rho(first:last)<=0).or.any(temperature(first:last)<=0).or.any(pressure(first:last)<=0)) return
    if(first==last) then
      edge=[rho(first),velocity(first),temperature(first),pressure(first)]
    else
      if(any(y(first+1:last)<=y(first:last-1))) return
      width=y(last)-y(first)
      if(.not.ieee_is_finite(width)) return
      weights=0
      do j=first,last-1
        weights(j-first+1)=weights(j-first+1)+.5e0_real_kind*((y(j+1)-y(j))/width)
        weights(j-first+2)=weights(j-first+2)+.5e0_real_kind*((y(j+1)-y(j))/width)
      enddo
      edge(1)=sum(weights*rho(first:last))
      edge(3)=sum(weights*temperature(first:last));edge(4)=sum(weights*pressure(first:last))
      density_scale=maxval(rho(first:last));mass=weights*(rho(first:last)/density_scale)
      edge(2)=sum((mass/sum(mass))*velocity(first:last))
    endif
    ok=all(ieee_is_finite(edge)).and.all(edge>0)
  end subroutine

  subroutine thickness_reynolds(rho_edge,u_edge,theta,mu_edge,value,ok)
    real(real_kind),intent(in) :: rho_edge,u_edge,theta,mu_edge
    real(real_kind),intent(out) :: value
    logical,intent(out) :: ok
    call product_ratio(theta,[rho_edge,u_edge],[mu_edge],value,ok)
  end subroutine

  ! Edge states are explicit inputs, not maxima or automatically selected top values.
  subroutine boundary_layer_thickness(y,rho,u,rho_edge,u_edge,edge_band,result,cumulative_d,cumulative_t,ok,explicit_upper)
    real(real_kind),intent(in) :: y(:),rho(:),u(:),rho_edge,u_edge
    integer,intent(in) :: edge_band(2)
    type(thickness_result),intent(out) :: result
    real(real_kind),intent(out) :: cumulative_d(size(y)),cumulative_t(size(y))
    logical,intent(out) :: ok
    real(real_kind),optional,intent(in) :: explicit_upper
    real(real_kind) :: integrands(2,size(y)),increments(2),duration,threshold,fraction,limit,scale
    integer :: n,j,left,first,last
    logical :: valid,above,previous
    result=thickness_result();cumulative_d=0;cumulative_t=0;ok=.false.;n=size(y)
    if(n<2.or.size(rho)/=n.or.size(u)/=n) return
    if(.not.all(ieee_is_finite(y)).or..not.all(ieee_is_finite(rho)).or..not.all(ieee_is_finite(u))) return
    if(.not.all(ieee_is_finite([rho_edge,u_edge]))) return
    if(any(y(2:)<=y(:n-1)).or.any(rho<=0).or.min(rho_edge,u_edge)<=0) return
    first=edge_band(1);last=edge_band(2)
    if(first<1.or.last<first.or.last>n) return
    integrands(1,:)=(rho/rho_edge)*(u/u_edge)
    integrands(2,:)=integrands(1,:)*(1.e0_real_kind-u/u_edge)
    integrands(1,:)=1.e0_real_kind-integrands(1,:)
    if(.not.all(ieee_is_finite(integrands))) return
    do j=2,n
      call clipped_trapezoid(y(j-1),y(j),integrands(:,j-1),integrands(:,j), &
        y(j-1),y(j),increments,duration,valid)
      if(.not.valid) return
      cumulative_d(j)=cumulative_d(j-1)+increments(1)
      cumulative_t(j)=cumulative_t(j-1)+increments(2)
    enddo
    if(.not.all(ieee_is_finite(cumulative_d)).or..not.all(ieee_is_finite(cumulative_t))) return
    threshold=.99e0_real_kind*u_edge;previous=.false.
    do j=1,last
      above=u(j)>=threshold
      if(above.and..not.previous) result%threshold_components=result%threshold_components+1
      previous=above
    enddo
    result%multiple_crossings=result%threshold_components>1
    if(all(u(first:last)>=threshold)) then
      left=first
      do while(left>1)
        if(u(left-1)<threshold) exit
        left=left-1
      enddo
      if(left>1) then
        scale=max(abs(threshold),abs(u(left-1)),abs(u(left)))
        fraction=(threshold/scale-u(left-1)/scale)/(u(left)/scale-u(left-1)/scale)
        limit=(1.e0_real_kind-fraction)*y(left-1)+fraction*y(left)
        result%delta99=limit-y(1)
        call integrate_to(limit,increments,valid)
        if(.not.valid) return
        result%delta_available=.true.
        result%displacement99=increments(1);result%momentum99=increments(2)
        call shape_factor(increments,result%shape99,result%shape99_available)
      endif
    endif
    if(present(explicit_upper)) then
      result%upper=explicit_upper
      call integrate_to(explicit_upper,increments,valid)
      if(valid) then
        result%explicit_available=.true.
        result%displacement_explicit=increments(1);result%momentum_explicit=increments(2)
        call shape_factor(increments,result%shape_explicit,result%shape_explicit_available)
      endif
    endif
    ok=.true.
  contains
    subroutine integrate_to(upper,values,valid)
      real(real_kind),intent(in) :: upper
      real(real_kind),intent(out) :: values(2)
      logical,intent(out) :: valid
      real(real_kind) :: partial(2),covered
      integer :: i
      valid=.false.;values=0
      if(.not.ieee_is_finite(upper)) return
      if(upper<y(1).or.upper>y(n)) return
      if(upper==y(1)) then
        valid=.true.;return
      endif
      do i=2,n
        if(upper>y(i)) cycle
        call clipped_trapezoid(y(i-1),y(i),integrands(:,i-1),integrands(:,i), &
          y(i-1),upper,partial,covered,valid)
        if(.not.valid) return
        values=[cumulative_d(i-1),cumulative_t(i-1)]+partial
        valid=all(ieee_is_finite(values));return
      enddo
    end subroutine
  end subroutine

  subroutine shape_factor(integrals,value,ok)
    real(real_kind),intent(in) :: integrals(2)
    real(real_kind),intent(out) :: value
    logical,intent(out) :: ok
    real(real_kind) :: candidate
    value=0;ok=.false.
    if(integrals(2)==0.e0_real_kind) return
    candidate=integrals(1)/integrals(2)
    if(.not.ieee_is_finite(candidate)) return
    value=candidate;ok=.true.
  end subroutine
end module
