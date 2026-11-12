module keep_operator
  use astr_precision, only: real_kind
  implicit none
  private
  public :: keep_two_point_flux5,keep_explicit_line,keep_explicit_volume
  public :: keep_split_plan,build_keep_split_plan,keep_split_variables,keep_monomial,keep_split_accumulate
  integer,parameter :: variables_count=9,max_split_terms=294
  ! Variables: density, u/v/w, internal energy, pressure, mapped area x/y/z.
  type :: keep_split_plan
    integer :: terms=0,bases=0
    integer :: left(variables_count,max_split_terms)=0,right(variables_count,max_split_terms)=0
    integer :: component(max_split_terms)=0,derivative(max_split_terms)=0
    real(real_kind) :: weight(max_split_terms)=0.e0_real_kind
  end type
contains
  subroutine build_keep_split_plan(plan)
    type(keep_split_plan),intent(out) :: plan
    integer :: c,m
    plan=keep_split_plan()
    do c=1,3
      call averaged_product([1,1+c,6+c],1,1.e0_real_kind)
      do m=1,3
        call averaged_product([1,1+c,6+c,1+m],1+m,1.e0_real_kind)
      enddo
      call averaged_product([6,6+c],1+c,1.e0_real_kind)
      call averaged_product([1,1+c,6+c,5],5,1.e0_real_kind)
      do m=1,3
        call kinetic_product([1,1+c,6+c],1+m)
      enddo
      call add_term([1+c,6+c],[6],5,0.5e0_real_kind)
      call add_term([1+c],[6,6+c],5,0.5e0_real_kind)
      call add_term([6,6+c],[1+c],5,0.5e0_real_kind)
      call add_term([6],[1+c,6+c],5,0.5e0_real_kind)
    enddo
  contains
    ! Expand 2*sum_j D_ij*product((a_i+a_j)/2); the empty derivative is D(1)=0.
    subroutine averaged_product(factors,target,factor)
      integer,intent(in) :: factors(:),target
      real(real_kind),intent(in) :: factor
      integer :: mask,j,left(size(factors)),right(size(factors)),nl,nr
      do mask=1,2**size(factors)-1
        nl=0;nr=0
        do j=1,size(factors)
          if(btest(mask,j-1)) then
            nr=nr+1;right(nr)=factors(j)
          else
            nl=nl+1;left(nl)=factors(j)
          endif
        enddo
        call add_term(left(:nl),right(:nr),target,factor/real(2**(size(factors)-1),real_kind))
      enddo
    end subroutine
    ! Cross-point kinetic energy leaves one velocity on each side of D.
    subroutine kinetic_product(factors,velocity)
      integer,intent(in) :: factors(:),velocity
      integer :: mask,j,left(size(factors)+1),right(size(factors)+1),nl,nr
      do mask=0,2**size(factors)-1
        nl=1;nr=1
        left(1)=velocity;right(1)=velocity
        do j=1,size(factors)
          if(btest(mask,j-1)) then
            nr=nr+1;right(nr)=factors(j)
          else
            nl=nl+1;left(nl)=factors(j)
          endif
        enddo
        call add_term(left(:nl),right(:nr),5,1.e0_real_kind/real(2**size(factors),real_kind))
      enddo
    end subroutine
    subroutine add_term(left,right,target,weight)
      integer,intent(in) :: left(:),right(:),target
      real(real_kind),intent(in) :: weight
      integer :: powers(variables_count),local_powers(variables_count),j,b
      powers=0;local_powers=0
      do j=1,size(right)
        powers(right(j))=powers(right(j))+1
      enddo
      do j=1,size(left)
        local_powers(left(j))=local_powers(left(j))+1
      enddo
      do b=1,plan%bases
        if(all(plan%right(:,b)==powers)) exit
      enddo
      if(b>plan%bases) then
        plan%bases=b
        plan%right(:,b)=powers
      endif
      plan%terms=plan%terms+1
      j=plan%terms
      plan%left(:,j)=local_powers
      plan%component(j)=target
      plan%derivative(j)=b
      plan%weight(j)=weight
    end subroutine
  end subroutine

  pure subroutine keep_split_variables(rho,velocity,pressure,metric,gamma,variables)
    real(real_kind),intent(in) :: rho,velocity(3),pressure,metric(3),gamma
    real(real_kind),intent(out) :: variables(9)
    variables(1)=rho
    variables(2:4)=velocity
    variables(5)=pressure/rho/(gamma-1.e0_real_kind)
    variables(6)=pressure
    variables(7:9)=metric
  end subroutine

  pure function keep_monomial(variables,powers) result(value)
    real(real_kind),intent(in) :: variables(9)
    integer,intent(in) :: powers(9)
    real(real_kind) :: value
    integer :: j,k
    value=1.e0_real_kind
    do j=1,9
      do k=1,powers(j)
        value=value*variables(j)
      enddo
    enddo
  end function

  pure subroutine keep_split_accumulate(plan,variables,derivatives,divergence)
    type(keep_split_plan),intent(in) :: plan
    real(real_kind),intent(in) :: variables(9),derivatives(:)
    real(real_kind),intent(out) :: divergence(5)
    integer :: t,c
    divergence=0.e0_real_kind
    do t=1,plan%terms
      c=plan%component(t)
      divergence(c)=divergence(c)+plan%weight(t)*keep_monomial(variables,plan%left(:,t))* &
        derivatives(plan%derivative(t))
    enddo
  end subroutine

  pure subroutine keep_two_point_flux5(rho_l,vel_l,p_l,metric_l, &
      rho_r,vel_r,p_r,metric_r,gamma,flux)
    real(real_kind),intent(in) :: rho_l,rho_r,p_l,p_r,gamma
    real(real_kind),intent(in) :: vel_l(3),vel_r(3),metric_l(3),metric_r(3)
    real(real_kind),intent(out) :: flux(5)
    real(real_kind) :: mass,kinetic,internal,velocity(3),metric(3),pressure_work(3)
    ! Original arithmetic-average KEEP, Kuya--Kawai 2021, Eqs. (69)--(70).
    velocity=0.5e0_real_kind*(vel_l+vel_r)
    metric=0.5e0_real_kind*(metric_l+metric_r)
    mass=0.5e0_real_kind*(rho_l+rho_r)*dot_product(velocity,metric)
    kinetic=0.5e0_real_kind*dot_product(vel_l,vel_r)
    internal=0.5e0_real_kind*(p_l/rho_l+p_r/rho_r)/(gamma-1.e0_real_kind)
    pressure_work=0.5e0_real_kind*(vel_l*p_r+vel_r*p_l)
    flux(1)=mass
    flux(2:4)=mass*velocity+0.5e0_real_kind*(p_l+p_r)*metric
    flux(5)=mass*(kinetic+internal)+dot_product(pressure_work,metric)
  end subroutine

  pure subroutine keep_explicit_volume(im,jm,km,hm,types,first,last,q,rho,velocity,pressure,jacobian,dxi,gamma,divergence)
    integer,intent(in) :: im,jm,km,hm,types(3),first(3),last(3)
    real(real_kind),intent(in) :: gamma
    real(real_kind),intent(in) :: rho(-hm:,-hm:,-hm:),velocity(-hm:,-hm:,-hm:,:)
    real(real_kind),intent(in) :: pressure(-hm:,-hm:,-hm:),jacobian(-hm:,-hm:,-hm:)
    real(real_kind),intent(in) :: dxi(-hm:,-hm:,-hm:,:,:)
    real(real_kind),intent(in) :: q(-hm:,-hm:,-hm:,:)
    real(real_kind),intent(inout) :: divergence(0:im,0:jm,0:km,5)
    real(real_kind),parameter :: a(3)=[0.75e0_real_kind,-0.15e0_real_kind,1.e0_real_kind/60.e0_real_kind]
    real(real_kind) :: plus(5),minus(5),ml(3),mr(3),coefficient,normal
    integer :: i,j,k,axis,l,point(3),s,node(3),dims(3),radius,distance
    ! The caller initializes RHS; characteristic boundary contributions are retained.
    dims=[im,jm,km]
    do k=first(3),last(3)
    do j=first(2),last(2)
    do i=first(1),last(1)
      node=[i,j,k]
      do axis=1,3
        ml=jacobian(i,j,k)*dxi(i,j,k,axis,:)
        distance=3
        if(types(axis)==1.or.types(axis)==4) distance=min(distance,node(axis))
        if(types(axis)==2.or.types(axis)==4) distance=min(distance,dims(axis)-node(axis))
        radius=min(distance,3)
        do l=1,radius
          do s=-1,1,2
            point=[i,j,k]
            point(axis)=point(axis)+s*l
            mr=jacobian(point(1),point(2),point(3))*dxi(point(1),point(2),point(3),axis,:)
            if(distance<3) then
              normal=dot_product(mr,velocity(point(1),point(2),point(3),:))
              plus(1)=q(point(1),point(2),point(3),1)*normal
              plus(2:4)=q(point(1),point(2),point(3),2:4)*normal &
                +pressure(point(1),point(2),point(3))*mr
              plus(5)=(q(point(1),point(2),point(3),5)+pressure(point(1),point(2),point(3)))*normal
            else
              call keep_two_point_flux5(rho(i,j,k),velocity(i,j,k,:),pressure(i,j,k),ml, &
                rho(point(1),point(2),point(3)),velocity(point(1),point(2),point(3),:), &
                pressure(point(1),point(2),point(3)),mr,gamma,plus)
            endif
            if(s==-1) then
              minus=plus
            else
              coefficient=2.e0_real_kind*a(l)
              if(distance==1) coefficient=0.5e0_real_kind
              if(distance==2) then
                coefficient=2.e0_real_kind/3.e0_real_kind
                if(l==2) coefficient=-1.e0_real_kind/12.e0_real_kind
              endif
              divergence(i,j,k,:)=divergence(i,j,k,:)+coefficient*(plus-minus)
            endif
          enddo
        enddo
      enddo
    enddo
    enddo
    enddo
  end subroutine

  pure subroutine keep_explicit_line(dim,spacing,rho,velocity,pressure,metric,gamma,divergence)
    integer,intent(in) :: dim
    real(real_kind),intent(in) :: spacing,gamma
    real(real_kind),intent(in) :: rho(-3:),velocity(-3:,:),pressure(-3:),metric(-3:,:)
    real(real_kind),intent(out) :: divergence(0:dim,5)
    real(real_kind),parameter :: coefficient(3)=[0.75e0_real_kind,-0.15e0_real_kind, &
      1.e0_real_kind/60.e0_real_kind]
    real(real_kind) :: plus(5),minus(5)
    integer :: i,l
    ! Positive mapped flux divergence; the caller owns the RHS sign and 1/J.
    divergence=0.e0_real_kind
    do i=0,dim
      do l=1,3
        call keep_two_point_flux5(rho(i),velocity(i,:),pressure(i),metric(i,:), &
          rho(i+l),velocity(i+l,:),pressure(i+l),metric(i+l,:),gamma,plus)
        call keep_two_point_flux5(rho(i),velocity(i,:),pressure(i),metric(i,:), &
          rho(i-l),velocity(i-l,:),pressure(i-l),metric(i-l,:),gamma,minus)
        divergence(i,:)=divergence(i,:)+2.e0_real_kind*coefficient(l)*(plus-minus)/spacing
      enddo
    enddo
  end subroutine
end module
