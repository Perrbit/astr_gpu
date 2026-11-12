contains
  GRID_PROC function sbp63_axis_weight(axis,node,offset,total) result(weight)
    integer,intent(in) :: axis,node,offset,total
    real(real_kind) :: weight
    if(axis==2) then
      weight=sbp63_weight(node,offset,total)
    else
      weight=sbp63_weight(6,offset,12)
    endif
  end function

  GRID_PROC subroutine sbp63_gradient_point(im,jm,km,hm,total,start,i,j,k,tw,velocity,temperature,metric,gu,gt)
    integer,intent(in) :: im,jm,km,hm,total,start,i,j,k
    real(real_kind),intent(in) :: tw(2)
    real(real_kind),intent(in) :: velocity(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3)
    real(real_kind),intent(in) :: temperature(-hm:im+hm,-hm:jm+hm,-hm:km+hm)
    real(real_kind),intent(in) :: metric(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3,3)
    real(real_kind),intent(out) :: gu(3,3),gt(3)
    real(real_kind) :: du(3),dt,weight,normal(3),length_inv,wall_t,s,v(3)
    integer :: axis,offset,p(3),c,d
    gu=0.e0_real_kind;gt=0.e0_real_kind
    do axis=1,3
      du=0.e0_real_kind;dt=0.e0_real_kind
      do offset=-5,5
        weight=sbp63_axis_weight(axis,start+j,offset,total)
        if(weight==0.e0_real_kind) cycle
        p=[i,j,k];p(axis)=p(axis)+offset
        do c=1,3
          du(c)=du(c)+weight*velocity(p(1),p(2),p(3),c)
        enddo
        dt=dt+weight*temperature(p(1),p(2),p(3))
      enddo
      do d=1,3
        gu(:,d)=gu(:,d)+du*metric(i,j,k,axis,d)
        gt(d)=gt(d)+dt*metric(i,j,k,axis,d)
      enddo
    enddo
    if(start+j==0.or.start+j==total) then
      s=1.e0_real_kind;wall_t=tw(2)
      if(start+j==0) then
        s=-1.e0_real_kind;wall_t=tw(1)
      endif
      normal=metric(i,j,k,2,:)
      length_inv=sqrt(dot_product(normal,normal))
      normal=s*normal/length_inv
      length_inv=length_inv/sbp63_norm(start+j,total)
      v=velocity(i,j,k,:)
      call sbp63_lift(normal,length_inv,[0.e0_real_kind,0.e0_real_kind,0.e0_real_kind], &
        v,wall_t,temperature(i,j,k),gu,gt)
    endif
  end subroutine

  GRID_PROC subroutine sbp63_convection_point(im,jm,km,hm,total,start,i,j,k,gamma,rho,velocity,pressure,jac,metric,rhs)
    integer,intent(in) :: im,jm,km,hm,total,start,i,j,k
    real(real_kind),intent(in) :: gamma
    real(real_kind),intent(in) :: rho(-hm:im+hm,-hm:jm+hm,-hm:km+hm)
    real(real_kind),intent(in) :: velocity(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3)
    real(real_kind),intent(in) :: pressure(-hm:im+hm,-hm:jm+hm,-hm:km+hm)
    real(real_kind),intent(in) :: jac(-hm:im+hm,-hm:jm+hm,-hm:km+hm)
    real(real_kind),intent(in) :: metric(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3,3)
    real(real_kind),intent(out) :: rhs(5)
    real(real_kind) :: vl(3),vr(3),ml(3),mr(3),flux(5),wall(5),normal(3),weight,s,area
    integer :: axis,offset,p(3)
    rhs=0.e0_real_kind;vl=velocity(i,j,k,:)
    do axis=1,3
      ml=jac(i,j,k)*metric(i,j,k,axis,:)
      do offset=-5,5
        weight=sbp63_axis_weight(axis,start+j,offset,total)
        if(weight==0.e0_real_kind) cycle
        p=[i,j,k];p(axis)=p(axis)+offset
        vr=velocity(p(1),p(2),p(3),:)
        mr=jac(p(1),p(2),p(3))*metric(p(1),p(2),p(3),axis,:)
        call sbp_pair(rho(i,j,k),vl,pressure(i,j,k),ml, &
          rho(p(1),p(2),p(3)),vr,pressure(p(1),p(2),p(3)),mr,gamma,flux)
        rhs=rhs+2.e0_real_kind*weight*flux
      enddo
    enddo
    if(start+j==0.or.start+j==total) then
      s=1.e0_real_kind
      if(start+j==0) s=-1.e0_real_kind
      ml=jac(i,j,k)*metric(i,j,k,2,:)
      area=sqrt(dot_product(ml,ml));normal=s*ml/area
      call sbp63_inviscid_wall(pressure(i,j,k),normal,wall)
      call sbp_pair(rho(i,j,k),vl,pressure(i,j,k),ml,rho(i,j,k),vl,pressure(i,j,k),ml,gamma,flux)
      ! Positive divergence here; CPU negates once, GPU subtracts once.
      rhs=rhs+s*(s*area*wall-flux)/sbp63_norm(start+j,total)
    endif
  end subroutine

  GRID_PROC subroutine sbp63_diffusion_point(im,jm,km,hm,total,start,i,j,k,jac,metric,sigma,heat,rhs)
    integer,intent(in) :: im,jm,km,hm,total,start,i,j,k
    real(real_kind),intent(in) :: jac(-hm:im+hm,-hm:jm+hm,-hm:km+hm)
    real(real_kind),intent(in) :: metric(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3,3)
    real(real_kind),intent(in) :: sigma(-hm:im+hm,-hm:jm+hm,-hm:km+hm,6)
    real(real_kind),intent(in) :: heat(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3)
    real(real_kind),intent(out) :: rhs(5)
    real(real_kind) :: stress(3,3),m(3),weight
    integer :: axis,offset,p(3)
    rhs=0.e0_real_kind
    do axis=1,3
      do offset=-5,5
        weight=sbp63_axis_weight(axis,start+j,offset,total)
        if(weight==0.e0_real_kind) cycle
        p=[i,j,k];p(axis)=p(axis)+offset
        stress(1,:)=[sigma(p(1),p(2),p(3),1),sigma(p(1),p(2),p(3),2),sigma(p(1),p(2),p(3),3)]
        stress(2,:)=[sigma(p(1),p(2),p(3),2),sigma(p(1),p(2),p(3),4),sigma(p(1),p(2),p(3),5)]
        stress(3,:)=[sigma(p(1),p(2),p(3),3),sigma(p(1),p(2),p(3),5),sigma(p(1),p(2),p(3),6)]
        m=jac(p(1),p(2),p(3))*metric(p(1),p(2),p(3),axis,:)
        rhs(2:4)=rhs(2:4)+weight*matmul(stress,m)
        rhs(5)=rhs(5)+weight*dot_product(heat(p(1),p(2),p(3),:),m)
      enddo
    enddo
  end subroutine

  GRID_PROC subroutine sbp63_viscous_sat(globalj,total,tw,mu,kappa,jac,metric,velocity,temperature,gu,gt,rhs)
    integer,intent(in) :: globalj,total
    real(real_kind),intent(in) :: tw(2),mu,kappa,jac,metric(3),velocity(3),temperature,gu(3,3),gt(3)
    real(real_kind),intent(out) :: rhs(5)
    real(real_kind) :: normal(3),inverse_length,area,s,wall_t,wall(5),raw(5),stress(3,3),heat(3)
    rhs=0.e0_real_kind
    if(globalj/=0.and.globalj/=total) return
    s=1.e0_real_kind;wall_t=tw(2)
    if(globalj==0) then
      s=-1.e0_real_kind;wall_t=tw(1)
    endif
    inverse_length=sqrt(dot_product(metric,metric));normal=s*metric/inverse_length
    area=jac*inverse_length;inverse_length=inverse_length/sbp63_norm(globalj,total)
    call sbp63_viscous_wall(mu,kappa,inverse_length,normal,[0.e0_real_kind,0.e0_real_kind,0.e0_real_kind], &
      velocity,wall_t,temperature,gu,gt,wall)
    call sbp63_material_flux(mu,kappa,velocity,gu,gt,stress,heat)
    raw=0.e0_real_kind;raw(2:4)=matmul(stress,normal);raw(5)=dot_product(heat,normal)
    rhs=area*(wall-raw)/sbp63_norm(globalj,total)
  end subroutine
