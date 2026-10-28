#ifndef INCIDENT_PROC
#define INCIDENT_PROC pure
#endif
  use astr_precision, only: real_kind
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: incident_shock_states,incident_reference
  public :: grcbc_incoming_rate,apply_incident_characteristic_rhs
  public :: incident_normal_characteristics,incident_convective_compatibility
contains
  INCIDENT_PROC logical function incident_admissible(q)
    real(real_kind),intent(in) :: q(5)
    incident_admissible=.false.
    if(.not.all(ieee_is_finite(q))) return
    if(q(1)<=0.e0_real_kind) return
    incident_admissible=q(5)>0.5e0_real_kind*sum(q(2:4)**2)/q(1)
  end function incident_admissible

  INCIDENT_PROC subroutine incident_shock_states(upstream,gamma,beta,downstream,status)
    real(real_kind),intent(in) :: upstream(5),gamma,beta
    real(real_kind),intent(inout) :: downstream(5)
    integer,intent(out) :: status
    real(real_kind) :: velocity(3),normal(3),pressure,un,mn2,ratio,p2,rho2,trial(5)
    status=1
    if(.not.all(ieee_is_finite([gamma,beta]))) return
    if(gamma<=1.e0_real_kind.or.beta<=0.e0_real_kind.or.beta>=acos(0.e0_real_kind)) return
    if(.not.incident_admissible(upstream)) return
    velocity=upstream(2:4)/upstream(1)
    pressure=(gamma-1.e0_real_kind)*(upstream(5)-0.5e0_real_kind*upstream(1)*sum(velocity**2))
    ! Positive beta denotes a stationary shock descending from the upper-y face.
    normal=[sin(beta),cos(beta),0.e0_real_kind]
    un=sum(velocity*normal)
    mn2=un*un*upstream(1)/(gamma*pressure)
    status=2
    if(un<=0.e0_real_kind.or.mn2<=1.e0_real_kind) return
    ratio=(gamma+1.e0_real_kind)*mn2/((gamma-1.e0_real_kind)*mn2+2.e0_real_kind)
    rho2=upstream(1)*ratio
    p2=pressure*(1.e0_real_kind+2.e0_real_kind*gamma/(gamma+1.e0_real_kind)*(mn2-1.e0_real_kind))
    velocity=velocity+(un/ratio-un)*normal
    trial(1)=rho2
    trial(2:4)=rho2*velocity
    trial(5)=p2/(gamma-1.e0_real_kind)+0.5e0_real_kind*rho2*sum(velocity**2)
    status=3
    if(.not.incident_admissible(trial)) return
    downstream=trial
    status=0
  end subroutine incident_shock_states

  INCIDENT_PROC subroutine incident_reference(position,entry_x,top_y,beta,pre,post,reference)
    real(real_kind),intent(in) :: position(3),entry_x,top_y,beta,pre(5),post(5)
    real(real_kind),intent(out) :: reference(5)
    real(real_kind) :: signed_distance
    ! Sine/cosine form avoids cot(beta); the exact interface belongs to pre.
    signed_distance=(position(1)-entry_x)*sin(beta)+(position(2)-top_y)*cos(beta)
    if(signed_distance>0.e0_real_kind) then
      reference=post
    else
      reference=pre
    endif
  end subroutine incident_reference

  INCIDENT_PROC subroutine grcbc_incoming_rate(face,external,gamma,metric,hn,pinv, &
                                               reference_rate,rate,lambda,incoming,status)
    real(real_kind),intent(in) :: face(5),external(5),gamma,metric(3),hn,pinv(5,5),reference_rate(5)
    real(real_kind),intent(out) :: rate(5),lambda(5)
    logical,intent(out) :: incoming(5)
    integer,intent(out) :: status
    real(real_kind) :: u(3),ug(3),p,pg,css,un,norm,drho,du(3),increment(5),tolerance
    integer :: m
    status=1
    incoming=.false.
    lambda=0.e0_real_kind
    rate=reference_rate
    if(.not.all(ieee_is_finite([gamma,hn,metric]))) return
    if(gamma<=1.e0_real_kind.or.hn<=0.e0_real_kind) return
    norm=sqrt(sum(metric*metric))
    if(norm<=0.e0_real_kind) return
    if(.not.incident_admissible(face).or..not.incident_admissible(external)) return
    u=face(2:4)/face(1)
    ug=external(2:4)/external(1)
    p=(gamma-1.e0_real_kind)*(face(5)-0.5e0_real_kind*face(1)*sum(u*u))
    pg=(gamma-1.e0_real_kind)*(external(5)-0.5e0_real_kind*external(1)*sum(ug*ug))
    css=sqrt(gamma*p/face(1))
    un=sum(metric*u)/norm
    lambda=[un,un,un,un+css,un-css]
    tolerance=64.e0_real_kind*epsilon(1.e0_real_kind)*max(css,abs(un))
    incoming=lambda<-tolerance
    ! P_b (v_b-v_g), with primitive order rho,u,v,w,p and P_b=dq/dv.
    drho=face(1)-external(1)
    du=u-ug
    increment(1)=drho
    increment(2:4)=u*drho+face(1)*du
    increment(5)=(p-pg)/(gamma-1.e0_real_kind)+0.5e0_real_kind*sum(u*u)*drho+face(1)*sum(u*du)
    do m=1,5
      if(incoming(m)) rate(m)=reference_rate(m)-abs(lambda(m))/hn*sum(pinv(m,:)*increment)
    enddo
    status=2
    if(.not.all(ieee_is_finite(rate))) return
    status=0
  end subroutine grcbc_incoming_rate

  INCIDENT_PROC subroutine incident_normal_characteristics(states,velocity,pressure,jacobian,metric,pinv,lodi,status)
    real(real_kind),intent(in) :: states(5,3),velocity(3,3),pressure(3),jacobian(3),metric(3,3),pinv(5,5)
    real(real_kind),intent(out) :: lodi(5)
    integer,intent(out) :: status
    real(real_kind) :: flux(5,3),cartesian(5,3),delta_metric(3),derivative(5),un
    integer :: j
    status=1
    lodi=0.e0_real_kind
    if(any(jacobian<=0.e0_real_kind)) return
    do j=1,3
      un=sum(metric(:,j)*velocity(:,j))
      flux(1,j)=jacobian(j)*states(1,j)*un
      flux(2:4,j)=jacobian(j)*(states(2:4,j)*un+metric(:,j)*pressure(j))
      flux(5,j)=jacobian(j)*(states(5,j)+pressure(j))*un
      cartesian(1,j)=states(j+1,1)
      cartesian(2:4,j)=states(2:4,1)*velocity(j,1)
      cartesian(j+1,j)=cartesian(j+1,j)+pressure(1)
      cartesian(5,j)=(states(5,1)+pressure(1))*velocity(j,1)
    enddo
    delta_metric=1.5e0_real_kind*metric(:,1)*jacobian(1)- &
      2.e0_real_kind*metric(:,2)*jacobian(2)+0.5e0_real_kind*metric(:,3)*jacobian(3)
    derivative=1.5e0_real_kind*flux(:,1)-2.e0_real_kind*flux(:,2)+ &
      0.5e0_real_kind*flux(:,3)-matmul(cartesian,delta_metric)
    lodi=matmul(pinv,derivative)/jacobian(1)
    status=2
    if(.not.all(ieee_is_finite(lodi))) return
    status=0
  end subroutine incident_normal_characteristics

  INCIDENT_PROC subroutine incident_convective_compatibility(rhs,jacobian,pinv,normal_lodi,incoming,rate)
    real(real_kind),intent(in) :: rhs(5),jacobian,pinv(5,5),normal_lodi(5)
    logical,intent(in) :: incoming(5)
    real(real_kind),intent(inout) :: rate(5)
    integer :: m
    ! f=-R L_n+G: retain G for entropy/shear even as the normal speed crosses zero.
    do m=1,3
      if(incoming(m)) rate(m)=rate(m)+sum(pinv(m,:)*rhs)/jacobian+normal_lodi(m)
    enddo
  end subroutine incident_convective_compatibility

  INCIDENT_PROC subroutine apply_incident_characteristic_rhs(rhs,jacobian,pnor,pinv,incoming,rate,status)
    real(real_kind),intent(inout) :: rhs(5)
    real(real_kind),intent(in) :: jacobian,pnor(5,5),pinv(5,5),rate(5)
    logical,intent(in) :: incoming(5)
    integer,intent(out) :: status
    real(real_kind) :: correction(5),trial(5)
    integer :: m
    status=1
    if(.not.ieee_is_finite(jacobian)) return
    if(jacobian<=0.e0_real_kind) return
    correction=0.e0_real_kind
    do m=1,5
      if(incoming(m)) correction(m)=rate(m)-sum(pinv(m,:)*rhs)/jacobian
    enddo
    trial=rhs+jacobian*matmul(pnor,correction)
    status=2
    if(.not.all(ieee_is_finite(trial))) return
    rhs=trial
    status=0
  end subroutine apply_incident_characteristic_rhs
