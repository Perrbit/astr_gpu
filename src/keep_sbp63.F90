module keep_sbp63
  use keep_operator, only: keep_two_point_flux5
#include "keep_sbp63_body.inc"
  pure subroutine sbp63_convection_line(dim,total,start,hm,spacing,rho,velocity,pressure,gamma,divergence)
    integer,intent(in) :: dim,total,start,hm
    real(real_kind),intent(in) :: spacing,rho(-hm:),velocity(-hm:,:),pressure(-hm:),gamma
    real(real_kind),intent(out) :: divergence(0:dim,5)
    real(real_kind) :: weight,flux(5),metric(3),normal(3),wall(5),sign
    integer :: i,offset
    metric=[0.e0_real_kind,1.e0_real_kind,0.e0_real_kind];divergence=0.e0_real_kind
    do i=0,dim
      do offset=-5,5
        weight=sbp63_weight(start+i,offset,total)
        if(weight==0.e0_real_kind) cycle
        call keep_two_point_flux5(rho(i),velocity(i,:),pressure(i),metric, &
          rho(i+offset),velocity(i+offset,:),pressure(i+offset),metric,gamma,flux)
        divergence(i,:)=divergence(i,:)-2.e0_real_kind*weight*flux/spacing
      enddo
      ! This new component returns physical q_t, including its unique wall SAT.
      if(start+i==0.or.start+i==total) then
        sign=1.e0_real_kind
        if(start+i==0) sign=-1.e0_real_kind
        normal=sign*metric
        call sbp63_inviscid_wall(pressure(i),normal,wall)
        call keep_two_point_flux5(rho(i),velocity(i,:),pressure(i),metric, &
          rho(i),velocity(i,:),pressure(i),metric,gamma,flux)
        divergence(i,:)=divergence(i,:)-sign*(sign*wall-flux)/(spacing*sbp63_norm(start+i,total))
      endif
    enddo
  end subroutine

  pure subroutine sbp63_viscous_line(dim,hm,spacing,velocity,temperature,mu,kappa,wall_temperature,rhs,flux,face,gradient)
    integer,intent(in) :: dim,hm
    real(real_kind),intent(in) :: spacing,velocity(-hm:,:),temperature(-hm:),mu(0:),kappa(0:),wall_temperature
    real(real_kind),intent(out) :: rhs(0:dim,5),flux(0:dim,4),face(0:dim,5),gradient(0:dim,12)
    real(real_kind) :: gu(3,3),gt(3),stress(3,3),energy(3),normal(3),weight,inverse_length,sign
    integer :: i,offset
    face=0.e0_real_kind;rhs=0.e0_real_kind
    do i=0,dim
      gu=0.e0_real_kind;gt=0.e0_real_kind
      do offset=-5,5
        weight=sbp63_weight(i,offset,dim)/spacing
        if(weight==0.e0_real_kind) cycle
        gu(:,2)=gu(:,2)+weight*velocity(i+offset,:)
        gt(2)=gt(2)+weight*temperature(i+offset)
      enddo
      if(i==0.or.i==dim) then
        sign=1.e0_real_kind
        if(i==0) sign=-1.e0_real_kind
        normal=[0.e0_real_kind,sign,0.e0_real_kind]
        inverse_length=1.e0_real_kind/(spacing*sbp63_norm(i,dim))
        call sbp63_lift(normal,inverse_length,[0.e0_real_kind,0.e0_real_kind,0.e0_real_kind], &
          velocity(i,:),wall_temperature,temperature(i),gu,gt)
        call sbp63_viscous_wall(mu(i),kappa(i),inverse_length,normal, &
          [0.e0_real_kind,0.e0_real_kind,0.e0_real_kind],velocity(i,:),wall_temperature,temperature(i),gu,gt,face(i,:))
      endif
      call sbp63_material_flux(mu(i),kappa(i),velocity(i,:),gu,gt,stress,energy)
      flux(i,1:3)=stress(:,2);flux(i,4)=energy(2)
      gradient(i,1:9)=reshape(gu,[9]);gradient(i,10:12)=gt
    enddo
    do i=0,dim
      do offset=-5,5
        weight=sbp63_weight(i,offset,dim)/spacing
        if(weight==0.e0_real_kind) cycle
        rhs(i,2:5)=rhs(i,2:5)+weight*flux(i+offset,:)
      enddo
      if(i==0.or.i==dim) then
        sign=1.e0_real_kind
        if(i==0) sign=-1.e0_real_kind
        rhs(i,2:5)=rhs(i,2:5)+sign*(sign*face(i,2:5)-flux(i,:))/(spacing*sbp63_norm(i,dim))
      endif
    enddo
  end subroutine
end module
