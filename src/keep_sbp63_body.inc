#ifndef SBP63_PROC
#define SBP63_PROC pure
#endif
  use astr_precision, only: real_kind
  implicit none
  private
  public :: sbp63_weight,sbp63_norm,sbp63_lift,sbp63_inviscid_wall,sbp63_viscous_wall,sbp63_material_flux
#ifdef SBP63_DEVICE
  public :: sbp63_convection_line_kernel,sbp63_flux_line_kernel,sbp63_divergence_line_kernel
#else
  public :: sbp63_convection_line,sbp63_viscous_line
#endif
  real(real_kind),parameter,public :: sbp63_theta=0.7049084023584565e0_real_kind
  real(real_kind),parameter :: hleft(0:5)=[13649.e0_real_kind/43200.e0_real_kind, &
    12013.e0_real_kind/8640.e0_real_kind,2711.e0_real_kind/4320.e0_real_kind, &
    5359.e0_real_kind/4320.e0_real_kind,7877.e0_real_kind/8640.e0_real_kind, &
    43801.e0_real_kind/43200.e0_real_kind]
  ! Rounded once from the exact H-weighted leading-error minimizer in design section 14.
  real(real_kind),parameter :: dleft(0:8,0:5)=reshape([ &
    -1.58253351893911631e+00_real_kind,2.04489044241717277e+00_real_kind, &
    -1.87559913610859469e-01_real_kind,-3.81327724279292901e-01_real_kind, &
    5.84410144180559574e-02_real_kind,4.80896999940402387e-02_real_kind,0.e0_real_kind,0.e0_real_kind,0.e0_real_kind, &
    -4.64675096121734577e-01_real_kind,0.e0_real_kind,3.13417627884012617e-01_real_kind, &
    2.06498077565308052e-01_real_kind,-2.98735581739810498e-02_real_kind, &
    -2.53670511536050539e-02_real_kind,0.e0_real_kind,0.e0_real_kind,0.e0_real_kind, &
    9.44302936508528562e-02_real_kind,-6.94409067460465490e-01_real_kind,0.e0_real_kind, &
    7.22151468254264239e-01_real_kind,-1.38818134920930952e-01_real_kind, &
    1.66454404762792808e-02_real_kind,0.e0_real_kind,0.e0_real_kind,0.e0_real_kind, &
    9.71215172362020623e-02_real_kind,-2.31448162510920485e-01_real_kind, &
    -3.65320513237042455e-01_real_kind,0.e0_real_kind,5.49581945988428577e-01_real_kind, &
    -6.33701298912972694e-02_real_kind,1.34353424146295957e-02_real_kind,0.e0_real_kind,0.e0_real_kind, &
    -2.02529238489791985e-02_real_kind,4.55593569054252023e-02_real_kind, &
    9.55531201652009138e-02_real_kind,-7.47799834594893631e-01_real_kind,0.e0_real_kind, &
    7.73188853164537782e-01_real_kind,-1.64529643265202502e-01_real_kind, &
    1.82810714739113887e-02_real_kind,0.e0_real_kind, &
    -1.49854184885882796e-02_real_kind,3.47862361028580952e-02_real_kind, &
    -1.03024563665654063e-02_real_kind,7.75325965360293412e-02_real_kind, &
    -6.95236249900352044e-01_real_kind,0.e0_real_kind,7.39709139060752086e-01_real_kind, &
    -1.47941827812150400e-01_real_kind,1.64379808680167119e-02_real_kind],[9,6])
contains
  SBP63_PROC function sbp63_norm(global_node,total) result(weight)
    integer,intent(in) :: global_node,total
    real(real_kind) :: weight
    integer :: distance
    distance=min(global_node,total-global_node)
    weight=1.e0_real_kind
    if(distance<6) weight=hleft(distance)
  end function

  SBP63_PROC function sbp63_weight(global_node,offset,total) result(weight)
    integer,intent(in) :: global_node,offset,total
    real(real_kind) :: weight
    integer :: column,distance
    column=global_node+offset;weight=0.e0_real_kind
    if(column<0.or.column>total) return
    if(global_node<6) then
      if(column<=8) weight=dleft(column,global_node)
    elseif(total-global_node<6) then
      column=total-column;distance=total-global_node
      if(column<=8) weight=-dleft(column,distance)
    else
      select case(offset)
      case(-3);weight=-1.e0_real_kind/60.e0_real_kind
      case(-2);weight=3.e0_real_kind/20.e0_real_kind
      case(-1);weight=-3.e0_real_kind/4.e0_real_kind
      case(1);weight=3.e0_real_kind/4.e0_real_kind
      case(2);weight=-3.e0_real_kind/20.e0_real_kind
      case(3);weight=1.e0_real_kind/60.e0_real_kind
      end select
    endif
  end function

  SBP63_PROC subroutine sbp63_lift(normal,inverse_length,wall_velocity,velocity,wall_temperature,temperature,gu,gt)
    real(real_kind),intent(in) :: normal(3),inverse_length,wall_velocity(3),velocity(3),wall_temperature,temperature
    real(real_kind),intent(inout) :: gu(3,3),gt(3)
    integer :: c,a
    do a=1,3
      do c=1,3
        gu(c,a)=gu(c,a)+inverse_length*(wall_velocity(c)-velocity(c))*normal(a)
      enddo
    enddo
    gt=gt+inverse_length*(wall_temperature-temperature)*normal
  end subroutine

  SBP63_PROC subroutine sbp63_inviscid_wall(pressure,normal,flux)
    real(real_kind),intent(in) :: pressure,normal(3)
    real(real_kind),intent(out) :: flux(5)
    flux=0.e0_real_kind;flux(2:4)=pressure*normal
  end subroutine

  SBP63_PROC subroutine sbp63_material_flux(mu,kappa,velocity,gu,gt,stress,energy_flux)
    real(real_kind),intent(in) :: mu,kappa,velocity(3),gu(3,3),gt(3)
    real(real_kind),intent(out) :: stress(3,3),energy_flux(3)
    real(real_kind) :: trace
    integer :: c
    trace=gu(1,1)+gu(2,2)+gu(3,3)
    stress=mu*(gu+transpose(gu))
    do c=1,3
      stress(c,c)=stress(c,c)-(2.e0_real_kind/3.e0_real_kind)*mu*trace
    enddo
    energy_flux=kappa*gt+matmul(velocity,stress)
  end subroutine

  SBP63_PROC subroutine sbp63_viscous_wall(mu,kappa,inverse_length,normal,wall_velocity,velocity, &
      wall_temperature,temperature,gu,gt,flux)
    real(real_kind),intent(in) :: mu,kappa,inverse_length,normal(3),wall_velocity(3),velocity(3)
    real(real_kind),intent(in) :: wall_temperature,temperature,gu(3,3),gt(3)
    real(real_kind),intent(out) :: flux(5)
    real(real_kind) :: stress(3,3),energy(3),delta(3),heat
    call sbp63_material_flux(mu,kappa,velocity,gu,gt,stress,energy)
    delta=velocity-wall_velocity
    flux=0.e0_real_kind
    flux(2:4)=matmul(stress,normal)-mu*inverse_length*(delta+normal*dot_product(delta,normal)/3.e0_real_kind)
    heat=kappa*dot_product(gt,normal)-kappa*inverse_length*(temperature-wall_temperature)
    flux(5)=dot_product(wall_velocity,flux(2:4))+heat
  end subroutine
