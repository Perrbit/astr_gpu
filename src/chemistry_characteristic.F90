module chemistry_characteristic
  use iso_fortran_env, only: real64
  implicit none
  private
  public :: air5_top_transport_rhs
  public :: air5_y_flux_differential
  public :: air5_top_normal_convection
contains
#ifdef _CUDA
  attributes(host,device) &
#endif
  pure subroutine air5_top_normal_convection(q0,q1,q2,spacing,pressure,dpdq,normal,valid,first_order)
    real(real64),intent(in) :: q0(11),q1(11),q2(11),spacing,pressure,dpdq(11)
    real(real64),intent(out) :: normal(11)
    logical,intent(out) :: valid
    logical,intent(in),optional :: first_order
    real(real64) :: gradient(11)

    normal=0.0_real64
    valid=spacing>0.0_real64.and.abs(spacing)<=huge(1.0_real64).and. &
      all(abs(q0)<=huge(1.0_real64)).and.all(abs(q1)<=huge(1.0_real64)).and. &
      all(abs(q2)<=huge(1.0_real64)).and.q0(1)>0.0_real64
    if(.not.valid) return
    ! Difference form preserves a uniform state exactly without large-state cancellation.
    gradient=(1.5_real64*(q0-q1)-0.5_real64*(q1-q2))/spacing
    if(present(first_order)) then
      if(first_order) gradient=(q0-q1)/spacing
    endif
    call air5_y_flux_differential(q0,pressure,dpdq,gradient,normal)
    valid=all(abs(normal)<=huge(1.0_real64))
  end subroutine air5_top_normal_convection

#ifdef _CUDA
  attributes(host,device) &
#endif
  pure subroutine air5_y_flux_differential(q,pressure,dpdq,dq,df)
    real(real64),intent(in) :: q(11),pressure,dpdq(11),dq(11)
    real(real64),intent(out) :: df(11)
    real(real64) :: v,dv,dp

    ! Exact directional derivative of F_y; thermodynamic validity is a caller contract.
    v=q(3)/q(1)
    dv=(dq(3)-v*dq(1))/q(1)
    dp=sum(dpdq*dq)
    df=v*dq+dv*q
    df(3)=df(3)+dp
    df(5)=df(5)+v*dp+pressure*dv
  end subroutine air5_y_flux_differential

#ifdef _CUDA
  attributes(host,device) &
#endif
  pure subroutine acoustic_amplitudes(difference,dpdq,v,sound,a2,amplitude)
    real(real64),intent(in) :: difference(11),dpdq(11),v,sound,a2
    real(real64),intent(out) :: amplitude(2)
    real(real64) :: dp,dm

    dp=sum(dpdq*difference)
    dm=sound*(difference(3)-v*difference(1))
    amplitude=[dp-dm,dp+dm]/(2.0_real64*a2)
  end subroutine acoustic_amplitudes

#ifdef _CUDA
  attributes(host,device) &
#endif
  pure subroutine air5_top_transport_rhs(q,target,pressure,sound,dpdq,normal_flux_gradient, &
      remaining_transport,relaxation_rate,rhs,valid,reference_sound)
    real(real64),intent(in) :: q(11),target(11),pressure,sound,dpdq(11)
    real(real64),intent(in) :: normal_flux_gradient(11),remaining_transport(11),relaxation_rate
    real(real64),intent(in) :: reference_sound
    real(real64),intent(out) :: rhs(11)
    logical,intent(out) :: valid
    real(real64) :: velocity(3),waves(11,2),difference(11),amplitude(2),a2,convective_rate
    logical :: convective_in,acoustic_in(2)
    integer :: mode,sign

    ! Caller supplies validated AIR5 thermodynamics at q. Upper-y Cartesian only.
    ! remaining_transport includes all diffusion, but never chemistry or V-T.
    rhs=0.0_real64
    valid=all(abs(q)<=huge(1.0_real64)).and.all(abs(target)<=huge(1.0_real64)).and. &
      all(abs(dpdq)<=huge(1.0_real64)).and.all(abs(normal_flux_gradient)<=huge(1.0_real64)).and. &
      all(abs(remaining_transport)<=huge(1.0_real64)).and. &
      abs(pressure)<=huge(1.0_real64).and.abs(sound)<=huge(1.0_real64).and. &
      abs(relaxation_rate)<=huge(1.0_real64).and.abs(reference_sound)<=huge(1.0_real64)
    if(.not.valid) return
    valid=q(1)>0.0_real64.and.target(1)>0.0_real64.and.pressure>0.0_real64.and. &
      sound>0.0_real64.and.relaxation_rate>=0.0_real64.and.reference_sound>0.0_real64
    if(.not.valid) return
    velocity=q(2:4)/q(1)
    convective_rate=(max(-velocity(2),0.0_real64)/reference_sound)*relaxation_rate
    valid=all(abs(velocity)<=huge(1.0_real64)).and.abs(convective_rate)<=huge(1.0_real64)
    if(.not.valid) return
    a2=sound*sound
    do mode=1,2
      sign=2*mode-3
      waves(1,mode)=1.0_real64
      waves(2:4,mode)=velocity
      waves(3,mode)=velocity(2)+sign*sound
      waves(5,mode)=(q(5)+pressure)/q(1)+sign*sound*velocity(2)
      waves(6:11,mode)=q(6:11)/q(1)
    enddo

    ! Assemble the outgoing normal contribution without subtracting the full incoming normal state.
    convective_in=velocity(2)<0.0_real64
    acoustic_in=[velocity(2)-sound<0.0_real64,velocity(2)+sound<0.0_real64]
    if(convective_in) then
      rhs=remaining_transport
      if(.not.all(acoustic_in)) then
        call acoustic_amplitudes(normal_flux_gradient,dpdq,velocity(2),sound,a2,amplitude)
        do mode=1,2
          if(.not.acoustic_in(mode)) rhs=rhs-waves(:,mode)*amplitude(mode)
        enddo
      endif
    else
      rhs=remaining_transport-normal_flux_gradient
      if(any(acoustic_in)) then
        call acoustic_amplitudes(normal_flux_gradient,dpdq,velocity(2),sound,a2,amplitude)
        do mode=1,2
          if(acoustic_in(mode)) rhs=rhs+waves(:,mode)*amplitude(mode)
        enddo
      endif
    endif
    if(convective_in.and.convective_rate/=0.0_real64) then
      difference=-convective_rate*(q-target)
      call acoustic_amplitudes(difference,dpdq,velocity(2),sound,a2,amplitude)
      rhs=rhs+(difference-waves(:,1)*amplitude(1)-waves(:,2)*amplitude(2))
    endif
    if(relaxation_rate/=0.0_real64.and.any(acoustic_in)) then
      difference=-relaxation_rate*(q-target)
      call acoustic_amplitudes(difference,dpdq,velocity(2),sound,a2,amplitude)
      do mode=1,2
        if(acoustic_in(mode)) rhs=rhs+waves(:,mode)*amplitude(mode)
      enddo
    endif
    valid=all(abs(rhs)<=huge(1.0_real64))
  end subroutine air5_top_transport_rhs
end module chemistry_characteristic
