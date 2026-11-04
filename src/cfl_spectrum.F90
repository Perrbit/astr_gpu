module cfl_spectrum
  use astr_precision, only: real_kind
  implicit none
  private
  public :: spectral_rates
  public :: spectral_failure_kind,spectral_failure_message,cfl_bad_thermodynamics
  integer,parameter :: cfl_bad_thermodynamics=5
contains
#ifdef _CUDA
  attributes(host,device) &
#endif
  pure subroutine spectral_rates(velocity,sound,metric,rates,valid)
    real(real_kind),intent(in) :: velocity(3),sound,metric(3,3)
    real(real_kind),intent(out) :: rates(4)
    logical,intent(out) :: valid
    integer :: axis

    rates=0.0_real_kind
    valid=all(abs(velocity)<=huge(1.0_real_kind)).and.abs(sound)<=huge(1.0_real_kind).and. &
      all(abs(metric)<=huge(1.0_real_kind)).and.sound>0.0_real_kind
    if(.not.valid) return
    do axis=1,3
      rates(axis)=abs(sum(velocity*metric(axis,:)))+ &
        sound*sqrt(sum(metric(axis,:)**2))
    enddo
    rates(4)=sum(rates(1:3))
    valid=all(abs(rates)<=huge(1.0_real_kind)).and.all(rates(1:3)>=0.0_real_kind)
  end subroutine spectral_rates

#ifdef _CUDA
  attributes(host,device) &
#endif
  pure integer function spectral_failure_kind(velocity,sound,metric) result(reason)
    real(real_kind),intent(in) :: velocity(3),sound,metric(3,3)
    ! Called only after the existing spectrum check fails; acceptance is unchanged.
    if(.not.all(abs(velocity)<=huge(1.0_real_kind))) then
      reason=1
    elseif(.not.(abs(sound)<=huge(1.0_real_kind).and.sound>0.0_real_kind)) then
      reason=2
    elseif(.not.all(abs(metric)<=huge(1.0_real_kind))) then
      reason=3
    else
      reason=4
    endif
  end function

  pure function spectral_failure_message(reason) result(message)
    integer,intent(in) :: reason
    character(32) :: message
    character(32),parameter :: messages(0:5)=[character(32) :: 'valid', &
      'nonfinite velocity','invalid sound speed','nonfinite metric','wave-rate arithmetic', &
      'thermodynamic state recovery']
    message=messages(reason)
  end function
end module cfl_spectrum
