module cfl_spectrum
  use astr_precision, only: real_kind
  implicit none
  private
  public :: spectral_rates
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
end module cfl_spectrum
