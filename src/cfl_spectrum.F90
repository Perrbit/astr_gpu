module cfl_spectrum
  use iso_fortran_env, only: real64
  implicit none
  private
  public :: spectral_rates
contains
#ifdef _CUDA
  attributes(host,device) &
#endif
  pure subroutine spectral_rates(velocity,sound,metric,rates,valid)
    real(real64),intent(in) :: velocity(3),sound,metric(3,3)
    real(real64),intent(out) :: rates(4)
    logical,intent(out) :: valid
    integer :: axis

    rates=0.0_real64
    valid=all(abs(velocity)<=huge(1.0_real64)).and.abs(sound)<=huge(1.0_real64).and. &
      all(abs(metric)<=huge(1.0_real64)).and.sound>0.0_real64
    if(.not.valid) return
    do axis=1,3
      rates(axis)=abs(sum(velocity*metric(axis,:)))+ &
        sound*sqrt(sum(metric(axis,:)**2))
    enddo
    rates(4)=sum(rates(1:3))
    valid=all(abs(rates)<=huge(1.0_real64)).and.all(rates(1:3)>=0.0_real64)
  end subroutine spectral_rates
end module cfl_spectrum
