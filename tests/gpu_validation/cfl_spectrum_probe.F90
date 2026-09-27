#ifdef _CUDA
module cfl_probe_gpu
  use cudafor
  use cfl_spectrum
  implicit none
contains
  attributes(global) subroutine evaluate(v,a,m,r,ok)
    real(8),device :: v(3),a,m(3,3),r(4)
    logical,device :: ok
    call spectral_rates(v,a,m,r,ok)
  end subroutine evaluate
end module cfl_probe_gpu
#endif
program cfl_spectrum_probe
  use cfl_spectrum
  use ieee_arithmetic
#ifdef _CUDA
  use cfl_probe_gpu
#endif
  implicit none
  real(8) :: v(3),a,m(3,3),r(4),expected(4),reverse(4)
  logical :: valid
  integer :: n
#ifdef _CUDA
  real(8),device :: vd(3),ad,md(3,3),rd(4)
  logical,device :: valid_d
  integer :: ierr
  real(8) :: gpu(4)
  logical :: gpu_valid
#endif

  a=3.d0
  v=(/2.d0,-4.d0,6.d0/)
  do n=1,3
    m=0.d0
    if(n==1) then
      m(1,1)=2.d0
      m(2,2)=4.d0
      m(3,3)=8.d0
      expected=(/10.d0,28.d0,72.d0,110.d0/)
    else if(n==2) then
      ! Nonorthogonal rows test geometric projection, not Cartesian spacing.
      m(1,:)=(/0.6d0,0.8d0,0.d0/)
      m(2,:)=(/0.d0,1.d0,1.d0/)
      m(3,:)=(/1.d0,0.d0,0.d0/)
      expected(1:3)=(/5.d0,2.d0+3.d0*sqrt(2.d0),5.d0/)
      expected(4)=sum(expected(1:3))
    else
      m(1,1)=1.d0
      expected=(/5.d0,0.d0,0.d0,5.d0/)
    endif
    call spectral_rates(v,a,m,r,valid)
    if(.not.valid.or.any(abs(r-expected)>1.d-13)) error stop 'CPU spectral rate mismatch'
    call spectral_rates(-v,a,m,reverse,valid)
    if(.not.valid.or.any(reverse/=r)) error stop 'direction reversal mismatch'
#ifdef _CUDA
    vd=v
    ad=a
    md=m
    call evaluate<<<1,1>>>(vd,ad,md,rd,valid_d)
    ierr=cudaDeviceSynchronize()
    if(ierr/=cudaSuccess) error stop 'GPU spectrum kernel failed'
    gpu=rd
    gpu_valid=valid_d
    if(.not.gpu_valid.or.any(abs(gpu-r)>1.d-13)) error stop 'GPU spectral rate mismatch'
    vd=-v
    call evaluate<<<1,1>>>(vd,ad,md,rd,valid_d)
    ierr=cudaDeviceSynchronize()
    if(ierr/=cudaSuccess) error stop 'GPU reversed spectrum kernel failed'
    gpu=rd
    gpu_valid=valid_d
    if(.not.gpu_valid.or.any(abs(gpu-r)>1.d-13)) error stop 'GPU reversed spectrum mismatch'
#endif
  enddo
  v(1)=ieee_value(0.d0,ieee_quiet_nan)
  call spectral_rates(v,a,m,r,valid)
  if(valid) error stop 'NaN accepted'
#ifdef _CUDA
  vd=v
  call evaluate<<<1,1>>>(vd,ad,md,rd,valid_d)
  ierr=cudaDeviceSynchronize()
  if(ierr/=cudaSuccess) error stop 'GPU invalid spectrum kernel failed'
  gpu_valid=valid_d
  if(gpu_valid) error stop 'GPU NaN accepted'
#endif
  print *, 'CFL_SPECTRUM_PASS'
end program cfl_spectrum_probe
