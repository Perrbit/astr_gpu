#ifdef _CUDA
module cfl_probe_gpu
  use astr_precision, only: real_kind
  use cudafor
  use cfl_spectrum
  implicit none
  private :: real_kind
contains
  attributes(global) subroutine evaluate(v,a,m,r,ok)
    real(real_kind),device :: v(3),a,m(3,3),r(4)
    logical,device :: ok
    call spectral_rates(v,a,m,r,ok)
  end subroutine evaluate
end module cfl_probe_gpu
#endif
program cfl_spectrum_probe
  use astr_precision, only: real_kind,real_bytes
  use cfl_spectrum
  use ieee_arithmetic
#ifdef _CUDA
  use cfl_probe_gpu
#endif
  implicit none
  real(real_kind) :: v(3),a,m(3,3),r(4),expected(4),reverse(4),tolerance,cpu_error=0
  logical :: valid
  integer :: n
#ifdef _CUDA
  real(real_kind),device :: vd(3),ad,md(3,3),rd(4)
  logical,device :: valid_d
  integer :: ierr
  real(real_kind) :: gpu(4),gpu_error=0
  logical :: gpu_valid
#endif

  a=3.e0_real_kind
  v=(/2.e0_real_kind,-4.e0_real_kind,6.e0_real_kind/)
  do n=1,3
    m=0.e0_real_kind
    if(n==1) then
      m(1,1)=2.e0_real_kind
      m(2,2)=4.e0_real_kind
      m(3,3)=8.e0_real_kind
      expected=(/10.e0_real_kind,28.e0_real_kind,72.e0_real_kind,110.e0_real_kind/)
    else if(n==2) then
      ! Nonorthogonal rows test geometric projection, not Cartesian spacing.
      m(1,:)=(/0.6e0_real_kind,0.8e0_real_kind,0.e0_real_kind/)
      m(2,:)=(/0.e0_real_kind,1.e0_real_kind,1.e0_real_kind/)
      m(3,:)=(/1.e0_real_kind,0.e0_real_kind,0.e0_real_kind/)
      expected(1:3)=(/5.e0_real_kind,2.e0_real_kind+3.e0_real_kind*sqrt(2.e0_real_kind),5.e0_real_kind/)
      expected(4)=sum(expected(1:3))
    else
      m(1,1)=1.e0_real_kind
      expected=(/5.e0_real_kind,0.e0_real_kind,0.e0_real_kind,5.e0_real_kind/)
    endif
    call spectral_rates(v,a,m,r,valid)
    tolerance=1.e-13_real_kind
    if(real_bytes==4) tolerance=64*epsilon(a)*max(1.e0_real_kind,maxval(abs(expected)))
    cpu_error=max(cpu_error,maxval(abs(r-expected)))
    if(.not.valid.or.any(abs(r-expected)>tolerance)) error stop 'CPU spectral rate mismatch'
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
    gpu_error=max(gpu_error,maxval(abs(gpu-r)))
    if(.not.gpu_valid.or.any(abs(gpu-r)>tolerance)) error stop 'GPU spectral rate mismatch'
    vd=-v
    call evaluate<<<1,1>>>(vd,ad,md,rd,valid_d)
    ierr=cudaDeviceSynchronize()
    if(ierr/=cudaSuccess) error stop 'GPU reversed spectrum kernel failed'
    gpu=rd
    gpu_valid=valid_d
    gpu_error=max(gpu_error,maxval(abs(gpu-r)))
    if(.not.gpu_valid.or.any(abs(gpu-r)>tolerance)) error stop 'GPU reversed spectrum mismatch'
#endif
  enddo
  v(1)=ieee_value(0.e0_real_kind,ieee_quiet_nan)
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
  write(*,'(a,i0,a,es24.16e3)') 'CFL_SPECTRUM_PASS bytes=',real_bytes,' cpu_analytic_maxabs=',cpu_error
#ifdef _CUDA
  write(*,'(a,es24.16e3)') 'CFL_SPECTRUM_GPU_MAXABS=',gpu_error
#endif
end program cfl_spectrum_probe
