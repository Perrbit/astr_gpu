module commvar
  use iso_fortran_env,only:int64
  implicit none
  integer :: ia,ja,ka,im,jm,km,nmonitor
  integer(int64) :: nstep
  integer,allocatable :: imon(:,:)
  logical :: use_gpu=.true.,lreadgrid=.false.
  character(32) :: flowtype='air5hbl'
end module
module commarray
  implicit none
  real(8),allocatable :: q(:,:,:,:),x(:,:,:,:)
end module
module parallel
  implicit none
  integer :: ig0,jg0,kg0,mpirank
end module
#ifdef ASTR_AIR5_MONITOR_REFERENCE
module commarray_gpu
  use cudafor
  implicit none
  real(8),allocatable,device :: q_d(:,:,:,:),x_d(:,:,:,:)
end module
module gpu_check
  use cudafor
  implicit none
contains
  subroutine sync_after_kernel(label,force_sync)
    character(*),intent(in) :: label
    logical,intent(in) :: force_sync
    integer :: status
    status=cudaDeviceSynchronize()
    if(status/=0)then
      write(*,*)trim(label),status
      error stop 'original monitor CUDA error'
    endif
  end subroutine
end module
#endif
