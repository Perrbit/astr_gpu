module commvar
  use iso_fortran_env,only:int64,real64
  implicit none
  integer :: im,jm,km,ia,ja,ka
  integer :: numq=11,num_species=5,num_modequ=1
  integer(int64) :: nstep
  real(real64) :: time
  logical :: nondimen=.false.,diffterm=.true.,lavg=.true.
end module
module parallel
  implicit none
  integer :: ig0,jg0,kg0
end module
module statistic
  use iso_fortran_env,only:int64
  implicit none
  integer(int64) :: nsamples
end module
#ifdef ASTR_MEAN_RUNTIME_REFERENCE
module commarray_gpu
  use cudafor
  implicit none
  real(8),allocatable,device :: rho_d(:,:,:),vel_d(:,:,:,:),prs_d(:,:,:),tmp_d(:,:,:), &
    tve_d(:,:,:),spc_d(:,:,:,:),dvel_d(:,:,:,:,:),q_d(:,:,:,:),x_d(:,:,:,:)
end module
module gpu_check
  use cudafor
  implicit none
contains
  subroutine sync_after_kernel(label,force_sync)
    character(*),intent(in) :: label
    logical,optional,intent(in) :: force_sync
    integer :: status
    status=cudaDeviceSynchronize()
    if(status/=0)then
      write(*,*)trim(label),status
      error stop 'original statistics CUDA error'
    endif
  end subroutine
  subroutine sync_gpu_boundary(label)
    character(*),intent(in) :: label
    call sync_after_kernel(label,.true.)
  end subroutine
end module
#endif
