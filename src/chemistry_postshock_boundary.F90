module chemistry_postshock_boundary
  use iso_fortran_env, only: real64
  use chemistry_state_layout, only: air5_num_conservative
  implicit none
  private

  real(real64), save :: left_q(air5_num_conservative)=0.0_real64
  real(real64), save :: right_q(air5_num_conservative)=0.0_real64
  logical, save :: boundary_configured=.false.

  public :: configure_air5_postshock_boundary
  public :: get_air5_postshock_boundary
  public :: apply_air5_postshock_boundary

contains

  subroutine configure_air5_postshock_boundary(left_state,right_state)
    real(real64), intent(in) :: left_state(air5_num_conservative)
    real(real64), intent(in) :: right_state(air5_num_conservative)

    left_q=left_state
    right_q=right_state
    boundary_configured=.true.
  end subroutine configure_air5_postshock_boundary

  subroutine get_air5_postshock_boundary(left_state,right_state)
    real(real64), intent(out) :: left_state(air5_num_conservative)
    real(real64), intent(out) :: right_state(air5_num_conservative)

    if(.not.boundary_configured) &
      error stop 'fixed air5 post-shock boundary is not configured'
    left_state=left_q
    right_state=right_q
  end subroutine get_air5_postshock_boundary

  subroutine apply_air5_postshock_boundary()
    use mpi, only: MPI_PROC_NULL
    use commvar, only: im,jm,km,hm,flowtype
    use commarray, only: q
    use parallel, only: mpileft,mpiright
    integer :: component

    if(trim(flowtype)/='air5postshock') return
    if(.not.boundary_configured) &
      error stop 'fixed air5 post-shock boundary is not configured'
    if(mpileft==MPI_PROC_NULL) then
      do component=1,air5_num_conservative
        q(-hm:0,0:jm,0:km,component)=left_q(component)
      enddo
    endif
    if(mpiright==MPI_PROC_NULL) then
      do component=1,air5_num_conservative
        q(im:im+hm,0:jm,0:km,component)=right_q(component)
      enddo
    endif
  end subroutine apply_air5_postshock_boundary

end module chemistry_postshock_boundary
