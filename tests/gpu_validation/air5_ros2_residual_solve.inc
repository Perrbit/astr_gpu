#ifdef ASTR_ROS2_RESIDUAL_GPU
  attributes(device) subroutine air5_probe_solve_once(matrix,lu,pivots,rhs,x,status)
#else
  pure subroutine air5_probe_solve_once(matrix,lu,pivots,rhs,x,status)
#endif
    use air5_ros2_residual_words, only: air5_probe_residual
    real(real64), intent(in) :: matrix(6,6),lu(6,6),rhs(6)
    integer, intent(in) :: pivots(6)
    real(real64), intent(out) :: x(6)
    integer, intent(out) :: status
    real(real64) :: r(6),delta(6)
    integer :: i
#ifdef ASTR_ROS2_RESIDUAL_GPU
    call air5_lu_solve_6_gpu(lu,pivots,rhs,x,status)
#else
    call air5_lu_solve_6(lu,pivots,rhs,x,status)
#endif
    if(status/=chemistry_status_ok)return
    call air5_probe_residual(matrix,rhs,x,r)
#ifdef ASTR_ROS2_RESIDUAL_GPU
    call air5_lu_solve_6_gpu(lu,pivots,r,delta,status)
#else
    call air5_lu_solve_6(lu,pivots,r,delta,status)
#endif
    if(status/=chemistry_status_ok)return
    do i=1,6
      x(i)=x(i)+delta(i)
#ifdef ASTR_ROS2_RESIDUAL_GPU
      if(.not.air5_finite_gpu(x(i)))status=chemistry_status_nonfinite
#else
      if(.not.ieee_is_finite(x(i)))status=chemistry_status_nonfinite
#endif
    end do
  end subroutine air5_probe_solve_once
