module chemistry_linear6
  use iso_fortran_env, only: real64
  use ieee_arithmetic, only: ieee_is_finite
  use chemistry_model, only: chemistry_status_ok, chemistry_status_linear_failure, &
    chemistry_status_nonfinite
  implicit none
  private

  integer, parameter :: linear_size = 6

  public :: air5_lu_factor_6
  public :: air5_lu_solve_6

contains

  pure subroutine air5_lu_factor_6(matrix, lu, pivots, status)
    real(real64), intent(in) :: matrix(linear_size,linear_size)
    real(real64), intent(out) :: lu(linear_size,linear_size)
    integer, intent(out) :: pivots(linear_size), status
    real(real64) :: row_scale(linear_size), scaled_pivot, best_scaled_pivot
    real(real64) :: row_buffer(linear_size), scale_buffer
    integer :: column, pivot, row

    lu = matrix
    pivots = [(row,row=1,linear_size)]
    if (.not. all(ieee_is_finite(matrix))) then
      status = chemistry_status_nonfinite
      return
    end if
    do row = 1, linear_size
      row_scale(row) = maxval(abs(matrix(row,:)))
    end do
    if (any(row_scale == 0.0_real64)) then
      status = chemistry_status_linear_failure
      return
    end if

    do column = 1, linear_size-1
      pivot = column
      best_scaled_pivot = abs(lu(column,column))/row_scale(column)
      do row = column+1, linear_size
        scaled_pivot = abs(lu(row,column))/row_scale(row)
        if (scaled_pivot > best_scaled_pivot) then
          pivot = row
          best_scaled_pivot = scaled_pivot
        end if
      end do
      if (best_scaled_pivot <= 64.0_real64*epsilon(1.0_real64)) then
        status = chemistry_status_linear_failure
        return
      end if
      pivots(column) = pivot
      if (pivot /= column) then
        row_buffer = lu(column,:)
        lu(column,:) = lu(pivot,:)
        lu(pivot,:) = row_buffer
        scale_buffer = row_scale(column)
        row_scale(column) = row_scale(pivot)
        row_scale(pivot) = scale_buffer
      end if
      do row = column+1, linear_size
        lu(row,column) = lu(row,column)/lu(column,column)
        lu(row,column+1:linear_size) = lu(row,column+1:linear_size) - &
          lu(row,column)*lu(column,column+1:linear_size)
      end do
    end do
    if (abs(lu(linear_size,linear_size))/row_scale(linear_size) <= &
        64.0_real64*epsilon(1.0_real64) .or. &
        .not. all(ieee_is_finite(lu))) then
      status = chemistry_status_linear_failure
      return
    end if
    status = chemistry_status_ok
  end subroutine air5_lu_factor_6

  pure subroutine air5_lu_solve_6(lu, pivots, rhs, solution, status)
    real(real64), intent(in) :: lu(linear_size,linear_size), rhs(linear_size)
    integer, intent(in) :: pivots(linear_size)
    real(real64), intent(out) :: solution(linear_size)
    integer, intent(out) :: status
    real(real64) :: value
    integer :: column, pivot, row

    solution = 0.0_real64
    if (.not. all(ieee_is_finite(lu)) .or. .not. all(ieee_is_finite(rhs))) then
      status = chemistry_status_nonfinite
      return
    end if
    if (any(pivots < 1) .or. any(pivots > linear_size)) then
      status = chemistry_status_linear_failure
      return
    end if

    solution = rhs
    do column = 1, linear_size-1
      pivot = pivots(column)
      if (pivot /= column) then
        value = solution(column)
        solution(column) = solution(pivot)
        solution(pivot) = value
      end if
    end do
    do row = 2, linear_size
      solution(row) = solution(row) - &
        dot_product(lu(row,1:row-1),solution(1:row-1))
    end do
    do row = linear_size, 1, -1
      if (lu(row,row) == 0.0_real64) then
        solution = 0.0_real64
        status = chemistry_status_linear_failure
        return
      end if
      solution(row) = (solution(row) - &
        dot_product(lu(row,row+1:linear_size),solution(row+1:linear_size)))/lu(row,row)
    end do
    if (.not. all(ieee_is_finite(solution))) then
      solution = 0.0_real64
      status = chemistry_status_nonfinite
    else
      status = chemistry_status_ok
    end if
  end subroutine air5_lu_solve_6

end module chemistry_linear6
