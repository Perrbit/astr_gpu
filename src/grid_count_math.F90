module grid_count_math
  use iso_fortran_env, only: int64
  implicit none
  private
  public :: grid_count64
contains
  pure integer(int64) function grid_count64(extents)
    integer,intent(in) :: extents(:)
    grid_count64=product(int(extents,int64))
  end function grid_count64
end module grid_count_math
