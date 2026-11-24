program weak_scaling_grid_count_probe
  use iso_fortran_env, only: int64
  use grid_count_math, only: grid_count64
  implicit none
  integer,parameter :: dims(3,3)=reshape([3904,1952,3904,3536,1768,3536,2688,1344,2688],[3,3])
  integer(int64),parameter :: expected(3)=[29750853632_int64,22105827328_int64,9710862336_int64]
  integer :: i
  do i=1,size(expected)
    if(grid_count64(dims(:,i))/=expected(i)) error stop 'incorrect full-size interval count'
    if(grid_count64(dims(:,i)+1)<=expected(i)) error stop 'incorrect endpoint-inclusive count'
    write(*,'(a,i0,1x,i0)') 'ASTR_WEAK_GRID_COUNT ',i,grid_count64(dims(:,i))
  enddo
end program weak_scaling_grid_count_probe
