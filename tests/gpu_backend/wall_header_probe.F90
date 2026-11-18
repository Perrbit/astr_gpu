program wall_header_probe
  use iso_fortran_env, only: int32,int64
  implicit none
  logical,allocatable :: owned(:,:,:)
  integer(int64) :: step
  integer :: rank,ig0,jg0,kg0,unit
  integer(int32) :: expected(9),statistics(11)
  character(1200) :: path
  allocate(owned(0:16,0:16,2))
  step=1;rank=0;ig0=11;jg0=22;kg0=33
  expected(1:3)=[1_int32,int(step,int32),int(rank,int32)]
  expected(4:6)=int(shape(owned),int32)
  expected(7:9)=int([ig0,jg0,kg0],int32)
  call get_command_argument(1,path)
  open(newunit=unit,file=trim(path),access='stream',form='unformatted',status='new')
  write(unit) expected
  close(unit)
  statistics(1:9)=expected;statistics(10)=3;statistics(11)=5
  open(newunit=unit,file=trim(path)//'.statistics',access='stream',form='unformatted',status='new')
  write(unit) statistics
  close(unit)
  print *, 'FIXED_HEADER_SIZE=',size(expected)
  print *, 'FIXED_STATISTICS_HEADER_SIZE=',size(statistics)
  print *, 'DECLARED_HEADER_SIZE=',size(int([integer(int64) :: 1,step,rank,shape(owned),ig0,jg0,kg0],int32))
  print *, 'EXPECTED_HEADER=',expected
  print *, 'DECLARED_STATISTICS_HEADER_SIZE=',size(int([integer(int64) :: 1,step,rank,shape(owned), &
    ig0,jg0,kg0,3,5],int32))
  print *, 'EXPECTED_STATISTICS_HEADER_SIZE=',11
end program
