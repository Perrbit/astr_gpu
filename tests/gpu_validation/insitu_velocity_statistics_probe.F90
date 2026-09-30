program velocity_statistics_probe
  use insitu_velocity_statistics
  use iso_fortran_env, only: int64
  implicit none
  type(velocity_statistics) :: s,copy
  type(velocity_statistics_result) :: r,other
  logical :: ok
  integer :: unit
  call configure_velocity_statistics(s,0.25d0,0.75d0,ok)
  call require(ok)
  call push_velocity_sample(s,0.d0,1.d0,[0.d0,0.d0,0.d0],ok)
  call require(ok)
  call read_velocity_statistics(s,r,ok)
  call require(.not.ok)
  call push_velocity_sample(s,1.d0,3.d0,[2.d0,4.d0,-2.d0],ok)
  call require(ok)
  call read_velocity_statistics(s,r,ok)
  call require(ok)
  call require(r%duration==0.5d0.and.r%mean_density==2.d0)
  call require(all(r%mean_r==[1.d0,2.d0,-1.d0]))
  call require(all(r%mean_f==[1.5d0,3.d0,-1.5d0]))
  call require(r%covariance_r(1,1)==1.d0.and.r%covariance_f(1,1)==0.75d0)
  call require(r%covariance_r(1,2)==2.d0.and.r%covariance_f(1,2)==1.5d0)
  call require(r%density_stress(1,1)==1.5d0.and.r%rms_r(1)==1.d0)
  ! Rejected samples must leave all accumulated values unchanged.
  call push_velocity_sample(s,1.d0,3.d0,[99.d0,0.d0,0.d0],ok)
  call require(.not.ok)
  call push_velocity_sample(s,2.d0,-1.d0,[0.d0,0.d0,0.d0],ok)
  call require(.not.ok)
  call read_velocity_statistics(s,other,ok)
  call require(ok.and.all(other%covariance_f==r%covariance_f))
  ! An exact large translation must not erase order-one variance.
  call configure_velocity_statistics(s,0.d0,1.d0,ok)
  call require(ok)
  call push_velocity_sample(s,0.d0,1.d0,[2.d0**40,0.d0,0.d0],ok)
  call require(ok)
  copy=s
  call push_velocity_sample(s,1.d0,1.d0,[2.d0**40+2.d0,0.d0,0.d0],ok)
  call require(ok)
  call read_velocity_statistics(s,r,ok)
  call require(ok.and.r%covariance_r(1,1)==1.d0.and.r%covariance_f(1,1)==1.d0)
  call push_velocity_sample(copy,1.d0,1.d0,[2.d0**40+2.d0,0.d0,0.d0],ok)
  call require(ok)
  call read_velocity_statistics(copy,other,ok)
  call require(ok.and.all(other%covariance_r==r%covariance_r))
  ! Save mid-window with a preceding sample, then continue across the saved boundary.
  call configure_velocity_statistics(s,0.d0,2.d0,ok)
  call require(ok)
  call push_velocity_sample(s,0.d0,1.d0,[0.d0,0.d0,0.d0],ok)
  call require(ok)
  call push_velocity_sample(s,1.d0,2.d0,[2.d0,0.d0,0.d0],ok)
  call require(ok)
  open(newunit=unit,status='scratch',access='stream',form='unformatted',convert='little_endian')
  call write_velocity_state(unit,s,'batch-a',1_int64,1.d0,ok)
  call require(ok)
  rewind(unit)
  call restore_velocity_state(unit,copy,'batch-a',1_int64,1.d0,0.d0,2.d0,ok)
  call require(ok)
  call push_velocity_sample(s,2.d0,3.d0,[4.d0,0.d0,0.d0],ok)
  call require(ok)
  call push_velocity_sample(copy,2.d0,3.d0,[4.d0,0.d0,0.d0],ok)
  call require(ok)
  call read_velocity_statistics(s,r,ok)
  call require(ok)
  call read_velocity_statistics(copy,other,ok)
  call require(ok.and.other%duration==r%duration.and.all(other%mean_r==r%mean_r))
  call require(all(other%covariance_r==r%covariance_r).and.all(other%covariance_f==r%covariance_f))
  rewind(unit)
  call restore_velocity_state(unit,copy,'batch-b',1_int64,1.d0,0.d0,2.d0,ok)
  call require(.not.ok)
  rewind(unit)
  call restore_velocity_state(unit,copy,'batch-a',2_int64,1.d0,0.d0,2.d0,ok)
  call require(.not.ok)
  rewind(unit)
  call restore_velocity_state(unit,copy,'batch-a',1_int64,1.5d0,0.d0,2.d0,ok)
  call require(.not.ok)
  rewind(unit)
  call restore_velocity_state(unit,copy,'batch-a',1_int64,1.d0,0.d0,3.d0,ok)
  call require(.not.ok)
  ! Failed restores do not replace valid state.
  call read_velocity_statistics(copy,other,ok)
  call require(ok.and.all(other%covariance_f==r%covariance_f))
  close(unit)
  open(newunit=unit,status='scratch',access='stream',form='unformatted',convert='little_endian')
  write(unit) 'ASTRVS01'
  rewind(unit)
  call restore_velocity_state(unit,copy,'batch-a',1_int64,1.d0,0.d0,2.d0,ok)
  call require(.not.ok)
  close(unit)
  print *, 'PASS: statistics stream roundtrip, continuation, identity/window/truncation rejection'
  print *, 'PASS: Reynolds/Favre definitions, clipping, central moments, rejection atomicity'
contains
  subroutine require(condition)
    logical,intent(in) :: condition
    if(.not.condition) error stop 'velocity statistics assertion failed'
  end subroutine
end program
