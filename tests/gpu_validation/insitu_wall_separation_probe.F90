program insitu_wall_separation_probe
  use iso_fortran_env, only: real64
  use ieee_arithmetic, only: ieee_value,ieee_quiet_nan
  use insitu_wall_separation
  implicit none
  type(wall_crossing),allocatable :: events(:)
  real(real64) :: nan
  integer :: pairs
  logical :: ok
  nan=ieee_value(0.d0,ieee_quiet_nan)
  call find_wall_crossings([0.d0,1.d0,3.d0],[1.d0,-1.d0,1.d0],events,pairs,ok)
  call require(ok.and.pairs==1.and.size(events)==2,'complete pair')
  call require(events(1)%kind==separation.and.events(2)%kind==reattachment,'signed transitions')
  call require(events(1)%lower==.5d0.and.events(2)%lower==2.d0,'linear positions')
  call require(all(events%pair==1).and.all(events%bubble_length==1.5d0),'length')
  call find_wall_crossings([0.d0,1.d0,2.d0,3.d0,4.d0],[1.d0,-1.d0,1.d0,-1.d0,1.d0],events,pairs,ok)
  call require(ok.and.pairs==2.and.size(events)==4,'multiple bubbles')
  call find_wall_crossings([0.d0,1.d0],[-1.d0,1.d0],events,pairs,ok)
  call require(ok.and.pairs==0.and.size(events)==1,'unpaired reattachment')
  call require(events(1)%pair==0,'unpaired identity')
  call find_wall_crossings([0.d0,1.d0,2.d0,3.d0],[1.d0,0.d0,0.d0,-1.d0],events,pairs,ok)
  call require(ok.and.pairs==0.and.size(events)==1,'zero plateau')
  call require(events(1)%kind==zero_interval.and.events(1)%lower==1.d0.and.events(1)%upper==2.d0,'interval')
  call find_wall_crossings([0.d0,1.d0,2.d0],[1.d0,0.d0,-1.d0],events,pairs,ok)
  call require(ok.and.pairs==0.and.size(events)==1,'isolated zero')
  call require(events(1)%lower==events(1)%upper.and.events(1)%pair==0,'ambiguous zero')
  call find_wall_crossings([0.d0,1.d0,2.d0,3.d0,4.d0],[1.d0,-1.d0,0.d0,-1.d0,1.d0],events,pairs,ok)
  call require(ok.and.pairs==0.and.size(events)==3.and.all(events%pair==0),'zero breaks pair')
  call find_wall_crossings([0.d0,1.d0],[1.d0,1.d0],events,pairs,ok)
  call require(ok.and.pairs==0.and.size(events)==0,'no crossing')
  call find_wall_crossings([0.d0,1.d0],[0.d0,0.d0],events,pairs,ok)
  call require(ok.and.size(events)==1.and.events(1)%kind==zero_interval,'all zero')
  call find_wall_crossings([0.d0,1.d0],[huge(1.d0),-huge(1.d0)],events,pairs,ok)
  call require(ok.and.events(1)%lower==.5d0,'no overflow in interpolation')
  call find_wall_crossings([0.d0,0.d0],[1.d0,-1.d0],events,pairs,ok)
  call require(.not.ok,'reject nonmonotone coordinates')
  call find_wall_crossings([0.d0,1.d0],[nan,-1.d0],events,pairs,ok)
  call require(.not.ok,'reject nonfinite shear')
  print '(A)','PASS: wall separation synthetic gates'
contains
  subroutine require(condition,message)
    logical,intent(in) :: condition
    character(*),intent(in) :: message
    if(condition) return
    print '(A)','FAIL: '//message
    error stop 1
  end subroutine
end program
