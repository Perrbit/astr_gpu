module insitu_wall_separation
  use iso_fortran_env, only: real64
  use ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: wall_crossing,find_wall_crossings,separation,reattachment,zero_interval
  integer,parameter :: separation=1,reattachment=2,zero_interval=3
  type :: wall_crossing
    integer :: kind=0,pair=0
    real(real64) :: lower=0.d0,upper=0.d0,bubble_length=0.d0
  end type
contains
  ! The caller must establish the positive streamwise direction and eligibility.
  subroutine find_wall_crossings(x,shear,events,pairs,ok)
    real(real64),intent(in) :: x(:),shear(:)
    type(wall_crossing),allocatable,intent(out) :: events(:)
    integer,intent(out) :: pairs
    logical,intent(out) :: ok
    type(wall_crossing),allocatable :: work(:)
    integer :: i,j,n,count,pending,status
    real(real64) :: scale,fraction,position,length
    ok=.false.; pairs=0
    n=size(x)
    if(n<2.or.size(shear)/=n) return
    if(.not.all(ieee_is_finite(x)).or..not.all(ieee_is_finite(shear))) return
    if(any(x(2:n)<=x(1:n-1))) return
    allocate(work(n),stat=status)
    if(status/=0) return
    i=1; count=0; pending=0
    do while(i<=n)
      if(shear(i)==0.d0) then
        j=i
        do while(j<n)
          if(shear(j+1)/=0.d0) exit
          j=j+1
        enddo
        count=count+1
        work(count)=wall_crossing(zero_interval,0,x(i),x(j),0.d0)
        pending=0; i=j+1
        cycle
      endif
      if(i<n) then
        if((shear(i)>0.d0.and.shear(i+1)<0.d0).or. &
           (shear(i)<0.d0.and.shear(i+1)>0.d0)) then
          scale=max(abs(shear(i)),abs(shear(i+1)))
          fraction=(abs(shear(i))/scale)/(abs(shear(i))/scale+abs(shear(i+1))/scale)
          position=(1.d0-fraction)*x(i)+fraction*x(i+1)
          if(.not.ieee_is_finite(position)) return
          count=count+1
          work(count)%lower=position; work(count)%upper=position
          if(shear(i)>0.d0) then
            work(count)%kind=separation
            pending=count
          else
            work(count)%kind=reattachment
            if(pending>0) then
              length=position-work(pending)%lower
              if(.not.ieee_is_finite(length).or.length<=0.d0) return
              pairs=pairs+1
              work(count)%pair=pairs; work(pending)%pair=pairs
              work(count)%bubble_length=length; work(pending)%bubble_length=length
              pending=0
            endif
          endif
        endif
      endif
      i=i+1
    enddo
    allocate(events(count),stat=status)
    if(status/=0) return
    events=work(1:count)
    ok=.true.
  end subroutine
end module
