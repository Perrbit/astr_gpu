module insitu_resource_budget
  use iso_fortran_env, only: int64
  implicit none
  private
  public :: resource_budget, configure_budget, reserve_bytes, release_bytes, checked_bytes
  integer,parameter :: slots=64
  type :: resource_budget
    private
    logical :: configured=.false.
    integer(int64) :: limit=0, headroom=0, used=0, peak=0
    integer(int64) :: reserved(slots)=0
  contains
    procedure :: current_bytes
    procedure :: peak_bytes
  end type
contains
  pure subroutine checked_bytes(extents,element_bytes,bytes,ok)
    integer(int64),intent(in) :: extents(:),element_bytes
    integer(int64),intent(out) :: bytes
    logical,intent(out) :: ok
    integer :: i
    bytes=0
    ok=.false.
    if(element_bytes<=0.or.any(extents<0)) return
    if(any(extents==0)) then
      ok=.true.
      return
    endif
    bytes=element_bytes
    do i=1,size(extents)
      if(extents(i)>huge(bytes)/bytes) then
        bytes=0
        return
      endif
      bytes=bytes*extents(i)
    enddo
    ok=.true.
  end subroutine

  pure subroutine configure_budget(budget,limit,headroom,ok)
    type(resource_budget),intent(inout) :: budget
    integer(int64),intent(in) :: limit,headroom
    logical,intent(out) :: ok
    ok=.false.
    if(budget%used/=0.or.limit<=0.or.headroom<0) return
    budget%limit=limit
    budget%headroom=headroom
    budget%peak=0
    budget%reserved=0
    budget%configured=.true.
    ok=.true.
  end subroutine

  pure subroutine reserve_bytes(budget,slot,bytes,available,ok)
    type(resource_budget),intent(inout) :: budget
    integer,intent(in) :: slot
    integer(int64),intent(in) :: bytes,available
    logical,intent(out) :: ok
    ok=.false.
    if(.not.budget%configured.or.slot<1.or.slot>slots) return
    if(bytes<=0.or.available<0) return
    if(budget%reserved(slot)/=0) return
    ! Subtraction avoids overflow in both the limit and free-memory checks.
    if(bytes>budget%limit-budget%used) return
    if(available<budget%headroom) return
    if(bytes>available-budget%headroom) return
    budget%reserved(slot)=bytes
    budget%used=budget%used+bytes
    budget%peak=max(budget%peak,budget%used)
    ok=.true.
  end subroutine

  pure subroutine release_bytes(budget,slot,ok)
    type(resource_budget),intent(inout) :: budget
    integer,intent(in) :: slot
    logical,intent(out) :: ok
    ok=.false.
    if(.not.budget%configured.or.slot<1.or.slot>slots) return
    if(budget%reserved(slot)==0) return
    budget%used=budget%used-budget%reserved(slot)
    budget%reserved(slot)=0
    ok=.true.
  end subroutine

  pure integer(int64) function current_bytes(budget)
    class(resource_budget),intent(in) :: budget
    current_bytes=budget%used
  end function

  pure integer(int64) function peak_bytes(budget)
    class(resource_budget),intent(in) :: budget
    peak_bytes=budget%peak
  end function
end module
