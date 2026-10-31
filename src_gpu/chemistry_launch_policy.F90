module chemistry_launch_policy
  use iso_fortran_env, only: real64
  use ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  integer, parameter, public :: air5_launch_max_candidates=5,air5_launch_rounds=6
  integer, parameter, public :: air5_launch_warmup=5,air5_launch_deadline=64

  type, public :: air5_launch_learning
    integer :: original=1,selected=1,eligible_count=0,calls=0,complete_rounds=0
    integer :: eligible(air5_launch_max_candidates)=0
    integer :: counts(air5_launch_max_candidates)=0
    real(real64) :: milliseconds(air5_launch_rounds,air5_launch_max_candidates)=0.0_real64
    logical :: frozen=.false.
    character(len=24) :: reason='learning'
  end type air5_launch_learning

  public :: initialize_air5_launch_learning,select_air5_launch_candidate
  public :: record_air5_launch_sample,finish_air5_launch_learning,air5_launch_median

contains

  subroutine initialize_air5_launch_learning(policy,legal,original)
    type(air5_launch_learning), intent(out) :: policy
    logical, intent(in) :: legal(:)
    integer, intent(in) :: original
    integer :: index

    if(size(legal)>air5_launch_max_candidates .or. original<1 .or. original>size(legal)) &
      error stop 'invalid AIR5 launch candidate table'
    if(.not.legal(original)) error stop 'original AIR5 launch is not legal on this device'
    policy=air5_launch_learning()
    policy%original=original
    policy%selected=original
    do index=1,size(legal)
      if(.not.legal(index)) cycle
      policy%eligible_count=policy%eligible_count+1
      policy%eligible(policy%eligible_count)=index
    enddo
  end subroutine initialize_air5_launch_learning

  integer function learning_candidate(policy) result(candidate)
    type(air5_launch_learning), intent(in) :: policy
    integer :: round,position
    round=policy%calls/policy%eligible_count
    position=modulo(policy%calls,policy%eligible_count)
    candidate=policy%eligible(modulo(position+round,policy%eligible_count)+1)
  end function learning_candidate

  subroutine select_air5_launch_candidate(policy,startup_step,candidate,sample)
    type(air5_launch_learning), intent(inout) :: policy
    integer, intent(in) :: startup_step
    integer, intent(out) :: candidate
    logical, intent(out) :: sample

    if(startup_step>air5_launch_deadline .and. .not.policy%frozen) &
      call finish_air5_launch_learning(policy,'deadline')
    sample=.not.policy%frozen .and. startup_step>air5_launch_warmup
    candidate=policy%selected
    if(sample) candidate=learning_candidate(policy)
  end subroutine select_air5_launch_candidate

  subroutine record_air5_launch_sample(policy,milliseconds)
    type(air5_launch_learning), intent(inout) :: policy
    real(real64), intent(in) :: milliseconds
    integer :: candidate,round

    if(.not.ieee_is_finite(milliseconds) .or. milliseconds<0.0_real64) &
      error stop 'invalid AIR5 GPU event duration'
    candidate=learning_candidate(policy)
    round=policy%calls/policy%eligible_count+1
    policy%milliseconds(round,candidate)=milliseconds
    policy%counts(candidate)=policy%counts(candidate)+1
    policy%calls=policy%calls+1
    policy%complete_rounds=policy%calls/policy%eligible_count
    if(policy%complete_rounds==air5_launch_rounds) &
      call finish_air5_launch_learning(policy,'samples-complete')
  end subroutine record_air5_launch_sample

  real(real64) function air5_launch_median(values) result(median)
    real(real64), intent(in) :: values(:)
    real(real64) :: sorted(size(values)),value
    integer :: item,index,count

    sorted=values
    count=size(values)
    do item=2,count
      value=sorted(item)
      index=item-1
      do while(index>=1)
        if(sorted(index)<=value) exit
        sorted(index+1)=sorted(index)
        index=index-1
      enddo
      sorted(index+1)=value
    enddo
    if(modulo(count,2)==1) then
      median=sorted((count+1)/2)
    else
      median=0.5_real64*sorted(count/2)+0.5_real64*sorted(count/2+1)
    endif
  end function air5_launch_median

  subroutine finish_air5_launch_learning(policy,reason)
    type(air5_launch_learning), intent(inout) :: policy
    character(*), intent(in) :: reason
    real(real64) :: best,score
    integer :: candidate,item,rounds

    if(policy%frozen) return
    policy%frozen=.true.
    policy%reason=reason
    rounds=policy%complete_rounds
    if(rounds==0) then
      policy%reason='no-complete-round'
      return
    endif
    policy%selected=policy%original
    best=air5_launch_median(policy%milliseconds(1:rounds,policy%original))
    do item=1,policy%eligible_count
      candidate=policy%eligible(item)
      score=air5_launch_median(policy%milliseconds(1:rounds,candidate))
      if(score<best) then
        best=score
        policy%selected=candidate
      endif
    enddo
  end subroutine finish_air5_launch_learning

end module chemistry_launch_policy
