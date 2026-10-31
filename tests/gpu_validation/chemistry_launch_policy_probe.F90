program chemistry_launch_policy_probe
  use iso_fortran_env, only: real64
  use chemistry_launch_policy
  implicit none
  type(air5_launch_learning) :: policies(5),policy
  integer :: step,candidate,round,position,context,expected
  logical :: sample
  real(real64) :: duration

  ! The first five complete steps must not create timing samples.
  call initialize_air5_launch_learning(policy,[.true.,.true.,.true.,.true.],3)
  do step=1,5
    call select_air5_launch_candidate(policy,step,candidate,sample)
    if(sample .or. candidate/=3 .or. policy%calls/=0) error stop 'warmup changed launch'
  enddo
  do round=1,6
    do position=1,4
      step=5+4*(round-1)+position
      call select_air5_launch_candidate(policy,step,candidate,sample)
      expected=modulo(position+round-2,4)+1
      if(.not.sample .or. candidate/=expected) error stop 'unbalanced rotating order'
      duration=real(candidate,real64)
      if(round==6 .and. candidate==1) duration=1000.0_real64
      call record_air5_launch_sample(policy,duration)
    enddo
  enddo
  if(.not.policy%frozen .or. policy%selected/=1) error stop 'median or freeze failure'
  if(any(policy%counts(1:4)/=6) .or. policy%milliseconds(6,1)/=1000.0_real64) &
    error stop 'sample or tail was lost'
  call select_air5_launch_candidate(policy,30,candidate,sample)
  if(sample .or. candidate/=1 .or. policy%calls/=24) error stop 'frozen launch sampled'

  ! Illegal candidates are excluded, not low-occupancy candidates.
  call initialize_air5_launch_learning(policy,[.true.,.false.,.true.,.false.,.true.],3)
  do position=1,3
    call select_air5_launch_candidate(policy,6,candidate,sample)
    if(candidate/=2*position-1 .or. .not.sample) error stop 'resource mask ignored'
    call record_air5_launch_sample(policy,1.0_real64)
  enddo
  call finish_air5_launch_learning(policy,'deadline')
  if(policy%selected/=3 .or. policy%complete_rounds/=1) error stop 'tie did not prefer original'

  ! A partial trailing round is retained but must not influence the choice.
  call initialize_air5_launch_learning(policy,[.true.,.true.,.true.,.true.,.true.],1)
  do position=1,7
    call select_air5_launch_candidate(policy,64,candidate,sample)
    duration=real(candidate,real64)
    if(position>5) duration=0.0_real64
    call record_air5_launch_sample(policy,duration)
  enddo
  call finish_air5_launch_learning(policy,'deadline')
  if(policy%selected/=1 .or. policy%complete_rounds/=1 .or. policy%calls/=7) &
    error stop 'partial round contaminated choice'
  if(sum(policy%counts)/=7) error stop 'partial samples discarded'

  call initialize_air5_launch_learning(policy,[.true.,.true.,.true.,.true.],3)
  call select_air5_launch_candidate(policy,64,candidate,sample)
  call record_air5_launch_sample(policy,0.0_real64)
  call select_air5_launch_candidate(policy,65,candidate,sample)
  if(sample .or. candidate/=3 .or. policy%reason/='no-complete-round') &
    error stop 'deadline failed to retain original'

  ! Independent contexts can choose different winners without sharing samples.
  do context=1,5
    call initialize_air5_launch_learning(policies(context),[.true.,.true.,.true.,.true.,.true.],1)
  enddo
  do position=1,30
    do context=1,5
      call select_air5_launch_candidate(policies(context),6,candidate,sample)
      duration=real(abs(candidate-context),real64)
      call record_air5_launch_sample(policies(context),duration)
    enddo
  enddo
  do context=1,5
    if(policies(context)%selected/=context .or. policies(context)%calls/=30) &
      error stop 'contexts shared learning state'
  enddo

  ! A restart reconstructs the policy; it does not retain a former winner.
  call initialize_air5_launch_learning(policies(5),[.true.,.true.,.true.,.true.,.true.],1)
  if(policies(5)%frozen .or. policies(5)%selected/=1 .or. policies(5)%calls/=0) &
    error stop 'restart did not relearn'
  call finish_air5_launch_learning(policies(5),'deadline')
  if(policies(5)%selected/=1 .or. policies(5)%reason/='no-complete-round') &
    error stop 'uncalled context failed'
  write(*,'(a)') 'AIR5_LAUNCH_POLICY_PASS'
end program chemistry_launch_policy_probe
