program me6opti_symbol_audit
  use astr_precision, only: real_kind
  implicit none
  real(real_kind),parameter :: me6opti_straight_table(9,3)=reshape((/          &
    -3.e0_real_kind/1250.e0_real_kind,89141.e0_real_kind/4480000.e0_real_kind, &
    -49133.e0_real_kind/640000.e0_real_kind,            &
    411173.e0_real_kind/1920000.e0_real_kind,-174629.e0_real_kind/128000.e0_real_kind, &
    851641.e0_real_kind/640000.e0_real_kind,    &
    -282149.e0_real_kind/1920000.e0_real_kind,18413.e0_real_kind/640000.e0_real_kind, &
    -13877.e0_real_kind/4480000.e0_real_kind,   &
    459.e0_real_kind/4480000.e0_real_kind,-547.e0_real_kind/4480000.e0_real_kind, &
    -1289.e0_real_kind/640000.e0_real_kind,          &
    2703.e0_real_kind/640000.e0_real_kind,18379.e0_real_kind/384000.e0_real_kind, &
    -738047.e0_real_kind/640000.e0_real_kind,        &
    742461.e0_real_kind/640000.e0_real_kind,-820391.e0_real_kind/13440000.e0_real_kind, &
    9167.e0_real_kind/2240000.e0_real_kind,   &
    -3377.e0_real_kind/2240000.e0_real_kind,36157.e0_real_kind/4480000.e0_real_kind, &
    -6141.e0_real_kind/640000.e0_real_kind,      &
    -20593.e0_real_kind/640000.e0_real_kind,16367.e0_real_kind/128000.e0_real_kind, &
    -296029.e0_real_kind/1920000.e0_real_kind,    &
    -618391.e0_real_kind/640000.e0_real_kind,4737907.e0_real_kind/4480000.e0_real_kind,                      &
    -400637.e0_real_kind/13440000.e0_real_kind /),(/9,3/))
  real(real_kind),parameter :: me6opti_interp_table(9,3)=reshape((/           &
    -661.e0_real_kind/819200.e0_real_kind,263.e0_real_kind/512000.e0_real_kind, &
    31573.e0_real_kind/1024000.e0_real_kind,           &
    -91107.e0_real_kind/512000.e0_real_kind,302761.e0_real_kind/409600.e0_real_kind, &
    43093.e0_real_kind/102400.e0_real_kind,       &
    6429.e0_real_kind/1024000.e0_real_kind,-12349.e0_real_kind/512000.e0_real_kind, &
    21511.e0_real_kind/4096000.e0_real_kind,      &
    -7673.e0_real_kind/4096000.e0_real_kind,9179.e0_real_kind/512000.e0_real_kind, &
    -15959.e0_real_kind/204800.e0_real_kind,       &
    106337.e0_real_kind/512000.e0_real_kind,-165879.e0_real_kind/409600.e0_real_kind, &
    456421.e0_real_kind/512000.e0_real_kind,    &
    408037.e0_real_kind/1024000.e0_real_kind,-3357.e0_real_kind/102400.e0_real_kind, &
    8279.e0_real_kind/4096000.e0_real_kind,      &
    8279.e0_real_kind/4096000.e0_real_kind,-10273.e0_real_kind/512000.e0_real_kind, &
    92869.e0_real_kind/1024000.e0_real_kind,      &
    -126827.e0_real_kind/512000.e0_real_kind,37877.e0_real_kind/81920.e0_real_kind, &
    -337743.e0_real_kind/512000.e0_real_kind,     &
    1086701.e0_real_kind/1024000.e0_real_kind,166763.e0_real_kind/512000.e0_real_kind,                       &
    -59769.e0_real_kind/4096000.e0_real_kind /),(/9,3/))
  real(real_kind),parameter :: me6opti_penalty_table(9,3)=reshape((/          &
    -13.e0_real_kind/1000.e0_real_kind,13.e0_real_kind/125.e0_real_kind, &
    -91.e0_real_kind/250.e0_real_kind,91.e0_real_kind/125.e0_real_kind,          &
    -91.e0_real_kind/100.e0_real_kind,91.e0_real_kind/125.e0_real_kind, &
    -91.e0_real_kind/250.e0_real_kind,13.e0_real_kind/125.e0_real_kind,           &
    -13.e0_real_kind/1000.e0_real_kind,-1.e0_real_kind/2000.e0_real_kind, &
    1.e0_real_kind/250.e0_real_kind,-7.e0_real_kind/500.e0_real_kind,           &
    7.e0_real_kind/250.e0_real_kind,-7.e0_real_kind/200.e0_real_kind,7.e0_real_kind/250.e0_real_kind, &
    -7.e0_real_kind/500.e0_real_kind,               &
    1.e0_real_kind/250.e0_real_kind,-1.e0_real_kind/2000.e0_real_kind, &
    -1.e0_real_kind/2000.e0_real_kind,1.e0_real_kind/250.e0_real_kind,             &
    -7.e0_real_kind/500.e0_real_kind,7.e0_real_kind/250.e0_real_kind, &
    -7.e0_real_kind/200.e0_real_kind,7.e0_real_kind/250.e0_real_kind,               &
    -7.e0_real_kind/500.e0_real_kind,1.e0_real_kind/250.e0_real_kind,-1.e0_real_kind/2000.e0_real_kind /),(/9,3/))

  real(real_kind),parameter :: pi=acos(-1.e0_real_kind)
  real(real_kind),parameter :: outer(3)=[75.e0_real_kind/64.e0_real_kind, &
    -25.e0_real_kind/384.e0_real_kind,3.e0_real_kind/640.e0_real_kind]
  real(real_kind) :: theta,straight_reference,mixed_reference
  real(real_kind) :: straight_error,donor_error,candidate_error,donor_cutoff,candidate_cutoff,paper_cutoff
  complex(real_kind) :: phase,interpolation,normal,penalty,dy,straight,donor,candidate
  integer :: q,p,h
  straight_error=0.e0_real_kind
  donor_error=0.e0_real_kind
  candidate_error=0.e0_real_kind
  do q=0,1024
    theta=pi*real(q,real_kind)/1024.e0_real_kind
    dy=cmplx(0.e0_real_kind,2.e0_real_kind*(0.75e0_real_kind*sin(theta) &
      -0.15e0_real_kind*sin(2.e0_real_kind*theta)+sin(3.e0_real_kind*theta)/60.e0_real_kind),real_kind)
    straight=0.e0_real_kind;donor=0.e0_real_kind;candidate=0.e0_real_kind
    do h=1,3
      interpolation=0.e0_real_kind;normal=0.e0_real_kind;penalty=0.e0_real_kind
      do p=-4,4
        phase=exp(cmplx(0.e0_real_kind,theta*real(p,real_kind),real_kind))
        interpolation=interpolation+me6opti_interp_table(p+5,h)*phase
        normal=normal+me6opti_straight_table(p+5,h)*phase
        penalty=penalty+me6opti_penalty_table(p+5,h)*phase
      enddo
      straight=straight+outer(h)*(normal+conjg(normal))
      ! Reproduce the donor's negative-half reflection without changing its code.
      donor=donor+outer(h)*((interpolation-conjg(interpolation))*dy+penalty-conjg(penalty))
      ! Diagnostic candidate only: negate the penalty on the negative half.
      candidate=candidate+outer(h)*((interpolation-conjg(interpolation))*dy+penalty+conjg(penalty))
    enddo
    straight_reference=-9223447.e0_real_kind/2880000.e0_real_kind &
      +2261149.e0_real_kind/600000.e0_real_kind*cos(theta) &
      -821149.e0_real_kind/1200000.e0_real_kind*cos(2.e0_real_kind*theta) &
      +1663447.e0_real_kind/12600000.e0_real_kind*cos(3.e0_real_kind*theta) &
      -461149.e0_real_kind/33600000.e0_real_kind*cos(4.e0_real_kind*theta)
    mixed_reference=-180127829.e0_real_kind/57600000.e0_real_kind &
      +28259327.e0_real_kind/7680000.e0_real_kind*cos(theta) &
      -81089207.e0_real_kind/115200000.e0_real_kind*cos(2.e0_real_kind*theta) &
      +7562747.e0_real_kind/38400000.e0_real_kind*cos(3.e0_real_kind*theta) &
      -671839.e0_real_kind/11520000.e0_real_kind*cos(4.e0_real_kind*theta) &
      +1784983.e0_real_kind/115200000.e0_real_kind*cos(5.e0_real_kind*theta) &
      -65173.e0_real_kind/23040000.e0_real_kind*cos(6.e0_real_kind*theta) &
      +25991.e0_real_kind/115200000.e0_real_kind*cos(7.e0_real_kind*theta)
    straight_error=max(straight_error,abs(straight-straight_reference))
    donor_error=max(donor_error,abs(donor-mixed_reference))
    candidate_error=max(candidate_error,abs(candidate-mixed_reference))
    if(q==1024) then
      donor_cutoff=real(donor,real_kind);candidate_cutoff=real(candidate,real_kind)
      paper_cutoff=mixed_reference
    endif
  enddo
  write(*,'(A,ES24.16)') 'ME6 straight Eq48 maximum_symbol_error=',straight_error
  write(*,'(A,ES24.16)') 'ME6 donor mixed Eq43 maximum_symbol_error=',donor_error
  write(*,'(A,ES24.16)') 'ME6 candidate mixed Eq43 maximum_symbol_error=',candidate_error
  write(*,'(A,3ES24.16)') 'ME6 cutoff donor/candidate/Eq43=',donor_cutoff,candidate_cutoff,paper_cutoff
  if(straight_error>2.e-10_real_kind) error stop 'ME6 straight reference mismatch'
  if(candidate_error>2.e-10_real_kind) error stop 'ME6 candidate does not recover Eq43'
  if(donor_error>2.e-10_real_kind) error stop 'ME6 donor mixed symbol mismatch; human decision required'
end program
