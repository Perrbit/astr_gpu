program chemistry_compensation_lifecycle_probe
  use iso_fortran_env, only: real64
  use mpi
  use chemistry_compensation
  implicit none
  real(real64) :: q,c,q0,c0,h,expected,mean_c
  real(real64) :: low(2,3,11),high(2,3,11),cl(2,3,11),ch(2,3,11)
  real(real64) :: state(11),carry(11),species(5),saved_carry
  integer :: iteration,ierr,rank,nranks,comm,lower,upper
  call MPI_Init(ierr)
  call MPI_Comm_dup(MPI_COMM_WORLD,comm,ierr)
  call MPI_Comm_rank(comm,rank,ierr)
  call MPI_Comm_size(comm,nranks,ierr)
  state=0.0_real64; carry=0.0_real64
  state(1)=1.0_real64
  state(6:10)=[0.5_real64,0.25_real64,0.125_real64,0.125_real64,0.0_real64]
  carry(7)=2.0_real64**(-57)
  carry(2)=2.0_real64**(-60)
  state(2)=1.0_real64
  saved_carry=carry(7)
  species=state(6:10)
  call compensated_filter_projection(state,carry,species,1)
  if(carry(7)/=saved_carry.or.carry(2)/=2.0_real64**(-60)) &
    error stop 'filter projection discarded unchanged low parts'
  if(carry(6)/=-saved_carry) error stop 'filter projection lost species carry closure'
  species=state(6:10); species(5)=2.0_real64**(-40)
  call compensated_filter_projection(state,carry,species,1)
  if(state(10)/=species(5).or.carry(10)/=0.0_real64) &
    error stop 'filter projection did not install corrected species target'
  if(carry(7)/=saved_carry) error stop 'filter projection changed unaffected species'
  if(rank==0) write(*,'(A)') 'COMPENSATED_FILTER_PROJECTION_PASS'
  q=1.0_real64; c=0.0_real64
  h=2.0_real64**(-55)
  do iteration=1,100000
    q0=q; c0=c
    call compensated_add(q,c,h)
    call compensated_rk(q,c,q0,c0,0.75_real64,0.25_real64,0.25_real64*h)
    call compensated_rk(q,c,q0,c0,1.0_real64/3.0_real64, &
      2.0_real64/3.0_real64,2.0_real64/3.0_real64*h)
  enddo
  expected=1.0_real64+100000.0_real64*h
  if(q/=expected) error stop 'SSP-RK constant RHS accumulation failed'
  if(abs(c)>epsilon(q)) error stop 'SSP-RK correction is not a low part'
  call compensated_mean(1.0_real64,1.0_real64+epsilon(q), &
    0.0_real64,0.0_real64,mean_c)
  if(mean_c/=-0.5_real64*epsilon(q)) error stop 'mean lost its rounding residual'
  ! A cancelling mean can retain a valid, non-normalized compensation pair.
  ! The checkpoint reader's spacing(high) bound is not guaranteed by this API.
  h=2.0_real64**(-53)
  call compensated_mean(1.0_real64,-1.0_real64,h,h,mean_c)
  if(mean_c/=h) error stop 'cancelling mean changed represented value'
  if(abs(mean_c)<=4.0_real64*spacing(0.0_real64)) &
    error stop 'cancellation diagnostic did not exercise the restart contract gap'
  if(rank==0) write(*,'(A)') 'COMPENSATION_MEAN_NONNORMAL_PAIR_REPRODUCED'
  q=0.0_real64; c=mean_c
  call compensated_normalize(q,c)
  if(q/=-h.or.c/=0.0_real64) error stop 'cancelling pair normalization lost value'
  q0=q; c0=c
  call compensated_normalize(q,c)
  if(q/=q0.or.c/=c0) error stop 'normalization is not idempotent'
  q=1.0_real64; c=2.0_real64**(-54)
  call compensated_normalize(q,c)
  if(q/=1.0_real64.or.c/=2.0_real64**(-54)) error stop 'normalization discarded low bits'
  q=3.3327160638660024e-23_real64; c=4.930380657631324e-32_real64
  q0=q; c0=c
  call compensated_normalize(q,c)
  ! Exact Decimal subtraction of the two binary64 inputs gives this binary64 value.
  if(q/=3.33271605893562170e-23_real64.or.c/=0.0_real64) &
    error stop 'observed checkpoint pair changed represented value'
  if(abs(c)>4.0_real64*spacing(q)) error stop 'observed checkpoint pair is not normalized'
  if(rank==0) write(*,'(A)') 'COMPENSATION_NORMALIZATION_PASS'
  call configure_air5_compensation(.true.,2,3,4)
  if(any(air5_carry/=0.0_real64)) error stop 'new state did not zero carry'
  air5_carry=1.0_real64
  call configure_air5_compensation(.true.,2,3,4)
  if(any(air5_carry/=0.0_real64)) error stop 'reinitialization retained old carry'
  call configure_air5_compensation(.false.,2,3,4)
  if(allocated(air5_carry)) error stop 'disabled state allocated carry'
  low=1.0_real64
  high=1.0_real64+epsilon(q)
  cl=2.0_real64**(-57)
  ch=-2.0_real64**(-56)
  lower=modulo(rank-1,nranks); upper=modulo(rank+1,nranks)
  call compensation_exchange_faces(low,high,cl,ch,lower,upper,comm)
  expected=0.5_real64*(2.0_real64**(-57)-2.0_real64**(-56))-0.5_real64*epsilon(q)
  if(any(cl/=expected) .or. any(ch/=expected)) error stop 'MPI duplicate-node carry failed'
  low=0.0_real64; high=0.0_real64; cl=h; ch=h
  call compensation_normalize_faces(low,high,cl,ch,lower,upper)
  if(any(low/=-h).or.any(high/=-h).or.any(cl/=0.0_real64).or.any(ch/=0.0_real64)) &
    error stop 'MPI face normalization failed'
  low=3.0_real64; high=5.0_real64; cl=7.0_real64; ch=9.0_real64
  call compensation_exchange_faces(low,high,cl,ch,MPI_PROC_NULL,MPI_PROC_NULL,comm)
  if(any(cl/=7.0_real64) .or. any(ch/=9.0_real64)) error stop 'physical face modified'
  call compensation_normalize_faces(low,high,cl,ch,MPI_PROC_NULL,MPI_PROC_NULL)
  if(any(low/=3.0_real64).or.any(high/=5.0_real64).or.any(cl/=7.0_real64).or.any(ch/=9.0_real64)) &
    error stop 'normalization modified physical face'
  if(rank==0) write(*,'(A,I0)') 'COMPENSATION_LIFECYCLE_PRIMITIVES_PASS ranks=',nranks
  call MPI_Comm_free(comm,ierr)
  call MPI_Finalize(ierr)
end program
