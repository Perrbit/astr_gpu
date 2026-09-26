program chemistry_compensation_lifecycle_probe
  use iso_fortran_env, only: real64
  use mpi
  use chemistry_compensation
  implicit none
  real(real64) :: q,c,q0,c0,h,expected,mean_c
  real(real64) :: low(2,3,11),high(2,3,11),cl(2,3,11),ch(2,3,11)
  integer :: iteration,ierr,rank,nranks,comm,lower,upper
  call MPI_Init(ierr)
  call MPI_Comm_dup(MPI_COMM_WORLD,comm,ierr)
  call MPI_Comm_rank(comm,rank,ierr)
  call MPI_Comm_size(comm,nranks,ierr)
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
  low=3.0_real64; high=5.0_real64; cl=7.0_real64; ch=9.0_real64
  call compensation_exchange_faces(low,high,cl,ch,MPI_PROC_NULL,MPI_PROC_NULL,comm)
  if(any(cl/=7.0_real64) .or. any(ch/=9.0_real64)) error stop 'physical face modified'
  if(rank==0) write(*,'(A,I0)') 'COMPENSATION_LIFECYCLE_PRIMITIVES_PASS ranks=',nranks
  call MPI_Comm_free(comm,ierr)
  call MPI_Finalize(ierr)
end program
