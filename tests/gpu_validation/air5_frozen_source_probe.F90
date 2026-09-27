#ifdef _CUDA
module frozen_probe_kernel
  use iso_fortran_env, only: real64
  use cudafor
  use chemistry_source, only: air5_source_mode_frozen
  use chemistry_source_gpu, only: air5_instantaneous_source_jacobian_gpu
  use chemistry_ros2_gpu, only: air5_ros2_advance_gpu
  use chemistry_air5_data, only: air5_num_reactions
  implicit none
contains
  attributes(global) subroutine frozen_probe(rho,momentum,energy,state,seed,atol,final,carry,counts,zero_source)
    real(real64), value :: rho,energy
    real(real64), device :: momentum(3),state(6),seed(6),atol(6),final(6),carry(6)
    integer, device :: counts(5),zero_source(1)
    real(real64) :: q(6),c(6),h,source(6),jacobian(6,6),progress(air5_num_reactions)
    integer :: status,accepted,rejected,rhs,jac
    if(threadIdx%x/=1) return
    c=seed
    call air5_instantaneous_source_jacobian_gpu(rho,momentum,energy,state,source,jacobian,progress, &
      status,air5_source_mode_frozen)
    zero_source(1)=0
    if(status==0.and.all(source==0).and.all(jacobian==0).and.all(progress==0)) zero_source(1)=1
    call air5_ros2_advance_gpu(rho,momentum,energy,state,1.e-6_real64,1.e-8_real64,1.e-9_real64, &
      atol,1,q,h,accepted,rejected,rhs,jac,status,air5_source_mode_frozen,c)
    final=q
    carry=c
    counts=[status,accepted,rejected,rhs,jac]
  end subroutine
end module
#endif

program air5_frozen_source_probe
  use iso_fortran_env, only: real64,int64
  use ieee_arithmetic, only: ieee_value,ieee_quiet_nan
  use chemistry_air5_data, only: air5_num_reactions
  use chemistry_thermo, only: air5_ev_from_tv,air5_q5_from_state
  use chemistry_source, only: air5_instantaneous_source_jacobian, &
    air5_source_mode_frozen,air5_source_mode_coupled
  use chemistry_ros2, only: air5_ros2_advance,air5_ros2_fixed_step
#ifdef _CUDA
  use frozen_probe_kernel
  use gpu_check, only: sync_after_kernel
#endif
  implicit none
  real(real64) :: rho,momentum(3),energy,state(6),seed(6),atol(6),final(6),carry(6),h
  real(real64) :: source(6),jacobian(6,6),progress(air5_num_reactions),error(6)
  real(real64) :: default_q(6),explicit_q(6),default_carry(6),explicit_carry(6)
  integer :: status,accepted,rejected,rhs,jac,counts(5),default_counts(5),case_id,enabled
#ifdef _CUDA
  real(real64), device :: m_d(3),s_d(6),seed_d(6),atol_d(6),q_d(6),carry_d(6)
  integer, device :: counts_d(5),zero_d(1)
  integer :: zero(1)
#endif
  rho=.05_real64
  momentum=rho*[1200._real64,10._real64,-20._real64]
  do case_id=1,2
    if(case_id==1) then
      state(1:5)=rho*[.767_real64,.233_real64,0._real64,0._real64,0._real64]
    else
      state(1:5)=rho*[.55_real64,.15_real64,.10_real64,.12_real64,.08_real64]
    endif
    call air5_ev_from_tv(state(1:5),1500._real64,state(6),status)
    if(status/=0) error stop 'setup Ev'
    call air5_q5_from_state(rho,momentum,state(1:5),state(6),3000._real64,energy,status)
    if(status/=0) error stop 'setup E'
    atol(1:5)=1.e-13_real64*rho
    atol(6)=1.e-13_real64*state(6)
    seed=epsilon(rho)*max(abs(state),tiny(rho))*.125_real64
    seed(2)=-seed(2)
    seed(3)=-0._real64
    call air5_instantaneous_source_jacobian(rho,momentum,energy,state,source,jacobian,status, &
      air5_source_mode_frozen,progress)
    if(status/=0.or.any(source/=0).or.any(jacobian/=0).or.any(progress/=0)) &
      error stop 'frozen source not exactly zero'
    do enabled=0,1
      carry=seed
      if(enabled==1) then
        call air5_ros2_advance(rho,momentum,energy,state,1.e-6_real64,1.e-8_real64,1.e-9_real64, &
          atol,1,final,h,accepted,rejected,rhs,jac,status,air5_source_mode_frozen,carry)
      else
        call air5_ros2_advance(rho,momentum,energy,state,1.e-6_real64,1.e-8_real64,1.e-9_real64, &
          atol,1,final,h,accepted,rejected,rhs,jac,status,air5_source_mode_frozen)
      endif
      counts=[status,accepted,rejected,rhs,jac]
      call check_frozen()
    enddo
    call air5_ros2_fixed_step(rho,momentum,energy,state,1.e-6_real64,final,error,status, &
      source_mode=air5_source_mode_frozen,compensation=carry)
    if(status/=0.or.any(error/=0)) error stop 'frozen fixed step failed'
    call check_frozen()
#ifdef _CUDA
    m_d=momentum
    s_d=state
    seed_d=seed
    atol_d=atol
    call frozen_probe<<<1,32>>>(rho,m_d,energy,s_d,seed_d,atol_d,q_d,carry_d,counts_d,zero_d)
    call sync_after_kernel('frozen_source_probe',.true.)
    final=q_d
    carry=carry_d
    counts=counts_d
    zero=zero_d
    if(zero(1)/=1) error stop 'GPU frozen source not zero'
    call check_frozen()
#endif
    ! An omitted mode must remain exactly the explicit coupled path.
    default_carry=seed
    explicit_carry=seed
    call air5_ros2_advance(rho,momentum,energy,state,1.e-11_real64,1.e-11_real64,1.e-9_real64, &
      atol,200000,default_q,h,accepted,rejected,rhs,jac,status,compensation=default_carry)
    default_counts=[status,accepted,rejected,rhs,jac]
    call air5_ros2_advance(rho,momentum,energy,state,1.e-11_real64,1.e-11_real64,1.e-9_real64, &
      atol,200000,explicit_q,h,accepted,rejected,rhs,jac,status,air5_source_mode_coupled,explicit_carry)
    if(status/=0.or.any(default_counts/=[status,accepted,rejected,rhs,jac])) error stop 'default counts'
    if(any(transfer(default_q,[0_int64],6)/=transfer(explicit_q,[0_int64],6)).or. &
       any(transfer(default_carry,[0_int64],6)/=transfer(explicit_carry,[0_int64],6))) &
      error stop 'default changed'
  enddo
  state(1)=-rho
  call air5_ros2_advance(rho,momentum,energy,state,1.e-6_real64,1.e-8_real64,1.e-9_real64, &
    atol,1,final,h,accepted,rejected,rhs,jac,status,air5_source_mode_frozen)
  if(status==0) error stop 'frozen accepted invalid state'
  state(1)=ieee_value(rho,ieee_quiet_nan)
  call air5_ros2_advance(rho,momentum,energy,state,1.e-6_real64,1.e-8_real64,1.e-9_real64, &
    atol,1,final,h,accepted,rejected,rhs,jac,status,air5_source_mode_frozen)
  if(status==0) error stop 'frozen accepted nonfinite state'
  write(*,'(A)') 'FROZEN_SOURCE_STATE_CARRY_DEFAULT_PASS'
contains
  subroutine check_frozen()
    if(any(counts/=0)) error stop 'frozen advance attempted chemistry'
    if(any(transfer(final,[0_int64],6)/=transfer(state,[0_int64],6))) error stop 'frozen state changed'
    if(any(transfer(carry,[0_int64],6)/=transfer(seed,[0_int64],6))) error stop 'frozen carry changed'
  end subroutine
end program
