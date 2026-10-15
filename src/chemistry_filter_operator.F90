module chemistry_filter_operator
  use iso_fortran_env, only: real64
  use chemistry_model, only: air5_ratio_bisection_iterations,air5_transport_roundoff_gamma
  use chemistry_filter_math, only: filter_lower_energy_admissible,filter_pressure_admissible, &
    filter_feasible_mass,filter_pressure_pair_safe
  implicit none
  private
  public :: filter_node,filter_high_face,face_bounds,lower_admissible,pair_admissible,thermal_limit
  real(real64), parameter :: coefficients(5)=[105.0_real64/512.0_real64, &
    -15.0_real64/128.0_real64,45.0_real64/1024.0_real64,-5.0_real64/512.0_real64,1.0_real64/1024.0_real64]
  integer, parameter :: difference4(0:4)=[1,-4,6,-4,1]
contains
#ifdef _CUDA
  attributes(host,device) &
#endif
  subroutine filter_node(axis,r,t1,t2,a)
    integer, value :: axis,r,t1,t2
    integer, intent(out) :: a(3)
    select case(axis)
    case(1); a=[r,t1,t2]
    case(2); a=[t1,r,t2]
    case(3); a=[t1,t2,r]
    end select
  end subroutine filter_node
#ifdef _CUDA
  attributes(host,device) &
#endif
  subroutine face_bounds(left,right,high,lower,upper,mass)
    real(real64), intent(in) :: left(11),right(11),high(11)
    real(real64), intent(out) :: lower(5),upper(5),mass
    real(real64) :: half
    integer :: s
    do s=1,5
      half=0.5_real64*left(s+5)
      if(half>0.0_real64) half=nearest(half,-1.0_real64)
      lower(s)=-half
      half=0.5_real64*right(s+5)
      if(half>0.0_real64) half=nearest(half,-1.0_real64)
      upper(s)=half
    enddo
    call filter_feasible_mass(high(6:10),lower,upper,high(1),mass)
  end subroutine

#ifdef _CUDA
  attributes(host,device) &
#endif
  logical function lower_admissible(state,tr_floor,v_floor)
    real(real64), intent(in) :: state(11),tr_floor(5),v_floor(5)
    lower_admissible=.false.
    if(any(state/=state).or.any(abs(state)>huge(1.0_real64))) return
    if(state(1)<=0.0_real64.or.any(state(6:10)<0.0_real64)) return
    lower_admissible=filter_lower_energy_admissible(state,tr_floor,v_floor)
    if(lower_admissible) lower_admissible=filter_pressure_admissible(state)
  end function

#ifdef _CUDA
  attributes(host,device) &
#endif
  logical function pair_admissible(left,right,flux,tr_floor,v_floor, &
    left_magnitude,right_magnitude,use_carry)
    real(real64), intent(in) :: left(11),right(11),flux(11),tr_floor(5),v_floor(5)
    real(real64), intent(in) :: left_magnitude(11),right_magnitude(11)
    logical, intent(in) :: use_carry
    real(real64) :: a(11),b(11),negative(11)
    negative=-flux
    a=left+2.0_real64*flux
    b=right-2.0_real64*flux
    pair_admissible=lower_admissible(a,tr_floor,v_floor).and.lower_admissible(b,tr_floor,v_floor)
    if(.not.pair_admissible) return
    pair_admissible=filter_pressure_pair_safe(left,left_magnitude,flux,use_carry).and. &
      filter_pressure_pair_safe(right,right_magnitude,negative,use_carry)
  end function

#ifdef _CUDA
  attributes(host,device) &
#endif
  subroutine thermal_limit(left,right,candidate,tr_floor,v_floor,flux,ratio,status, &
    left_magnitude,right_magnitude,use_carry)
    real(real64), intent(in) :: left(11),right(11),candidate(11),tr_floor(5),v_floor(5)
    real(real64), intent(in) :: left_magnitude(11),right_magnitude(11)
    logical, intent(in) :: use_carry
    real(real64), intent(out) :: flux(11),ratio
    integer, intent(out) :: status
    real(real64) :: lower,upper,middle,zero(11)
    integer :: iteration
    flux=candidate
    ratio=1.0_real64
    status=0
    if(.not.lower_admissible(left,tr_floor,v_floor).or..not.lower_admissible(right,tr_floor,v_floor)) then
      status=3
      return
    endif
    zero=0.0_real64
    ! A near-bound node freezes every incident face, not just one failed exchange.
    if(.not.filter_pressure_pair_safe(left,left_magnitude,zero,use_carry).or. &
       .not.filter_pressure_pair_safe(right,right_magnitude,zero,use_carry)) then
      flux=0.0_real64; ratio=0.0_real64
      return
    endif
    if(pair_admissible(left,right,flux,tr_floor,v_floor,left_magnitude,right_magnitude,use_carry)) return
    lower=0.0_real64
    upper=1.0_real64
    do iteration=1,air5_ratio_bisection_iterations
      middle=0.5_real64*(lower+upper)
      ! Test the rounded exchange that the node update will actually consume.
      flux=middle*candidate
      if(pair_admissible(left,right,flux,tr_floor,v_floor,left_magnitude,right_magnitude,use_carry)) then
        lower=middle
      else
        upper=middle
      endif
    enddo
    ! Reuse the existing operation-derived inward allowance, not a state floor.
    ratio=lower/(1.0_real64+air5_transport_roundoff_gamma)
    if(ratio>0.0_real64) ratio=nearest(ratio,-1.0_real64)
    flux=ratio*candidate
    if(.not.pair_admissible(left,right,flux,tr_floor,v_floor,left_magnitude,right_magnitude,use_carry)) status=3
  end subroutine



#ifdef _CUDA
  attributes(host,device) &
#endif
  subroutine filter_high_face(stencil,carry,global_face,global_intervals,periodic,use_carry,high)
    real(real64), intent(in) :: stencil(11,-4:5),carry(11,-4:5)
    integer, value :: global_face,global_intervals
    logical, intent(in) :: periodic,use_carry
    real(real64), intent(out) :: high(11)
    real(real64) :: pairs,difference5,pair
    integer :: m,r,s,row,k
    high=0.0_real64
    if(periodic) then
      do s=1,11
        do m=1,5
          pairs=0.0_real64
          do r=0,m-1
            pair=stencil(s,r+1)-stencil(s,-r)
            if(use_carry) pair=pair-(carry(s,r+1)-carry(s,-r))
            pairs=pairs+pair
          enddo
          high(s)=high(s)+coefficients(m)*pairs
        enddo
      enddo
    else
      if(global_face<=0.or.global_face>=global_intervals-1) return
      do s=1,11
        do row=max(1,global_face-4),min(global_face,global_intervals-6)
          difference5=0.0_real64
          do k=0,4
            r=row+k-global_face
            pair=stencil(s,r+1)-stencil(s,r)
            if(use_carry) pair=pair-(carry(s,r+1)-carry(s,r))
            difference5=difference5+difference4(k)*pair
          enddo
          high(s)=high(s)+difference4(global_face-row)*difference5
        enddo
        high(s)=high(s)*(1.0_real64/1024.0_real64)
      enddo
    endif
  end subroutine filter_high_face
end module chemistry_filter_operator
