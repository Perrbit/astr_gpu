module chemistry_filter_math
  use iso_fortran_env, only: real64
  use chemistry_air5_data, only: air5_formation_energy,air5_cv_tr_factor,air5_pressure_min_pa
  implicit none
  private
  public :: filter_lower_energy_admissible,filter_pressure_admissible,filter_feasible_mass,filter_pressure_energy_floor
  public :: filter_pressure_pair_safe,filter_pressure_represented_admissible,filter_state_magnitude
  real(real64), parameter :: filter_pressure_energy_floor=maxval(air5_cv_tr_factor)*air5_pressure_min_pa
  integer, parameter :: expansion_size=32
  real(real64), parameter :: splitter=134217729.0_real64
  real(real64), parameter :: unit_roundoff=epsilon(1.0_real64)/2.0_real64
  real(real64), parameter :: sign_gamma=32.0_real64*unit_roundoff/(1.0_real64-32.0_real64*unit_roundoff)
contains

#ifdef _CUDA
  attributes(host,device) &
#endif
  subroutine two_product(a,b,high,low)
    real(real64), intent(in) :: a,b
    real(real64), intent(out) :: high,low
    real(real64) :: split_a,split_b,ah,al,bh,bl,error1,error2,error3
    high=a*b
    split_a=splitter*a; split_b=splitter*b
    ah=split_a-(split_a-a); al=a-ah
    bh=split_b-(split_b-b); bl=b-bh
    error1=high-ah*bh
    error2=error1-al*bh
    error3=error2-ah*bl
    low=al*bl-error3
  end subroutine

#ifdef _CUDA
  attributes(host,device) &
#endif
  subroutine grow_expansion(expansion,length,term)
    real(real64), intent(inout) :: expansion(*)
    integer, intent(inout) :: length
    real(real64), intent(in) :: term
    real(real64) :: q,total,virtual_a,virtual_b,error_a,error_b,error,b
    integer :: i,out
    q=term; out=0
    do i=1,length
      b=expansion(i)
      total=q+b
      virtual_b=total-q
      virtual_a=total-virtual_b
      error_b=b-virtual_b
      error_a=q-virtual_a
      error=error_a+error_b
      if(error/=0.0_real64) then
        out=out+1; expansion(out)=error
      endif
      q=total
    enddo
    if(q/=0.0_real64.or.out==0) then
      out=out+1; expansion(out)=q
    endif
    length=out
  end subroutine

#ifdef _CUDA
  attributes(host,device) &
#endif
  subroutine add_product(expansion,length,a,b)
    real(real64), intent(inout) :: expansion(*)
    integer, intent(inout) :: length
    real(real64), intent(in) :: a,b
    real(real64) :: high,low
    call two_product(a,b,high,low)
    call grow_expansion(expansion,length,low)
    call grow_expansion(expansion,length,high)
  end subroutine

#ifdef _CUDA
  attributes(host,device) &
#endif
  logical function vibration_sign(state,floor)
    real(real64), intent(in) :: state(11),floor(5)
    real(real64) :: expansion(expansion_size)
    integer :: length,s
    length=0
    call grow_expansion(expansion,length,state(11))
    do s=1,5
      call add_product(expansion,length,-state(s+5),floor(s))
    enddo
    vibration_sign=expansion(length)>=0.0_real64
  end function

#ifdef _CUDA
  attributes(host,device) &
#endif
  logical function mass_in_exact_bounds(mass,lower,upper)
    real(real64), intent(in) :: mass,lower(5),upper(5)
    real(real64) :: low(expansion_size),high(expansion_size)
    integer :: nl,nh,s
    nl=0; nh=0
    call grow_expansion(low,nl,mass)
    call grow_expansion(high,nh,-mass)
    do s=1,5
      call grow_expansion(low,nl,-lower(s))
      call grow_expansion(high,nh,upper(s))
    enddo
    mass_in_exact_bounds=low(nl)>=0.0_real64.and.high(nh)>=0.0_real64
  end function

#ifdef _CUDA
  attributes(host,device) &
#endif
  subroutine filter_feasible_mass(candidate,lower,upper,raw_mass,mass)
    real(real64), intent(in) :: candidate(5),lower(5),upper(5),raw_mass
    real(real64), intent(out) :: mass
    real(real64) :: magnitude,lo(5),hi(5),target
    mass=max(sum(lower),min(sum(upper),raw_mass))
    do
      magnitude=max(abs(mass),maxval(abs(candidate)),maxval(abs(lower)),maxval(abs(upper)))
      if(magnitude==0.0_real64) return
      lo=lower/magnitude; hi=upper/magnitude; target=mass/magnitude
      if(mass_in_exact_bounds(mass,lower,upper).and.mass_in_exact_bounds(target,lo,hi).and. &
         target>=sum(lo).and.target<=sum(hi)) return
      ! Here lower<=0<=upper. Move an outward-rounded target one ULP inward.
      ! Zero is feasible, so this finite-representation projection terminates.
      mass=nearest(mass,-sign(1.0_real64,mass))
    enddo
  end subroutine

#ifdef _CUDA
  attributes(host,device) &
#endif
  logical function thermal_sign(state,floor,offset)
    real(real64), intent(in) :: state(11),floor(5),offset
    real(real64) :: thermal(expansion_size),result(expansion_size),twice_rho
    integer :: length,result_length,s,i
    length=0; result_length=0
    call grow_expansion(thermal,length,state(5))
    call grow_expansion(thermal,length,-state(11))
    call grow_expansion(thermal,length,-offset)
    do s=1,5
      call add_product(thermal,length,-state(s+5),floor(s))
    enddo
    ! rho is positive: test 2*rho*(E-Ev-sum(z*h)-offset)-sum(momentum^2).
    ! At most 13 terms before scaling, then 26+6 product terms fit 32 slots.
    twice_rho=2.0_real64*state(1)
    do i=1,length
      call add_product(result,result_length,twice_rho,thermal(i))
    enddo
    do i=2,4
      call add_product(result,result_length,-state(i),state(i))
    enddo
    thermal_sign=result(result_length)>=0.0_real64
  end function

#ifdef _CUDA
  attributes(host,device) &
#endif
  logical function filter_lower_energy_admissible(state,tr_floor,v_floor)
    real(real64), intent(in) :: state(11),tr_floor(5),v_floor(5)
    real(real64) :: thermal,vibration,kinetic,tr_scale,v_scale,term,margin,bound
    logical :: vibration_ok,thermal_ok
    integer :: s
    thermal=state(5)-state(11); vibration=state(11)
    tr_scale=abs(state(5))+abs(state(11)); v_scale=abs(state(11))
    do s=1,5
      term=state(s+5)*tr_floor(s)
      thermal=thermal-term; tr_scale=tr_scale+abs(term)
      term=state(s+5)*v_floor(s)
      vibration=vibration-term; v_scale=v_scale+abs(term)
    enddo
    kinetic=((state(2)*state(2)+state(3)*state(3))+state(4)*state(4))/(2.0_real64*state(1))
    margin=thermal-kinetic; tr_scale=tr_scale+abs(kinetic)
    ! gamma_32 bounds this bounded probe's product/subtraction/scale operations.
    ! It chooses the exact branch; it never accepts a negative residual.
    bound=nearest(sign_gamma*v_scale,1.0_real64)
    if(abs(vibration)>bound) then
      vibration_ok=vibration>0.0_real64
    else
      vibration_ok=vibration_sign(state,v_floor)
    endif
    if(.not.vibration_ok) then
      filter_lower_energy_admissible=.false.
      return
    endif
    bound=nearest(sign_gamma*tr_scale,1.0_real64)
    if(abs(margin)>bound) then
      thermal_ok=margin>0.0_real64
    else
      thermal_ok=thermal_sign(state,tr_floor,0.0_real64)
    endif
    filter_lower_energy_admissible=thermal_ok
  end function

#ifdef _CUDA
  attributes(host,device) &
#endif
  logical function filter_pressure_admissible(state)
    real(real64), intent(in) :: state(11)
    real(real64) :: residual,scale,term,kinetic,bound
    integer :: s
    residual=(state(5)-state(11))-filter_pressure_energy_floor
    scale=abs(state(5))+abs(state(11))+filter_pressure_energy_floor
    do s=1,5
      term=state(s+5)*air5_formation_energy(s)
      residual=residual-term; scale=scale+abs(term)
    enddo
    kinetic=((state(2)*state(2)+state(3)*state(3))+state(4)*state(4))/(2.0_real64*state(1))
    residual=residual-kinetic; scale=scale+abs(kinetic)
    bound=nearest(sign_gamma*scale,1.0_real64)
    if(abs(residual)>bound) then
      filter_pressure_admissible=residual>0.0_real64
    else
      filter_pressure_admissible=thermal_sign(state,air5_formation_energy,filter_pressure_energy_floor)
    endif
  end function

#ifdef _CUDA
  attributes(host,device) &
#endif
  subroutine filter_state_magnitude(high,carry,magnitude)
    real(real64), intent(in) :: high(11),carry(11)
    real(real64), intent(out) :: magnitude(11)
    integer :: s
    do s=1,11
      magnitude(s)=abs(high(s))+abs(carry(s))
      if(magnitude(s)>0.0_real64) magnitude(s)=nearest(magnitude(s),1.0_real64)
    enddo
  end subroutine

#ifdef _CUDA
  attributes(host,device) &
#endif
  logical function filter_pressure_pair_safe(state,magnitude,flux,use_carry)
    real(real64), intent(in) :: state(11),magnitude(11),flux(11)
    logical, intent(in) :: use_carry
    real(real64) :: center(11),worst(11),radius(11),scale,coefficient,commit_gamma,pair_gamma
    integer :: s
    ! Componentwise construction + commit boxes; see the T3 finite-precision derivation.
    if(use_carry) then
      commit_gamma=nearest(13.0_real64*unit_roundoff/(1.0_real64-13.0_real64*unit_roundoff),1.0_real64)
      pair_gamma=nearest(3.0_real64*unit_roundoff/(1.0_real64-3.0_real64*unit_roundoff),1.0_real64)
    else
      commit_gamma=nearest(2.0_real64*unit_roundoff/(1.0_real64-2.0_real64*unit_roundoff),1.0_real64)
      pair_gamma=unit_roundoff
    endif
    coefficient=nearest(commit_gamma+pair_gamma,1.0_real64)
    center=state+2.0_real64*flux; worst=center
    do s=1,11
      scale=magnitude(s)+2.0_real64*abs(flux(s))
      radius(s)=0.0_real64
      if(scale>0.0_real64) then
        scale=nearest(scale,1.0_real64)
        radius(s)=nearest(coefficient*scale,1.0_real64)
      endif
    enddo
    do s=1,11
      if(radius(s)==0.0_real64) cycle
      select case(s)
      case(1,5)
        worst(s)=nearest(center(s)-radius(s),-1.0_real64)
      case(2:4)
        worst(s)=nearest(abs(center(s))+radius(s),1.0_real64)
      case(6:10)
        if(air5_formation_energy(s-5)>=0.0_real64) then
          worst(s)=nearest(center(s)+radius(s),1.0_real64)
        else
          worst(s)=nearest(center(s)-radius(s),-1.0_real64)
        endif
      case(11)
        worst(s)=nearest(center(s)+radius(s),1.0_real64)
      end select
    enddo
    filter_pressure_pair_safe=.false.
    if(worst(1)<=0.0_real64) return
    filter_pressure_pair_safe=filter_pressure_admissible(worst)
  end function

#ifdef _CUDA
  attributes(host,device) &
#endif
  logical function represented_pressure_sign(high,carry)
    real(real64), intent(in) :: high(11),carry(11)
    real(real64) :: thermal(128),result(128)
    integer :: length,result_length,s,i
    length=0; result_length=0
    call grow_expansion(thermal,length,high(5))
    call grow_expansion(thermal,length,-carry(5))
    call grow_expansion(thermal,length,-high(11))
    call grow_expansion(thermal,length,carry(11))
    call grow_expansion(thermal,length,-filter_pressure_energy_floor)
    do s=1,5
      call add_product(thermal,length,-high(s+5),air5_formation_energy(s))
      call add_product(thermal,length,carry(s+5),air5_formation_energy(s))
    enddo
    ! 25 thermal terms * two rho parts * two product parts + 18 kinetic terms <= 128.
    do i=1,length
      call add_product(result,result_length,2.0_real64*high(1),thermal(i))
      call add_product(result,result_length,-2.0_real64*carry(1),thermal(i))
    enddo
    do s=2,4
      call add_product(result,result_length,-high(s),high(s))
      call add_product(result,result_length,2.0_real64*high(s),carry(s))
      call add_product(result,result_length,-carry(s),carry(s))
    enddo
    represented_pressure_sign=result(result_length)>=0.0_real64
  end function

#ifdef _CUDA
  attributes(host,device) &
#endif
  logical function filter_pressure_represented_admissible(high,carry)
    real(real64), intent(in) :: high(11),carry(11)
    real(real64) :: state(11),magnitude(11),residual,scale,kinetic,bound
    integer :: s
    state=high-carry
    filter_pressure_represented_admissible=.false.
    if(state(1)<=0.0_real64) return
    call filter_state_magnitude(high,carry,magnitude)
    residual=(state(5)-state(11))-filter_pressure_energy_floor
    scale=magnitude(5)+magnitude(11)+filter_pressure_energy_floor
    do s=1,5
      residual=residual-state(s+5)*air5_formation_energy(s)
      scale=scale+magnitude(s+5)*abs(air5_formation_energy(s))
    enddo
    kinetic=((state(2)*state(2)+state(3)*state(3))+state(4)*state(4))/(2.0_real64*state(1))
    residual=residual-kinetic; scale=scale+abs(kinetic)
    bound=nearest((64.0_real64*unit_roundoff/(1.0_real64-64.0_real64*unit_roundoff))*scale,1.0_real64)
    if(abs(residual)>bound) then
      filter_pressure_represented_admissible=residual>0.0_real64
    else
      filter_pressure_represented_admissible=represented_pressure_sign(high,carry)
    endif
  end function
end module
