module wall_blowing_random
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use iso_fortran_env, only: int64
  implicit none
  private
  public :: sample_legacy_wall_velocity
  public :: legacy_random_state_size,get_legacy_random_state,put_legacy_random_state
  public :: legacy_random_runtime_signature
  logical,save :: first_call=.true.
contains
  integer function legacy_random_state_size() result(count)
    call random_seed(size=count)
    count=count+1
  end function

  subroutine get_legacy_random_state(state)
    integer,intent(out) :: state(:)
    if(size(state)/=legacy_random_state_size()) error stop 'wall RNG snapshot size mismatch'
    ! Unused ranks have no authoritative intrinsic seed until their first wall call.
    state=0
    state(1)=merge(0,1,first_call)
    if(.not.first_call) call random_seed(get=state(2:))
  end subroutine

  subroutine put_legacy_random_state(state,ok)
    integer,intent(in) :: state(:)
    logical,intent(out) :: ok
    ok=.false.
    if(size(state)/=legacy_random_state_size()) return
    if(state(1)/=0.and.state(1)/=1) return
    if(state(1)==0.and.any(state(2:)/=0)) return
    first_call=state(1)==0
    if(.not.first_call) call random_seed(put=state(2:))
    ok=.true.
  end subroutine

  subroutine legacy_random_runtime_signature(bits)
    integer(int64),intent(out) :: bits(8)
    integer,allocatable :: saved(:),probe(:)
    real(8) :: values(8)
    integer :: count,i
    call random_seed(size=count)
    allocate(saved(count),probe(count))
    call random_seed(get=saved)
    probe=1
    call random_seed(put=probe)
    do i=1,8
      call random_number(values(i))
    enddo
    bits=transfer(values,bits)
    call random_seed(put=saved)
  end subroutine

  subroutine sample_legacy_wall_velocity(rank,xcoord,zcoord,lz,uinf,amplitude, &
                                        xa,xb,xc,nmod_z,velocity)
    integer,intent(in) :: rank,nmod_z
    real(8),intent(in) :: xcoord(0:,0:),zcoord(0:,0:)
    real(8),intent(in) :: lz,uinf,amplitude,xa,xb,xc
    real(8),intent(out) :: velocity(0:,0:)
    integer :: i,k,m,seed_size
    integer,allocatable :: seed(:)
    real(8) :: theta,fx,gz,rfluc,sqrt27,unused_phase(15)
    real(8),parameter :: pi=3.1415926535897932384626433832795d0

    if(any(shape(xcoord)/=shape(zcoord)) .or. &
       any(shape(xcoord)/=shape(velocity))) error stop 'legacy wall array shape mismatch'
    if(.not.all(ieee_is_finite(xcoord)) .or. &
       .not.all(ieee_is_finite(zcoord)) .or. &
       .not.all(ieee_is_finite([lz,uinf,amplitude,xa,xb,xc]))) &
      error stop 'legacy wall inputs must be finite'
    if(lz<=0.d0 .or. xa>=xb .or. xb>=xc .or. nmod_z<1) &
      error stop 'legacy wall requires Lz>0, xa<xb<xc and nmod_z>=1'

    ! NVHPC rejects all-zero seeds; retain the approved rank-local draw order.
    if(first_call) then
      call random_seed(size=seed_size)
      allocate(seed(seed_size))
      seed=1
      call random_seed(put=seed)
      do m=1,15
        call random_number(unused_phase(m))
      enddo
      seed=rank+1
      call random_seed(put=seed)
      deallocate(seed)
      first_call=.false.
    endif

    sqrt27=1.d0/sqrt(27.d0)
    do k=0,ubound(velocity,2)
      do i=0,ubound(velocity,1)
        if(xcoord(i,k)<=xb .and. xcoord(i,k)>=xa) then
          theta=2.d0*pi*(xcoord(i,k)-xa)/(xb-xa)
          fx=4.d0*sin(theta)*(1.d0-cos(theta))*sqrt27
          gz=sin(2.d0*nmod_z*pi*(zcoord(i,k)/lz))
        elseif(xcoord(i,k)<=xc .and. xcoord(i,k)>=xb) then
          theta=2.d0*pi*(xcoord(i,k)-xb)/(xc-xb)
          fx=4.d0*sin(theta)*(1.d0-cos(theta))*sqrt27
          gz=sin(2.d0*nmod_z*pi*(zcoord(i,k)/lz)+0.5d0*pi)
        else
          velocity(i,k)=0.d0
          cycle
        endif
        call random_number(rfluc)
        rfluc=(rfluc*2.d0-1.d0)*0.1d0
        velocity(i,k)=amplitude*uinf*fx*gz*1.d0*(1.d0+rfluc)
      enddo
    enddo
    if(.not.all(ieee_is_finite(velocity))) &
      error stop 'legacy wall velocity must be finite'
  end subroutine sample_legacy_wall_velocity
end module wall_blowing_random
