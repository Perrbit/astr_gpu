module insitu_velocity_statistics
  use iso_fortran_env, only: real64,int64,int32
  use ieee_arithmetic, only: ieee_is_finite
  use insitu_time_integral, only: clipped_trapezoid
  implicit none
  private
  public :: velocity_statistics,velocity_statistics_result
  public :: configure_velocity_statistics,push_velocity_sample,read_velocity_statistics
  public :: write_velocity_state,restore_velocity_state
  type :: velocity_statistics
    private
    real(real64) :: window_start=0,window_end=0,last_time=0,last_rho=0,last_u(3)=0
    real(real64) :: duration=0,mass_duration=0,mean_r(3)=0,mean_f(3)=0
    real(real64) :: central_r(3,3)=0,central_f(3,3)=0
    logical :: configured=.false.,has_previous=.false.
  end type
  type :: velocity_statistics_result
    real(real64) :: duration=0,mean_density=0
    real(real64) :: mean_r(3)=0,mean_f(3)=0,rms_r(3)=0,rms_f(3)=0
    real(real64) :: covariance_r(3,3)=0,covariance_f(3,3)=0,density_stress(3,3)=0
  end type
contains
  subroutine write_velocity_state(unit,s,batch,step,time,ok)
    integer,intent(in) :: unit
    type(velocity_statistics),intent(in) :: s
    character(*),intent(in) :: batch
    integer(int64),intent(in) :: step
    real(real64),intent(in) :: time
    logical,intent(out) :: ok
    character(64) :: identity
    integer :: ios
    ok=.false.
    if(.not.s%configured.or.len_trim(batch)==0.or.len_trim(batch)>64) return
    if(step<0.or..not.ieee_is_finite(time)) return
    if(s%has_previous.and.s%last_time>time) return
    identity=batch
    write(unit,iostat=ios) 'ASTRVS01',identity,step,time, &
      s%window_start,s%window_end,merge(1_int32,0_int32,s%has_previous), &
      s%last_time,s%last_rho,s%last_u,s%duration,s%mass_duration, &
      s%mean_r,s%mean_f,s%central_r,s%central_f
    ok=ios==0
  end subroutine

  subroutine restore_velocity_state(unit,s,batch,step,time,window_start,window_end,ok)
    integer,intent(in) :: unit
    type(velocity_statistics),intent(inout) :: s
    character(*),intent(in) :: batch
    integer(int64),intent(in) :: step
    real(real64),intent(in) :: time,window_start,window_end
    logical,intent(out) :: ok
    type(velocity_statistics) :: candidate
    character(8) :: magic
    character(64) :: identity
    integer(int64) :: saved_step
    integer(int32) :: previous
    real(real64) :: saved_time
    integer :: ios,i
    ok=.false.
    if(len_trim(batch)==0.or.len_trim(batch)>64.or.step<0) return
    if(.not.all(ieee_is_finite([time,window_start,window_end]))) return
    if(window_end<=window_start) return
    read(unit,iostat=ios) magic,identity,saved_step,saved_time, &
      candidate%window_start,candidate%window_end,previous, &
      candidate%last_time,candidate%last_rho,candidate%last_u,candidate%duration,candidate%mass_duration, &
      candidate%mean_r,candidate%mean_f,candidate%central_r,candidate%central_f
    if(ios/=0) return
    if(magic/='ASTRVS01'.or.identity/=batch.or.saved_step/=step) return
    if(.not.ieee_is_finite(saved_time).or.saved_time/=time) return
    if(candidate%window_start/=window_start.or.candidate%window_end/=window_end) return
    if(previous/=0.and.previous/=1) return
    if(.not.all(ieee_is_finite([candidate%last_time,candidate%last_rho,candidate%last_u, &
         candidate%duration,candidate%mass_duration,candidate%mean_r,candidate%mean_f]))) return
    if(.not.all(ieee_is_finite(candidate%central_r)).or. &
       .not.all(ieee_is_finite(candidate%central_f))) return
    if(candidate%duration<0.or.candidate%mass_duration<0) return
    if((candidate%duration==0).neqv.(candidate%mass_duration==0)) return
    if(previous==1) then
      if(candidate%last_rho<=0.or.candidate%last_time>time) return
    else
      if(candidate%duration/=0.or.candidate%mass_duration/=0) return
    endif
    if(any(candidate%central_r/=transpose(candidate%central_r))) return
    if(any(candidate%central_f/=transpose(candidate%central_f))) return
    do i=1,3
      if(candidate%central_r(i,i)<0.or.candidate%central_f(i,i)<0) return
    enddo
    candidate%has_previous=previous==1
    candidate%configured=.true.
    s=candidate
    ok=.true.
  end subroutine
  subroutine configure_velocity_statistics(s,a,b,ok)
    type(velocity_statistics),intent(out) :: s
    real(real64),intent(in) :: a,b
    logical,intent(out) :: ok
    ok=.false.
    if(.not.all(ieee_is_finite([a,b]))) return
    if(b<=a) return
    s%window_start=a
    s%window_end=b
    s%configured=.true.
    ok=.true.
  end subroutine

  subroutine add_weighted_state(weight,u,total,mean,central,ok)
    real(real64),intent(in) :: weight,u(3)
    real(real64),intent(inout) :: total,mean(3),central(3,3)
    logical,intent(out) :: ok
    real(real64) :: new_total,delta(3),ratio,factor
    integer :: i,j
    ok=.false.
    if(weight==0) then
      ok=.true.
      return
    endif
    new_total=total+weight
    if(.not.ieee_is_finite(new_total).or.new_total<=0) return
    if(total==0) then
      mean=u
    else
      delta=u-mean
      ratio=weight/new_total
      factor=(total/new_total)*weight
      mean=mean+ratio*delta
      do j=1,3
        do i=1,j
          central(i,j)=central(i,j)+(factor*delta(i))*delta(j)
          central(j,i)=central(i,j)
        enddo
      enddo
    endif
    if(.not.all(ieee_is_finite(mean)).or..not.all(ieee_is_finite(central))) return
    total=new_total
    ok=.true.
  end subroutine

  subroutine push_velocity_sample(s,t,rho,u,ok)
    type(velocity_statistics),intent(inout) :: s
    real(real64),intent(in) :: t,rho,u(3)
    logical,intent(out) :: ok
    type(velocity_statistics) :: next
    real(real64) :: weights(2),duration,weighted_density,velocity(3),density
    integer :: endpoint
    ok=.false.
    if(.not.s%configured.or..not.all(ieee_is_finite([t,rho,u]))) return
    if(rho<=0) return
    next=s
    if(s%has_previous) then
      if(t<=s%last_time) return
      ! These weights integrate endpoint integrands, not products of interpolated fields.
      call clipped_trapezoid(s%last_time,t,[1.d0,0.d0],[0.d0,1.d0], &
                            s%window_start,s%window_end,weights,duration,ok)
      if(.not.ok) return
      do endpoint=1,2
        if(weights(endpoint)==0) cycle
        velocity=u
        density=rho
        if(endpoint==1) then
          velocity=s%last_u
          density=s%last_rho
        endif
        weighted_density=weights(endpoint)*density
        ok=.false.
        if(.not.ieee_is_finite(weighted_density).or.weighted_density<=0) return
        call add_weighted_state(weights(endpoint),velocity,next%duration,next%mean_r,next%central_r,ok)
        if(.not.ok) return
        call add_weighted_state(weighted_density,velocity,next%mass_duration,next%mean_f,next%central_f,ok)
        if(.not.ok) return
      enddo
    endif
    next%last_time=t
    next%last_rho=rho
    next%last_u=u
    next%has_previous=.true.
    s=next
    ok=.true.
  end subroutine

  subroutine read_velocity_statistics(s,result,ok)
    type(velocity_statistics),intent(in) :: s
    type(velocity_statistics_result),intent(out) :: result
    logical,intent(out) :: ok
    integer :: i
    ok=.false.
    if(.not.s%configured.or.s%duration<=0.or.s%mass_duration<=0) return
    result%duration=s%duration
    result%mean_density=s%mass_duration/s%duration
    result%mean_r=s%mean_r
    result%mean_f=s%mean_f
    result%covariance_r=s%central_r/s%duration
    result%covariance_f=s%central_f/s%mass_duration
    result%density_stress=s%central_f/s%duration
    if(.not.ieee_is_finite(result%mean_density)) return
    if(.not.all(ieee_is_finite(result%covariance_r))) return
    if(.not.all(ieee_is_finite(result%covariance_f))) return
    if(.not.all(ieee_is_finite(result%density_stress))) return
    do i=1,3
      if(result%covariance_r(i,i)<0.or.result%covariance_f(i,i)<0) return
      result%rms_r(i)=sqrt(result%covariance_r(i,i))
      result%rms_f(i)=sqrt(result%covariance_f(i,i))
    enddo
    ok=.true.
  end subroutine
end module
