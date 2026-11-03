module insitu_sample_moments
  use astr_precision, only: real_kind
  use insitu_velocity_statistics, only: velocity_statistics_result
  use ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: sample_moment_size,push_sample_moments,update_sample_moments,read_sample_means,read_sample_velocity
  public :: decode_sample_primitives
  public :: sample_profile_size,sample_profile_terms,finish_sample_profile
  integer,parameter :: max_fields=5,max_components=30
contains
#ifdef _CUDA
  attributes(host,device) integer function sample_profile_size(nfields,covariance) result(n)
#else
  integer function sample_profile_size(nfields,covariance) result(n)
#endif
    integer,intent(in) :: nfields
    logical,intent(in) :: covariance
    n=0
    if(sample_moment_size(nfields,covariance)==0) return
    n=2+2*nfields+merge(12,0,covariance)
  end function

#ifdef _CUDA
  attributes(host,device) subroutine sample_profile_terms(ncomponents,nfields,state,weight,covariance,terms,ok)
#else
  subroutine sample_profile_terms(ncomponents,nfields,state,weight,covariance,terms,ok)
#endif
    integer,intent(in) :: ncomponents,nfields
    logical,intent(in) :: covariance
    real(real_kind),intent(in) :: state(ncomponents),weight
    real(real_kind),intent(out) :: terms(24)
    logical,intent(out) :: ok
    real(real_kind) :: density,weighted_density
    integer :: c,offset,output,n
    ok=.false.;terms=0.e0_real_kind
    n=sample_profile_size(nfields,covariance)
    if(n==0.or.ncomponents/=sample_moment_size(nfields,covariance)) return
    if(.not.(weight>=0.e0_real_kind.and.weight<=huge(weight))) return
    if(.not.(state(2)>0.e0_real_kind.and.state(2)<=huge(weight).and. &
      state(3)>0.e0_real_kind.and.state(3)<=huge(weight))) return
    density=state(3)/state(2);weighted_density=weight*density
    terms(1)=weight;terms(2)=weighted_density
    do c=1,nfields
      terms(2+c)=weight*state(3+nfields+c)
      terms(2+nfields+c)=weighted_density*state(3+2*nfields+c)
    enddo
    if(covariance) then
      offset=3+3*nfields;output=2+2*nfields
      do c=1,6
        terms(output+c)=weight*(state(offset+c)/state(2))
        terms(output+6+c)=weight*(state(offset+6+c)/state(2))
      enddo
    endif
    ok=all(abs(terms(:n))<=huge(weight))
  end subroutine

  subroutine finish_sample_profile(nfields,covariance,terms,measure,density,mean_r,mean_f, &
      covariance_r,covariance_f,density_stress,rms_r,rms_f,ok)
    integer,intent(in) :: nfields
    logical,intent(in) :: covariance
    real(real_kind),intent(in) :: terms(:)
    real(real_kind),intent(out) :: measure,density,mean_r(nfields),mean_f(nfields), &
      covariance_r(6),covariance_f(6),density_stress(6),rms_r(3),rms_f(3)
    logical,intent(out) :: ok
    integer :: offset,n
    ok=.false.;measure=0;density=0;mean_r=0;mean_f=0
    covariance_r=0;covariance_f=0;density_stress=0;rms_r=0;rms_f=0
    n=sample_profile_size(nfields,covariance)
    if(n==0.or.size(terms)/=n) return
    if(.not.all(ieee_is_finite(terms))) return
    if(any(terms(:2)<=0.e0_real_kind)) return
    measure=terms(1);density=terms(2)/measure
    mean_r=terms(3:2+nfields)/measure
    mean_f=terms(3+nfields:2+2*nfields)/terms(2)
    if(covariance) then
      offset=2+2*nfields
      covariance_r=terms(offset+1:offset+6)/measure
      density_stress=terms(offset+7:offset+12)/measure
      covariance_f=terms(offset+7:offset+12)/terms(2)
      if(any(covariance_r([1,3,6])<0).or.any(covariance_f([1,3,6])<0)) return
      rms_r=sqrt(covariance_r([1,3,6]));rms_f=sqrt(covariance_f([1,3,6]))
    endif
    ok=all(ieee_is_finite([density,mean_r,mean_f,covariance_r,covariance_f,density_stress,rms_r,rms_f]))
  end subroutine

#ifdef _CUDA
  attributes(host,device) integer function sample_moment_size(nfields,covariance) result(n)
#else
  integer function sample_moment_size(nfields,covariance) result(n)
#endif
    integer,intent(in) :: nfields
    logical,intent(in) :: covariance
    n=0
    if(nfields<0.or.nfields>max_fields) return
    if(covariance.and.nfields<3) return
    n=3+3*nfields
    if(covariance) n=n+12
  end function

#ifdef _CUDA
  attributes(host,device) subroutine decode_sample_primitives(q,nfields,field_ids,const2,const6,values,density,ok)
#else
  subroutine decode_sample_primitives(q,nfields,field_ids,const2,const6,values,density,ok)
#endif
    integer,intent(in) :: nfields,field_ids(nfields)
    real(real_kind),intent(in) :: q(5),const2,const6
    real(real_kind),intent(out) :: values(nfields),density
    logical,intent(out) :: ok
    real(real_kind) :: velocity(3),pressure
    integer :: c
    ok=.false.;values=0.e0_real_kind;density=q(1)
    if(nfields<0.or.nfields>max_fields) return
    if(any(field_ids<2).or.any(field_ids>6)) return
    if(.not.(density>0.e0_real_kind.and.density<=huge(density))) return
    pressure=0.e0_real_kind
    if(any(field_ids>=5)) then
      if(.not.(const6>0.e0_real_kind.and.const6<=huge(density))) return
      if(.not.all(abs(q(2:5))<=huge(density))) return
      velocity=q(2:4)/density
      pressure=(q(5)-0.5e0_real_kind*density*sum(velocity**2))/const6
    endif
    do c=1,nfields
      select case(field_ids(c))
      case(2:4)
        values(c)=q(field_ids(c))/density
      case(5)
        values(c)=pressure
      case(6)
        if(.not.(const2>0.e0_real_kind.and.const2<=huge(density))) return
        values(c)=(pressure/density)*const2
      end select
    enddo
    ok=all(abs(values)<=huge(density))
  end subroutine

  ! Layout: previous rho, time/mass weights, previous fields, two means,
  ! then optional upper-triangular velocity central moments (six per mean).
  subroutine push_sample_moments(state,values,rho,weights,previous,velocity_slots,ok)
    real(real_kind),intent(inout) :: state(:)
    real(real_kind),intent(in) :: values(:),rho,weights(2)
    logical,intent(in) :: previous
    integer,intent(in) :: velocity_slots(3)
    logical,intent(out) :: ok
    integer :: n
    n=sample_moment_size(size(values),any(velocity_slots/=0));ok=.false.
    if(n==0.or.size(state)/=n) return
    call update_sample_moments(n,size(values),state,values,rho,weights,previous,velocity_slots,ok)
  end subroutine

#ifdef _CUDA
  attributes(host,device) subroutine update_sample_moments(ncomponents,nfields,state,values,rho,weights, &
      previous,velocity_slots,ok)
#else
  subroutine update_sample_moments(ncomponents,nfields,state,values,rho,weights,previous,velocity_slots,ok)
#endif
    integer,intent(in) :: ncomponents,nfields,velocity_slots(3)
    real(real_kind),intent(inout) :: state(ncomponents)
    real(real_kind),intent(in) :: values(nfields),rho,weights(2)
    logical,intent(in) :: previous
    logical,intent(out) :: ok
    real(real_kind) :: next(max_components),sample(max_fields),delta(max_fields)
    real(real_kind) :: density,weight,total,new_total,ratio,factor
    integer :: nf,n,e,f,c,a,b,slot,mean_start,central_start,entry
    logical :: covariance
    ok=.false.;nf=nfields;covariance=any(velocity_slots/=0)
    n=sample_moment_size(nf,covariance)
    if(n==0.or.ncomponents/=n) return
    if(covariance) then
      if(any(velocity_slots<1).or.any(velocity_slots>nf)) return
      if(velocity_slots(1)==velocity_slots(2).or.velocity_slots(1)==velocity_slots(3).or. &
        velocity_slots(2)==velocity_slots(3)) return
    endif
    ! Use the established device finite comparison in this dual-target routine.
    if(.not.(rho>0.e0_real_kind.and.rho<=huge(rho))) return
    if(.not.all(abs(values)<=huge(rho)).or..not.all(abs(weights)<=huge(rho))) return
    if(any(weights<0.e0_real_kind)) return
    next=0.e0_real_kind
    if(previous) then
      if(.not.all(abs(state)<=huge(rho))) return
      if(state(1)<=0.e0_real_kind.or.any(state(2:3)<0.e0_real_kind)) return
      if((state(2)==0.e0_real_kind).neqv.(state(3)==0.e0_real_kind)) return
      next(:n)=state
    elseif(any(weights/=0.e0_real_kind)) then
      return
    endif
    do e=1,2
      if(weights(e)==0.e0_real_kind) cycle
      density=rho;sample(:nf)=values
      if(e==1) then
        density=next(1);sample(:nf)=next(4:3+nf)
      endif
      do f=1,2
        weight=weights(e)
        if(f==2) weight=weight*density
        total=next(1+f);new_total=total+weight
        if(.not.(weight>0.e0_real_kind.and.weight<=huge(rho).and. &
          new_total>0.e0_real_kind.and.new_total<=huge(rho))) return
        mean_start=3+f*nf
        delta(:nf)=sample(:nf)-next(mean_start+1:mean_start+nf)
        if(total==0.e0_real_kind) then
          next(mean_start+1:mean_start+nf)=sample(:nf)
        else
          ratio=weight/new_total
          factor=(total/new_total)*weight
          do c=1,nf
            next(mean_start+c)=next(mean_start+c)+ratio*delta(c)
          enddo
          if(covariance) then
            central_start=3+3*nf+6*(f-1);entry=0
            do b=1,3
              do a=1,b
                entry=entry+1
                slot=central_start+entry
                next(slot)=next(slot)+(factor*delta(velocity_slots(a)))*delta(velocity_slots(b))
              enddo
            enddo
          endif
        endif
        next(1+f)=new_total
      enddo
    enddo
    next(1)=rho;next(4:3+nf)=values
    if(.not.all(abs(next(:n))<=huge(rho))) return
    state=next(:n)
    ok=.true.
  end subroutine

  subroutine read_sample_means(state,nfields,density,mean_r,mean_f,ok)
    integer,intent(in) :: nfields
    real(real_kind),intent(in) :: state(:)
    real(real_kind),intent(out) :: density,mean_r(nfields),mean_f(nfields)
    logical,intent(out) :: ok
    ok=.false.;density=0.e0_real_kind;mean_r=0.e0_real_kind;mean_f=0.e0_real_kind
    if(nfields<0.or.nfields>max_fields.or.size(state)<3+3*nfields) return
    if(.not.all(ieee_is_finite(state(2:3)))) return
    if(any(state(2:3)<=0.e0_real_kind)) return
    density=state(3)/state(2)
    mean_r=state(4+nfields:3+2*nfields)
    mean_f=state(4+2*nfields:3+3*nfields)
    ok=ieee_is_finite(density).and.all(ieee_is_finite(mean_r)).and.all(ieee_is_finite(mean_f))
  end subroutine

  subroutine read_sample_velocity(state,nfields,velocity_slots,result,ok)
    real(real_kind),intent(in) :: state(:)
    integer,intent(in) :: nfields,velocity_slots(3)
    type(velocity_statistics_result),intent(out) :: result
    logical,intent(out) :: ok
    real(real_kind) :: density,mean_r(nfields),mean_f(nfields)
    integer :: a,b,entry,offset
    result=velocity_statistics_result();ok=.false.
    if(size(state)/=sample_moment_size(nfields,.true.)) return
    if(any(velocity_slots<1).or.any(velocity_slots>nfields)) return
    call read_sample_means(state,nfields,density,mean_r,mean_f,ok)
    if(.not.ok) return
    result%duration=state(2);result%mean_density=density
    result%mean_r=mean_r(velocity_slots);result%mean_f=mean_f(velocity_slots)
    entry=0;offset=3+3*nfields
    do b=1,3
      do a=1,b
        entry=entry+1
        result%covariance_r(a,b)=state(offset+entry)/state(2)
        result%covariance_f(a,b)=state(offset+6+entry)/state(3)
        result%density_stress(a,b)=state(offset+6+entry)/state(2)
        result%covariance_r(b,a)=result%covariance_r(a,b)
        result%covariance_f(b,a)=result%covariance_f(a,b)
        result%density_stress(b,a)=result%density_stress(a,b)
      enddo
    enddo
    ok=.false.
    if(.not.all(ieee_is_finite(result%covariance_r)).or. &
      .not.all(ieee_is_finite(result%covariance_f)).or..not.all(ieee_is_finite(result%density_stress))) return
    do a=1,3
      if(result%covariance_r(a,a)<0.e0_real_kind.or.result%covariance_f(a,a)<0.e0_real_kind) return
      result%rms_r(a)=sqrt(result%covariance_r(a,a))
      result%rms_f(a)=sqrt(result%covariance_f(a,a))
    enddo
    ok=.true.
  end subroutine
end module
