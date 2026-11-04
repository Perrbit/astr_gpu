module index_slice_source
  use iso_fortran_env, only: int64
  use astr_precision, only: real_kind,real_bytes
  use commvar, only: use_gpu,numq,num_species,num_modequ,lcomb,lavg,lreadgrid,turbmode
  use bc, only: bctype
  use insitu_run_config, only: catalog_product,catalog_wall_mean
  use insitu_session, only: slice_velocity_statistics_info,slice_velocity_statistics_value
  use insitu_selected_runtime, only: selected_runtime_info,selected_runtime_value,selected_runtime_cartesian
  implicit none
  private
  public :: slice_source_info,slice_source_value_cpu,slice_source_on_gpu
contains
  subroutine slice_source_info(definition,source,ready,ok,window,measure,count,message)
    type(catalog_product),intent(in) :: definition
    integer,intent(out) :: source
    logical,intent(out) :: ready,ok
    real(real_kind),intent(out) :: window(2),measure
    integer(int64),intent(out) :: count
    character(*),intent(out) :: message
    logical :: active,covered,formal_available,legacy_available,wants_favre
    integer :: fields(size(definition%fields)),i
    source=0;ready=.false.;ok=.false.;window=0;measure=0;count=0;message=''
    if(.not.(numq==5.and.num_species==0.and..not.lcomb)) then
#ifdef ASTR_AIR5_CHEMISTRY
      if(.not.(numq==11.and.num_species==5.and.num_modequ==1.and.lcomb)) then
#endif
        message='numerical slices require an implemented five-variable or AIR5 field layout';return
#ifdef ASTR_AIR5_CHEMISTRY
      endif
#endif
    endif
    if(any(definition%fields>=28).or.definition%view%kind=='wall_profile') then
      if(numq/=5.or.lcomb.or..not.selected_runtime_cartesian().or.real_bytes/=8) then
        message='new thermal/wall diagnostics currently require FP64 nonreacting Cartesian states';return
      endif
    endif
    if(any(definition%fields>=35)) then
      if(definition%source/='instantaneous') then
        message='lambda2/lambda_ci use the instantaneous physical velocity gradient';return
      endif
    endif
    if(any(definition%fields>=31.and.definition%fields<=34).or.definition%view%kind=='wall_profile') then
      if(definition%source/='instantaneous'.and.definition%view%kind/='wall_profile'.and. &
        .not.catalog_wall_mean(definition)) then
        message='wall transport means require accumulating instantaneous wall quantities';return
      endif
      if(numq/=5.or.lcomb.or..not.selected_runtime_cartesian().or.trim(turbmode)/='none') then
        message='new molecular wall transport requires a nonreacting Cartesian perfect-gas state';return
      endif
      if(any(definition%fields>=31.and.definition%fields<=33).or.definition%view%kind=='wall_profile'.or. &
        catalog_wall_mean(definition)) then
        if(.not.selected_runtime_cartesian(bottom_wall=.true.)) then
          message='complete wall transport currently requires the lower wall inward normal to point along +y';return
        endif
        if(.not.(definition%view%kind=='wall_profile'.or. &
          (definition%view%kind=='slice'.and.definition%view%sampling=='grid_index')).or. &
           definition%axis/=2.or.definition%index/=0.or..not.any(bctype(3)==[41,42,411,421])) then
          message='complete wall transport selects the physical lower Cartesian wall at global j=0';return
        endif
      endif
    endif
    if(definition%source=='instantaneous') then
      ok=.true.;ready=.true.;return
    endif
    if(definition%statistics_source=='selected-time') then
      call selected_runtime_info(definition,source,ready,ok,window,measure,count,message)
      if(ok.and.definition%fluctuation) source=source+2
      return
    endif
    fields=definition%fields
    wants_favre=definition%source=='favre'
    call formal_info(active,covered,window,measure)
    formal_available=active
    legacy_available=lavg
    if(len(definition%statistics_source)>0) then
      select case(definition%statistics_source)
      case('velocity-time')
        legacy_available=.false.
      case('solver-samples')
        formal_available=.false.
      case default
        message='statistics_source must identify velocity-time or solver-samples';return
      end select
    elseif(formal_available.and.legacy_available) then
      message='multiple mean providers are enabled; select statistics_source explicitly';return
    endif
    if(formal_available) then
      if(wants_favre.and.any(fields==1.or.fields==27)) then
        message='velocity-time does not accumulate Favre density';return
      endif
      if(any(fields>=5.and.fields<=12).or.any(fields>=28)) then
        message='velocity-time contains density and velocity, not pressure/temperature/species';return
      endif
      source=merge(2,1,wants_favre);ready=covered;ok=.true.;return
    endif
    if(legacy_available) then
      ! Legacy moments store rho, rho*u, p, rho*T and unweighted u, not every primitive mean.
      do i=1,size(fields)
        if(fields(i)>=7.and.fields(i)<=12.or. &
          (wants_favre.and.(fields(i)==1.or.fields(i)==5.or.fields(i)==27)).or. &
          (.not.wants_favre.and.(fields(i)==6.or.fields(i)>=28))) then
          message='solver-samples does not accumulate the requested mean field';return
        endif
      enddo
      call legacy_info(active,covered,window,count,measure)
      if(.not.active) then
        message='pointwise solver-samples is unavailable; spanwise-reduced statistics are not 3D means';return
      endif
      source=merge(4,3,wants_favre);ready=covered;ok=.true.;return
    endif
    message='requested mean accumulation is not enabled'
  end subroutine

  subroutine formal_info(active,covered,window,duration)
#ifdef _CUDA
    use insitu_statistics_gpu, only: slice_statistics_gpu_info
#endif
    logical,intent(out) :: active,covered
    real(real_kind),intent(out) :: window(2),duration
    logical :: device_active,device_covered
    real(real_kind) :: device_window(2),device_duration
    call slice_velocity_statistics_info(active,covered,window,duration)
#ifdef _CUDA
    if(use_gpu) then
      call slice_statistics_gpu_info(device_active,device_covered,device_window,device_duration)
      if(device_active) then
        covered=device_covered;window=device_window;duration=device_duration
      endif
      return
    endif
#endif
  end subroutine

  logical function slice_source_on_gpu(source) result(on_device)
#ifdef _CUDA
    use insitu_statistics_gpu, only: slice_statistics_gpu_active
#endif
    integer,intent(in) :: source
    on_device=use_gpu
#ifdef _CUDA
    if(use_gpu.and.(source==1.or.source==2)) then
      on_device=slice_statistics_gpu_active()
    endif
#endif
  end function

  subroutine legacy_info(active,covered,window,count,duration)
    use statistic, only: nsamples,lmeanallocated,liosta,time_sbeg,mean_last_sample_identity,mean_time_weight
#if defined(_CUDA) && defined(ASTR_AIR5_CHEMISTRY)
    use chemistry_mean_statistics_gpu, only: slice_air5_mean_info_gpu
#endif
    logical,intent(out) :: active,covered
    real(real_kind),intent(out) :: window(2)
    real(real_kind),intent(out) :: duration
    integer(int64),intent(out) :: count
#if defined(_CUDA) && defined(ASTR_AIR5_CHEMISTRY)
    if(use_gpu.and.numq==11) then
      call slice_air5_mean_info_gpu(active,covered,count,window,duration);return
    endif
#endif
    active=lavg.and..not.use_gpu
    count=int(nsamples,int64);window=[time_sbeg,mean_last_sample_identity()]
    duration=mean_time_weight
    covered=active.and.lmeanallocated.and.liosta.and.mean_time_weight>0.e0_real_kind
  end subroutine

  real(real_kind) function slice_source_value_cpu(i,j,k,field,source) result(value)
    use commarray, only: rho,vel,prs,tmp,q
    use commvar, only: const2,const6
    use insitu_sample_moments, only: decode_sample_primitives
    use ieee_arithmetic, only: ieee_value,ieee_quiet_nan
#ifdef ASTR_AIR5_CHEMISTRY
    use commarray, only: tve,spc
#endif
    use statistic, only: mean_time_weight,rom,u1m,u2m,u3m,pm,tm,u1rem,u2rem,u3rem
    integer,intent(in) :: i,j,k,field,source
    real(real_kind) :: current(1),density,local_q(5)
    logical :: ok
    if(source==7.or.source==8) then
      local_q=q(i,j,k,1:5)
      call decode_sample_primitives(local_q,1,[field],const2,const6,current,density,ok)
      value=ieee_value(0.e0_real_kind,ieee_quiet_nan)
      if(ok) value=current(1)-selected_runtime_value(i,j,k,field,source==8)
      return
    endif
    if(source==5.or.source==6) then
      value=selected_runtime_value(i,j,k,field,source==6);return
    endif
    if(source==1.or.source==2) then
      value=slice_velocity_statistics_value(i,j,k,field,source==2);return
    endif
    if(source==3.or.source==4) then
      select case(field)
      case(1)
        value=rom(i,j,k)/mean_time_weight
      case(2)
        if(source==4) then
          value=u1m(i,j,k)/rom(i,j,k)
        else
          value=u1rem(i,j,k)/mean_time_weight
        endif
      case(3)
        if(source==4) then
          value=u2m(i,j,k)/rom(i,j,k)
        else
          value=u2rem(i,j,k)/mean_time_weight
        endif
      case(4)
        if(source==4) then
          value=u3m(i,j,k)/rom(i,j,k)
        else
          value=u3rem(i,j,k)/mean_time_weight
        endif
      case(5)
        value=pm(i,j,k)/mean_time_weight
      case(6)
        value=tm(i,j,k)/rom(i,j,k)
      end select
      return
    endif
    select case(field)
    case(1)
      value=rho(i,j,k)
    case(2:4)
      value=vel(i,j,k,field-1)
    case(5)
      value=prs(i,j,k)
    case(6)
      value=tmp(i,j,k)
#ifdef ASTR_AIR5_CHEMISTRY
    case(7)
      value=tve(i,j,k)
    case(8:12)
      value=spc(i,j,k,field-7)
#endif
    end select
  end function
end module
