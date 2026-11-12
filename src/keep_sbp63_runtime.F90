module keep_sbp63_runtime
  use astr_precision, only: real_kind,numerical_precision
  use keep_sbp63
  use keep_operator, only: sbp_pair=>keep_two_point_flux5
  implicit none
  private
  public :: configure_keep_sbp63,keep_sbp63_enabled,keep_sbp63_wall_temperature
  public :: keep_sbp63_gradient_cpu,keep_sbp63_convection_cpu,keep_sbp63_diffusion_cpu
  logical,save :: keep_sbp63_enabled=.false.
  real(real_kind),save :: keep_sbp63_wall_temperature(2)=1.e0_real_kind
#ifdef ASTR_BUILD_TESTING
  public :: keep_sbp63_mms_enabled,keep_sbp63_source_cpu
  logical,save :: keep_sbp63_mms_enabled=.false.
#endif
#define GRID_PROC pure
#include "keep_sbp63_grid_body.inc"
#ifdef ASTR_BUILD_TESTING
#include "keep_sbp63_mms_body.inc"
#endif

  subroutine configure_keep_sbp63(types,wall_temperature)
    use commvar, only: ndims,numq,num_species,num_modequ,lcomb,nondimen,conschm,difschm, &
      rkscheme,diffterm,turbmode,lfilter,ja,hm
    use numerical_forms, only: convection_form,viscous_form
    use parallel, only: lio
    integer,intent(in) :: types(6)
    real(real_kind),intent(in) :: wall_temperature(2)
#ifdef ASTR_BUILD_TESTING
    character(16) :: flag
    integer :: status
#endif
    if(numerical_precision/='fp64'.or.ndims/=3.or.numq/=5.or.num_species/=0.or.num_modequ/=0.or. &
       lcomb.or..not.nondimen.or.conschm/='643e'.or.difschm/='643e'.or.rkscheme/='rk3'.or. &
       .not.diffterm.or.turbmode/='none'.or.convection_form/='keep'.or.viscous_form/='standard'.or. &
       any(types/=[1,1,41,41,1,1]).or.lfilter) &
      error stop 'KEEP SBP63 SAT requires FP64 explicit KEEP/standard viscosity, two y bc41 walls and no filter'
    if(ja<11.or.hm<5) error stop 'KEEP SBP63 SAT needs disjoint six-row wall bands and five halo layers'
    keep_sbp63_wall_temperature=wall_temperature
    keep_sbp63_enabled=.true.
    if(lio) write(*,'(a)') 'KEEP WALL s63sat-v1: SBP 6/3 volume, weak inviscid/viscous isothermal SAT'
#ifdef ASTR_BUILD_TESTING
    call get_environment_variable('ASTR_TEST_KEEP_SBP63_STEADY_MMS',flag,status=status)
    if(status/=1) then
      if(status/=0.or.trim(flag)/='1') error stop 'ASTR_TEST_KEEP_SBP63_STEADY_MMS must be 1 when set'
      if(any(wall_temperature/=1.e0_real_kind)) error stop 'Steady KEEP SBP63 MMS requires Tw=1'
      keep_sbp63_mms_enabled=.true.
      if(lio) write(*,'(a)') 'KEEP SBP63 steady MMS source: on; identity mms-v1'
    endif
#endif
  end subroutine

#ifdef ASTR_BUILD_TESTING
  subroutine keep_sbp63_source_cpu()
    use commvar, only: im,jm,km,gamma,mach,reynolds
    use commarray, only: x,jacob,qrhs
    integer :: i,j,k
    real(real_kind) :: source(5),position(3)
    if(.not.keep_sbp63_mms_enabled) return
    do k=0,km
    do j=0,jm
    do i=0,im
      position=x(i,j,k,:)
      call sbp63_steady_mms_source(position,gamma,mach,reynolds,source)
      qrhs(i,j,k,:)=qrhs(i,j,k,:)+jacob(i,j,k)*source
    enddo
    enddo
    enddo
  end subroutine
#endif

  subroutine keep_sbp63_gradient_cpu()
    use commvar, only: im,jm,km,hm,ja
    use parallel, only: jg0
    use commarray, only: vel,tmp,dxi,dvel,dtmp
    integer :: i,j,k
    real(real_kind) :: gu(3,3),gt(3)
    do k=0,km
    do j=0,jm
    do i=0,im
      call sbp63_gradient_point(im,jm,km,hm,ja,jg0,i,j,k,keep_sbp63_wall_temperature,vel,tmp,dxi,gu,gt)
      dvel(i,j,k,:,:)=gu;dtmp(i,j,k,:)=gt
    enddo
    enddo
    enddo
  end subroutine

  subroutine keep_sbp63_convection_cpu()
    use commvar, only: im,jm,km,hm,ja,gamma
    use parallel, only: jg0
    use commarray, only: rho,vel,prs,jacob,dxi,qrhs
    integer :: i,j,k
    real(real_kind) :: rhs(5)
    do k=0,km
    do j=0,jm
    do i=0,im
      call sbp63_convection_point(im,jm,km,hm,ja,jg0,i,j,k,gamma,rho,vel,prs,jacob,dxi,rhs)
      qrhs(i,j,k,:)=qrhs(i,j,k,:)+rhs
    enddo
    enddo
    enddo
  end subroutine

  subroutine keep_sbp63_diffusion_cpu(sigma,heat)
    use commvar, only: im,jm,km,hm,ja,reynolds,prandtl,const5
    use parallel, only: jg0
    use commarray, only: vel,tmp,jacob,dxi,dvel,dtmp,qrhs
    use fludyna, only: miucal
    real(real_kind),intent(in) :: sigma(-hm:im+hm,-hm:jm+hm,-hm:km+hm,6)
    real(real_kind),intent(in) :: heat(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3)
    integer :: i,j,k
    real(real_kind) :: rhs(5),sat(5),mu,gu(3,3),gt(3),metric(3),v(3)
    do k=0,km
    do j=0,jm
    do i=0,im
      call sbp63_diffusion_point(im,jm,km,hm,ja,jg0,i,j,k,jacob,dxi,sigma,heat,rhs)
      if(jg0+j==0.or.jg0+j==ja) then
        mu=miucal(tmp(i,j,k))/reynolds
        gu=dvel(i,j,k,:,:);gt=dtmp(i,j,k,:);metric=dxi(i,j,k,2,:);v=vel(i,j,k,:)
        call sbp63_viscous_sat(jg0+j,ja,keep_sbp63_wall_temperature,mu,(mu/prandtl)/const5, &
          jacob(i,j,k),metric,v,tmp(i,j,k),gu,gt,sat)
        rhs=rhs+sat
      endif
      qrhs(i,j,k,:)=qrhs(i,j,k,:)+rhs
    enddo
    enddo
    enddo
  end subroutine
end module
