module keep_sbp63_runtime
  use iso_c_binding, only: c_double,c_int,c_ptr,c_loc,c_null_ptr
  use astr_precision, only: real_kind,numerical_precision
  use keep_sbp63
  use keep_operator, only: sbp_pair=>keep_two_point_flux5
  implicit none
  private
  public :: configure_keep_sbp63,keep_sbp63_enabled,keep_sbp63_wall_temperature
  public :: keep_sbp63_gradient_cpu,keep_sbp63_convection_cpu,keep_sbp63_diffusion_cpu
  public :: keep_w1_enabled,keep_w1_configure_grid,keep_w1_inverse,keep_w1_diffusion_cpu
  public :: keep_w1_end_widths
  logical,save :: keep_sbp63_enabled=.false.
  logical,save :: keep_w1_enabled=.false.,keep_w1_grid_ready=.false.
  real(real_kind),save :: keep_sbp63_wall_temperature(2)=1.e0_real_kind
  real(c_double),save :: keep_w1_inverse(3)=1.0_c_double
  integer(c_int),save :: w1_send_width(2)=0,w1_recv_width(2)=0
  type :: w1_end_buffer
    real(real_kind),allocatable :: data(:,:,:,:)
  end type
  interface
    subroutine w1_host(stage,im,jm,km,halo,first_y,total_y,split, &
        velocity,temperature,metric,volume,inverse,wall,re,pr,c5,tc,tc1,scratch,rhs,width,lower,upper) &
        bind(C,name='astr_keep_w1_cpu')
      import c_int,c_double,c_ptr,real_kind
      integer(c_int),value :: stage,im,jm,km,halo,first_y,total_y
      integer(c_int),intent(in) :: split(3)
      real(real_kind),intent(in) :: velocity(*),temperature(*),metric(*),volume(*)
      real(c_double),intent(in) :: inverse(3),wall(2)
      real(c_double),value :: re,pr,c5,tc,tc1
      real(real_kind),intent(inout) :: scratch(*),rhs(*)
      integer(c_int),intent(in) :: width(2)
      type(c_ptr),value :: lower,upper
    end subroutine
    integer(c_int) function w1_missing_y(first,span,last,halo,received,required) bind(C,name='astr_keep_w1_missing_y')
      import c_int
      integer(c_int),value :: first,span,last,halo
      integer(c_int),intent(in) :: received(2)
      integer(c_int),intent(out) :: required(2)
    end function
  end interface
#ifdef ASTR_BUILD_TESTING
  public :: keep_sbp63_mms_enabled,keep_sbp63_source_cpu
  logical,save :: keep_sbp63_mms_enabled=.false.
#endif
#define GRID_PROC pure
#include "keep_sbp63_grid_body.inc"
#ifdef ASTR_BUILD_TESTING
#include "keep_sbp63_mms_body.inc"
#endif

  subroutine configure_keep_sbp63(types,wall_temperature,paired_viscosity)
    use commvar, only: ndims,numq,num_species,num_modequ,lcomb,nondimen,conschm,difschm, &
      rkscheme,diffterm,turbmode,lfilter,ja,hm,use_gpu
    use numerical_forms, only: convection_form,viscous_form
    use parallel, only: lio
    integer,intent(in) :: types(6)
    real(real_kind),intent(in) :: wall_temperature(2)
    logical,intent(in) :: paired_viscosity
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
    if(paired_viscosity) then
      if(ja<18) error stop 'W1 paired viscosity requires disjoint nine-row end bands'
      call keep_w1_check_halo()
#ifdef _CUDA
      if(use_gpu) error stop 'W1 paired viscosity uses the CPU or C++ GPU backend'
#endif
    endif
    keep_sbp63_wall_temperature=wall_temperature
    keep_sbp63_enabled=.true.
    keep_w1_enabled=paired_viscosity
    if(lio) then
      if(keep_w1_enabled) then
        write(*,'(a)') 'KEEP WALL s63pair-v1: paired normal viscosity and traction SAT'
      else
        write(*,'(a)') 'KEEP WALL s63sat-v1: SBP 6/3 volume, weak inviscid/viscous isothermal SAT'
      endif
    endif
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

  subroutine keep_w1_check_halo()
    use commvar, only: im,jm,km,ja,hm
    use parallel, only: jg0,isize,jsize,ksize,mpirank
    use mpi
    integer :: invalid,any_invalid,ierr,ignored
    integer(c_int) :: row,needed(2)
    invalid=0
    if(jsize>1) then
      ! Only the second rank from each wall can lack fixed-width halo samples.
      if(jm<9.and.jg0==0) w1_send_width(1)=max(0,jm-hm)
      if(jm<9.and.jg0+jm==ja) w1_send_width(2)=max(0,jm-hm)
      if(jg0>0.and.jg0<9) w1_recv_width(1)=max(0,jg0-hm)
      if(ja-jg0-jm>0.and.ja-jg0-jm<9) w1_recv_width(2)=max(0,ja-jg0-jm-hm)
    endif
    if(any([isize,jsize,ksize]>1.and.[im,jm,km]<hm)) then
      write(*,'(a,i0,a)') 'W1 rank ',mpirank,': fixed halo exceeds the local intervals'
      invalid=1
    endif
    row=w1_missing_y(int(jg0,c_int),int(jm,c_int),int(ja,c_int),int(hm,c_int),w1_recv_width,needed)
    if(row>=0) then
      write(*,'(a,i0,a,i0,a,2(1x,i0),a,2(1x,i0))') 'W1 rank ',mpirank,', global y row ',row, &
        ' requires',needed,'; available',max(0,jg0-hm),min(ja,jg0+jm+hm)
      invalid=1
    endif
    call MPI_Allreduce(invalid,any_invalid,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,ierr,ignored)
    if(any_invalid/=0) error stop 'W1 paired wall stencil exceeds the supplied halo'
  end subroutine

  subroutine keep_w1_end_widths(send_width,recv_width)
    integer(c_int),intent(out) :: send_width(2),recv_width(2)
    send_width=w1_send_width
    recv_width=w1_recv_width
  end subroutine

  subroutine keep_w1_configure_grid()
    use commvar, only: im,jm,km,ia,ka
    use commarray, only: x,dxi
    use mpi
    use astr_precision_mpi, only: mpi_real_type
    use parallel, only: ig0,kg0
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    real(real_kind) :: dx,dz,scale,tolerance,local(2),origin(2),finish(2)
    integer :: i,k,ierr,invalid,global_invalid,ignored
    if(keep_w1_grid_ready) return
    local=[x(0,0,0,1),x(0,0,0,3)]
    call MPI_Allreduce(local,origin,2,mpi_real_type,MPI_MIN,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,ierr,ignored)
    local=[x(im,0,0,1),x(0,0,km,3)]
    call MPI_Allreduce(local,finish,2,mpi_real_type,MPI_MAX,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,ierr,ignored)
    dx=(finish(1)-origin(1))/ia
    dz=(finish(2)-origin(2))/ka
    scale=max(1.e0_real_kind,maxval(abs(origin)),maxval(abs(finish)))
    tolerance=128.e0_real_kind*epsilon(1.e0_real_kind)*scale
    if(.not.ieee_is_finite(dx).or..not.ieee_is_finite(dz).or.dx<=0.or.dz<=0) &
      error stop 'W1 paired viscosity needs positive periodic Cartesian steps'
    invalid=0
    do i=0,im
      if(abs(x(i,0,0,1)-(origin(1)+(ig0+i)*dx))>tolerance) invalid=1
    enddo
    do k=0,km
      if(abs(x(0,0,k,3)-(origin(2)+(kg0+k)*dz))>tolerance) invalid=1
    enddo
    if(any(dxi(0:im,0:jm,0:km,2,2)<=0).or..not.all(ieee_is_finite(dxi(0:im,0:jm,0:km,2,2)))) &
      invalid=1
    call MPI_Allreduce(invalid,global_invalid,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,ierr,ignored)
    if(global_invalid/=0) error stop 'W1 paired viscosity needs uniform x/z and positive finite y metric'
    keep_w1_inverse=[1.0_c_double/dx,1.0_c_double,1.0_c_double/dz]
    keep_w1_grid_ready=.true.
  end subroutine

  subroutine keep_w1_diffusion_cpu()
    use commvar, only: im,jm,km,hm,ja,reynolds,prandtl,const5,tempconst,tempconst1
    use commarray, only: vel,tmp,dxi,jacob,qrhs
    use parallel, only: jg0,jrk,isize,jsize,ksize,mpi_jgroup,dataswap
    use astr_precision_mpi, only: mpi_real_type
    use mpi
    real(real_kind),allocatable,save :: work(:,:,:,:)
    type(w1_end_buffer),target,save :: send(2),recv(2)
    type(c_ptr) :: received(2)
    integer(c_int) :: split(3)
    integer,parameter :: wall_sample_tag=61
    integer :: axis,side,p,j,peer,ierr,ignored
    call keep_w1_configure_grid()
    split=int(merge(1,0,[isize,jsize,ksize]>1),c_int)
    received=c_null_ptr
    if(.not.allocated(work)) allocate(work(-hm:im+hm,-hm:jm+hm,-hm:km+hm,14))
    call apply(0_c_int)
    do axis=1,3
      if(split(axis)/=0) call dataswap(work,direction=axis)
    enddo
    do side=1,2
      if(w1_send_width(side)>0) then
        if(.not.allocated(send(side)%data)) allocate(send(side)%data(7,0:im,0:km,0:w1_send_width(side)-1))
        do p=0,w1_send_width(side)-1
          j=merge(p,jm-p,side==1)
          do axis=1,3
            send(side)%data(axis,:,:,p)=vel(0:im,j,0:km,axis)
          enddo
          send(side)%data(4,:,:,p)=tmp(0:im,j,0:km)
          send(side)%data(5,:,:,p)=work(0:im,j,0:km,13)
          send(side)%data(6,:,:,p)=work(0:im,j,0:km,14)
          send(side)%data(7,:,:,p)=1.e0_real_kind/dxi(0:im,j,0:km,2,2)
        enddo
        peer=jrk+merge(1,-1,side==1)
        call MPI_Send(send(side)%data,size(send(side)%data),mpi_real_type,peer,wall_sample_tag,mpi_jgroup,ierr)
        if(ierr/=MPI_SUCCESS) call MPI_Abort(mpi_jgroup,ierr,ignored)
      else if(w1_recv_width(side)>0) then
        if(.not.allocated(recv(side)%data)) allocate(recv(side)%data(7,0:im,0:km,0:w1_recv_width(side)-1))
        peer=jrk+merge(-1,1,side==1)
        call MPI_Recv(recv(side)%data,size(recv(side)%data),mpi_real_type,peer,wall_sample_tag,mpi_jgroup,MPI_STATUS_IGNORE,ierr)
        if(ierr/=MPI_SUCCESS) call MPI_Abort(mpi_jgroup,ierr,ignored)
        received(side)=c_loc(recv(side)%data)
      endif
    enddo
    call apply(1_c_int)
  contains
    subroutine apply(stage)
      integer(c_int),intent(in) :: stage
      call w1_host(stage,int(im,c_int),int(jm,c_int),int(km,c_int),int(hm,c_int),int(jg0,c_int),int(ja,c_int),split, &
        vel,tmp,dxi,jacob,keep_w1_inverse,real(keep_sbp63_wall_temperature,c_double),real(reynolds,c_double), &
        real(prandtl,c_double),real(const5,c_double),real(tempconst,c_double),real(tempconst1,c_double), &
        work,qrhs,w1_recv_width,received(1),received(2))
    end subroutine
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
