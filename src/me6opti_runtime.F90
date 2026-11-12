module me6opti_runtime
  use astr_precision, only: real_kind
  use commvar, only: im,jm,km,hm,ia,ja,ka,lihomo,ljhomo,lkhomo
  use commarray, only: vel,tmp,jacob,dxi,qrhs
  use parallel, only: isize,jsize,ksize,mpileft,mpiright,mpidown,mpiup,mpiback,mpifront, &
    lio,ig0,jg0,kg0
  use me6opti_operator, only: me6_mapped_direction,me6_coefficients
  use me6opti_halo, only: extend_me6_edges_cpu,release_me6_edges_cpu
  use mpi
  implicit none
  private
  public :: prepare_me6_geometry,me6_diffusion_cpu,release_me6_runtime,me6_parts,me6_peers
  public :: me6_nodal_boundary_point
  integer,save :: me6_parts(3),me6_peers(2,3)
  real(real_kind),allocatable,save :: viscosity(:,:,:),conductivity(:,:,:)
  logical,save :: prepared=.false.
contains
  pure logical function me6_nodal_boundary_point(i,j,k) result(nodal)
    integer,intent(in) :: i,j,k
    nodal=(.not.lihomo.and.min(ig0+i,ia-ig0-i)<4).or. &
      (.not.ljhomo.and.min(jg0+j,ja-jg0-j)<4).or. &
      (.not.lkhomo.and.min(kg0+k,ka-kg0-k)<4)
  end function

  subroutine prepare_me6_geometry()
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    real(real_kind) :: metric(3,3),norms(3),ci,cd,cp
    integer :: i,j,k,ierr,ignored,axis,half,p,point(3),d,invalid,global_invalid
    if(prepared) return
    if(min(ia,ja,ka)<=0.or.min(im,jm,km)<hm.or.hm<4) &
      error stop 'ME6 requires three active directions and local interior at least hm >= 4'
    invalid=0
    do k=0,km
    do j=0,jm
    do i=0,im
    if(me6_nodal_boundary_point(i,j,k)) cycle
      do axis=1,3
      do half=-5,5,2
        metric=0
        do p=-4,4
          point=[i,j,k];point(axis)=point(axis)+p
          if(.not.ieee_is_finite(jacob(point(1),point(2),point(3))).or. &
            jacob(point(1),point(2),point(3))<=0.e0_real_kind) invalid=1
          call me6_coefficients(half,p,ci,cd,cp)
          metric=metric+ci*dxi(point(1),point(2),point(3),:,:)
        enddo
        do d=1,3
          norms(d)=sqrt(sum(metric(d,:)**2))
        enddo
        if(.not.all(ieee_is_finite(norms)).or.any(norms<=0.e0_real_kind)) invalid=1
      enddo
      enddo
    enddo
    enddo
    enddo
    call mpi_allreduce(invalid,global_invalid,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call mpi_abort(MPI_COMM_WORLD,ierr,ignored)
    if(global_invalid/=0) error stop 'ME6 G2 requires positive finite J and nonzero finite half-point metric norms'
    me6_parts=[isize,jsize,ksize]
    me6_peers=reshape([mpileft,mpiright,mpidown,mpiup,mpiback,mpifront],[2,3])
    if(lio) write(*,'(A)') 'ASTR_ME6 G2 full half-point metric projection; norm-ratio penalty'
    prepared=.true.
  end subroutine

  subroutine me6_diffusion_cpu()
    use commvar, only: nondimen,reynolds,prandtl,const5,cp
    use fludyna, only: miucal
    real(real_kind) :: contribution(4)
    integer :: i,j,k,axis
    call prepare_me6_geometry()
    call extend_me6_edges_cpu(im,jm,km,hm,vel,tmp,me6_parts,me6_peers,[lihomo,ljhomo,lkhomo])
    if(.not.allocated(viscosity)) allocate(viscosity(-hm:im+hm,-hm:jm+hm,-hm:km+hm), &
      conductivity(-hm:im+hm,-hm:jm+hm,-hm:km+hm))
    do k=-hm,km+hm
    if(.not.lkhomo.and.(kg0+k<0.or.kg0+k>ka)) cycle
    do j=-hm,jm+hm
    if(.not.ljhomo.and.(jg0+j<0.or.jg0+j>ja)) cycle
    do i=-hm,im+hm
      if(.not.lihomo.and.(ig0+i<0.or.ig0+i>ia)) cycle
      viscosity(i,j,k)=miucal(tmp(i,j,k))
      if(nondimen) then
        viscosity(i,j,k)=viscosity(i,j,k)/reynolds
        conductivity(i,j,k)=(viscosity(i,j,k)/prandtl)/const5
      else
        conductivity(i,j,k)=cp*viscosity(i,j,k)/prandtl
      endif
    enddo
    enddo
    enddo
    do k=0,km
    do j=0,jm
    do i=0,im
    if(me6_nodal_boundary_point(i,j,k)) cycle
      do axis=1,3
        call me6_mapped_direction(axis,[i,j,k],hm,vel,tmp,viscosity,conductivity,dxi,jacob,contribution)
        qrhs(i,j,k,2:5)=qrhs(i,j,k,2:5)+contribution
      enddo
    enddo
    enddo
    enddo
  end subroutine

  subroutine release_me6_runtime()
    if(allocated(viscosity)) deallocate(viscosity,conductivity)
    call release_me6_edges_cpu()
    prepared=.false.
  end subroutine
end module
