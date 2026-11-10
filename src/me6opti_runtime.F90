module me6opti_runtime
  use astr_precision, only: real_kind
  use astr_precision_mpi, only: mpi_real_type
  use commvar, only: im,jm,km,hm,ia,ja,ka,xmin,xmax,ymin,ymax,zmin,zmax
  use commarray, only: x,vel,tmp,jacob,qrhs
  use parallel, only: isize,jsize,ksize,mpileft,mpiright,mpidown,mpiup,mpiback,mpifront, &
    ig0,jg0,kg0,lio
  use me6opti_operator, only: me6_direction
  use me6opti_halo, only: extend_me6_edges_cpu,release_me6_edges_cpu
  use mpi
  implicit none
  private
  public :: prepare_me6_geometry,me6_diffusion_cpu,release_me6_runtime,me6_spacing,me6_parts,me6_peers
  real(real_kind),save :: me6_spacing(3)
  integer,save :: me6_parts(3),me6_peers(2,3)
  real(real_kind),allocatable,save :: viscosity(:,:,:),conductivity(:,:,:)
  logical,save :: prepared=.false.
contains
  subroutine prepare_me6_geometry()
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    real(real_kind) :: origin(3),expected(3),error,global_error,scale
    integer :: i,j,k,ierr,ignored
    if(prepared) return
    if(min(ia,ja,ka)<=0.or.min(im,jm,km)<hm.or.hm<4) &
      error stop 'ME6 requires three active directions and local interior at least hm >= 4'
    origin=[xmin,ymin,zmin]
    me6_spacing=([xmax,ymax,zmax]-origin)/real([ia,ja,ka],real_kind)
    if(.not.all(ieee_is_finite(me6_spacing)).or.any(me6_spacing<=0.e0_real_kind)) &
      error stop 'ME6 requires finite positive Cartesian spacing'
    error=0.e0_real_kind
    scale=max(1.e0_real_kind,maxval(abs(origin)),maxval(abs([xmax,ymax,zmax])))
    do k=0,km
    do j=0,jm
    do i=0,im
      expected=origin+me6_spacing*real([ig0+i,jg0+j,kg0+k],real_kind)
      if(.not.all(ieee_is_finite(x(i,j,k,:)))) error stop 'ME6 nonfinite mesh coordinate'
      error=max(error,maxval(abs(x(i,j,k,:)-expected))/scale)
    enddo
    enddo
    enddo
    call mpi_allreduce(error,global_error,1,mpi_real_type,MPI_MAX,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call mpi_abort(MPI_COMM_WORLD,ierr,ignored)
    if(global_error>128.e0_real_kind*epsilon(1.e0_real_kind)) &
      error stop 'ME6 nonuniform or non-Cartesian geometry extension is not implemented yet'
    me6_parts=[isize,jsize,ksize]
    me6_peers=reshape([mpileft,mpiright,mpidown,mpiup,mpiback,mpifront],[2,3])
    if(lio) write(*,'(A,3ES23.15)') 'ASTR_ME6 uniform Cartesian physical spacing=',me6_spacing
    prepared=.true.
  end subroutine

  subroutine me6_diffusion_cpu()
    use commvar, only: nondimen,reynolds,prandtl,const5,cp
    use fludyna, only: miucal
    real(real_kind) :: contribution(4)
    integer :: i,j,k,axis
    call prepare_me6_geometry()
    call extend_me6_edges_cpu(im,jm,km,hm,vel,tmp,me6_parts,me6_peers)
    if(.not.allocated(viscosity)) allocate(viscosity(-hm:im+hm,-hm:jm+hm,-hm:km+hm), &
      conductivity(-hm:im+hm,-hm:jm+hm,-hm:km+hm))
    do k=-hm,km+hm
    do j=-hm,jm+hm
    do i=-hm,im+hm
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
      do axis=1,3
        call me6_direction(axis,[i,j,k],hm,vel,tmp,viscosity,conductivity,me6_spacing,contribution)
        qrhs(i,j,k,2:5)=qrhs(i,j,k,2:5)+jacob(i,j,k)*contribution
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
