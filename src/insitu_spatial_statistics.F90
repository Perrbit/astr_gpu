module insitu_spatial_statistics
  use mpi
  use iso_fortran_env, only: real64
  use ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: spatial_mean
contains
  ! Values are component x local node. Caller supplies physical measure and unique ownership.
  subroutine spatial_mean(weights,values,comm,volume,mean,ok)
    real(real64),intent(in) :: weights(:),values(:,:)
    integer,intent(in) :: comm
    real(real64),intent(out) :: volume,mean(size(values,1))
    logical,intent(out) :: ok
    real(real64) :: local(size(values,1)+1),global(size(values,1)+1)
    integer :: bad,any_bad,i,j,ierr,nmin,nmax,n
    ok=.false.
    volume=0.d0
    mean=0.d0
    n=size(values,1)
    call MPI_Allreduce(n,nmin,1,MPI_INTEGER,MPI_MIN,comm,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(comm,1,ierr)
    call MPI_Allreduce(n,nmax,1,MPI_INTEGER,MPI_MAX,comm,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(comm,1,ierr)
    bad=0
    if(n<=0.or.nmin/=nmax.or.size(values,2)/=size(weights)) bad=1
    if(.not.all(ieee_is_finite(weights)).or.any(weights<0)) bad=1
    if(.not.all(ieee_is_finite(values))) bad=1
    call MPI_Allreduce(bad,any_bad,1,MPI_INTEGER,MPI_MAX,comm,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(comm,1,ierr)
    if(any_bad/=0) return
    local=0.d0
    local(1)=sum(weights)
    do j=1,size(weights)
      do i=1,n
        local(i+1)=local(i+1)+weights(j)*values(i,j)
      enddo
    enddo
    bad=merge(0,1,all(ieee_is_finite(local)))
    call MPI_Allreduce(bad,any_bad,1,MPI_INTEGER,MPI_MAX,comm,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(comm,1,ierr)
    if(any_bad/=0) return
    call MPI_Allreduce(local,global,n+1,MPI_DOUBLE_PRECISION,MPI_SUM,comm,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(comm,1,ierr)
    if(.not.all(ieee_is_finite(global)).or.global(1)<=0) return
    volume=global(1)
    mean=global(2:)/volume
    ok=all(ieee_is_finite(mean))
  end subroutine
end module
