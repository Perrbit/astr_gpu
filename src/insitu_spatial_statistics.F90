module insitu_spatial_statistics
  use mpi
  use astr_precision, only: real_kind
  use astr_precision_mpi, only: mpi_real_type
  use insitu_velocity_statistics, only: velocity_statistics_result
  use ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: spatial_mean,summarize_velocity_profile
  public :: span_node_weights
  public :: canonical_node_intervals
contains
  ! A selected terminal node belongs to the periodic origin, not a second owner.
  pure subroutine canonical_node_intervals(first,last,intervals,periodic,ranges,count)
    integer,intent(in) :: first,last,intervals
    logical,intent(in) :: periodic
    integer,intent(out) :: ranges(2,2),count
    ranges=0;count=0
    if(first<0.or.last<first.or.last>intervals.or.intervals<1) return
    count=1;ranges(:,1)=[first,last]
    if(.not.periodic.or.last<intervals) return
    if(first==0) then
      ranges(:,1)=[0,intervals-1]
    elseif(first==intervals) then
      ranges(:,1)=[0,0]
    else
      count=2;ranges(:,1)=[0,0];ranges(:,2)=[first,intervals-1]
    endif
  end subroutine

  subroutine span_node_weights(coordinates,first,intervals,periodic,nodes,averaged,weights,ok)
    real(real_kind),intent(in) :: coordinates(:)
    integer,intent(in) :: first,intervals,nodes(:)
    logical,intent(in) :: periodic,averaged
    real(real_kind),intent(out) :: weights(size(nodes))
    logical,intent(out) :: ok
    integer :: n,i,k,last,canonical
    real(real_kind) :: weight
    ok=.false.;weights=0;n=size(coordinates);last=first+n-1
    if(n<1.or.first<0.or.last>intervals.or..not.all(ieee_is_finite(coordinates))) return
    if(.not.averaged) then
      if(n/=1) return
      canonical=first
      if(periodic) canonical=modulo(first,intervals)
      where(nodes==canonical) weights=1.e0_real_kind
      ok=.true.;return
    endif
    if(n<2) return
    if(.not.all(coordinates(2:)>coordinates(:n-1)).and..not.all(coordinates(2:)<coordinates(:n-1))) return
    do i=1,n
      if(i==1) then
        weight=.5e0_real_kind*abs(coordinates(2)-coordinates(1))
      elseif(i==n) then
        weight=.5e0_real_kind*abs(coordinates(n)-coordinates(n-1))
      else
        weight=.5e0_real_kind*abs(coordinates(i+1)-coordinates(i-1))
      endif
      canonical=first+i-1
      if(periodic) canonical=modulo(canonical,intervals)
      do k=1,size(nodes)
        if(nodes(k)==canonical) weights(k)=weights(k)+weight
      enddo
    enddo
    ok=all(ieee_is_finite(weights)).and.all(weights>=0.e0_real_kind)
  end subroutine

  ! Summarize local temporal moments, without adding variance of spatially varying means.
  ! The caller must also match the actual window and sampling identity of the inputs.
  subroutine summarize_velocity_profile(weights,points,comm,measure,summary,ok)
    real(real_kind),intent(in) :: weights(:)
    type(velocity_statistics_result),intent(in) :: points(:)
    integer,intent(in) :: comm
    real(real_kind),intent(out) :: measure
    type(velocity_statistics_result),intent(out) :: summary
    logical,intent(out) :: ok
    real(real_kind) :: values(25,size(points)),means(25),first_duration,last_duration,low,high
    real(real_kind) :: diagonal_r(3),diagonal_f(3)
    integer :: i,j,bad,any_bad,ierr
    ok=.false.;measure=0.e0_real_kind;summary=velocity_statistics_result()
    values=0.e0_real_kind;first_duration=huge(0.e0_real_kind);last_duration=0.e0_real_kind
    bad=0
    if(size(weights)/=size(points)) bad=1
    if(.not.all(ieee_is_finite(weights)).or.any(weights<0.e0_real_kind)) bad=1
    if(bad==0) then
      do i=1,size(points)
        if(weights(i)==0.e0_real_kind) cycle
        if(.not.ieee_is_finite(points(i)%duration).or.points(i)%duration<=0.e0_real_kind.or. &
          .not.ieee_is_finite(points(i)%mean_density).or.points(i)%mean_density<=0.e0_real_kind) then
          bad=1
          cycle
        endif
        first_duration=min(first_duration,points(i)%duration)
        last_duration=max(last_duration,points(i)%duration)
        values(:,i)=[points(i)%mean_density,points(i)%mean_r, &
          points(i)%mean_density*points(i)%mean_f,reshape(points(i)%covariance_r,[9]), &
          reshape(points(i)%density_stress,[9])]
        do j=1,3
          if(points(i)%covariance_r(j,j)<0.e0_real_kind.or.points(i)%density_stress(j,j)<0.e0_real_kind) bad=1
        enddo
      enddo
    endif
    call MPI_Allreduce(bad,any_bad,1,MPI_INTEGER,MPI_MAX,comm,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(comm,1,ierr)
    if(any_bad/=0) return
    call MPI_Allreduce(first_duration,low,1,mpi_real_type,MPI_MIN,comm,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(comm,1,ierr)
    call MPI_Allreduce(last_duration,high,1,mpi_real_type,MPI_MAX,comm,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(comm,1,ierr)
    if(low/=high.or.high<=0.e0_real_kind) return
    call spatial_mean(weights,values,comm,measure,means,ok)
    if(.not.ok) return
    if(means(1)<=0.e0_real_kind) then
      ok=.false.
      return
    endif
    summary%duration=low;summary%mean_density=means(1)
    summary%mean_r=means(2:4);summary%mean_f=means(5:7)/summary%mean_density
    summary%covariance_r=reshape(means(8:16),[3,3])
    summary%density_stress=reshape(means(17:25),[3,3])
    summary%covariance_f=summary%density_stress/summary%mean_density
    do j=1,3
      diagonal_r(j)=summary%covariance_r(j,j)
      diagonal_f(j)=summary%covariance_f(j,j)
    enddo
    summary%rms_r=sqrt(diagonal_r);summary%rms_f=sqrt(diagonal_f)
    ok=all(ieee_is_finite([summary%mean_r,summary%mean_f,summary%rms_r,summary%rms_f, &
      reshape(summary%covariance_f,[9])]))
  end subroutine

  ! Values are component x local node. Caller supplies physical measure and unique ownership.
  subroutine spatial_mean(weights,values,comm,volume,mean,ok)
    real(real_kind),intent(in) :: weights(:),values(:,:)
    integer,intent(in) :: comm
    real(real_kind),intent(out) :: volume,mean(size(values,1))
    logical,intent(out) :: ok
    real(real_kind) :: local(size(values,1)+1),global(size(values,1)+1)
    integer :: bad,any_bad,i,j,ierr,nmin,nmax,n
    ok=.false.
    volume=0.e0_real_kind
    mean=0.e0_real_kind
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
    local=0.e0_real_kind
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
    call MPI_Allreduce(local,global,n+1,mpi_real_type,MPI_SUM,comm,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(comm,1,ierr)
    if(.not.all(ieee_is_finite(global)).or.global(1)<=0) return
    volume=global(1)
    mean=global(2:)/volume
    ok=all(ieee_is_finite(mean))
  end subroutine
end module
