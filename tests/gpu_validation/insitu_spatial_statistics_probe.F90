program spatial_statistics_probe
  use mpi
  use iso_fortran_env, only: real64
  use insitu_spatial_statistics
  use insitu_velocity_statistics
  implicit none
  integer :: rank,nrank,ierr,nx,i,j,k,p,n
  real(real64),allocatable :: weights(:),vel(:,:),variance(:,:)
  real(real64) :: volume,mean(3),mean_variance(3),sign_value
  type(velocity_statistics) :: point,region
  type(velocity_statistics_result) :: result
  logical :: ok
  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)
  call MPI_Comm_size(MPI_COMM_WORLD,nrank,ierr)
  call require(nrank==1.or.nrank==2)
  nx=4/nrank
  n=(nx+1)*5*5
  allocate(weights(n),vel(3,n),variance(3,n))
  weights=0.d0
  vel=0.d0
  variance=0.d0
  p=0
  do k=0,4
    do j=0,4
      do i=0,nx
        p=p+1
        if(i<nx.and.j<4.and.k<4) weights(p)=1.d0/64.d0
        sign_value=real(1-2*mod(rank*nx+i,2),real64)
        vel(:,p)=[sign_value,0.d0,0.d0]
        call configure_velocity_statistics(point,0.d0,1.d0,ok)
        call require(ok)
        call push_velocity_sample(point,0.d0,1.d0,vel(:,p),ok)
        call require(ok)
        call push_velocity_sample(point,1.d0,1.d0,-vel(:,p),ok)
        call require(ok)
        call read_velocity_statistics(point,result,ok)
        call require(ok)
        variance(:,p)=[result%covariance_r(1,1),result%covariance_r(2,2),result%covariance_r(3,3)]
      enddo
    enddo
  enddo
  call spatial_mean(weights,variance,MPI_COMM_WORLD,volume,mean_variance,ok)
  call require(ok.and.volume==1.d0.and.mean_variance(1)==1.d0)
  call configure_velocity_statistics(region,0.d0,1.d0,ok)
  call require(ok)
  call spatial_mean(weights,vel,MPI_COMM_WORLD,volume,mean,ok)
  call require(ok.and.all(mean==0.d0))
  call push_velocity_sample(region,0.d0,1.d0,mean,ok)
  call require(ok)
  call spatial_mean(weights,-vel,MPI_COMM_WORLD,volume,mean,ok)
  call require(ok)
  call push_velocity_sample(region,1.d0,1.d0,mean,ok)
  call require(ok)
  call read_velocity_statistics(region,result,ok)
  call require(ok.and.result%rms_r(1)==0.d0.and.sqrt(mean_variance(1))==1.d0)
  ! Empty local partitions are valid if another rank owns the region.
  weights=0.d0
  if(rank==0) weights(1)=2.d0
  vel=3.d0
  call spatial_mean(weights,vel,MPI_COMM_WORLD,volume,mean,ok)
  call require(ok.and.volume==2.d0.and.all(mean==3.d0))
  ! Invalid weights on one rank must fail collectively, not hang peers.
  if(rank==0) weights(1)=-1.d0
  call spatial_mean(weights,vel,MPI_COMM_WORLD,volume,mean,ok)
  call require(.not.ok)
  weights=0.d0
  call spatial_mean(weights,vel,MPI_COMM_WORLD,volume,mean,ok)
  call require(.not.ok)
  if(rank==0) print *, 'PASS: unique periodic volume, local RMS=1, regional-signal RMS=0, collective rejection'
  call MPI_Finalize(ierr)
contains
  subroutine require(condition)
    logical,intent(in) :: condition
    if(.not.condition) call MPI_Abort(MPI_COMM_WORLD,1,ierr)
  end subroutine
end program
