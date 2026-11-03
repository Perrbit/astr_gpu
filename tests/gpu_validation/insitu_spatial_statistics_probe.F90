program spatial_statistics_probe
  use mpi
  use iso_fortran_env, only: real64
  use insitu_spatial_statistics, only: spatial_mean,summarize_velocity_profile,span_node_weights,canonical_node_intervals
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
  call require(nrank==1.or.nrank==2.or.nrank==4)
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
  call check_velocity_profile()
  call check_span_weights()
  weights=0.d0
  call spatial_mean(weights,vel,MPI_COMM_WORLD,volume,mean,ok)
  call require(.not.ok)
  if(rank==0) print *, 'PASS: unique periodic volume, local RMS=1, regional-signal RMS=0, collective rejection'
  call MPI_Finalize(ierr)
contains
  subroutine check_span_weights()
    real(real64) :: z(5),w(4),partial(2),point_weight(1),none(0)
    logical :: valid
    integer :: ranges(2,2),parts
    call canonical_node_intervals(0,4,4,.true.,ranges,parts)
    call require(parts==1.and.all(ranges(:,1)==[0,3]))
    call canonical_node_intervals(2,4,4,.true.,ranges,parts)
    call require(parts==2.and.all(ranges(:,1)==[0,0]).and.all(ranges(:,2)==[2,3]))
    call canonical_node_intervals(4,4,4,.true.,ranges,parts)
    call require(parts==1.and.all(ranges(:,1)==[0,0]))
    call canonical_node_intervals(2,4,4,.false.,ranges,parts)
    call require(parts==1.and.all(ranges(:,1)==[2,4]))
    z=[0.d0,.25d0,1.d0,2.d0,4.d0]
    call span_node_weights(z,0,4,.true.,[0,1,2,3],.true.,w,valid)
    call require(valid.and.all(w==[1.125d0,.5d0,.875d0,1.5d0]).and.sum(w)==4.d0)
    call span_node_weights(z(3:),2,4,.true.,[0,3],.true.,partial,valid)
    call require(valid.and.all(partial==[1.d0,1.5d0]))
    call span_node_weights(-z,0,4,.true.,[0,1,2,3],.true.,w,valid)
    call require(valid.and.all(w==[1.125d0,.5d0,.875d0,1.5d0]))
    call span_node_weights(z(5:),4,4,.true.,[0],.false.,point_weight,valid)
    call require(valid.and.point_weight(1)==1.d0)
    call span_node_weights(z,0,4,.true.,[integer::],.true.,none,valid)
    call require(valid)
    z(3)=z(2)
    call span_node_weights(z,0,4,.true.,[0,1,2,3],.true.,w,valid)
    call require(.not.valid)
    if(rank==0) print *, 'PASS: nonuniform span lengths, periodic aliases, pointwise and empty selections'
  end subroutine

  subroutine check_velocity_profile()
    type(velocity_statistics_result),allocatable :: points(:)
    type(velocity_statistics_result) :: summary
    type(velocity_statistics) :: temporal
    real(real64),allocatable :: lengths(:)
    real(real64) :: measure,base(3),amplitude(3),expected_r(3,3),expected_density(3,3)
    integer :: count_nodes,node,local
    logical :: valid
    count_nodes=0
    do node=1,2
      if(mod(node-1,nrank)==rank) count_nodes=count_nodes+1
    enddo
    allocate(points(count_nodes),lengths(count_nodes))
    local=0
    do node=1,2
      if(mod(node-1,nrank)/=rank) cycle
      local=local+1
      base=[real(4*node-3,real64),0.d0,0.d0]
      if(node==1) then
        lengths(local)=1.d0
        amplitude=[1.d0,-2.d0,3.d0]
      else
        lengths(local)=3.d0
        amplitude=[3.d0,-4.d0,5.d0]
      endif
      call configure_velocity_statistics(temporal,0.d0,2.d0,valid)
      call require(valid)
      call push_velocity_sample(temporal,0.d0,.5d0*node,base-amplitude,valid)
      call require(valid)
      call push_velocity_sample(temporal,2.d0,1.5d0*node,base+amplitude,valid)
      call require(valid)
      call read_velocity_statistics(temporal,points(local),valid)
      call require(valid)
    enddo
    call summarize_velocity_profile(lengths,points,MPI_COMM_WORLD,measure,summary,valid)
    call require(valid.and.measure==4.d0.and.summary%duration==2.d0)
    call require(summary%mean_density==1.75d0)
    call require(summary%mean_r(1)==4.d0)
    call require(maxval(abs(summary%mean_f-[81.d0/14.d0,-13.d0/7.d0,33.d0/14.d0]))<2.d-10)
    expected_r=reshape([7.d0,-9.5d0,12.d0,-9.5d0,13.d0,-16.5d0,12.d0,-16.5d0,21.d0],[3,3])
    expected_density=(3.d0/16.d0)*reshape([55.d0,-74.d0,93.d0,-74.d0,100.d0,-126.d0, &
      93.d0,-126.d0,159.d0],[3,3])
    call require(maxval(abs(summary%covariance_r-expected_r))<2.d-10)
    call require(maxval(abs(summary%density_stress-expected_density))<2.d-10)
    call require(maxval(abs(summary%covariance_f-expected_density/1.75d0))<2.d-10)
    call require(abs(summary%rms_r(1)**2-7.d0)<2.d-10)
    call require(abs(summary%rms_f(1)**2-165.d0/28.d0)<2.d-10)
    ! Spatial variation of local means must not become temporal fluctuation energy.
    call require(abs(summary%rms_r(1)-2.5d0)>1.d-3)
    call require(summary%covariance_r(1,1)/=10.d0)
    if(rank==0) print *, 'PASS: local temporal profile, physical lengths, Favre weights, empty ranks'
    if(size(points)>0.and.rank==0) points(1)%duration=1.d0
    call summarize_velocity_profile(lengths,points,MPI_COMM_WORLD,measure,summary,valid)
    call require(.not.valid)
    if(size(points)>0.and.rank==0) points(1)%duration=2.d0
    lengths=0.d0
    call summarize_velocity_profile(lengths,points,MPI_COMM_WORLD,measure,summary,valid)
    call require(.not.valid)
  end subroutine

  subroutine require(condition)
    logical,intent(in) :: condition
    if(.not.condition) call MPI_Abort(MPI_COMM_WORLD,1,ierr)
  end subroutine
end program
