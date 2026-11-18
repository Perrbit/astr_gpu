program profile_wall_probe
  use iso_c_binding
  use mpi
  use ieee_arithmetic, only: ieee_is_finite
  use device_state
  use fluid_explicit, only: fluid_prepare
  use insitu_wall_cpp, only: profile_wall_create,profile_wall_capture,profile_wall_metrics,profile_wall_destroy
  implicit none
  integer,parameter :: global=24,jm=8,km=8,hm=5
  integer :: rank,ranks,local_rank,err,status,im,offset,debug,i,j,k,g
  type(c_ptr) :: parent,sample
  real(c_device_real_kind),allocatable :: q(:,:,:,:),metric(:,:,:,:,:),weights(:,:),bad_weights(:,:), &
    before(:),after(:),expected(:,:),actual(:,:)
  real(c_device_real_kind) :: material(7),reference(2),rho,u,v,t,p,mu,weight,dz,previous,span,cf,heat
  real(c_double) :: maximum=0,error
  integer(c_int64_t) :: usage(5)
  integer(c_size_t) :: qbytes,metric_bytes,line_bytes,scalar_bytes
  character(64) :: argument
  call get_environment_variable('OMPI_COMM_WORLD_LOCAL_RANK',argument,status=status)
  if(status/=0) error stop 'profile wall probe needs matching Open MPI launcher'
  read(argument,*) local_rank
  call check(device_bind(int(local_rank,c_int)),'bind')
  call MPI_Init(err);call MPI_Comm_rank(MPI_COMM_WORLD,rank,err);call MPI_Comm_size(MPI_COMM_WORLD,ranks,err)
  if(ranks/=1.and.ranks/=2) error stop 'profile wall component uses one/two ranks'
  im=global/ranks;offset=rank*im
  material=[2._c_device_real_kind,.4_c_device_real_kind,2.5_c_device_real_kind,20._c_device_real_kind, &
    .7_c_device_real_kind,.4_c_device_real_kind,1.4_c_device_real_kind]
  reference=[3._c_device_real_kind,2._c_device_real_kind]
  allocate(q(-hm:im+hm,-hm:jm+hm,-hm:km+hm,5),metric(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3,3), &
    weights(0:im,0:km-1),bad_weights(0:im,0:km-1),actual(3,0:im),expected(3,0:im))
  q=-99;metric=0
  metric(:,:,:,1,1)=2.5_c_device_real_kind;metric(:,:,:,2,2)=1.75_c_device_real_kind
  do k=0,km-1
  do j=0,2
  do i=-hm,im+hm
    g=offset+i
    if(g<0.or.g>global) cycle
    rho=1+.001_c_device_real_kind*(g+k)
    u=.1_c_device_real_kind+.2_c_device_real_kind*j+.01_c_device_real_kind*j*j
    v=.05_c_device_real_kind*g
    t=1+.2_c_device_real_kind*j+.03_c_device_real_kind*j*j+.01_c_device_real_kind*k
    p=rho*t/material(1)
    q(i,j,k,:)=[rho,rho*u,rho*v,0._c_device_real_kind,material(3)*p+.5_c_device_real_kind*rho*(u*u+v*v)]
  enddo
  enddo
  enddo
  expected=0
  do i=0,im
    do k=0,km-1
      dz=1+.05_c_device_real_kind*(2*k+1)
      previous=1+.05_c_device_real_kind*(2*modulo(k-1,km)+1)
      weight=.5_c_device_real_kind*(dz+previous)*(1+.001_c_device_real_kind*(offset+i))
      weights(i,k)=weight;t=1+.01_c_device_real_kind*k
      mu=t*sqrt(t)*material(7)/(t+material(6))/material(4)
      cf=mu*(.2_c_device_real_kind*1.75_c_device_real_kind+.05_c_device_real_kind*2.5_c_device_real_kind)/ &
        (.5_c_device_real_kind*reference(1)*reference(2)*reference(2))
      heat=((mu/material(5))/material(2))*(.2_c_device_real_kind*1.75_c_device_real_kind)/ &
        (reference(1)*reference(2)*reference(2)*reference(2))
      expected(:,i)=expected(:,i)+weight*[1._c_device_real_kind,cf,heat]
    enddo
    span=expected(1,i);expected(2:3,i)=expected(2:3,i)/span
  enddo
  before=reshape(q,[size(q)]);allocate(after(size(q)))
  scalar_bytes=c_sizeof(q(0,0,0,1));qbytes=size(q,kind=c_size_t)*scalar_bytes
  metric_bytes=size(metric,kind=c_size_t)*scalar_bytes;line_bytes=size(actual,kind=c_size_t)*scalar_bytes
  do debug=0,1
    call check(device_state_create(int(local_rank,c_int),int(im,c_int),jm,km,hm,int(debug,c_int),parent),'create')
    call check(fluid_prepare(parent,1_c_int),'prepare')
    call check(device_state_upload(parent,device_q,q,qbytes),'upload q')
    call check(device_state_upload(parent,device_metric,metric,metric_bytes),'upload metric')
    bad_weights=weights;bad_weights(0,0)=0
    status=profile_wall_create(parent,int(offset,c_int),global,material,reference,bad_weights,sample)
    if(status/=device_invalid_argument.or.c_associated(sample)) error stop 'zero weight accepted'
    call check(profile_wall_create(parent,int(offset,c_int),global,material,reference,weights,sample),'create wall')
    status=device_state_destroy(parent)
    if(status/=device_invalid_argument.or..not.c_associated(parent)) error stop 'wall parent lifetime lost'
    call check(profile_wall_capture(sample,actual,line_bytes),'capture')
    error=maxval(abs(real(actual,c_double)-real(expected,c_double)));maximum=max(maximum,error)
    if(.not.all(ieee_is_finite(actual))) error stop 'nonfinite profile wall'
    if(scalar_bytes==8.and.error>2.e-10_c_double) error stop 'analytic shear/heat/span'
    q(1,0,0,1)=-1
    call check(device_state_upload(parent,device_q,q,qbytes),'bad selected state')
    status=profile_wall_capture(sample,actual,line_bytes)
    if(status/=device_nonphysical) error stop 'invalid wall density not reported'
    q=reshape(before,shape(q))
    call check(device_state_upload(parent,device_q,q,qbytes),'restore wall state')
    call check(profile_wall_capture(sample,actual,line_bytes),'capture after bad state')
    call check(profile_wall_metrics(sample,usage),'byte counters')
    if(any(usage/=[int((im+1)*(km+3)*scalar_bytes+4,c_int64_t),int((im+1)*km*scalar_bytes,c_int64_t), &
      2_c_int64_t,12_c_int64_t,int(2*line_bytes,c_int64_t)])) error stop 'wall bytes/capture count'
    call check(device_state_download(parent,device_q,after,qbytes),'read parent state')
    if(any(transfer(before,[0_c_int8_t],size(before)*scalar_bytes)/= &
           transfer(after,[0_c_int8_t],size(after)*scalar_bytes))) error stop 'wall observer changed parent'
    call check(profile_wall_destroy(sample),'destroy wall')
    call check(device_state_destroy(parent),'destroy parent')
  enddo
  call MPI_Allreduce(MPI_IN_PLACE,maximum,1,MPI_DOUBLE_PRECISION,MPI_MAX,MPI_COMM_WORLD,err)
  if(rank==0) print '(a,i0,a,es24.16)','CPP_PROFILE_WALL_PASS ranks=',ranks,' maximum_absolute=',maximum
  call MPI_Finalize(err)
contains
  subroutine check(code,label)
    integer(c_int),intent(in) :: code
    character(*),intent(in) :: label
    if(code/=0) then
      print *,label,code
      error stop 'profile wall API failure'
    endif
  end subroutine
end program
