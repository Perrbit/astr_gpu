program sparse_probe
  use iso_c_binding
  use mpi
  use ieee_arithmetic, only: ieee_is_finite
  use device_state
  use fluid_explicit, only: fluid_prepare
  use insitu_sampling_cpp, only: probe_create,probe_capture,probe_metrics,probe_destroy
  implicit none
  integer,parameter :: upper=8,hm=5,count=5
  type(c_ptr) :: parent,probes
  integer :: rank,ranks,local_rank,err,status,debug,i,j,k,n,a
  integer(c_int) :: indices(3,count),invalid_indices(3,count)
  integer(c_int64_t) :: usage(5)
  real(c_device_real_kind) :: q(-hm:upper+hm,-hm:upper+hm,-hm:upper+hm,5),before(size(q)),after(size(q)), &
    expected(6,count),actual(6,count),constants(6),rho,u,v,w,pressure,temperature,error
  real(c_double) :: maximum
  character(64) :: argument
  call get_environment_variable('OMPI_COMM_WORLD_LOCAL_RANK',argument,status=status)
  if(status/=0) error stop 'sparse probe needs its matching Open MPI launcher'
  read(argument,*) local_rank
  call check(device_bind(int(local_rank,c_int)),'bind')
  call MPI_Init(err);call MPI_Comm_rank(MPI_COMM_WORLD,rank,err);call MPI_Comm_size(MPI_COMM_WORLD,ranks,err)
  constants=[2._c_device_real_kind,2.5_c_device_real_kind,3._c_device_real_kind, &
    4._c_device_real_kind,5._c_device_real_kind,6._c_device_real_kind]
  indices=reshape([0,0,0,3,2,1,8,8,8,3,2,1,-1,-1,-1],[3,count])
  do k=-hm,upper+hm
  do j=-hm,upper+hm
  do i=-hm,upper+hm
    rho=1+.01_c_device_real_kind*i+.02_c_device_real_kind*j+.03_c_device_real_kind*k
    u=.2_c_device_real_kind+.01_c_device_real_kind*i
    v=-.3_c_device_real_kind+.01_c_device_real_kind*j
    w=.4_c_device_real_kind+.01_c_device_real_kind*k
    pressure=1+.03_c_device_real_kind*rank
    q(i,j,k,:)=[rho,rho*u,rho*v,rho*w,constants(2)*pressure+.5_c_device_real_kind*rho*(u*u+v*v+w*w)]
  enddo
  enddo
  enddo
  before=reshape(q,[size(q)]);expected=0
  do n=1,count-1
    i=indices(1,n);j=indices(2,n);k=indices(3,n)
    rho=1+.01_c_device_real_kind*i+.02_c_device_real_kind*j+.03_c_device_real_kind*k
    u=.2_c_device_real_kind+.01_c_device_real_kind*i
    v=-.3_c_device_real_kind+.01_c_device_real_kind*j
    w=.4_c_device_real_kind+.01_c_device_real_kind*k
    pressure=1+.03_c_device_real_kind*rank;temperature=(pressure/rho)*constants(1)
    expected(:,n)=[u/constants(3),v/constants(3),w/constants(3),rho/constants(4), &
      pressure/constants(5),temperature/constants(6)]
  enddo
  maximum=0
  do debug=0,1
    call check(device_state_create(int(local_rank,c_int),upper,upper,upper,hm,int(debug,c_int),parent),'create parent')
    call check(fluid_prepare(parent,1_c_int),'prepare parent')
    call check(device_state_upload(parent,device_q,q,int(size(q),c_size_t)*c_sizeof(q(0,0,0,1))),'upload state')
    invalid_indices=indices;invalid_indices(:,1)=[-1,0,-1]
    status=probe_create(parent,count,invalid_indices,constants,probes)
    if(status/=device_invalid_argument.or.c_associated(probes)) error stop 'mixed absent index accepted'
    invalid_indices=indices;invalid_indices(1,1)=upper+1
    status=probe_create(parent,count,invalid_indices,constants,probes)
    if(status/=device_invalid_argument.or.c_associated(probes)) error stop 'physical index overflow accepted'
    call check(probe_create(parent,count,indices,constants,probes),'create sparse probes')
    status=device_state_destroy(parent)
    if(status/=device_invalid_argument.or..not.c_associated(parent)) error stop 'probe parent lifetime lost'
    call check(probe_capture(probes,actual,int(size(actual),c_size_t)*c_sizeof(actual(1,1))),'capture')
    error=maxval(abs(actual-expected));maximum=max(maximum,real(error,kind(maximum)))
    if(.not.all(ieee_is_finite(actual))) error stop 'nonfinite normalized primitives'
    if(c_sizeof(actual(1,1))==8.and.error>2.e-10_c_device_real_kind) error stop 'analytic normalized primitives'
    if(any(actual(:,count)/=0)) error stop 'non-owner contribution not zero'
    q(3,2,1,1)=-1
    call check(device_state_upload(parent,device_q,q,int(size(q),c_size_t)*c_sizeof(q(0,0,0,1))),'invalid selected state')
    status=probe_capture(probes,actual,int(size(actual),c_size_t)*c_sizeof(actual(1,1)))
    if(status/=device_nonphysical) error stop 'invalid selected density not reported'
    q=reshape(before,shape(q))
    call check(device_state_upload(parent,device_q,q,int(size(q),c_size_t)*c_sizeof(q(0,0,0,1))),'restore selected state')
    call check(probe_capture(probes,actual,int(size(actual),c_size_t)*c_sizeof(actual(1,1))),'capture restored state')
    call check(probe_metrics(probes,usage),'transfer counters')
    if(any(usage/=[int(count*(3*4+6*c_sizeof(actual(1,1)))+4,c_int64_t),int(3*count*4,c_int64_t), &
      2_c_int64_t,12_c_int64_t,int(2*size(actual)*c_sizeof(actual(1,1)),c_int64_t)])) error stop 'sparse traffic counters'
    call check(device_state_download(parent,device_q,after,int(size(after),c_size_t)*c_sizeof(after(1))),'observer isolation')
    if(any(transfer(before,[0_c_int8_t],size(before)*c_sizeof(before(1)))/= &
           transfer(after,[0_c_int8_t],size(after)*c_sizeof(after(1))))) &
      error stop 'sparse observer modified parent state'
    call check(probe_destroy(probes),'release probes')
    call check(device_state_destroy(parent),'release parent')
  enddo
  call MPI_Allreduce(MPI_IN_PLACE,maximum,1,MPI_DOUBLE_PRECISION,MPI_MAX,MPI_COMM_WORLD,err)
  if(rank==0) print '(a,i0,a,es24.16)','CPP_SPARSE_PROBE_PASS ranks=',ranks,' maximum_absolute=',maximum
  call MPI_Finalize(err)
contains
  subroutine check(code,label)
    integer(c_int),intent(in) :: code
    character(*),intent(in) :: label
    if(code/=0) then
      print *,label,code
      error stop 'sparse probe API failure'
    endif
  end subroutine
end program
