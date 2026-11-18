program gradient_axis_reference_probe
  use astr_precision, only: real_kind
  use cudafor
  use mpi
  use ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_is_finite
  use commvar, only: im,jm,km,hm,lihomo,ljhomo,lkhomo,flowtype
  use bc, only: bctype
  use parallel, only: mpileft,mpiright,mpidown,mpiup,mpiback,mpifront
  use commarray_gpu, only: vel_d,tmp_d,dxi_d,dvel_d,dtmp_d
  use gradcal_gpu, only: compute_gradcal_gpu
  implicit none
  real(real_kind),allocatable :: velocity(:,:,:,:),temperature(:,:,:),metric(:,:,:,:,:), &
    gradient(:,:,:,:,:),thermal(:,:,:,:)
  real(real_kind) :: slopes(3,3),projection(3,3),dt(3),expected(3,3),expected_t(3),nan,error,maximum
  integer :: ierr,rank,ranks,axis,i,j,k,c,d,n,point(3),dims(3)
  ierr=cudaSetDevice(0);if(ierr/=cudaSuccess) error stop 'gradient-axis device selection'
  call MPI_Init(ierr);call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr);call MPI_Comm_size(MPI_COMM_WORLD,ranks,ierr)
  if(ranks/=1) error stop 'gradient-axis manufactured test is NP1'
  im=16;jm=12;km=8;dims=[im,jm,km];flowtype='tgv'
  mpileft=MPI_PROC_NULL;mpiright=MPI_PROC_NULL;mpidown=MPI_PROC_NULL
  mpiup=MPI_PROC_NULL;mpiback=MPI_PROC_NULL;mpifront=MPI_PROC_NULL
  allocate(velocity(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3),temperature(-hm:im+hm,-hm:jm+hm,-hm:km+hm), &
    metric(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3,3),gradient(0:im,0:jm,0:km,3,3),thermal(0:im,0:jm,0:km,3))
  allocate(vel_d,mold=velocity);allocate(tmp_d,mold=temperature);allocate(dxi_d,mold=metric)
  allocate(dvel_d,mold=gradient);allocate(dtmp_d,mold=thermal)
  slopes=reshape([1._real_kind,2._real_kind,4._real_kind,2._real_kind,-1._real_kind,.5_real_kind, &
    .5_real_kind,1._real_kind,-2._real_kind],[3,3])
  projection=reshape([1._real_kind,.25_real_kind,.5_real_kind,.5_real_kind,2._real_kind,.25_real_kind, &
    .25_real_kind,.5_real_kind,1._real_kind],[3,3])
  dt=[.125_real_kind,.25_real_kind,.5_real_kind]
  expected=matmul(transpose(slopes),projection);expected_t=matmul(dt,projection)
  nan=ieee_value(0._real_kind,ieee_quiet_nan);maximum=0
  do axis=1,3
    lihomo=axis/=1;ljhomo=axis/=2;lkhomo=axis/=3
    bctype=1
    select case(axis)
    case(1);bctype(1:2)=[11,21]
    case(2);bctype(3:4)=[41,52]
    case(3);bctype(5:6)=[50,50]
    end select
    do k=-hm,km+hm
    do j=-hm,jm+hm
    do i=-hm,im+hm
      point=[i,j,k]
      velocity(i,j,k,:)=matmul(real(point,real_kind),slopes)
      temperature(i,j,k)=32._real_kind+dot_product(real(point,real_kind),dt)
      metric(i,j,k,:,:)=projection
      if(point(axis)<0.or.point(axis)>dims(axis)) then
        velocity(i,j,k,:)=nan;temperature(i,j,k)=nan
      endif
    enddo
    enddo
    enddo
    vel_d=velocity;tmp_d=temperature;dxi_d=metric
    call compute_gradcal_gpu()
    ierr=cudaDeviceSynchronize();if(ierr/=cudaSuccess) error stop 'gradient-axis kernel execution'
    gradient=dvel_d;thermal=dtmp_d
    if(.not.all(ieee_is_finite(gradient)).or..not.all(ieee_is_finite(thermal))) &
      error stop 'single physical axis consumed poisoned exterior halos'
    error=0
    do c=1,3
      do d=1,3
        error=max(error,maxval(abs(gradient(:,:,:,c,d)-expected(c,d)))/max(1._real_kind,abs(expected(c,d))))
      enddo
      error=max(error,maxval(abs(thermal(:,:,:,c)-expected_t(c)))/max(1._real_kind,abs(expected_t(c))))
    enddo
    if(error>2.e-10_real_kind) error stop 'single physical axis affine derivative mismatch'
    maximum=max(maximum,error)
    write(*,'(a,i0,a,es24.16e3)') 'GRADIENT_AXIS_PHYSICAL axis=',axis,' max_scaled=',error
  enddo
  deallocate(vel_d,tmp_d,dxi_d,dvel_d,dtmp_d)
  write(*,'(a,es24.16e3)') 'GRADIENT_AXIS_PHYSICAL_PASS max_scaled=',maximum
  call MPI_Finalize(ierr)
end program
