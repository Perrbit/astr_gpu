program keep_geometry_probe
  use astr_precision, only: real_kind
  use keep_operator, only: keep_two_point_flux5
  use compact_operator
  use compact_line_solver, only: compact_matrix
  use mpi
  use hdf5
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  type :: derivative_matrix
    real(real_kind),allocatable :: d(:,:)
  end type
  type(derivative_matrix) :: operators(3)
  real(real_kind),allocatable :: jac(:,:,:),field(:,:,:),area(:,:,:,:,:)
  real(real_kind) :: velocity(3),flux(5),rhs(5),metric_rhs(3),max_rhs(5),max_metric(3),weight
  integer(hid_t) :: file,dataset,space
  integer(hsize_t) :: shape3(3),maximum(3)
  integer :: ierr,hdferr,rank,extent,last(3),point(3),other(3),i,j,k,axis,b,m
  character(1024) :: path
  character(16) :: scheme
  character(5) :: key
  call mpi_init(ierr)
  call mpi_comm_size(MPI_COMM_WORLD,rank,ierr)
  if(rank/=1) error stop 'Geometry component probe uses one rank and the full saved geometry'
  call get_command_argument(1,path)
  call get_command_argument(2,scheme)
  if(scheme/='643e'.and.scheme/='643c') error stop 'Geometry probe requires 643e or 643c'
  call h5open_f(hdferr);call check_hdf()
  call h5fopen_f(trim(path),H5F_ACC_RDONLY_F,file,hdferr);call check_hdf()
  call h5dopen_f(file,'q0004',dataset,hdferr);call check_hdf()
  call h5dget_space_f(dataset,space,hdferr);call check_hdf()
  call h5sget_simple_extent_ndims_f(space,extent,hdferr);call check_hdf()
  if(extent/=3) error stop 'Geometry probe expects a three-dimensional resource'
  call h5sget_simple_extent_dims_f(space,shape3,maximum,hdferr)
  if(hdferr<0) error stop 'Geometry shape read failed'
  last=int(shape3)-1
  allocate(jac(0:last(1),0:last(2),0:last(3)),field(0:last(1),0:last(2),0:last(3)), &
    area(0:last(1),0:last(2),0:last(3),3,3))
  call h5dread_f(dataset,H5T_NATIVE_DOUBLE,jac,shape3,hdferr);call check_hdf()
  call h5sclose_f(space,hdferr);call check_hdf()
  call h5dclose_f(dataset,hdferr);call check_hdf()
  if(.not.all(ieee_is_finite(jac)).or.any(jac<=0.e0_real_kind)) error stop 'Invalid saved Jacobian'
  do b=1,3
    do axis=1,3
      write(key,'(A,I4.4)') 'q',4+axis+3*(b-1)
      call h5dopen_f(file,key,dataset,hdferr);call check_hdf()
      call h5dread_f(dataset,H5T_NATIVE_DOUBLE,field,shape3,hdferr);call check_hdf()
      call h5dclose_f(dataset,hdferr);call check_hdf()
      if(.not.all(ieee_is_finite(field))) error stop 'Invalid saved inverse metric'
      area(:,:,:,axis,b)=jac*field
    enddo
  enddo
  call h5fclose_f(file,hdferr);call check_hdf()
  call h5close_f(hdferr);call check_hdf()
  do axis=1,3
    allocate(operators(axis)%d(last(axis),last(axis)))
    call build_derivative(last(axis),scheme=='643c',operators(axis)%d)
  enddo
  velocity=[0.2e0_real_kind,0.13e0_real_kind,-0.17e0_real_kind]
  max_rhs=0.e0_real_kind;max_metric=0.e0_real_kind
  do k=0,last(3)-1
  do j=0,last(2)-1
  do i=0,last(1)-1
    point=[i,j,k];rhs=0.e0_real_kind;metric_rhs=0.e0_real_kind
    do axis=1,3
      other=point
      do m=0,last(axis)-1
        weight=operators(axis)%d(point(axis)+1,m+1)
        if(weight==0.e0_real_kind) cycle
        other(axis)=m
        call keep_two_point_flux5(1.e0_real_kind,velocity,1.e0_real_kind,area(i,j,k,axis,:), &
          1.e0_real_kind,velocity,1.e0_real_kind,area(other(1),other(2),other(3),axis,:),1.4e0_real_kind,flux)
        rhs=rhs-2.e0_real_kind*weight*flux/jac(i,j,k)
        metric_rhs=metric_rhs+weight*area(other(1),other(2),other(3),axis,:)
      enddo
    enddo
    max_rhs=max(max_rhs,abs(rhs));max_metric=max(max_metric,abs(metric_rhs))
  enddo
  enddo
  enddo
  if(.not.all(ieee_is_finite(max_rhs))) error stop 'Nonfinite constant-state KEEP diagnostic'
  write(*,'(A,A)') 'KEEP_ACTUAL_GEOMETRY scheme=',trim(scheme)
  write(*,'(A,3ES24.16)') 'metric_identity_max=',max_metric
  write(*,'(A,5ES24.16)') 'native_constant_state_rhs_max=',max_rhs
  ! No zero-residual admission threshold is inferred from this audit.
  call mpi_finalize(ierr)
contains
  subroutine check_hdf()
    if(hdferr/=0) error stop 'Geometry diagnostic HDF5 read failed'
  end subroutine

  subroutine build_derivative(n,compact,d)
    integer,intent(in) :: n
    logical,intent(in) :: compact
    real(real_kind),intent(out) :: d(n,n)
    type(compact_matrix) :: matrix
    type(compact_line_partition) :: partition
    type(compact_cpu_plan) :: plan
    real(real_kind) :: input(-3:n+2,n),rhs(n,n),owned(n,n),edge(2)
    real(real_kind),parameter :: coefficients(3)=[0.75e0_real_kind,-0.15e0_real_kind,1.e0_real_kind/60.e0_real_kind]
    integer :: i,j,l
    d=0.e0_real_kind
    if(.not.compact) then
      do i=1,n
        do l=1,3
          d(i,modulo(i-1+l,n)+1)=coefficients(l)
          d(i,modulo(i-1-l,n)+1)=-coefficients(l)
        enddo
      enddo
      return
    endif
    partition=compact_line_partition(intervals=n,global_n=n,first=0,local_intervals=n-1, &
      owned=n,rank=0,ranks=1,periodic=.true.)
    input=0.e0_real_kind
    do j=1,n
      do i=-3,n+2
        if(modulo(i,n)+1==j) input(i,j)=1.e0_real_kind
      enddo
    enddo
    call compact_derivative_core(partition,matrix,edge)
    call prepare_compact_cpu(matrix,edge,partition,MPI_COMM_SELF,n,plan)
    call compact_derivative_rhs(partition,n-1,3,input,rhs)
    call solve_compact_cpu(plan,rhs,owned,d,n)
    call release_compact_cpu(plan)
  end subroutine
end program
