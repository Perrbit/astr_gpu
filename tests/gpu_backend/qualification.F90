program backend_qualification
  use iso_c_binding, only: c_int, c_double
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use mpi
  use hdf5
  implicit none
  interface
    integer(c_int) function evaluate(rank,n,x,y,d,s) bind(C,name='astr_mb0_evaluate')
      import c_int, c_double
      integer(c_int), value :: rank,n
      real(c_double), intent(in) :: x(*)
      real(c_double), intent(out) :: y(*),d(*),s
    end function
  end interface
  integer, parameter :: lengths(2) = [1031,65537]
  integer :: rank, nranks, ierr, test, n, i, j, status, major, minor, release
  integer(hid_t) :: file_id, space_id, dataset_id
  integer(hsize_t) :: dims(1)
  real(c_double), allocatable :: x(:), original(:), y(:), d(:), expected(:), dy(:)
  real(c_double) :: s, scaled(3), global(3), stored(6), restored(6)
  character(len=1024) :: path
  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)
  call MPI_Comm_size(MPI_COMM_WORLD,nranks,ierr)
  call get_command_argument(1,path)
  if (len_trim(path)==0) call fail('explicit HDF5 output path required')
  do test=1,size(lengths)
    n=lengths(test)
    allocate(x(n),original(n),y(n),d(n),expected(n),dy(n))
    do i=1,n
      x(i)=1.d0+real(mod(7*(i-1),257)-128,c_double)/512.d0
    end do
    original=x
    expected=1.5d0*x+0.25d0
    do i=1,n
      j=i-1
      dy(i)=(45.d0*(expected(mod(j+1,n)+1)-expected(mod(j+n-1,n)+1)) &
             -9.d0*(expected(mod(j+2,n)+1)-expected(mod(j+n-2,n)+1)) &
                   +expected(mod(j+3,n)+1)-expected(mod(j+n-3,n)+1))/60.d0
    end do
    status=evaluate(int(rank,c_int),int(n,c_int),x,y,d,s)
    if (status/=0) call fail('device qualification call failed')
    if (any(x/=original)) call fail('C ABI modified input')
    if (any(y/=expected)) call fail('cross-TU dyadic transform not exact')
    if (.not.all(ieee_is_finite(d)).or..not.ieee_is_finite(s)) call fail('nonfinite device result')
    scaled=[maxval(abs(y-expected))/max(1.d0,maxval(abs(expected))), &
            maxval(abs(d-dy))/max(1.d0,maxval(abs(dy))), &
            abs(s-sum(expected))/max(1.d0,abs(sum(expected)))]
    call MPI_Allreduce(scaled,global,3,MPI_DOUBLE_PRECISION,MPI_MAX,MPI_COMM_WORLD,ierr)
    if (maxval(global)>2.d-10) call fail('scaled operator/reduction error above 2e-10')
    stored(3*test-2:3*test)=global
    if (rank==0) write(*,'(a,i0,a,3(es24.16,1x))') 'MB0 n=',n,' errors=',global
    deallocate(x,original,y,d,expected,dy)
  end do
  if (rank==0) then
    call h5open_f(ierr)
    call hdf_check(ierr)
    call h5get_libversion_f(major,minor,release,ierr)
    call hdf_check(ierr)
    write(*,'(a,3(i0,1x))') 'MB0 HDF5 version=',major,minor,release
    dims=[6_hsize_t]
    call h5fcreate_f(trim(path),H5F_ACC_TRUNC_F,file_id,ierr)
    call hdf_check(ierr)
    call h5screate_simple_f(1,dims,space_id,ierr)
    call hdf_check(ierr)
    call h5dcreate_f(file_id,'scaled_errors',H5T_NATIVE_DOUBLE,space_id,dataset_id,ierr)
    call hdf_check(ierr)
    call h5dwrite_f(dataset_id,H5T_NATIVE_DOUBLE,stored,dims,ierr)
    call hdf_check(ierr)
    call h5dclose_f(dataset_id,ierr)
    call hdf_check(ierr)
    call h5sclose_f(space_id,ierr)
    call hdf_check(ierr)
    call h5fclose_f(file_id,ierr)
    call hdf_check(ierr)
    call h5fopen_f(trim(path),H5F_ACC_RDONLY_F,file_id,ierr)
    call hdf_check(ierr)
    call h5dopen_f(file_id,'scaled_errors',dataset_id,ierr)
    call hdf_check(ierr)
    call h5dread_f(dataset_id,H5T_NATIVE_DOUBLE,restored,dims,ierr)
    call hdf_check(ierr)
    if (any(restored/=stored)) call fail('HDF5 summary roundtrip differs')
    call h5dclose_f(dataset_id,ierr)
    call hdf_check(ierr)
    call h5fclose_f(file_id,ierr)
    call hdf_check(ierr)
    call h5close_f(ierr)
    call hdf_check(ierr)
    write(*,'(a,i0,a)') 'MB0 PASS ranks=',nranks,' exact copy/transform; D6/reduction <= 2e-10; HDF5 exact'
  end if
  call MPI_Finalize(ierr)
contains
  subroutine fail(message)
    character(len=*), intent(in) :: message
    write(*,'(a,i0,2a)') 'MB0 FAIL rank=',rank,': ',message
    call MPI_Abort(MPI_COMM_WORLD,1,ierr)
    error stop 1
  end subroutine
  subroutine hdf_check(code)
    integer, intent(in) :: code
    if (code/=0) call fail('HDF5 operation failed')
  end subroutine
end program
