module mpiio

   use, intrinsic :: iso_fortran_env, only : real64
   use mpi
   
   implicit none
   private

   integer, parameter, public :: rk = real64

   public :: mpiio_write_2d_real64
   public :: mpiio_read_2d_real64
   public :: mpiio_write_3d_real64
   public :: mpiio_read_3d_real64

contains

   subroutine mpi_check(ierr, where)
      integer, intent(in) :: ierr
      character(*), intent(in) :: where
      integer :: ierr2
      if (ierr /= MPI_SUCCESS) then
         write(*,'(a,1x,a,1x,i0)') 'MPI error in', trim(where), ierr
         call MPI_Abort(MPI_COMM_WORLD, ierr, ierr2)
      end if
   end subroutine mpi_check

   !===========================================================
   ! Write 2D real(8) array
   !===========================================================
   subroutine mpiio_write_2d_real64(filename, a, im, jm, ia, ja, is, js, comm)
      character(*), intent(in) :: filename
      integer,      intent(in) :: im, jm          ! local upper bounds
      integer,      intent(in) :: ia, ja          ! global upper bounds
      integer,      intent(in) :: is, js          ! global start indices of local block
      integer,      intent(in) :: comm
      real(rk),     intent(in) :: a(0:im, 0:jm)

      integer :: fh, ierr, filetype
      integer :: gsizes(2), lsizes(2), starts(2)
      integer :: status(MPI_STATUS_SIZE)
      integer :: nlocal
      integer(kind=MPI_OFFSET_KIND) :: disp

      gsizes = [ia+1, ja+1]
      lsizes = [im+1, jm+1]
      starts = [is,   js  ]
      nlocal = (im+1)*(jm+1)
      disp   = 0_MPI_OFFSET_KIND

      call MPI_Type_create_subarray(2, gsizes, lsizes, starts, MPI_ORDER_FORTRAN, &
                                    MPI_DOUBLE_PRECISION, filetype, ierr)
      call mpi_check(ierr, 'MPI_Type_create_subarray(2D write)')
      call MPI_Type_commit(filetype, ierr)
      call mpi_check(ierr, 'MPI_Type_commit(2D write)')

      call MPI_File_open(comm, filename, MPI_MODE_WRONLY + MPI_MODE_CREATE, &
                         MPI_INFO_NULL, fh, ierr)
      call mpi_check(ierr, 'MPI_File_open(2D write)')

      call MPI_File_set_view(fh, disp, MPI_DOUBLE_PRECISION, filetype, &
                             'native', MPI_INFO_NULL, ierr)
      call mpi_check(ierr, 'MPI_File_set_view(2D write)')

      call MPI_File_write_all(fh, a, nlocal, MPI_DOUBLE_PRECISION, status, ierr)
      call mpi_check(ierr, 'MPI_File_write_all(2D)')

      call MPI_File_close(fh, ierr)
      call mpi_check(ierr, 'MPI_File_close(2D write)')

      call MPI_Type_free(filetype, ierr)
      call mpi_check(ierr, 'MPI_Type_free(2D write)')
   end subroutine mpiio_write_2d_real64

   !===========================================================
   ! Read 2D real(8) array
   !===========================================================
   subroutine mpiio_read_2d_real64(filename, a, im, jm, ia, ja, is, js)
      character(*), intent(in) :: filename
      integer,      intent(in) :: im, jm
      integer,      intent(in) :: ia, ja
      integer,      intent(in) :: is, js
      real(rk),     intent(out) :: a(0:im, 0:jm)

      integer :: fh, ierr, filetype
      integer :: gsizes(2), lsizes(2), starts(2)
      integer :: status(MPI_STATUS_SIZE)
      integer :: nlocal
      integer(kind=MPI_OFFSET_KIND) :: disp

      integer :: comm

      comm=MPI_COMM_WORLD

      gsizes = [ia+1, ja+1]
      lsizes = [im+1, jm+1]
      starts = [is,   js  ]
      nlocal = (im+1)*(jm+1)
      disp   = 0_MPI_OFFSET_KIND

      call MPI_Type_create_subarray(2, gsizes, lsizes, starts, MPI_ORDER_FORTRAN, &
                                    MPI_DOUBLE_PRECISION, filetype, ierr)
      call mpi_check(ierr, 'MPI_Type_create_subarray(2D read)')
      call MPI_Type_commit(filetype, ierr)
      call mpi_check(ierr, 'MPI_Type_commit(2D read)')

      call MPI_File_open(comm, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, fh, ierr)
      call mpi_check(ierr, 'MPI_File_open(2D read)')

      call MPI_File_set_view(fh, disp, MPI_DOUBLE_PRECISION, filetype, &
                             'native', MPI_INFO_NULL, ierr)
      call mpi_check(ierr, 'MPI_File_set_view(2D read)')

      call MPI_File_read_all(fh, a, nlocal, MPI_DOUBLE_PRECISION, status, ierr)
      call mpi_check(ierr, 'MPI_File_read_all(2D)')

      call MPI_File_close(fh, ierr)
      call mpi_check(ierr, 'MPI_File_close(2D read)')

      call MPI_Type_free(filetype, ierr)
      call mpi_check(ierr, 'MPI_Type_free(2D read)')
   end subroutine mpiio_read_2d_real64

   !===========================================================
   ! Write 3D real(8) array
   !===========================================================
   subroutine mpiio_write_3d_real64(filename, a, im, jm, km, ia, ja, ka, is, js, ks)
      character(*), intent(in) :: filename
      integer,      intent(in) :: im, jm, km
      integer,      intent(in) :: ia, ja, ka
      integer,      intent(in) :: is, js, ks
      real(rk),     intent(in) :: a(0:im, 0:jm, 0:km)

      integer :: fh, ierr, filetype
      integer :: gsizes(3), lsizes(3), starts(3)
      integer :: status(MPI_STATUS_SIZE)
      integer :: nlocal
      integer(kind=MPI_OFFSET_KIND) :: disp
      integer :: comm

      comm=MPI_COMM_WORLD

      gsizes = [ia+1, ja+1, ka+1]
      lsizes = [im+1, jm+1, km+1]
      starts = [is,   js,   ks  ]
      nlocal = (im+1)*(jm+1)*(km+1)
      disp   = 0_MPI_OFFSET_KIND

      call MPI_Type_create_subarray(3, gsizes, lsizes, starts, MPI_ORDER_FORTRAN, &
                                    MPI_DOUBLE_PRECISION, filetype, ierr)
      call mpi_check(ierr, 'MPI_Type_create_subarray(3D write)')
      call MPI_Type_commit(filetype, ierr)
      call mpi_check(ierr, 'MPI_Type_commit(3D write)')

      call MPI_File_open(comm, filename, MPI_MODE_WRONLY + MPI_MODE_CREATE, &
                         MPI_INFO_NULL, fh, ierr)
      call mpi_check(ierr, 'MPI_File_open(3D write)')

      call MPI_File_set_view(fh, disp, MPI_DOUBLE_PRECISION, filetype, &
                             'native', MPI_INFO_NULL, ierr)
      call mpi_check(ierr, 'MPI_File_set_view(3D write)')

      call MPI_File_write_all(fh, a, nlocal, MPI_DOUBLE_PRECISION, status, ierr)
      call mpi_check(ierr, 'MPI_File_write_all(3D)')

      call MPI_File_close(fh, ierr)
      call mpi_check(ierr, 'MPI_File_close(3D write)')

      call MPI_Type_free(filetype, ierr)
      call mpi_check(ierr, 'MPI_Type_free(3D write)')
   end subroutine mpiio_write_3d_real64

   !===========================================================
   ! Read 3D real(8) array
   !===========================================================
   subroutine mpiio_read_3d_real64(filename, a, im, jm, km, ia, ja, ka, is, js, ks)
      character(*), intent(in) :: filename
      integer,      intent(in) :: im, jm, km
      integer,      intent(in) :: ia, ja, ka
      integer,      intent(in) :: is, js, ks
      real(rk),     intent(out) :: a(0:im, 0:jm, 0:km)

      integer :: fh, ierr, filetype
      integer :: gsizes(3), lsizes(3), starts(3)
      integer :: status(MPI_STATUS_SIZE)
      integer :: nlocal
      integer(kind=MPI_OFFSET_KIND) :: disp
      integer :: comm

      comm=MPI_COMM_WORLD

      gsizes = [ia+1, ja+1, ka+1]
      lsizes = [im+1, jm+1, km+1]
      starts = [is,   js,   ks  ]
      nlocal = (im+1)*(jm+1)*(km+1)
      disp   = 0_MPI_OFFSET_KIND

      call MPI_Type_create_subarray(3, gsizes, lsizes, starts, MPI_ORDER_FORTRAN, &
                                    MPI_DOUBLE_PRECISION, filetype, ierr)
      call mpi_check(ierr, 'MPI_Type_create_subarray(3D read)')
      call MPI_Type_commit(filetype, ierr)
      call mpi_check(ierr, 'MPI_Type_commit(3D read)')

      call MPI_File_open(comm, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, fh, ierr)
      call mpi_check(ierr, 'MPI_File_open(3D read)')

      call MPI_File_set_view(fh, disp, MPI_DOUBLE_PRECISION, filetype, &
                             'native', MPI_INFO_NULL, ierr)
      call mpi_check(ierr, 'MPI_File_set_view(3D read)')

      call MPI_File_read_all(fh, a, nlocal, MPI_DOUBLE_PRECISION, status, ierr)
      call mpi_check(ierr, 'MPI_File_read_all(3D)')

      call MPI_File_close(fh, ierr)
      call mpi_check(ierr, 'MPI_File_close(3D read)')

      call MPI_Type_free(filetype, ierr)
      call mpi_check(ierr, 'MPI_Type_free(3D read)')
   end subroutine mpiio_read_3d_real64

end module mpiio