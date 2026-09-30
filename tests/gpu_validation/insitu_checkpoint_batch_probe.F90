program batch_probe
  use mpi
  use iso_fortran_env, only: int64
  use insitu_checkpoint_batch
  implicit none
  character(1024) :: path
  character(96),allocatable :: files(:)
  integer :: rank,ranks,ierr,unit,i
  integer(int64) :: bytes,crc
  logical :: ok
  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)
  call MPI_Comm_size(MPI_COMM_WORLD,ranks,ierr)
  call get_command_argument(1,path)
  allocate(files(ranks))
  do i=1,ranks
    write(files(i),'("rank",I8.8,".bin")') i-1
  enddo
  call create_batch(trim(path),MPI_COMM_WORLD,ok)
  call require(ok)
  call create_batch(trim(path),MPI_COMM_WORLD,ok)
  call require(.not.ok)
  call publish(.true.)
  call require(.not.ok)
  open(newunit=unit,file=trim(path)//'/'//trim(files(rank+1)),status='new', &
    access='stream',form='unformatted')
  write(unit) '123456789'
  close(unit)
  call file_fingerprint(trim(path)//'/'//trim(files(rank+1)),bytes,crc,ok)
  call require(ok.and.bytes==9.and.crc==int(z'6C40DF5F0B497347',int64))
  call copy_batch_file(trim(path)//'/'//trim(files(rank+1)), &
    trim(path)//'/copy-'//trim(files(rank+1)),ok)
  call require(ok)
  call copy_batch_file(trim(path)//'/'//trim(files(rank+1)), &
    trim(path)//'/copy-'//trim(files(rank+1)),ok)
  call require(.not.ok)
  call file_fingerprint(trim(path)//'/copy-'//trim(files(rank+1)),bytes,crc,ok)
  call require(ok.and.bytes==9.and.crc==int(z'6C40DF5F0B497347',int64))
  call validate('case-a')
  call require(.not.ok)
  call publish(rank/=ranks-1)
  call require(.not.ok)
  call publish(.true.)
  call require(ok)
  call validate('case-a')
  call require(ok)
  call validate('case-b')
  call require(.not.ok)
  call publish(.true.)
  call require(.not.ok)
  call validate('case-a')
  call require(ok)
  if(rank==0) then
    open(newunit=unit,file=trim(path)//'/'//trim(files(1)),status='old',access='stream',form='unformatted')
    write(unit,pos=1) '923456789'
    close(unit)
  endif
  call MPI_Barrier(MPI_COMM_WORLD,ierr)
  call validate('case-a')
  call require(.not.ok)
  if(rank==0) print *, 'PASS: CRC64, verified non-overwrite copy/publication, partial/readiness/config/corruption rejection'
  call MPI_Finalize(ierr)
contains
  subroutine require(condition)
    logical,intent(in) :: condition
    if(.not.condition) then
      print *, 'batch probe assertion failed on rank ',rank
      call MPI_Abort(MPI_COMM_WORLD,1,ierr)
    endif
  end subroutine
  subroutine publish(ready)
    logical,intent(in) :: ready
    call publish_batch(trim(path),'batch-a',1_int64,0.001d0,[ranks,1,1],'case-a',files,ready,MPI_COMM_WORLD,ok)
  end subroutine
  subroutine validate(config)
    character(*),intent(in) :: config
    call validate_batch(trim(path),'batch-a',1_int64,0.001d0,[ranks,1,1],config,files,MPI_COMM_WORLD,ok)
  end subroutine
end program
