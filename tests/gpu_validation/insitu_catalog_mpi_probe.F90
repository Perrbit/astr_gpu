program insitu_catalog_mpi_probe
  use mpi
  use iso_fortran_env, only: int64,real64
  use iso_c_binding, only: c_char,c_int,c_null_char
  use insitu_catalog
  use insitu_product_schedule, only: runtime_catalog_state,runtime_product_state, &
    runtime_catalog_due,runtime_catalog_success,runtime_catalog_disable,runtime_active
  implicit none
  interface
    integer(c_int) function rename_file(a,b) bind(C,name='rename')
      import c_char,c_int
      character(c_char),intent(in) :: a(*),b(*)
    end function
  end interface
  type(runtime_product_state) :: state
  character(4096) :: formal,candidate,repeat_candidate
  character(32) :: mode
  character(:),allocatable :: message
  integer :: rank,err,status,unit
  integer(int64) :: reads,checks
  logical :: updated,ok
  call MPI_Init(err)
  call MPI_Comm_rank(MPI_COMM_WORLD,rank,err)
  call get_command_argument(1,formal)
  call get_command_argument(2,candidate)
  call get_command_argument(3,mode)
  call get_command_argument(4,repeat_candidate)
  call configure_catalog_watch(trim(formal),10_int64,0_int64,MPI_COMM_WORLD)
  call poll_catalog_watch(1_int64,.001_real64,[2250,260,480],5,MPI_COMM_WORLD,updated,message)
  call require(updated,'initial file was not applied')
  call runtime_catalog_state(1,state)
  call require(state%version==1.and.state%definition%index==20,'initial normalized metadata differs')
  call runtime_catalog_success(1,1_int64,.001_real64)
  if(mode=='feedback') then
    call runtime_catalog_disable(1,2_int64,.002_real64,'injected "disk" error'//achar(10)//'line2')
    call write_catalog_status(trim(formal)//'.status.json',2_int64,.002_real64,MPI_COMM_WORLD,'',ok)
    call require(ok,'status feedback was not published')
    call MPI_Finalize(err)
    stop
  endif
  if(mode=='restart-ignore') then
    open(newunit=unit,status='scratch',access='stream',form='unformatted')
    call catalog_watch_state_file(unit,.true.,5_int64,.005_real64,MPI_COMM_WORLD,ok)
    call require(ok,'cannot save watcher/control state')
    if(rank==0) then
      status=rename_file(trim(candidate)//c_null_char,trim(formal)//c_null_char)
      call require(status==0,'pre-restart publication failed')
    endif
    call MPI_Barrier(MPI_COMM_WORLD,err)
    call configure_catalog_watch(trim(formal),10_int64,5_int64,MPI_COMM_WORLD)
    rewind(unit)
    call catalog_watch_state_file(unit,.false.,5_int64,.005_real64,MPI_COMM_WORLD,ok)
    close(unit)
    call require(ok,'cannot restore watcher/control state')
    call poll_catalog_watch(11_int64,.011_real64,[2250,260,480],5,MPI_COMM_WORLD,updated,message)
    call runtime_catalog_state(1,state)
    call require(.not.updated.and.state%definition%index==20,'disk configuration replaced checkpoint authority')
    if(rank==0) then
      status=rename_file(trim(repeat_candidate)//c_null_char,trim(formal)//c_null_char)
      call require(status==0,'explicit republication failed')
    endif
    call MPI_Barrier(MPI_COMM_WORLD,err)
    call poll_catalog_watch(21_int64,.021_real64,[2250,260,480],5,MPI_COMM_WORLD,updated,message)
    call runtime_catalog_state(1,state)
    call require(updated.and.state%definition%index==21,'explicit republication was not applied')
    call catalog_watch_counts(reads,checks)
    if(rank==0) call require(reads==1.and.checks==3,'restore triggered an implicit catalog read')
    if(rank/=0) call require(reads==0.and.checks==0,'restore caused nonroot catalog access')
    print '(a,i0,a)','RANK ',rank,' RESTORE_OK'
    call MPI_Finalize(err)
    stop
  endif
  call poll_catalog_watch(2_int64,.002_real64,[2250,260,480],5,MPI_COMM_WORLD,updated,message)
  call require(.not.updated,'poll ran before its interval')
  call poll_catalog_watch(11_int64,.011_real64,[2250,260,480],5,MPI_COMM_WORLD,updated,message)
  call require(.not.updated,'unchanged file must not be parsed again')
  if(rank==0) then
    status=rename_file(trim(candidate)//c_null_char,trim(formal)//c_null_char)
    call require(status==0,'candidate publication failed')
  endif
  call MPI_Barrier(MPI_COMM_WORLD,err)
  call poll_catalog_watch(21_int64,.021_real64,[2250,260,480],5,MPI_COMM_WORLD,updated,message)
  call runtime_catalog_state(1,state)
  if(mode=='change') then
    call require(updated,'new publication was not applied')
    call require(state%version==2.and.state%definition%index==21,'updated metadata differs')
    call require(runtime_catalog_due(1,21_int64,.021_real64),'new definition first frame missing')
  else
    call require(.not.updated.and.len(message)>0,'bad header must reject the file update')
    call require(state%version==1.and.state%status==runtime_active,'file rejection changed effective state')
    call poll_catalog_watch(22_int64,.022_real64,[2250,260,480],5,MPI_COMM_WORLD,updated,message)
    call write_catalog_status(trim(formal)//'.status.json',22_int64,.022_real64,MPI_COMM_WORLD,message,ok)
    call require(ok,'retained file error feedback was not published')
  endif
  call catalog_watch_counts(reads,checks)
  if(rank==0) then
    call require(reads==2.and.checks==3,'root read/stat counts differ')
  else
    call require(reads==0.and.checks==0,'non-root accessed the catalog file')
  endif
  print '(a,i0,a,i0,a,i0)','RANK ',rank,' READS ',reads,' CHECKS ',checks
  call MPI_Finalize(err)
contains
  subroutine require(condition,message)
    logical,intent(in) :: condition
    character(*),intent(in) :: message
    integer :: ierr
    if(condition) return
    print '(a,i0,2a)','FAIL rank=',rank,': ',message
    call MPI_Abort(MPI_COMM_WORLD,1,ierr)
  end subroutine
end program
