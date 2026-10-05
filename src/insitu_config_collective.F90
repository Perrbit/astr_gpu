module insitu_config_collective
  use mpi
  use iso_fortran_env, only: int64,real64
  use insitu_run_config, only: insitu_options,read_insitu_options
  implicit none
  private
  public :: read_insitu_options_collective
contains
  subroutine read_insitu_options_collective(filename,comm,options,ok,message)
    character(*),intent(in) :: filename
    integer,intent(in) :: comm
    type(insitu_options),intent(inout) :: options
    logical,intent(out) :: ok
    character(*),intent(out) :: message
    type(insitu_options) :: candidate
    logical :: parsed,flags(9),root_flags(9),same,all_same
    integer(int64) :: counts(5),root_counts(5)
    real(real64) :: times(3),root_times(3)
    character(1024) :: paths(8),root_paths(8)
    character(16) :: root_mode
    character(1024) :: parse_message
    integer :: rank,first_bad,local_bad,ierr,int_type,real_type

    ok=.false.
    call MPI_Comm_rank(comm,rank,ierr)
    call check_mpi(ierr,comm)
    call read_insitu_options(filename,candidate,parsed,parse_message)
    local_bad=huge(local_bad)
    if(.not.parsed) local_bad=rank
    call MPI_Allreduce(local_bad,first_bad,1,MPI_INTEGER,MPI_MIN,comm,ierr)
    call check_mpi(ierr,comm)
    if(first_bad/=huge(local_bad)) then
      call MPI_Bcast(parse_message,len(parse_message),MPI_CHARACTER,first_bad,comm,ierr)
      call check_mpi(ierr,comm)
      message=parse_message
      return
    endif

    ! Typed values avoid derived-type padding and compiler-specific serialization.
    flags=[candidate%enabled,candidate%statistics,candidate%render, &
      candidate%initial_frame,candidate%final_frame,candidate%air5_volume_statistics,candidate%air5_volume_reduction, &
      candidate%wall_mean_render,candidate%wall_separation]
    counts=[candidate%step_interval,candidate%host_budget_bytes, &
      candidate%device_budget_bytes,candidate%device_reserve_bytes,int(candidate%slice_index,int64)]
    times=[candidate%time_interval,candidate%statistics_window]
    paths=[candidate%implementation_path,candidate%pipeline_file,candidate%output_directory, &
      candidate%batch_prefix,candidate%restore_batch,candidate%derivative_backend,candidate%products,candidate%slice_axis]
    root_flags=flags; root_counts=counts; root_times=times; root_paths=paths
    root_mode=candidate%schedule_mode
    call MPI_Type_match_size(MPI_TYPECLASS_INTEGER,storage_size(counts(1))/8,int_type,ierr)
    call check_mpi(ierr,comm)
    call MPI_Type_match_size(MPI_TYPECLASS_REAL,storage_size(times(1))/8,real_type,ierr)
    call check_mpi(ierr,comm)
    call MPI_Bcast(root_flags,size(root_flags),MPI_LOGICAL,0,comm,ierr)
    call check_mpi(ierr,comm)
    call MPI_Bcast(root_counts,size(root_counts),int_type,0,comm,ierr)
    call check_mpi(ierr,comm)
    call MPI_Bcast(root_times,size(root_times),real_type,0,comm,ierr)
    call check_mpi(ierr,comm)
    call MPI_Bcast(root_paths,len(root_paths)*size(root_paths),MPI_CHARACTER,0,comm,ierr)
    call check_mpi(ierr,comm)
    call MPI_Bcast(root_mode,len(root_mode),MPI_CHARACTER,0,comm,ierr)
    call check_mpi(ierr,comm)
    same=all(flags.eqv.root_flags).and.all(counts==root_counts).and. &
      all(times==root_times).and.all(paths==root_paths).and.candidate%schedule_mode==root_mode
    call MPI_Allreduce(same,all_same,1,MPI_LOGICAL,MPI_LAND,comm,ierr)
    call check_mpi(ierr,comm)
    if(.not.all_same) then
      message='in-situ configuration values differ between MPI ranks'
      return
    endif
    options=candidate
    message=''
    ok=.true.
  end subroutine

  subroutine check_mpi(status,comm)
    integer,intent(in) :: status,comm
    integer :: ignored
    if(status/=MPI_SUCCESS) call MPI_Abort(comm,status,ignored)
  end subroutine
end module
