module insitu_config_collective
  use mpi
  use iso_fortran_env, only: int64,real64
  use insitu_run_config, only: insitu_options,read_insitu_options,insitu_max_products,canonical_slice_plane
  use adaptive_output_collective, only: agree_adaptive_bindings
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
    logical :: parsed,flags(10),root_flags(10),same,all_same
    integer(int64) :: counts(5),root_counts(5)
    real(real64) :: times(9),root_times(9),plane(6)
    character(1024) :: paths(13),root_paths(13)
    character(16) :: root_mode
    character(64) :: root_ids(insitu_max_products)
    character(16) :: root_product_modes(insitu_max_products)
    integer(int64) :: root_steps(insitu_max_products)
    real(real64) :: root_times_product(insitu_max_products)
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
      candidate%wall_mean_render,candidate%wall_separation,candidate%mean_streamline_render]
    counts=[candidate%step_interval,candidate%host_budget_bytes, &
      candidate%device_budget_bytes,candidate%device_reserve_bytes,int(candidate%slice_index,int64)]
    times=[candidate%time_interval,candidate%statistics_window,candidate%slice_origin,candidate%slice_normal]
    paths=[candidate%implementation_path,candidate%pipeline_file,candidate%output_directory, &
      candidate%batch_prefix,candidate%restore_batch,candidate%derivative_backend,candidate%products,candidate%slice_axis, &
      candidate%processing_backend,candidate%postprocess_transport,candidate%rendering_pipeline,candidate%slice_definition, &
      candidate%streamline_seeds]
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
    root_ids=candidate%product_ids; root_product_modes=candidate%product_modes
    root_steps=candidate%product_steps; root_times_product=candidate%product_times
    call MPI_Bcast(root_ids,len(root_ids)*size(root_ids),MPI_CHARACTER,0,comm,ierr)
    call check_mpi(ierr,comm)
    call MPI_Bcast(root_product_modes,len(root_product_modes)*size(root_product_modes),MPI_CHARACTER,0,comm,ierr)
    call check_mpi(ierr,comm)
    call MPI_Bcast(root_steps,size(root_steps),int_type,0,comm,ierr)
    call check_mpi(ierr,comm)
    call MPI_Bcast(root_times_product,size(root_times_product),real_type,0,comm,ierr)
    call check_mpi(ierr,comm)
    same=all(flags.eqv.root_flags).and.all(counts==root_counts).and. &
      all(times==root_times).and.all(paths==root_paths).and.candidate%schedule_mode==root_mode
    same=same.and.all(candidate%product_ids==root_ids).and.all(candidate%product_modes==root_product_modes).and. &
      all(candidate%product_steps==root_steps).and.all(candidate%product_times==root_times_product)
    call MPI_Allreduce(same,all_same,1,MPI_LOGICAL,MPI_LAND,comm,ierr)
    call check_mpi(ierr,comm)
    if(.not.all_same) then
      message='in-situ configuration values differ between MPI ranks'
      return
    endif
    call agree_adaptive_bindings(candidate%product_adaptive,comm,ok)
    if(.not.ok) then
      message='adaptive in-situ bindings differ between MPI ranks'; return
    endif
    if(candidate%slice_definition=='plane') then
      plane=[candidate%slice_origin,candidate%slice_normal]
      if(rank==0) call canonical_slice_plane(plane(4:6),ok)
      call MPI_Bcast(ok,1,MPI_LOGICAL,0,comm,ierr)
      call check_mpi(ierr,comm)
      if(.not.ok) then
        message='physical plane unit normal is not representable'; return
      endif
      call MPI_Bcast(plane,size(plane),real_type,0,comm,ierr)
      call check_mpi(ierr,comm)
      candidate%slice_origin=plane(1:3); candidate%slice_normal=plane(4:6)
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
