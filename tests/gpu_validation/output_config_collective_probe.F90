program output_config_collective_probe
  use mpi
  use output_config
  use output_config_collective
  implicit none
  type(output_options) :: actual,expected
  character(1024) :: filename,mode,message
  integer :: rank,status
  logical :: ok
  call MPI_Init(status)
  call MPI_Comm_rank(MPI_COMM_WORLD,rank,status)
  call get_command_argument(1,filename)
  call get_command_argument(2,mode)
  actual%format_version=77
  if(mode=='mismatch'.and.rank==1) filename=trim(filename)//'.other'
  call read_output_options_collective(trim(filename),actual,MPI_COMM_WORLD,ok,message)
  if(.not.ok) then
    if(actual%format_version/=77) call MPI_Abort(MPI_COMM_WORLD,2,status)
    print '(A,I0,2A)', 'REJECT unchanged rank ',rank,': ',trim(message)
    call MPI_Finalize(status)
    stop 1
  endif
  ! Compare every component to a separately parsed reference on every rank.
  call read_output_options(trim(filename),expected,ok,message)
  if(.not.ok) call MPI_Abort(MPI_COMM_WORLD,3,status)
  if(actual%format_version/=expected%format_version.or.actual%keep/=expected%keep.or. &
     actual%directory/=expected%directory.or.actual%restore_directory/=expected%restore_directory.or. &
     actual%restart_output/=expected%restart_output.or.actual%buffer_bytes/=expected%buffer_bytes.or. &
     actual%host_budget_bytes/=expected%host_budget_bytes.or.actual%device_budget_bytes/=expected%device_budget_bytes.or. &
     any(actual%i_indices/=expected%i_indices).or.any(actual%j_indices/=expected%j_indices).or. &
     any(actual%k_indices/=expected%k_indices)) call MPI_Abort(MPI_COMM_WORLD,4,status)
  call compare_product(actual%checkpoint,expected%checkpoint)
  call compare_product(actual%volume,expected%volume)
  call compare_product(actual%slices,expected%slices)
  print '(A,I0)', 'PASS collective rank ',rank
  call MPI_Finalize(status)
contains
  subroutine compare_product(a,b)
    type(output_product_options),intent(in) :: a,b
    if(a%mode/=b%mode.or.a%fields/=b%fields.or.a%interval_steps/=b%interval_steps.or. &
       a%interval_time/=b%interval_time) call MPI_Abort(MPI_COMM_WORLD,5,status)
    if(any([a%enabled,a%initial_frame,a%final_frame,a%velocity_gradient,a%vorticity,a%qcriterion].neqv. &
           [b%enabled,b%initial_frame,b%final_frame,b%velocity_gradient,b%vorticity,b%qcriterion])) &
      call MPI_Abort(MPI_COMM_WORLD,6,status)
  end subroutine
end program
