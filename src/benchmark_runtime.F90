module benchmark_runtime
  use mpi
  implicit none
  private
  public :: configure_benchmark_runtime,benchmark_field_io_disabled, &
            benchmark_cpu_rk_timing_enabled,begin_complete_step_timing,end_complete_step_timing
  logical,save :: configured=.false.,field_io_disabled=.false.
  logical,save :: cpu_rk_timing_enabled_flag=.false.
  logical,save :: complete_step_timing_enabled=.false.
  real(8),save :: complete_step_start=0.d0
  integer,save :: complete_step_rank=-1
contains
  integer function switch_choice(name)
    character(*),intent(in) :: name
    character(32) :: value
    integer :: length,status
    value=''
    call get_environment_variable(name,value,length=length,status=status)
    switch_choice=0
    if(status==1 .or. length==0) return
    if(status/=0) then
      switch_choice=-1
      return
    endif
    select case(trim(adjustl(value)))
    case('1','t','T','true','TRUE','on','ON')
      switch_choice=1
    case('0','f','F','false','FALSE','off','OFF')
      switch_choice=0
    case default
      switch_choice=-1
    end select
  end function switch_choice

  subroutine configure_benchmark_runtime(use_gpu,flowtype,ndims,lihomo,ljhomo,lkhomo,periodic_boundary_case)
    logical,intent(in) :: use_gpu,lihomo,ljhomo,lkhomo,periodic_boundary_case
    character(*),intent(in) :: flowtype
    integer,intent(in) :: ndims
    integer :: requested,rk_timing,cpu_rk_timing,lowest,highest,complete_timing
    integer :: rk_lowest,rk_highest,cpu_rk_lowest,cpu_rk_highest
    integer :: ierr,rank,ignored
    logical :: supported_flow
    if(configured) return
    complete_timing=switch_choice('ASTR_COMPLETE_STEP_TIMING')
    call mpi_allreduce(complete_timing,lowest,1,MPI_INTEGER,MPI_MIN,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call mpi_abort(MPI_COMM_WORLD,ierr,ignored)
    call mpi_allreduce(complete_timing,highest,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call mpi_abort(MPI_COMM_WORLD,ierr,ignored)
    if(lowest<0 .or. lowest/=highest) then
      print *, 'Invalid or inconsistent ASTR_COMPLETE_STEP_TIMING environment'
      call mpi_abort(MPI_COMM_WORLD,1,ignored)
    endif
    complete_step_timing_enabled=complete_timing==1
    if(complete_step_timing_enabled) then
      call mpi_comm_rank(MPI_COMM_WORLD,complete_step_rank,ierr)
      if(ierr/=MPI_SUCCESS) call mpi_abort(MPI_COMM_WORLD,ierr,ignored)
    endif
    requested=switch_choice('ASTR_GPU_BENCHMARK_NO_FIELD_IO')
    rk_timing=switch_choice('ASTR_GPU_RK_TIMING')
    cpu_rk_timing=switch_choice('ASTR_CPU_RK_TIMING')
    call mpi_allreduce(requested,lowest,1,MPI_INTEGER,MPI_MIN,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call mpi_abort(MPI_COMM_WORLD,ierr,ignored)
    call mpi_allreduce(requested,highest,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call mpi_abort(MPI_COMM_WORLD,ierr,ignored)
    call mpi_allreduce(rk_timing,rk_lowest,1,MPI_INTEGER,MPI_MIN,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call mpi_abort(MPI_COMM_WORLD,ierr,ignored)
    call mpi_allreduce(rk_timing,rk_highest,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call mpi_abort(MPI_COMM_WORLD,ierr,ignored)
    call mpi_allreduce(cpu_rk_timing,cpu_rk_lowest,1,MPI_INTEGER,MPI_MIN,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call mpi_abort(MPI_COMM_WORLD,ierr,ignored)
    call mpi_allreduce(cpu_rk_timing,cpu_rk_highest,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call mpi_abort(MPI_COMM_WORLD,ierr,ignored)
    if(lowest<0 .or. lowest/=highest) then
      print *, 'Invalid or inconsistent GPU benchmark environment'
      call mpi_abort(MPI_COMM_WORLD,1,ignored)
    endif
    if(rk_lowest<0 .or. rk_lowest/=rk_highest) then
      print *, 'Invalid or inconsistent ASTR_GPU_RK_TIMING environment'
      call mpi_abort(MPI_COMM_WORLD,1,ignored)
    endif
    if(cpu_rk_lowest<0 .or. cpu_rk_lowest/=cpu_rk_highest) then
      print *, 'Invalid or inconsistent ASTR_CPU_RK_TIMING environment'
      call mpi_abort(MPI_COMM_WORLD,1,ignored)
    endif
    field_io_disabled=requested==1
    if(cpu_rk_timing==1 .and. (use_gpu .or. .not.field_io_disabled)) then
      print *, 'ASTR_CPU_RK_TIMING requires a CPU no-field-I/O benchmark run'
      call mpi_abort(MPI_COMM_WORLD,1,ignored)
    endif
    cpu_rk_timing_enabled_flag=cpu_rk_timing==1
    supported_flow=trim(flowtype)=='tgv' .or. trim(flowtype)=='shuosher'
    if(field_io_disabled .and. (ndims/=3 .or. .not.supported_flow .or. &
       .not.(lihomo.and.ljhomo.and.lkhomo) .or. &
       .not.periodic_boundary_case .or. &
       (use_gpu.and.rk_timing/=1))) then
      print *, 'ASTR_GPU_BENCHMARK_NO_FIELD_IO requires periodic 3-D TGV/Shu-Osher; GPU runs require RK timing'
      call mpi_abort(MPI_COMM_WORLD,1,ignored)
    endif
    configured=.true.
    if(field_io_disabled) then
      call mpi_comm_rank(MPI_COMM_WORLD,rank,ierr)
      if(ierr/=MPI_SUCCESS) call mpi_abort(MPI_COMM_WORLD,ierr,ignored)
      if(rank==0) write(*,'(A)') 'ASTR_GPU_BENCHMARK_NO_FIELD_IO enabled'
    endif
  end subroutine configure_benchmark_runtime

  logical function benchmark_field_io_disabled()
    benchmark_field_io_disabled=configured.and.field_io_disabled
  end function benchmark_field_io_disabled

  logical function benchmark_cpu_rk_timing_enabled()
    benchmark_cpu_rk_timing_enabled=configured.and.cpu_rk_timing_enabled_flag
  end function benchmark_cpu_rk_timing_enabled

  subroutine begin_complete_step_timing()
    if(complete_step_timing_enabled) complete_step_start=MPI_Wtime()
  end subroutine begin_complete_step_timing

  subroutine end_complete_step_timing(step)
    integer,intent(in) :: step
    real(8) :: elapsed
    if(.not.complete_step_timing_enabled) return
    elapsed=MPI_Wtime()-complete_step_start
    write(*,'(A,1X,I0,1X,I0,1X,ES24.16E3)') &
      'ASTR_COMPLETE_STEP_TIMING',step,complete_step_rank,elapsed
  end subroutine end_complete_step_timing
end module benchmark_runtime
