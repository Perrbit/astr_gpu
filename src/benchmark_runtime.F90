module benchmark_runtime
  use iso_fortran_env, only: int64
  use mpi
  implicit none
  private
  public :: configure_benchmark_runtime,benchmark_field_io_disabled, &
            benchmark_cpu_rk_timing_enabled,benchmark_gpu_rk_timing_enabled, &
            begin_complete_step_timing,end_complete_step_timing, &
            configure_insitu_timing,insitu_clock,report_insitu_timing,benchmark_buffered_output_enabled,insitu_timing_active
  public :: benchmark_window_due,benchmark_mark_window
  logical,save :: configured=.false.,field_io_disabled=.false.
  logical,save :: cpu_rk_timing_enabled_flag=.false.
  logical,save :: gpu_rk_timing_enabled_flag=.false.
  logical,save :: complete_step_timing_enabled=.false.
  logical,save :: buffered_output=.false.
  real(8),save :: complete_step_start=0.d0
  integer,save :: complete_step_rank=-1
  logical,save :: insitu_timing_enabled=.false.
  integer,save :: insitu_timing_rank=-1
  integer,save :: window_steps(2)=0,window_rank=-1
  real(8),save :: wall_window_start=0.d0
contains
  subroutine configure_wall_window(use_gpu)
    logical,intent(in) :: use_gpu
    integer :: options(2),low(2),high(2),status,ierr,ignored
    character(64) :: value
    options=0
    call get_environment_variable('ASTR_BENCHMARK_WALL_WINDOW',value,status=status)
    if(status==0.and.len_trim(value)>0) then
      read(value,*,iostat=status) options
      if(status/=0.or.options(1)<0.or.options(2)<=0) options=-1
#ifdef ASTR_GPU_CPP
      if(.not.use_gpu) options=-1
#else
      options=-1
#endif
    elseif(status/=0.and.status/=1) then
      options=-1
    endif
    call MPI_Allreduce(options,low,2,MPI_INTEGER,MPI_MIN,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,ierr,ignored)
    call MPI_Allreduce(options,high,2,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,ierr,ignored)
    if(any(low<0).or.any(low/=high)) then
      print *, 'ASTR_BENCHMARK_WALL_WINDOW requires matching warmup,timed steps and C++ GPU execution'
      call MPI_Abort(MPI_COMM_WORLD,1,ignored)
    endif
    window_steps=low
    if(window_steps(2)>0) then
      call MPI_Comm_rank(MPI_COMM_WORLD,window_rank,ierr)
      if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,ierr,ignored)
    endif
  end subroutine

  pure logical function benchmark_window_due(step)
    integer(int64),intent(in) :: step
    benchmark_window_due=window_steps(2)>0.and. &
      (step==window_steps(1).or.step==int(window_steps(1),int64)+window_steps(2))
  end function

  subroutine benchmark_mark_window(step)
    integer(int64),intent(in) :: step
    integer :: ierr,ignored
    real(8) :: now,elapsed
    call MPI_Barrier(MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,ierr,ignored)
    if(window_rank/=0) return
    now=MPI_Wtime()
    if(step==window_steps(1)) then
      wall_window_start=now
    else
      elapsed=now-wall_window_start
      write(*,'(a,2(1x,i0),2(1x,es24.16e3))') 'ASTR_BENCHMARK_WALL_WINDOW', &
        window_steps,elapsed,elapsed/window_steps(2)
    endif
  end subroutine

  subroutine configure_insitu_timing()
    integer :: choice,lowest,highest,ierr,ignored
    choice=switch_choice('ASTR_INSITU_TIMING')
    call MPI_Allreduce(choice,lowest,1,MPI_INTEGER,MPI_MIN,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,ierr,ignored)
    call MPI_Allreduce(choice,highest,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,ierr,ignored)
    if(lowest<0.or.lowest/=highest) then
      print *, 'Invalid or inconsistent ASTR_INSITU_TIMING environment'
      call MPI_Abort(MPI_COMM_WORLD,1,ignored)
    endif
    insitu_timing_enabled=choice==1
    call MPI_Comm_rank(MPI_COMM_WORLD,insitu_timing_rank,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,ierr,ignored)
  end subroutine

  real(8) function insitu_clock()
    insitu_clock=0.d0
    if(insitu_timing_enabled) insitu_clock=MPI_Wtime()
  end function

  logical function insitu_timing_active()
    insitu_timing_active=insitu_timing_enabled
  end function

  subroutine report_insitu_timing(stage,started,step)
    character(*),intent(in) :: stage
    real(8),intent(in) :: started
    integer(int64),intent(in) :: step
    if(.not.insitu_timing_enabled) return
    ! Local durations only: aggregate whole windows offline, without new barriers.
    write(*,'(A,1X,A,1X,I0,1X,I0,1X,ES24.16E3)') &
      'ASTR_INSITU_STAGE_TIMING',stage,step,insitu_timing_rank,MPI_Wtime()-started
  end subroutine

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
    integer :: requested,rk_timing,cpu_rk_timing,lowest,highest,complete_timing,buffer_choice
    integer :: rk_lowest,rk_highest,cpu_rk_lowest,cpu_rk_highest
    integer :: ierr,rank,ignored
    logical :: supported_flow
    if(configured) return
    call configure_wall_window(use_gpu)
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
    buffer_choice=switch_choice('ASTR_BENCHMARK_BUFFER_OUTPUT')
    call MPI_Allreduce(buffer_choice,lowest,1,MPI_INTEGER,MPI_MIN,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,ierr,ignored)
    call MPI_Allreduce(buffer_choice,highest,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,ierr,ignored)
    if(lowest<0.or.lowest/=highest.or.(buffer_choice==1.and.(.not.use_gpu.or.rk_timing/=1))) then
      error stop 'ASTR_BENCHMARK_BUFFER_OUTPUT requires consistent GPU RK timing'
    endif
    buffered_output=buffer_choice==1
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
    gpu_rk_timing_enabled_flag=rk_timing==1
    supported_flow=trim(flowtype)=='bl'.or. &
      ((trim(flowtype)=='tgv'.or.trim(flowtype)=='air5tgv'.or.trim(flowtype)=='shuosher').and. &
       lihomo.and.ljhomo.and.lkhomo.and.periodic_boundary_case)
    if(field_io_disabled .and. (ndims/=3 .or. .not.supported_flow .or. &
       (use_gpu.and.rk_timing/=1.and.window_steps(2)==0))) then
      print *, 'ASTR_GPU_BENCHMARK_NO_FIELD_IO requires 3-D flat plate or periodic TGV/AIR5 TGV/Shu-Osher'
      if(use_gpu.and.rk_timing/=1.and.window_steps(2)==0) &
        print *, 'GPU no-field-I/O benchmarks require an RK or wall window timer'
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

  logical function benchmark_gpu_rk_timing_enabled()
    benchmark_gpu_rk_timing_enabled=configured.and.gpu_rk_timing_enabled_flag
  end function benchmark_gpu_rk_timing_enabled

  logical function benchmark_buffered_output_enabled()
    benchmark_buffered_output_enabled=configured.and.buffered_output
  end function benchmark_buffered_output_enabled

  subroutine begin_complete_step_timing()
    if(complete_step_timing_enabled) complete_step_start=MPI_Wtime()
  end subroutine begin_complete_step_timing

  subroutine end_complete_step_timing(step)
    integer(int64),intent(in) :: step
    real(8) :: elapsed
    if(.not.complete_step_timing_enabled) return
    elapsed=MPI_Wtime()-complete_step_start
    write(*,'(A,1X,I0,1X,I0,1X,ES24.16E3)') &
      'ASTR_COMPLETE_STEP_TIMING',step,complete_step_rank,elapsed
  end subroutine end_complete_step_timing
end module benchmark_runtime
