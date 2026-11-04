module run_control_config
  use iso_fortran_env, only: int64,iostat_end
  use astr_precision, only: real_kind,real_to_word
  use timestep_control, only: timestep_options,timestep_automatic,timestep_fixed, &
    validate_timestep_options
  use ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: statistics_options,run_control_options
  public :: read_run_control,validate_run_control,same_run_control,run_finished
  public :: read_run_control_collective
  public :: pack_control,unpack_control

  type :: statistics_options
    logical :: enabled=.false.
    character(32) :: sample_clock='steps',write_clock='steps'
    integer(int64) :: sample_steps=1,write_steps=0
    real(real_kind) :: sample_time=0.0_real_kind,write_time=0.0_real_kind
  end type

  type :: run_control_options
    integer(int64) :: max_steps=-1,reload_steps=100,list_steps=100
    real(real_kind) :: stop_time=-1.0_real_kind
    logical :: crashfix=.false.
    type(timestep_options) :: timestep
    type(statistics_options) :: statistics
  end type
contains
  pure function lowercase(value) result(result)
    character(*),intent(in) :: value
    character(len(value)) :: result
    integer :: i,c
    result=value
    do i=1,len(value)
      c=iachar(value(i:i))
      if(c>=iachar('A').and.c<=iachar('Z')) result(i:i)=achar(c+32)
    enddo
  end function lowercase

  subroutine read_run_control(filename,options,ok,message)
    character(*),intent(in) :: filename
    type(run_control_options),intent(inout) :: options
    logical,intent(out) :: ok
    character(*),intent(out) :: message
    type(run_control_options) :: candidate
    integer(int64) :: max_steps,reload_steps,list_steps,sample_steps,write_steps
    real(real_kind) :: stop_time,cfl_target,growth_factor,max_dt,initial_dt,fixed_dt
    real(real_kind) :: sample_time,write_time
    logical :: crashfix,enabled,seen(5),parsed
    character(32) :: mode,sample_clock,write_clock,group
    character(4096) :: line
    character(512) :: detail
    integer :: unit,status,i,n,group_index
    namelist /run_control/ max_steps,stop_time,reload_steps
    namelist /timestep/ mode,cfl_target,growth_factor,max_dt,initial_dt,fixed_dt
    namelist /logging/ list_steps
    namelist /repair/ crashfix
    namelist /statistics/ enabled,sample_clock,sample_steps,sample_time, &
      write_clock,write_steps,write_time

    ok=.false.
    message=''
    candidate=run_control_options()
    max_steps=candidate%max_steps;stop_time=candidate%stop_time
    reload_steps=candidate%reload_steps;list_steps=candidate%list_steps
    crashfix=candidate%crashfix
    mode='auto'
    cfl_target=candidate%timestep%cfl_target;growth_factor=candidate%timestep%growth_factor
    max_dt=candidate%timestep%max_dt;initial_dt=candidate%timestep%initial_dt
    fixed_dt=candidate%timestep%fixed_dt
    enabled=candidate%statistics%enabled
    sample_clock=candidate%statistics%sample_clock;sample_steps=candidate%statistics%sample_steps
    sample_time=candidate%statistics%sample_time
    write_clock=candidate%statistics%write_clock;write_steps=candidate%statistics%write_steps
    write_time=candidate%statistics%write_time

    open(newunit=unit,file=filename,status='old',action='read',iostat=status,iomsg=message)
    if(status/=0) return
    seen=.false.;parsed=.true.
    do
      read(unit,'(A)',iostat=status,iomsg=detail) line
      if(status==iostat_end) exit
      if(status/=0) then
        message=trim(detail);parsed=.false.;exit
      endif
      do i=1,len_trim(line)
        if(line(i:i)==achar(13).or.line(i:i)==achar(9)) line(i:i)=' '
      enddo
      line=adjustl(line)
      if(len_trim(line)==0) cycle
      if(line(1:1)=='!') cycle
      if(line(1:1)/='&') then
        message='expected a named runtime-control group';parsed=.false.;exit
      endif
      line=lowercase(line)
      n=scan(line(2:),' ,/!')
      if(n==0) n=len_trim(line)
      group=line(2:n)
      select case(trim(group))
      case('run_control');group_index=1
      case('timestep');group_index=2
      case('logging');group_index=3
      case('repair');group_index=4
      case('statistics');group_index=5
      case default
        message='unknown runtime-control group: '//trim(group);parsed=.false.;exit
      end select
      if(seen(group_index)) then
        message='duplicate runtime-control group: '//trim(group);parsed=.false.;exit
      endif
      seen(group_index)=.true.
      backspace(unit,iostat=status,iomsg=detail)
      if(status/=0) then
        message=trim(detail);parsed=.false.;exit
      endif
      select case(group_index)
      case(1);read(unit,nml=run_control,iostat=status,iomsg=detail)
      case(2);read(unit,nml=timestep,iostat=status,iomsg=detail)
      case(3);read(unit,nml=logging,iostat=status,iomsg=detail)
      case(4);read(unit,nml=repair,iostat=status,iomsg=detail)
      case(5);read(unit,nml=statistics,iostat=status,iomsg=detail)
      end select
      if(status/=0) then
        message='invalid &'//trim(group)//': '//trim(detail);parsed=.false.;exit
      endif
    enddo
    close(unit)
    if(.not.parsed) return

    candidate%max_steps=max_steps;candidate%stop_time=stop_time
    candidate%reload_steps=reload_steps;candidate%list_steps=list_steps
    candidate%crashfix=crashfix
    select case(trim(lowercase(mode)))
    case('auto');candidate%timestep%mode=timestep_automatic
    case('fixed');candidate%timestep%mode=timestep_fixed
    case default
      message='timestep mode must be auto or fixed'
      return
    end select
    candidate%timestep%cfl_target=cfl_target;candidate%timestep%growth_factor=growth_factor
    candidate%timestep%max_dt=max_dt;candidate%timestep%initial_dt=initial_dt
    candidate%timestep%fixed_dt=fixed_dt
    candidate%statistics%enabled=enabled
    candidate%statistics%sample_clock=lowercase(sample_clock)
    candidate%statistics%sample_steps=sample_steps;candidate%statistics%sample_time=sample_time
    candidate%statistics%write_clock=lowercase(write_clock)
    candidate%statistics%write_steps=write_steps;candidate%statistics%write_time=write_time
    call validate_run_control(candidate,ok,message)
    if(ok) options=candidate
  end subroutine read_run_control

  pure subroutine validate_run_control(options,ok,message)
    type(run_control_options),intent(in) :: options
    logical,intent(out) :: ok
    character(*),intent(out) :: message
    ok=.false.
    message=''
    if(options%max_steps< -1_int64) then
      message='max_steps must be -1 or a nonnegative complete-step target'
      return
    endif
    if(.not.ieee_is_finite(options%stop_time)) then
      message='stop_time must be finite'
      return
    endif
    if(options%stop_time<0.0_real_kind.and.options%stop_time/=-1.0_real_kind) then
      message='stop_time must be -1 or a nonnegative physical-time target'
      return
    endif
    if(options%max_steps<0.and.options%stop_time<0.0_real_kind) then
      message='set at least one of max_steps and stop_time'
      return
    endif
    if(options%reload_steps<=0) then
      message='reload_steps must be positive'
      return
    endif
    if(options%list_steps<0) then
      message='list_steps must be nonnegative (zero disables ordinary reports)'
      return
    endif
    call validate_timestep_options(options%timestep,ok,message)
    if(.not.ok) return
    ok=.false.
    if(options%statistics%enabled) then
      select case(trim(options%statistics%sample_clock))
      case('steps')
        if(options%statistics%sample_steps<=0) then
          message='sample_steps must be positive'
          return
        endif
      case('time')
        if(.not.ieee_is_finite(options%statistics%sample_time)) then
          message='sample_time must be finite'
          return
        endif
        if(options%statistics%sample_time<=0.0_real_kind) then
          message='sample_time must be positive'
          return
        endif
      case default
        message='sample_clock must be steps or time'
        return
      end select
      select case(trim(options%statistics%write_clock))
      case('steps')
        if(options%statistics%write_steps<0) then
          message='write_steps must be nonnegative'
          return
        endif
      case('time')
        if(.not.ieee_is_finite(options%statistics%write_time)) then
          message='write_time must be finite'
          return
        endif
        if(options%statistics%write_time<0.0_real_kind) then
          message='write_time must be nonnegative'
          return
        endif
      case default
        message='write_clock must be steps or time'
        return
      end select
    endif
    ok=.true.
  end subroutine validate_run_control

  pure logical function run_finished(options,step,time) result(finished)
    type(run_control_options),intent(in) :: options
    integer(int64),intent(in) :: step
    real(real_kind),intent(in) :: time
    finished=.false.
    if(options%max_steps>=0) finished=step>=options%max_steps
    if(options%stop_time>=0.0_real_kind) finished=finished.or.time>=options%stop_time
  end function run_finished

  pure subroutine pack_control(options,integers,reals,clocks)
    type(run_control_options),intent(in) :: options
    integer(int64),intent(out) :: integers(8)
    real(real_kind),intent(out) :: reals(8)
    character(32),intent(out) :: clocks(2)
    integers=[options%max_steps,options%reload_steps,options%list_steps,int(options%timestep%mode,int64), &
      int(merge(1,0,options%crashfix),int64),int(merge(1,0,options%statistics%enabled),int64), &
      options%statistics%sample_steps,options%statistics%write_steps]
    reals=[options%stop_time,options%timestep%cfl_target,options%timestep%growth_factor, &
      options%timestep%max_dt,options%timestep%initial_dt,options%timestep%fixed_dt, &
      options%statistics%sample_time,options%statistics%write_time]
    clocks=[options%statistics%sample_clock,options%statistics%write_clock]
  end subroutine pack_control

  pure subroutine unpack_control(integers,reals,clocks,options)
    integer(int64),intent(in) :: integers(8)
    real(real_kind),intent(in) :: reals(8)
    character(32),intent(in) :: clocks(2)
    type(run_control_options),intent(out) :: options
    options%max_steps=integers(1);options%reload_steps=integers(2);options%list_steps=integers(3)
    options%timestep%mode=int(integers(4))
    options%crashfix=integers(5)/=0;options%statistics%enabled=integers(6)/=0
    options%statistics%sample_steps=integers(7);options%statistics%write_steps=integers(8)
    options%stop_time=reals(1);options%timestep%cfl_target=reals(2)
    options%timestep%growth_factor=reals(3);options%timestep%max_dt=reals(4)
    options%timestep%initial_dt=reals(5);options%timestep%fixed_dt=reals(6)
    options%statistics%sample_time=reals(7);options%statistics%write_time=reals(8)
    options%statistics%sample_clock=clocks(1);options%statistics%write_clock=clocks(2)
  end subroutine unpack_control

  pure logical function same_run_control(a,b) result(same)
    type(run_control_options),intent(in) :: a,b
    integer(int64) :: integers_a(8),integers_b(8)
    real(real_kind) :: reals_a(8),reals_b(8)
    character(32) :: clocks_a(2),clocks_b(2)
    call pack_control(a,integers_a,reals_a,clocks_a)
    call pack_control(b,integers_b,reals_b,clocks_b)
    same=all(integers_a==integers_b).and.all(clocks_a==clocks_b).and. &
      all(real_to_word(reals_a)==real_to_word(reals_b))
  end function same_run_control

  subroutine read_run_control_collective(filename,communicator,options,ok,message)
    use mpi
    use astr_precision_mpi, only: mpi_real_type
    character(*),intent(in) :: filename
    integer,intent(in) :: communicator
    type(run_control_options),intent(inout) :: options
    logical,intent(out) :: ok
    character(*),intent(out) :: message
    type(run_control_options) :: candidate
    integer(int64) :: integers(8)
    real(real_kind) :: reals(8)
    character(32) :: clocks(2)
    character(512) :: detail
    integer :: rank,status,accepted
    call MPI_Comm_rank(communicator,rank,status)
    call require_mpi(status,communicator)
    detail='';accepted=0
    if(rank==0) then
      call read_run_control(filename,candidate,ok,detail)
      accepted=merge(1,0,ok)
    endif
    call MPI_Bcast(accepted,1,MPI_INTEGER,0,communicator,status)
    call require_mpi(status,communicator)
    call MPI_Bcast(detail,len(detail),MPI_CHARACTER,0,communicator,status)
    call require_mpi(status,communicator)
    ok=accepted/=0
    message=detail
    if(.not.ok) return
    ! Serialize named fields, never derived-type padding or process-local memory.
    if(rank==0) call pack_control(candidate,integers,reals,clocks)
    call MPI_Bcast(integers,size(integers),MPI_INTEGER8,0,communicator,status)
    call require_mpi(status,communicator)
    call MPI_Bcast(reals,size(reals),mpi_real_type,0,communicator,status)
    call require_mpi(status,communicator)
    call MPI_Bcast(clocks,len(clocks(1))*size(clocks),MPI_CHARACTER,0,communicator,status)
    call require_mpi(status,communicator)
    call unpack_control(integers,reals,clocks,candidate)
    options=candidate
  end subroutine read_run_control_collective

  subroutine require_mpi(status,communicator)
    use mpi, only: MPI_SUCCESS,MPI_Abort
    integer,intent(in) :: status,communicator
    integer :: abort_status
    if(status/=MPI_SUCCESS) call MPI_Abort(communicator,status,abort_status)
  end subroutine require_mpi
end module run_control_config
