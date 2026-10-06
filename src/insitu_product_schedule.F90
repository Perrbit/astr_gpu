module insitu_product_schedule
  use iso_fortran_env, only: int64,real64
  use iso_c_binding, only: c_int,c_char,c_null_char,c_int64_t,c_double
  use ieee_arithmetic, only: ieee_is_finite
  use insitu_run_config, only: insitu_options,insitu_max_products
  use insitu_schedule, only: sample_schedule,configure_schedule,poll_schedule,write_schedule_state, &
    restore_schedule_state,resume_schedule,schedule_next_target
  use adaptive_output, only: adaptive_clock,adaptive_binding_equal,adaptive_dependencies_reset, &
    configure_adaptive_clock,poll_adaptive_clock,report_adaptive_clock,adaptive_clock_file,adaptive_shared_config
  implicit none
  private
  public :: configure_product_schedules,poll_product_schedules,product_scene_due,product_due
  public :: product_results_complete,product_state_file,product_schedules_active
  public :: product_schedule_equal
  type :: product_state
    type(sample_schedule) :: clock
    type(adaptive_clock) :: adaptive
    integer(int64) :: origin_step=0
    real(real64) :: origin_time=0.d0
    integer(int64) :: attempt=-1,success=-1,missing=-1
    real(real64) :: attempt_time=-1.d0,success_time=-1.d0,missing_time=-1.d0
    integer :: code=0,phase=0,missing_code=0,missing_phase=0
    logical :: due=.false.,reported=.true.
  end type
  type(insitu_options),save :: configuration
  type(product_state),save :: states(insitu_max_products)
  logical,save :: configured=.false.
contains
  logical function product_schedules_active()
    product_schedules_active=configured.and.configuration%product_count>0
  end function

  subroutine configure_product_schedules(options,origin_step,origin_time,ok)
    type(insitu_options),intent(in) :: options
    integer(int64),intent(in) :: origin_step
    real(real64),intent(in) :: origin_time
    logical,intent(out) :: ok
    integer :: i
    configuration=options; states=product_state(); configured=.false.; ok=.true.
    do i=1,options%product_count
      states(i)%origin_step=origin_step; states(i)%origin_time=origin_time
      call configure_schedule(states(i)%clock,trim(options%product_modes(i)),options%product_steps(i), &
        options%product_times(i),origin_step,origin_time,options%initial_frame,options%final_frame,ok)
      if(.not.ok) return
      if(options%product_adaptive(i)%enabled) then
        call configure_adaptive_clock(states(i)%adaptive,options%product_adaptive(i),options%product_modes(i), &
          options%product_steps(i),options%product_times(i),origin_step,origin_time, &
          options%initial_frame,options%final_frame,ok)
        if(.not.ok) return
      endif
    enddo
    configured=.true.
  end subroutine

  subroutine poll_product_schedules(step,time,final,emit,ok)
    integer(int64),intent(in) :: step
    real(real64),intent(in) :: time
    logical,intent(in) :: final
    logical,intent(out) :: emit,ok
    integer(int64) :: crossed
    integer :: i
    emit=.false.; ok=configured
    if(.not.ok) return
    do i=1,configuration%product_count
      if(.not.states(i)%reported) then
        ok=.false.; return
      endif
      if(configuration%product_adaptive(i)%enabled) then
        call poll_adaptive_clock(states(i)%adaptive,configuration%product_adaptive(i),step,time,final,states(i)%due,ok)
      else
        call poll_schedule(states(i)%clock,step,time,final,states(i)%due,crossed,ok)
      endif
      if(.not.ok) return
      if(states(i)%due) then
        states(i)%attempt=step; states(i)%attempt_time=time
        states(i)%reported=.false.
        emit=.true.
      endif
    enddo
  end subroutine

  logical function product_due(id) result(due)
    character(*),intent(in) :: id
    integer :: i
    due=.not.product_schedules_active()
    if(due) return
    do i=1,configuration%product_count
      if(configuration%product_ids(i)==id) then
        due=states(i)%due; return
      endif
    enddo
  end function

  logical function product_scene_due(scene) result(due)
    character(*),intent(in) :: scene
    due=product_due(trim(scene)//'.image').or.product_due(trim(scene)//'.geometry')
  end function

  integer(c_int) function product_clock_active_c() bind(C,name='astr_insitu_product_clocks_active') result(active)
    active=merge(1_c_int,0_c_int,product_schedules_active())
  end function

  function c_string(value) result(text)
    character(c_char),intent(in) :: value(*)
    character(64) :: text
    integer :: i
    text=''
    do i=1,len(text)
      if(value(i)==c_null_char) exit
      text(i:i)=value(i)
    enddo
  end function

  integer(c_int) function product_due_c(id) bind(C,name='astr_insitu_product_due') result(due)
    character(c_char),intent(in) :: id(*)
    due=merge(1_c_int,0_c_int,product_due(trim(c_string(id))))
  end function

  integer(c_int) function product_scene_due_c(id) bind(C,name='astr_insitu_product_scene_due') result(due)
    character(c_char),intent(in) :: id(*)
    due=merge(1_c_int,0_c_int,product_scene_due(trim(c_string(id))))
  end function

  integer(c_int) function adaptive_info_c(id,integers,times) bind(C,name='astr_insitu_product_adaptive_info') result(active)
    character(c_char),intent(in) :: id(*)
    integer(c_int64_t),intent(out) :: integers(8)
    real(c_double),intent(out) :: times(4)
    character(64) :: name
    integer :: i
    active=0; integers=0; times=0
    if(.not.product_schedules_active()) return
    name=c_string(id)
    do i=1,configuration%product_count
      if(configuration%product_ids(i)/=name.or..not.configuration%product_adaptive(i)%enabled) cycle
      active=1
      integers=[merge(1_int64,0_int64,states(i)%adaptive%dense),int(states(i)%adaptive%reason,int64), &
        states(i)%adaptive%requested_step,states(i)%adaptive%next_step,states(i)%adaptive%success, &
        states(i)%adaptive%attempt,int(states(i)%adaptive%code,int64),int(states(i)%adaptive%phase,int64)]
      times=[states(i)%adaptive%requested_time,states(i)%adaptive%next_time, &
        states(i)%adaptive%success_time,states(i)%adaptive%attempt_time]
      return
    enddo
  end function

  integer(c_int) function product_report_c(id,code,phase) bind(C,name='astr_insitu_product_report') result(status)
    character(c_char),intent(in) :: id(*)
    integer(c_int),value :: code,phase
    character(64) :: name
    integer :: i
    logical :: ok
    status=0
    if(.not.product_schedules_active()) return
    status=1; name=c_string(id)
    do i=1,configuration%product_count
      if(configuration%product_ids(i)/=name) cycle
      if(.not.states(i)%due.or.states(i)%reported.or.code< -1.or.phase<0.or.phase>4) return
      if(code>0.and.index(name,'.geometry')>0) return
      if(code>0.and.code/=5.and.code/=13.and.code/=28) return
      if((code>0).neqv.(phase>0)) return
      if(configuration%product_adaptive(i)%enabled) then
        call report_adaptive_clock(states(i)%adaptive,int(code),int(phase),ok)
        if(.not.ok) return
      endif
      states(i)%reported=.true.; states(i)%code=code; states(i)%phase=phase
      if(code==0) then
        states(i)%success=states(i)%attempt; states(i)%success_time=states(i)%attempt_time
      else if(code>0) then
        states(i)%missing=states(i)%attempt; states(i)%missing_time=states(i)%attempt_time
        states(i)%missing_code=code; states(i)%missing_phase=phase
      endif
      status=0; return
    enddo
  end function

  logical function product_results_complete() result(ok)
    ok=.true.
    if(product_schedules_active()) ok=all(states(1:configuration%product_count)%reported)
  end function

  logical function product_schedule_equal(a,b) result(same)
    type(insitu_options),intent(in) :: a,b
    integer :: i
    same=a%product_count==b%product_count.and.all(a%product_ids==b%product_ids).and. &
      all(a%product_modes==b%product_modes).and.all(a%product_steps==b%product_steps).and. &
      all(a%product_times==b%product_times)
    do i=1,insitu_max_products
      same=same.and.adaptive_binding_equal(a%product_adaptive(i),b%product_adaptive(i))
    enddo
  end function

  subroutine product_state_file(unit,writing,step,time,same,ok,override)
    integer,intent(in) :: unit
    logical,intent(in) :: writing
    integer(int64),intent(in) :: step
    real(real64),intent(in) :: time
    logical,intent(out) :: same,ok
    logical,optional,intent(in) :: override
    type(insitu_options) :: saved
    type(product_state) :: restored(insitu_max_products)
    integer(int64) :: next_step,expected_step,history(7),count_saved,flags(2)
    real(real64) :: next_time,expected_time,times(3)
    character(8) :: magic
    logical :: initial,final,allow,entry_same(insitu_max_products),clock_same,compatible
    integer :: i,j,status
    ok=.false.; same=.false.
    allow=.false.
    if(present(override)) allow=override
    if(writing) then
      if(.not.product_results_complete()) return
      flags=[merge(1_int64,0_int64,configuration%initial_frame),merge(1_int64,0_int64,configuration%final_frame)]
      magic='ASTRPF01'
      if(any(configuration%product_adaptive%enabled)) magic='ASTRPF02'
      write(unit,iostat=status) magic,int(configuration%product_count,int64),flags
      if(status/=0) return
      do i=1,configuration%product_count
        call schedule_next_target(states(i)%clock,next_step,next_time,ok)
        if(.not.ok) return
        write(unit,iostat=status) configuration%product_ids(i),configuration%product_modes(i), &
          configuration%product_steps(i),configuration%product_times(i),states(i)%attempt,states(i)%success, &
          states(i)%missing,int(states(i)%code,int64),int(states(i)%phase,int64), &
          int(states(i)%missing_code,int64),int(states(i)%missing_phase,int64), &
          states(i)%attempt_time,states(i)%success_time,states(i)%missing_time,next_step,next_time
        if(status/=0) return
        write(unit,iostat=status) states(i)%origin_step,states(i)%origin_time
        if(status/=0) return
        call write_schedule_state(unit,states(i)%clock,configuration%product_ids(i),step,time,ok)
        if(.not.ok) return
      enddo
      if(magic=='ASTRPF02') then
        do i=1,configuration%product_count
          write(unit,iostat=status) configuration%product_adaptive(i)%enabled
          if(status/=0) return
          if(.not.configuration%product_adaptive(i)%enabled) cycle
          call adaptive_clock_file(unit,.true.,states(i)%adaptive,configuration%product_adaptive(i), &
            step,time,clock_same,ok)
          if(.not.ok) return
        enddo
      endif
      same=.true.; ok=.true.; return
    endif
    read(unit,iostat=status) magic,count_saved,flags
    if(status/=0.or.(magic/='ASTRPF01'.and.magic/='ASTRPF02').or.count_saved<0.or.count_saved>insitu_max_products) return
    if(any(flags<0).or.any(flags>1)) return
    initial=flags(1)==1; final=flags(2)==1
    saved%product_count=int(count_saved); saved%initial_frame=initial; saved%final_frame=final
    do i=1,saved%product_count
      read(unit,iostat=status) saved%product_ids(i),saved%product_modes(i),saved%product_steps(i), &
        saved%product_times(i),history,times,next_step,next_time
      if(status/=0) return
      if(i>1) then
        if(saved%product_ids(i)<=saved%product_ids(i-1)) return
      endif
      if(any(history(1:3)< -1).or.any(history(1:3)>step).or.history(2)>history(1).or. &
        history(3)>history(1).or.history(4)< -1.or.history(4)>28.or.history(5)<0.or.history(5)>4.or. &
        history(6)<0.or.history(6)>28.or.history(7)<0.or.history(7)>4) return
      if(history(4)>0.and.all(history(4)/=[5_int64,13_int64,28_int64])) return
      if(history(6)>0.and.all(history(6)/=[5_int64,13_int64,28_int64])) return
      if((history(4)>0).neqv.(history(5)>0)) return
      if((history(6)>0).neqv.(history(7)>0)) return
      if((history(3)>=0).neqv.(history(6)>0)) return
      if(history(4)==0.and.history(1)/=history(2)) return
      if(history(4)>0.and.history(1)/=history(3)) return
      if(.not.all(ieee_is_finite(times)).or.any(times< -1.d0).or.any(times>time)) return
      do status=1,3
        if((history(status)==-1).neqv.(times(status)==-1.d0)) return
      enddo
      ! The saved schedule carries its origin and independently validates history.
      call restore_product_clock(unit,restored(i),saved,i,step,time,ok)
      if(.not.ok) return
      call schedule_next_target(restored(i)%clock,expected_step,expected_time,ok)
      if(.not.ok.or.next_step/=expected_step.or.next_time/=expected_time) return
      restored(i)%attempt=history(1); restored(i)%success=history(2); restored(i)%missing=history(3)
      restored(i)%code=int(history(4)); restored(i)%phase=int(history(5))
      restored(i)%missing_code=int(history(6)); restored(i)%missing_phase=int(history(7))
      restored(i)%attempt_time=times(1); restored(i)%success_time=times(2); restored(i)%missing_time=times(3)
      call resume_schedule(restored(i)%clock,ok)
      if(.not.ok) return
    enddo
    entry_same=.true.
    if(magic=='ASTRPF02') then
      do i=1,saved%product_count
        read(unit,iostat=status) saved%product_adaptive(i)%enabled
        if(status/=0) return
        j=findloc(configuration%product_ids,saved%product_ids(i),dim=1)
        if(saved%product_adaptive(i)%enabled) then
          if(j>0) restored(i)%adaptive=states(j)%adaptive
          if(j==0) j=insitu_max_products
          call adaptive_clock_file(unit,.false.,restored(i)%adaptive,configuration%product_adaptive(j), &
            step,time,clock_same,ok)
          if(.not.ok) return
          entry_same(i)=clock_same
          ! Detailed binding identity is validated by ASTRAC01, not the legacy PF header.
          saved%product_adaptive(i)=configuration%product_adaptive(j)
        else if(j>0) then
          entry_same(i)=.not.configuration%product_adaptive(j)%enabled
        endif
      enddo
    endif
    same=product_schedule_equal(configuration,saved).and.(initial.eqv.configuration%initial_frame).and. &
      (final.eqv.configuration%final_frame).and.all(entry_same)
    if(same) then
      do i=1,configuration%product_count
        if(adaptive_dependencies_reset(configuration%product_adaptive(i))) same=.false.
      enddo
    endif
    if(.not.same.and..not.allow) then
      ok=.true.; return
    endif
    do j=1,configuration%product_count
      i=findloc(saved%product_ids,configuration%product_ids(j),dim=1)
      compatible=.false.
      if(i>0) compatible=entry_same(i).and.saved%product_modes(i)==configuration%product_modes(j).and. &
        saved%product_steps(i)==configuration%product_steps(j).and.saved%product_times(i)==configuration%product_times(j).and. &
        (initial.eqv.configuration%initial_frame).and.(final.eqv.configuration%final_frame).and. &
        (saved%product_adaptive(i)%enabled.eqv.configuration%product_adaptive(j)%enabled).and. &
        .not.adaptive_dependencies_reset(configuration%product_adaptive(j))
      ! Fixed-only overrides retain their pre-AP whole-registry reset contract.
      if(.not.same.and..not.adaptive_shared_config%enabled.and.magic=='ASTRPF01') compatible=.false.
      if(compatible) then
        states(j)=restored(i)
      else
        states(j)=product_state(); states(j)%origin_step=step; states(j)%origin_time=time
        if(adaptive_shared_config%enabled) write(*,'(3a)') 'ASTR_AP_RESET product ',trim(configuration%product_ids(j)), &
          ' reason=product_configuration_or_dependency_change'
        call configure_schedule(states(j)%clock,configuration%product_modes(j),configuration%product_steps(j), &
          configuration%product_times(j),step,time,.false.,configuration%final_frame,ok)
        if(.not.ok) return
        if(configuration%product_adaptive(j)%enabled) then
          call configure_adaptive_clock(states(j)%adaptive,configuration%product_adaptive(j),configuration%product_modes(j), &
            configuration%product_steps(j),configuration%product_times(j),step,time,.false.,configuration%final_frame,ok)
          if(.not.ok) return
        endif
      endif
    enddo
    ok=.true.
  end subroutine

  subroutine restore_product_clock(unit,state,saved,i,step,time,ok)
    integer,intent(in) :: unit,i
    type(product_state),intent(out) :: state
    type(insitu_options),intent(in) :: saved
    integer(int64),intent(in) :: step
    real(real64),intent(in) :: time
    logical,intent(out) :: ok
    integer(int64) :: origin_step
    real(real64) :: origin_time
    integer :: status
    ! Origin is stored alongside the normal schedule; never infer it from restart time.
    read(unit,iostat=status) origin_step,origin_time
    ok=status==0
    if(.not.ok) return
    state%origin_step=origin_step; state%origin_time=origin_time
    call configure_schedule(state%clock,trim(saved%product_modes(i)),saved%product_steps(i),saved%product_times(i), &
      origin_step,origin_time,saved%initial_frame,saved%final_frame,ok)
    if(ok) call restore_schedule_state(unit,state%clock,saved%product_ids(i),step,time,ok)
  end subroutine
end module
