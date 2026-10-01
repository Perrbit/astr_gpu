program schedule_probe
  use iso_fortran_env, only: int64,real64
  use ieee_arithmetic, only: ieee_value,ieee_quiet_nan
  use insitu_schedule
  implicit none
  type(sample_schedule) :: s,restored
  logical :: ok,emit,other_emit
  integer(int64) :: crossed,other_crossed
  integer :: unit
  call configure_schedule(s,'steps',2_int64,0.d0,0_int64,0.d0,.false.,.false.,ok)
  call require(ok)
  call poll(0,0.d0,.false.,.false.,0)
  call poll(1,0.1d0,.false.,.false.,0)
  call poll(2,0.3d0,.false.,.true.,1)
  call poll(2,0.3d0,.false.,.false.,0)
  call poll(7,0.4d0,.true.,.true.,2)
  call poll_schedule(s,8_int64,0.5d0,.false.,emit,crossed,ok)
  call require(.not.ok)
  call configure_schedule(s,'time',0_int64,0.125d0,10_int64,2.d0,.true.,.true.,ok)
  call require(ok)
  call poll(10,2.d0,.false.,.true.,0)
  call poll(11,2.05d0,.false.,.false.,0)
  call poll(12,2.375d0,.false.,.true.,3)
  call poll(13,2.4d0,.true.,.true.,0)
  call configure_schedule(s,'steps',2_int64,0.d0,0_int64,0.d0,.false.,.true.,ok)
  call require(ok)
  call poll(2,0.2d0,.false.,.true.,1)
  call poll(2,0.2d0,.true.,.false.,0)
  call configure_schedule(s,'time',0_int64,0.1d0,0_int64,0.d0,.false.,.false.,ok)
  call require(ok)
  call poll(1,nearest(0.1d0,-1.d0),.false.,.false.,0)
  call poll(2,0.1d0,.false.,.true.,1)
  call poll_schedule(s,1_int64,0.2d0,.false.,emit,crossed,ok)
  call require(.not.ok)
  call poll_schedule(s,3_int64,ieee_value(0.d0,ieee_quiet_nan),.false.,emit,crossed,ok)
  call require(.not.ok)
  call configure_schedule(s,'steps',0_int64,0.d0,0_int64,0.d0,.false.,.false.,ok)
  call require(.not.ok)
  call configure_schedule(s,'time',1_int64,0.1d0,0_int64,0.d0,.false.,.false.,ok)
  call require(.not.ok)
  call configure_schedule(s,'time',0_int64,1.d-20,0_int64,1.d20,.false.,.false.,ok)
  call require(.not.ok)
  call checkpoint_test('steps',2_int64,0.d0)
  call checkpoint_test('time',0_int64,0.125d0)
  print *, 'PASS: schedule modes, crossings, endpoints, deduplication, invalid clocks'
  print *, 'PASS: schedule file continuation, mismatch rejection, final-state preservation'
contains
  subroutine checkpoint_test(mode,interval,period)
    character(*),intent(in) :: mode
    integer(int64),intent(in) :: interval
    real(real64),intent(in) :: period
    call configure_schedule(s,mode,interval,period,0_int64,0.d0,.true.,.true.,ok)
    call require(ok)
    restored=s
    call poll_schedule(s,2_int64,0.25d0,.false.,emit,crossed,ok)
    call require(ok.and.emit)
    open(newunit=unit,status='scratch',access='stream',form='unformatted',convert='little_endian')
    call write_schedule_state(unit,s,'batch-a',2_int64,0.25d0,ok)
    call require(ok)
    rewind(unit)
    call restore_schedule_state(unit,restored,'batch-b',2_int64,0.25d0,ok)
    call require(.not.ok)
    rewind(unit)
    call restore_schedule_state(unit,restored,'batch-a',3_int64,0.25d0,ok)
    call require(.not.ok)
    rewind(unit)
    call restore_schedule_state(unit,restored,'batch-a',2_int64,0.5d0,ok)
    call require(.not.ok)
    rewind(unit)
    call restore_schedule_state(unit,restored,'batch-a',2_int64,0.25d0,ok)
    call require(ok)
    call poll_schedule(restored,2_int64,0.25d0,.false.,emit,crossed,ok)
    call require(ok.and..not.emit.and.crossed==0)
    call poll_schedule(s,5_int64,0.625d0,.false.,emit,crossed,ok)
    call require(ok)
    call poll_schedule(restored,5_int64,0.625d0,.false.,other_emit,other_crossed,ok)
    call require(ok.and.(emit.eqv.other_emit).and.crossed==other_crossed)
    call configure_schedule(restored,mode,interval,period,0_int64,0.d0,.false.,.true.,ok)
    rewind(unit)
    call restore_schedule_state(unit,restored,'batch-a',2_int64,0.25d0,ok)
    call require(.not.ok)
    call poll_schedule(s,5_int64,0.625d0,.true.,emit,crossed,ok)
    call require(ok.and..not.emit)
    rewind(unit)
    call write_schedule_state(unit,s,'batch-final',5_int64,0.625d0,ok)
    call require(ok)
    restored=s
    rewind(unit)
    call restore_schedule_state(unit,restored,'batch-final',5_int64,0.625d0,ok)
    call require(ok)
    call poll_schedule(restored,6_int64,0.75d0,.false.,emit,crossed,ok)
    call require(.not.ok)
    call resume_schedule(restored,ok)
    call require(ok)
    call poll_schedule(restored,5_int64,0.625d0,.false.,emit,crossed,ok)
    call require(ok.and..not.emit)
    call poll_schedule(restored,6_int64,0.75d0,.false.,emit,crossed,ok)
    call require(ok)
    close(unit)
    open(newunit=unit,status='scratch',access='stream',form='unformatted',convert='little_endian')
    write(unit) 'ASTRSC01'
    rewind(unit)
    call restore_schedule_state(unit,restored,'batch-final',5_int64,0.625d0,ok)
    call require(.not.ok)
    close(unit)
  end subroutine
  subroutine require(condition)
    logical,intent(in) :: condition
    if(.not.condition) error stop 'schedule assertion failed'
  end subroutine
  subroutine poll(step,t,final,want,count)
    integer,intent(in) :: step,count
    real(real64),intent(in) :: t
    logical,intent(in) :: final,want
    call poll_schedule(s,int(step,int64),t,final,emit,crossed,ok)
    call require(ok)
    call require(emit.eqv.want)
    call require(crossed==count)
  end subroutine
end program
