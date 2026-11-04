program run_control_config_probe
  use iso_fortran_env, only: int64
  use astr_precision, only: real_kind,numerical_precision
  use timestep_control, only: timestep_automatic,timestep_fixed,choose_timestep
  use run_control_config, only: run_control_options,read_run_control,same_run_control,run_finished
  implicit none
  type(run_control_options) :: options,before
  character(128),parameter :: statistics_fixture(11)=[character(128) :: &
    '&run_control max_steps=3, stop_time=0.75 /', '&timestep', 'mode="auto"', '/', &
    '&statistics', 'enabled=.true.', 'sample_clock="time"', 'sample_time=0.1', &
    'write_clock="time"', 'write_time=0.3', '/']
  character(2048) :: filename
  character(512) :: message
  real(real_kind) :: dt
  logical :: ok
  integer :: unit

  filename='run_control_config_probe.nml'
  if(command_argument_count()>0) call get_command_argument(1,filename)
  call save_text([character(128) :: '! named fields, not positional lines', &
    '&RUN_CONTROL reload_steps=1, max_steps=12 /', &
    '&TIMESTEP growth_factor=1.1, cfl_target=0.3, mode="AUTO" /', &
    '&repair crashfix=.true. /'])
  call read_run_control(trim(filename),options,ok,message)
  call require(ok,'initial named input: '//trim(message))
  call require(options%max_steps==12_int64.and.options%reload_steps==1_int64,'run fields')
  call require(options%timestep%mode==timestep_automatic,'automatic mode')
  call close_to(options%timestep%cfl_target,0.3_real_kind,'explicit target')
  call require(options%crashfix,'repair switch')
  call require(.not.run_finished(options,11_int64,0.0_real_kind),'step before target')
  call require(run_finished(options,12_int64,0.0_real_kind),'cumulative complete step stop')

  call save_text([character(128) :: '&timestep mode="fixed", fixed_dt=0.02, max_dt=0.001 /', &
    '&run_control max_steps=20 /'])
  call read_run_control(trim(filename),options,ok,message)
  call require(ok,'reordered groups: '//trim(message))
  call require(options%timestep%mode==timestep_fixed,'fixed mode')
  call choose_timestep(options%timestep,[0.0_real_kind,0.0_real_kind,0.0_real_kind], &
    0.0_real_kind,.true.,dt,ok,message)
  call require(ok,'parsed fixed selection')
  call close_to(dt,0.02_real_kind,'fixed_dt greater than max_dt')

  call save_text([character(128) :: '&run_control max_steps=13 /'])
  call read_run_control(trim(filename),options,ok,message)
  call require(ok,'full snapshot defaults: '//trim(message))
  call require(options%max_steps==13_int64,'new stop target')
  call require(options%timestep%mode==timestep_automatic,'omitted mode resets to automatic')
  call close_to(options%timestep%cfl_target,0.5_real_kind,'omitted target resets to default')
  call require(.not.options%crashfix,'omitted switch does not retain old value')
  before=options
  call save_text([character(128) :: '&run_control max_steps=999, reload_steps=0 /'])
  call read_run_control(trim(filename),options,ok,message)
  call require(.not.ok.and.len_trim(message)>0,'invalid reload interval')
  call require(same_run_control(options,before),'invalid candidate must not partially publish')

  call save_text([character(128) :: '&run_control max_steps=999 /', '&timestep cfl_target=0 /'])
  call read_run_control(trim(filename),options,ok,message)
  call require(.not.ok,'invalid automatic target')
  call require(options%max_steps==13_int64.and.same_run_control(options,before),'atomic target rejection')
  call save_text([character(128) :: '&run_control max_steps=999 /', '&timestep', &
    'mode="fixed", fixed_dt=0.01'])
  call read_run_control(trim(filename),options,ok,message)
  call require(.not.ok,'incomplete namelist must not be treated as an absent group')
  call require(same_run_control(options,before),'incomplete file retains valid configuration')
  call read_run_control(trim(filename)//'.missing',options,ok,message)
  call require(.not.ok.and.same_run_control(options,before),'missing file retains valid configuration')

  call save_text([character(128) :: '&run_control /'])
  call read_run_control(trim(filename),options,ok,message)
  call require(.not.ok.and.same_run_control(options,before),'missing both stop conditions')
  call save_text([character(128) :: '&run_control max_steps=2 /', '&run_control max_steps=3 /'])
  call read_run_control(trim(filename),options,ok,message)
  call require(.not.ok.and.same_run_control(options,before),'duplicate group has no implicit precedence')
  call save_text([character(128) :: '&run_control max_steps=2 /', '&timstep cfl_target=0.1 /'])
  call read_run_control(trim(filename),options,ok,message)
  call require(.not.ok.and.same_run_control(options,before),'unknown group cannot silently select defaults')

  call save_text(statistics_fixture)
  call read_run_control(trim(filename),options,ok,message)
  call require(ok,'LF and quoted character values: '//trim(message))
  call save_text(statistics_fixture,.true.)
  call read_run_control(trim(filename),options,ok,message)
  call require(ok,'CRLF and statistics input: '//trim(message))
  call require(options%statistics%enabled,'statistics switch')
  call require(trim(options%statistics%sample_clock)=='time','physical sampling clock')
  call close_to(options%statistics%sample_time,0.1_real_kind,'sampling interval')
  call close_to(options%statistics%write_time,0.3_real_kind,'independent writing interval')
  call require(.not.run_finished(options,2_int64,0.5_real_kind),'neither stop target reached')
  call require(run_finished(options,3_int64,0.5_real_kind),'step target reached first')
  call require(run_finished(options,1_int64,0.76_real_kind),'time target crossed without clipping')
  before=options
  call save_text([character(128) :: '&run_control max_steps=3 /', &
    '&statistics enabled=.true., sample_clock="time", sample_time=0 /'])
  call read_run_control(trim(filename),options,ok,message)
  call require(.not.ok.and.same_run_control(options,before),'invalid active statistics interval')

  call save_text([character(128) :: '&run_control max_steps=0 /'])
  call read_run_control(trim(filename),options,ok,message)
  call require(ok,'zero-step target is valid')
  call require(run_finished(options,0_int64,0.0_real_kind),'zero-step run must not advance once')
  open(newunit=unit,file=trim(filename),status='old')
  close(unit,status='delete')
  print '(A,A)', 'RUN_CONTROL_CONFIG_PASS ',numerical_precision
contains
  subroutine save_text(lines,crlf)
    character(*),intent(in) :: lines(:)
    logical,optional,intent(in) :: crlf
    logical :: windows
    integer :: n,fh
    windows=.false.
    if(present(crlf)) windows=crlf
    open(newunit=fh,file=trim(filename),status='replace',action='write')
    do n=1,size(lines)
      if(windows) then
        write(fh,'(A)') trim(lines(n))//achar(13)
      else
        write(fh,'(A)') trim(lines(n))
      endif
    enddo
    close(fh)
  end subroutine

  subroutine require(condition,reason)
    logical,intent(in) :: condition
    character(*),intent(in) :: reason
    if(.not.condition) then
      print '(A)', trim(reason)
      error stop 'run control config probe failed'
    endif
  end subroutine

  subroutine close_to(actual,expected,reason)
    real(real_kind),intent(in) :: actual,expected
    character(*),intent(in) :: reason
    call require(abs(actual-expected)<=32.0_real_kind*epsilon(actual)* &
      max(1.0_real_kind,abs(expected)),reason)
  end subroutine
end program run_control_config_probe
