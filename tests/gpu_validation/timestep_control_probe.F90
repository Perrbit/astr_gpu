program timestep_control_probe
  use astr_precision, only: real_kind,numerical_precision
  use timestep_control, only: timestep_options,timestep_automatic,timestep_fixed, &
    validate_timestep_options,choose_timestep
  use ieee_arithmetic, only: ieee_is_finite,ieee_value,ieee_quiet_nan
  implicit none
  type(timestep_options) :: options
  real(real_kind) :: rates(3),dt,previous,nan_value
  logical :: ok
  character(160) :: message

  nan_value=ieee_value(0.0_real_kind,ieee_quiet_nan)
  options=timestep_options()
  call validate_timestep_options(options,ok,message)
  call require(ok,'default configuration: '//trim(message))
  rates=[2.0_real_kind,3.0_real_kind,5.0_real_kind]
  call choose_timestep(options,rates,0.0_real_kind,.true.,dt,ok,message)
  call require(ok,'initial choice: '//trim(message))
  call close_to(dt,0.05_real_kind,'initial wave step')

  previous=0.01_real_kind
  call choose_timestep(options,rates,previous,.false.,dt,ok,message)
  call require(ok,'growth choice: '//trim(message))
  call close_to(dt,0.011_real_kind,'growth cap and fixed-to-auto transition')
  rates=[100.0_real_kind,0.0_real_kind,0.0_real_kind]
  call choose_timestep(options,rates,previous,.false.,dt,ok,message)
  call require(ok,'shrink choice: '//trim(message))
  call close_to(dt,0.005_real_kind,'immediate shrinking')

  rates=[2.0_real_kind,3.0_real_kind,5.0_real_kind]
  options%max_dt=0.003_real_kind
  options%initial_dt=0.002_real_kind
  call choose_timestep(options,rates,0.0_real_kind,.true.,dt,ok,message)
  call require(ok,'initial cap: '//trim(message))
  call close_to(dt,0.002_real_kind,'initial and global caps')
  call choose_timestep(options,rates,previous,.false.,dt,ok,message)
  call require(ok,'later cap: '//trim(message))
  call close_to(dt,0.003_real_kind,'initial cap does not constrain later steps')
  options%max_dt=0.0_real_kind
  call choose_timestep(options,rates,0.0_real_kind,.true.,dt,ok,message)
  call require(ok,'non-exact restart choice: '//trim(message))
  call close_to(dt,0.002_real_kind,'non-exact restart uses new first-step cap')
  options%initial_dt=0.0_real_kind
  options%growth_factor=1.0_real_kind
  call choose_timestep(options,rates,previous,.false.,dt,ok,message)
  call require(ok,'unit growth: '//trim(message))
  call close_to(dt,previous,'unit growth prohibits increase')

  options%mode=timestep_fixed
  options%fixed_dt=0.02_real_kind
  options%max_dt=0.001_real_kind
  options%initial_dt=0.0001_real_kind
  rates=nan_value
  call validate_timestep_options(options,ok,message)
  call require(ok,'fixed configuration: '//trim(message))
  call choose_timestep(options,rates,nan_value,.false.,dt,ok,message)
  call require(ok,'fixed choice must not consume wave rates or history')
  call require(dt==options%fixed_dt,'fixed step must not be truncated')
  options%fixed_dt=0.0_real_kind
  call validate_timestep_options(options,ok,message)
  call require(.not.ok,'zero fixed_dt must be rejected in fixed mode')

  options=timestep_options()
  options%cfl_target=0.0_real_kind
  call validate_timestep_options(options,ok,message)
  call require(.not.ok,'zero target must be rejected in automatic mode')
  options=timestep_options()
  options%growth_factor=0.9_real_kind
  call validate_timestep_options(options,ok,message)
  call require(.not.ok,'growth below one must be rejected')
  options=timestep_options()
  options%max_dt=-1.0_real_kind
  call validate_timestep_options(options,ok,message)
  call require(.not.ok,'negative automatic cap must be rejected')
  options=timestep_options()
  options%mode=-1
  call validate_timestep_options(options,ok,message)
  call require(.not.ok,'unknown timestep mode must be rejected')

  options=timestep_options()
  rates=[nan_value,1.0_real_kind,0.0_real_kind]
  call choose_timestep(options,rates,previous,.false.,dt,ok,message)
  call require(.not.ok.and.len_trim(message)>0,'invalid rates must be reported')
  rates=0.0_real_kind
  call choose_timestep(options,rates,previous,.false.,dt,ok,message)
  call require(.not.ok,'zero total rate has no finite automatic choice')
  rates=[-1.0_real_kind,1.0_real_kind,1.0_real_kind]
  call choose_timestep(options,rates,previous,.false.,dt,ok,message)
  call require(.not.ok,'negative spectral rate must be rejected')
  rates=1.0_real_kind
  call choose_timestep(options,rates,0.0_real_kind,.false.,dt,ok,message)
  call require(.not.ok,'non-initial automatic choice needs valid history')

  options%cfl_target=huge(dt)/2.0_real_kind
  options%growth_factor=8.0_real_kind
  rates=[1.0_real_kind,0.0_real_kind,0.0_real_kind]
  previous=huge(dt)/4.0_real_kind
  call choose_timestep(options,rates,previous,.false.,dt,ok,message)
  call require(ok,'irrelevant growth product must not overflow')
  call close_to(dt,options%cfl_target,'finite minimum without overflowing growth product')
  print '(A,A)', 'TIMESTEP_CONTROL_PASS ',numerical_precision
contains
  subroutine require(condition,reason)
    logical,intent(in) :: condition
    character(*),intent(in) :: reason
    if(.not.condition) then
      print '(A)', trim(reason)
      error stop 'timestep control probe failed'
    endif
  end subroutine

  subroutine close_to(actual,expected,reason)
    real(real_kind),intent(in) :: actual,expected
    character(*),intent(in) :: reason
    call require(ieee_is_finite(actual),reason//' is nonfinite')
    call require(abs(actual-expected)<=64.0_real_kind*epsilon(actual)* &
      max(1.0_real_kind,abs(expected)),reason)
  end subroutine
end program timestep_control_probe
