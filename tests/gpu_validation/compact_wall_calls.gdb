set pagination off
set confirm off
set $wall_calls = 0
break wall_blowing_random_sample_legacy_wall_velocity_
commands
  silent
  set $wall_calls = $wall_calls + 1
  continue
end
run
printf "CP6_WALL_CALLS=%d\n", $wall_calls
