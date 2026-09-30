import math

import pytest

from run_complete_step_clock import check_clock


def test_used_steps_define_clock():
    records = []
    for step, (time, dt) in enumerate(((0.001, 0.001), (0.003, 0.002), (0.0035, 0.0005))):
        records.append(check_clock(records, step, time, dt))
    assert records[-1]['expected_time'] == 0.0035


def test_controller_reload_bug_is_rejected():
    records = [check_clock([], 0, 0.001, 0.001)]
    records.append(check_clock(records, 1, 0.002, 0.001))
    with pytest.raises(ValueError, match='sum\\(dt_used\\)'):
        check_clock(records, 2, 0.005, 0.002)


@pytest.mark.parametrize('step,time,dt', [
    (1, 0.001, 0.001), (0, math.nan, 0.001), (0, math.inf, 0.001),
    (0, 0.0, 0.0), (0, 0.001, -0.001), (0, 0.001, math.nan),
])
def test_invalid_or_missing_record_rejected(step, time, dt):
    with pytest.raises(ValueError):
        check_clock([], step, time, dt)
