"""Long TGV diagnostics use matched sample identity, not accumulated clocks."""

import numpy as np
import pytest

from run_native_precision_tgv_long import clock_at_steps, compare_curves
from run_compact_execution import output_configuration,retained_checkpoints


def test_complete_checkpoints_are_opt_in_and_not_limited_by_old_timing_budget():
    disabled=output_configuration()
    assert 'enabled=t' not in disabled
    enabled=output_configuration(5000)
    assert 'exact_restart=t' in enabled and 'keep=2' in enabled
    assert 'interval_steps=5000' in enabled and 'initial_frame=f' in enabled
    assert 'host_budget_bytes=68719476736' in enabled


def test_retention_checks_only_the_last_two_complete_batches(tmp_path):
    root=tmp_path/'outdat/output'
    (root/'resources').mkdir(parents=True)
    (root/'resources/geometry.h5').touch()
    for step in (35000,40000):
        batch=root/'checkpoints'/f'step{step:012d}'
        batch.mkdir(parents=True)
        for name in ('state.h5','COMPLETE'):
            (batch/name).touch()
    result=retained_checkpoints(tmp_path,40000,5000)
    assert [p.rsplit('/',1)[-1] for p in result]==['step000000035000','step000000040000']


def test_native_time_is_repeated_source_kind_addition():
    steps = [0, 40, 40000]
    for precision, real in (('fp32', np.float32), ('fp64', np.float64)):
        value = real(0)
        expected = [value]
        for step in range(1, steps[-1]+1):
            value = real(value+real(2.5e-4))
            if step in steps:
                expected.append(value)
        np.testing.assert_array_equal(clock_at_steps(steps, 2.5e-4, precision), expected)
    assert clock_at_steps([40000], 2.5e-4, 'fp32')[0] != 10


def test_peak_time_comparison_is_not_clock_drift():
    steps = np.array([40, 80, 120, 160])
    values = np.array([1., 2., 4., 3.])
    a = np.column_stack((steps, steps*.00025, values, values, values))
    b = a.copy()
    b[:, 1] += .001
    b[:, 2:] *= 1.1
    report = compare_curves(a, b)
    for record in report.values():
        assert record['sampled_peak_nominal_time_difference'] == 0
        assert record['sampled_peak_difference'] == pytest.approx(.4)
        assert record['max_difference_over_reference_peak'] == pytest.approx(.1)
        assert record['max_difference_step'] == 120


def test_different_sample_steps_are_not_interpolated_silently():
    a = np.array([[40, .01, 1., 1., 1.], [80, .02, 2., 2., 2.]])
    b = a.copy()
    b[0, 0] = 41
    with pytest.raises(ValueError, match='sample identities'):
        compare_curves(a, b)


def test_clock_alignment_never_clamps_to_a_nonoverlapping_endpoint():
    steps = np.array([40,80,120,160])
    t = steps*.00025
    a = np.column_stack((steps,t,t,t,t))
    shifted = t+.005
    b = np.column_stack((steps,shifted,shifted,shifted,shifted))
    for item in compare_curves(a,b).values():
        assert item['same_native_clock_interpolated_max_abs_difference']==pytest.approx(0,abs=1e-16)
        assert item['same_native_clock_overlap']==[.015,.04]
        assert item['same_native_clock_reference_samples']==3
