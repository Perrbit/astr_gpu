"""Host-only checks of the read-only G2 paper-evidence extraction."""
import importlib.util
from pathlib import Path
import statistics

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/gpu_validation/plot_air5_stage_coordination.py'
SPEC = importlib.util.spec_from_file_location('air5_report', SCRIPT)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def test_slow_sample_is_kept_in_range_and_dispersion():
    values = [2.45, 2.46, 2.77, 2.44, 2.45]
    case = dict(rounds=[dict(seconds_per_step=x) for x in values],
        median_seconds=statistics.median(values), min_seconds=min(values),
        max_seconds=max(values), stdev_seconds=statistics.stdev(values))
    result = report.timing_stats(case, 5)
    assert result['median_seconds'] == 2.45
    assert result['max_seconds'] == 2.77
    case['max_seconds'] = 2.46
    with pytest.raises(ValueError, match='all samples'):
        report.timing_stats(case, 5)


@pytest.mark.parametrize('field,value', [('phase_timing', True), ('profiled', True), ('status', 'failed')])
def test_attribution_or_failed_run_cannot_be_performance(field, value):
    baseline = {key: 1 for key in report.PROTOCOL}
    baseline.update(status='bounded-timing-collected', phase_timing=False, profiled=False)
    actual = dict(baseline, **{field: value})
    with pytest.raises(ValueError, match='unprofiled'):
        report.validate_protocol(actual, baseline, {})


def test_different_filter_does_not_share_an_error_reference():
    baseline = {key: 1 for key in report.PROTOCOL}
    baseline.update(status='bounded-timing-collected', phase_timing=False, profiled=False)
    actual = dict(baseline, air5_filter_mode='legacy')
    with pytest.raises(ValueError, match='air5_filter_mode'):
        report.validate_protocol(actual, baseline, {})


def test_error_envelope_includes_both_compensated_representations():
    error = dict(candidate={'q': {'max_normalized_error': [1e-5, 2e-5]},
                            'q-minus-carry': {'max_normalized_error': [3e-5, 1e-5]}})
    assert report.max_error(error) == 3e-5
