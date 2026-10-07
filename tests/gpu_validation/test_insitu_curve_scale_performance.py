"""Timing report controls do not change solver or integration settings."""
import pytest

from run_insitu_curve_scale_performance import (BACKENDS, METRICS, backend_key,
                                              production_configuration, round_order, summarize)


def records():
    return [dict(backend=backend_key(*backend), round=round_id,
                 timing={metric: dict(seconds=10.+index+round_id*.1) for metric in METRICS},
                 step4_output_seconds=2.+index+round_id*.1)
            for index, backend in enumerate(BACKENDS) for round_id in range(-1, 5)]


def test_order_visits_each_backend_once():
    for round_id in range(-1, 5):
        assert len(round_order(round_id)) == 4 and set(round_order(round_id)) == set(BACKENDS)


def test_summary_excludes_warmup_and_selects_complete_window():
    rows = records()
    for row in rows:
        if row['round'] == -1:
            row['timing']['completed_window']['seconds'] = 1000.
    result = summarize(rows, 5)
    assert result['selected'] == backend_key(*BACKENDS[0])
    assert result['backends'][result['selected']]['completed_window']['median'] == 10.2
    assert result['nested_timings_are_not_additive']


def test_summary_rejects_missing_round():
    with pytest.raises(ValueError, match='Missing or repeated'):
        summarize(records()[:-1], 5)


def test_every_step_production_preserves_other_configuration():
    from test_insitu_curve_demo import scale_configuration
    for pipeline, mode in BACKENDS:
        assert production_configuration(pipeline, mode).replace('step_interval=1', 'step_interval=2') == \
            scale_configuration(pipeline, mode)
