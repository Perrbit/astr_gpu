"""Timing report controls do not change solver or integration settings."""
import pytest

from run_insitu_curve_scale_performance import (BACKENDS, METRICS, backend_key,
                                              group_bytes, production_configuration, round_order, summarize)


def records():
    return [dict(backend=backend_key(*backend), round=round_id,
                 timing={metric: dict(seconds=10.+index+round_id*.1) for metric in METRICS},
                 step4_output_seconds=2.+index+round_id*.1)
            for index, backend in enumerate(BACKENDS) for round_id in range(-1, 5)]


def test_order_visits_each_backend_once():
    for round_id in range(-1, 5):
        assert len(round_order(round_id)) == 4 and set(round_order(round_id)) == set(BACKENDS)
    for position in range(4):
        assert {round_order(round_id)[position] for round_id in range(4)} == set(BACKENDS)
    pairs = [(first, second) for round_id in range(4)
             for first, second in zip(round_order(round_id), round_order(round_id)[1:])]
    assert len(set(pairs)) == 12


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


def test_disk_budget_counts_receipts_once(tmp_path):
    case = tmp_path/'curve_case'
    case.mkdir()
    (case/'data.bin').write_bytes(b'abcd')
    (tmp_path/'curve_receipt.xml').write_bytes(b'abc')
    (tmp_path/'curve_current').symlink_to(case, target_is_directory=True)
    (tmp_path/'unrelated.bin').write_bytes(b'0123456789')
    assert group_bytes(tmp_path) == 7
