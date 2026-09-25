import pytest

from run_air5_performance_diagnosis import step_samples
from summarize_air5_nsys import merged, duration


def test_slowest_rank_and_restart_step():
    text = '\n'.join(['ASTR_COMPLETE_STEP_TIMING 1091 1 0.4',
                      'ASTR_COMPLETE_STEP_TIMING 1090 0 0.3',
                      'ASTR_COMPLETE_STEP_TIMING 1091 0 0.2',
                      'ASTR_COMPLETE_STEP_TIMING 1090 1 0.1'])
    assert step_samples(text, 2, 1090, 2) == [.3, .4]


@pytest.mark.parametrize('text', ['', 'ASTR_COMPLETE_STEP_TIMING 1090 0 nan',
    'ASTR_COMPLETE_STEP_TIMING 1090 0 -1', 'ASTR_COMPLETE_STEP_TIMING 1091 0 1',
    'ASTR_COMPLETE_STEP_TIMING 1090 1 1', 'ASTR_COMPLETE_STEP_TIMING 1090 0',
    'ASTR_COMPLETE_STEP_TIMING 1090 0 1\nASTR_COMPLETE_STEP_TIMING 1090 0 1'])
def test_invalid_or_missing_records_rejected(text):
    with pytest.raises(ValueError):
        step_samples(text, 1, 1090, 1)


def test_trace_union_does_not_double_count_compute_and_wait():
    busy = [(10, 20), (15, 30), (40, 50)]
    mpi = [(25, 45)]
    assert merged(busy) == [[10, 30], [40, 50]]
    assert duration(busy) == 30
    assert duration(busy+mpi)-duration(busy) == 10
    assert duration([]) == 0
