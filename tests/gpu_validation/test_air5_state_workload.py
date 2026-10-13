import h5py
import numpy as np
import pytest

from run_air5_state_workload import rank_inputs, chemistry_regions, distribution, parse_results, compare_history_work


@pytest.mark.parametrize('components', [34, 35])
def test_exact_shared_copy_and_compensation_are_not_global_owner_values(tmp_path, components):
    path = tmp_path / 'state.h5'
    cells, halo = np.array([2, 2, 2]), 1
    shape = cells + 2 * halo + 1
    z, y, x = np.indices(tuple(shape[::-1]))
    owner = (x >= 1) & (x < 3) & (y >= 1) & (y < 4) & (z >= 1) & (z < 4)
    full = np.arange(components * int(np.prod(shape)), dtype=float).reshape((components, *shape[::-1]))
    full[16, 1:4, 1:4, 3] = 2.37e-322  # An exact subnormal carry, not padding.
    extras = full[:, ~owner]
    with h5py.File(path, 'w') as state:
        state['identity'] = [0, 2, 1, 5, 3, 3, components, 1, 2, 0, 0, 0, 7]
        state['metadata'] = [1, 1]
        state['partitions'] = [0, 0, 0, *cells, halo, extras.size]
        state['rank_extras'] = extras.ravel()
        for component in range(1, components + 1):
            state[f'q{component:04d}'] = np.full((3, 3, 5), component, dtype=float)
    with h5py.File(path) as state:
        values, coordinates, global_cells = rank_inputs(state, 0, include_history=components == 35)
        if components == 34:
            with pytest.raises(ValueError, match='saved history is missing'):
                rank_inputs(state, 0, include_history=True)
    shared = coordinates[:, 0] == 2
    assert np.array_equal(values[shared, 0], full[0, 1:4, 1:4, 3].ravel())
    assert np.all(values[shared, 11] == 2.37e-322)
    assert np.all(values[~shared, 0] == 1)
    if components == 35:
        assert values.shape[1] == 18
        assert np.array_equal(values[shared, -1], full[34, 1:4, 1:4, 3].ravel())
        assert np.all(values[~shared, -1] == 35)
    assert global_cells.tolist() == [4, 2, 2]


def test_work_regions_include_shared_and_periodic_endpoints_but_not_prescribed_faces():
    coordinates = np.array([[0, 1, 0], [1, 0, 0], [2, 1, 2], [2, 2, 2], [4, 1, 0], [4, 2, 2]])
    regions = chemistry_regions(coordinates, np.array([4, 2, 2]), True)
    assert np.flatnonzero(regions['interior']).tolist() == [2]
    assert np.flatnonzero(regions['dynamic_top']).tolist() == [3, 5]
    assert 'dynamic_top' not in chemistry_regions(coordinates, np.array([4, 2, 2]), False)


def test_distribution_reports_counts_and_tail():
    row = distribution([1, 1, 1, 5])
    assert row['total'] == 8 and row['max'] == 5 and row['median'] == 1
    assert row['histogram'] == {'1': 3, '5': 1}


def test_probe_result_failure_does_not_become_valid_distribution(tmp_path):
    path = tmp_path / 'batch.out'
    row = np.array([[1, 0, 1, 0, 2, 1, *([1.] * 13)]])
    np.savetxt(path, row)
    assert parse_results(path, 1)['logical_rhs']['total'] == 2
    row[0, 1] = 202
    np.savetxt(path, row)
    with pytest.raises(ValueError, match='failed or reordered'):
        parse_results(path, 1)


def paired_work():
    initial = np.ones((3, 17))
    initial[0, 11] = 2.37e-322
    history = np.column_stack((initial, [1., .25, .5]))
    old = np.ones((3, 19))
    old[:, 0], old[:, 1], old[:, 2:6] = [1, 2, 3], 0, [1, 0, 2, 1]
    new = old.copy()
    new[1, 2:6] = [3, 0, 6, 3]
    new[2, 2:6] = [2, 1, 6, 3]
    return initial, history, old, new


def test_history_work_attribution_preserves_counts_and_task_identity():
    report = compare_history_work(*paired_work(), 1.)
    assert report['tasks'] == 3 and report['changed_tasks'] == 2
    assert report['reduced_history_tasks'] == 2 and report['changed_full_duration_tasks'] == 0
    assert report['more_attempt_tasks'] == 2 and report['fewer_attempt_tasks'] == 0
    assert report['delta_totals'] == [3, 1, 8, 4]
    assert report['reset_max_attempts'] == 1 and report['history_max_attempts'] == 3
    assert [row['row'] for row in report['changed_rows']] == [2, 3]
    assert report['attempt_simt_proxy']['reset']['lane_attempt_slots'] == 32
    assert report['attempt_simt_proxy']['history']['lane_attempt_slots'] == 96


def test_attempt_proxy_keeps_region_order_and_counts_partial_warp_slots():
    initial = np.ones((65, 17))
    history = np.column_stack((initial, np.ones(65)))
    old = np.ones((65, 19))
    old[:, 0], old[:, 1], old[:, 2:6] = np.arange(1, 66), 0, [1, 0, 2, 1]
    new = old.copy()
    new[32, 2:6] = [5, 0, 10, 5]
    report = compare_history_work(initial, history, old, new, 1.)
    assert report['attempt_simt_proxy']['reset'] == dict(
        warps=3, warp_loop_bound_sum=3, lane_attempt_slots=96, useful_attempts=65)
    assert report['attempt_simt_proxy']['history'] == dict(
        warps=3, warp_loop_bound_sum=7, lane_attempt_slots=224, useful_attempts=69)


@pytest.mark.parametrize('fault, message', [
    ('carry', 'bitwise identical'), ('failed', 'successful'), ('order', 'ordered'),
    ('history', 'invalid saved history'), ('counts', 'invalid work counts'),
])
def test_history_work_comparison_rejects_unmatched_or_failed_inputs(fault, message):
    initial, history, old, new = paired_work()
    if fault == 'carry':
        history[0, 11] = 0.
    elif fault == 'failed':
        new[0, 1] = 202
    elif fault == 'order':
        new[0, 0] = 2
    elif fault == 'history':
        history[0, 17] = np.nan
    else:
        new[0, 2] = .5
    with pytest.raises(ValueError, match=message):
        compare_history_work(initial, history, old, new, 1.)
