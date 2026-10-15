"""Synthetic host-only coverage of the approved G2 full-flow error contract."""
from pathlib import Path

import h5py
import numpy as np
import pytest

from air5_radau_reference import Air5RadauReference
from check_air5_flow_actual_error import compare_node, compare_checkpoints, require_flow_error


@pytest.fixture
def state():
    model = Air5RadauReference(Path(__file__).resolve().parents[2] / 'chemMech/air5_kimjo12.json')
    rho = .05
    species = rho * np.array([.7, .2, .03, .04, .03])
    momentum = rho * np.array([2000., 0., 0.])
    ev = model.ev_from_tv(species, 1500.)
    q5 = model.q5_from_state(rho, momentum, species, ev, 3000.)
    q = np.r_[rho, momentum, q5, species, ev]
    free = dict(rho=rho, u=2000., p=model.pressure(species, 3000.), q5=q5)
    return model, q, ev/rho, free


def test_exact_full_state_and_nonzero_carry(state):
    model, q, initial_ev, free = state
    carry = q * 1e-17
    result = compare_node(model, q, carry, q, carry, initial_ev, free)
    assert set(result) == {'q', 'q-minus-carry'}
    for row in result.values():
        assert max(row['normalized_error']) == 0.
        assert len(row['normalized_error']) == 14


def test_sub_ulp_carry_survives_density_and_quotient_errors(state):
    model, q, initial_ev, free = state
    carry = np.zeros(11)
    carry[5] = np.spacing(q[5]) / 4
    assert q[5] - carry[5] == q[5]
    result = compare_node(model, q, carry, q, np.zeros(11), initial_ev, free)
    assert result['q']['absolute_error'][0] == 0.
    assert result['q-minus-carry']['absolute_error'][0] == abs(carry[5]) / q[0]
    carry = q * 1e-17
    result = compare_node(model, q, carry, q, np.zeros(11), initial_ev, free)
    assert result['q-minus-carry']['absolute_error'][8] == abs(carry[0])


def test_zero_target_relative_error_is_undefined(state):
    model, q, initial_ev, free = state
    actual = q.copy()
    actual[2] = q[0] * 1e-9
    result = compare_node(model, actual, np.zeros(11), q, np.zeros(11), initial_ev, free)
    assert result['q']['relative_error'][10] is None
    assert result['q']['normalized_error'][10] == pytest.approx(.005)


def test_density_carry_changes_fraction_and_velocity_even_when_rounded_state_is_unchanged(state):
    model, q, initial_ev, free = state
    carry = np.zeros(11)
    carry[0] = np.spacing(q[0]) / 4
    assert q[0] - carry[0] == q[0]
    result = compare_node(model, q, carry, q, np.zeros(11), initial_ev, free)['q-minus-carry']
    assert result['absolute_error'][0] == pytest.approx(q[5]*carry[0]/q[0]**2, rel=1e-14, abs=0)
    assert result['absolute_error'][9] == pytest.approx(q[1]*carry[0]/q[0]**2, rel=1e-14, abs=0)


def test_negative_formation_inclusive_total_energy_is_not_rejected_when_physics_is_valid(state):
    model, q, _, free = state
    q[1:4] = 0.
    q[5:10] = q[0] * np.array([.767, .233, 0., 0., 0.])
    q[10] = model.ev_from_tv(q[5:10], 350.)
    q[4] = model.q5_from_state(q[0], q[1:4], q[5:10], q[10], 350.)
    assert q[4] < 0
    result = compare_node(model, q, np.zeros(11), q, np.zeros(11), q[10]/q[0], free)
    assert max(result['q']['normalized_error']) == 0.


@pytest.mark.parametrize('kind', ['negative-species', 'negative-represented', 'nan-carry', 'closure', 'temperature'])
def test_invalid_physics_is_not_hidden_by_error_norm(state, kind):
    model, q, initial_ev, free = state
    actual, carry = q.copy(), np.zeros(11)
    if kind == 'negative-species':
        actual[7] = -1e-30
    elif kind == 'negative-represented':
        carry[7] = 2*actual[7]
    elif kind == 'nan-carry':
        carry[1] = np.nan
    elif kind == 'closure':
        actual[5] += 1e-7
    else:
        actual[4] *= 100
    with pytest.raises(ValueError):
        compare_node(model, actual, carry, q, np.zeros(11), initial_ev, free)


def write_checkpoint(path, q, step=24, ranks=1, history=False):
    shape = (2, 3, 4)
    with h5py.File(path, 'w') as file:
        identity = np.array([int('4153545251533031', 16), 2, 1, 4, 3, 2,
                             35 if history else 34, ranks, step, 0, 0, 0, 7], dtype=np.int64)
        identity[9:12] = np.array([step*1e-10, 1e-10, 1e-10]).view(np.int64)
        file['identity'], file['metadata'] = identity, [1, 1]
        for component in range(1, 35):
            file[f'q{component:04}'] = np.full(shape, q[component-1] if component <= 11 else 0.)
        if history:
            file['q0035'] = np.full(shape, 1e-11)


def checkpoints(tmp_path, q):
    paths = {name: tmp_path / f'{name}.h5' for name in ('initial', 'reference', 'preceding', 'candidate')}
    for name, path in paths.items():
        write_checkpoint(path, q, step=0 if name == 'initial' else 24,
                         ranks=2 if name == 'candidate' else 1, history=name == 'candidate')
    return paths


def test_streamed_checkpoints_allow_different_rank_count_and_history(state, tmp_path):
    model, q, _, free = state
    paths = checkpoints(tmp_path, q)
    result = compare_checkpoints(model, **paths, freestream=free)
    require_flow_error(result)
    assert result['physical_nodes'] == 24
    assert result['within_reference_margin'] and result['within_candidate_error']
    assert max(result['candidate']['q']['max_normalized_error']) == 0.


@pytest.mark.parametrize('dataset,index', [('identity', 8), ('identity', 9), ('identity', 10), ('identity', 11),
                                         ('identity', 2), ('identity', 12), ('q0001', None)])
def test_phase_layout_and_nonfinite_input_mismatches_stop(state, tmp_path, dataset, index):
    model, q, _, free = state
    paths = checkpoints(tmp_path, q)
    with h5py.File(paths['candidate'], 'r+') as file:
        if index is None:
            file[dataset][0, 0, 0] = np.nan
        else:
            row = file[dataset][:]
            row[index] += 1
            file[dataset][:] = row
    with pytest.raises(ValueError):
        compare_checkpoints(model, **paths, freestream=free)


@pytest.mark.parametrize('which,amount,expected', [('preceding', .02, 'reference margin'),
                                                ('candidate', 2., 'candidate error')])
def test_reference_failure_is_separate_from_candidate_failure(state, tmp_path, which, amount, expected):
    model, q, _, free = state
    paths = checkpoints(tmp_path, q)
    with h5py.File(paths[which], 'r+') as file:
        file['q0003'][0, 0, 0] = q[0] * amount * 1e-10 * abs(free['u'])
    result = compare_checkpoints(model, **paths, freestream=free)
    with pytest.raises(ValueError, match=expected):
        require_flow_error(result)
    row = result['reference_pair' if which == 'preceding' else 'candidate']['q']
    assert row['max_error_node_zyx'][10] == [0, 0, 0]
