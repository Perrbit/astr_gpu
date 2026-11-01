"""Test-only positivity scaling of the existing H1 outlet state filter."""
import json
import os

import h5py
import numpy as np
import pytest

from run_native_precision_tgv_gate import rhs_state,primitive_state,cached_state_report
from tests.gpu_validation.compare_q_validation_snapshots import read_q_snapshot
from test_grcbc_oblique_shock import EXE, launch
from test_h1_low_order_audit import admissibility
from test_sbli_joint_nscbc import fields

pytestmark = pytest.mark.skipif(not EXE, reason='Set ASTR_SBLI_GRCBC_EXE to a root-CMake test executable')
VARIANT = os.environ.get('ASTR_H1_FILTER_VARIANT', 'admissible-filter')


def physical_filter_state(path):
    snapshot = read_q_snapshot(path, precision='fp64')
    im, jm, km, halo, components = snapshot.header
    state = snapshot.values.reshape((im+2*halo+1, jm+2*halo+1, km+2*halo+1, components), order='F')
    state = state[halo:halo+im+1, halo:halo+jm+1, halo:halo+km+1]
    # Match both production/budget energy orderings. Squaring momentum first
    # is a third ordering that can round a positive near-limit value to zero.
    assert admissibility(state)['invalid_nodes'] == 0, path
    cached = primitive_state(path.with_name(path.name.replace('pre_rhs.', 'pre_rhs_primitives.')),
                             'fp64', halo_sides=((False, False),)*3)
    cached_state_report(state, cached, 'fp64', mach=2.)
    return state


def write_hot_layers(source):
    temperature = np.full((17, 33, 33), .01)
    temperature[:, [18, 30], :] = 10.
    state = np.empty(temperature.shape+(5,))
    state[..., 0] = 1; state[..., 1] = 1; state[..., 2] = .2; state[..., 3] = 0
    state[..., 4] = temperature/(5.6*.4)+.5*(1+.2**2)
    assert admissibility(state)['invalid_nodes'] == 0
    with h5py.File(source, 'w') as file:
        for m in range(5):
            file[f'q{m+1:04d}'] = state[..., m]
    return source


@pytest.mark.parametrize('topology', [(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2)])
def test_hot_layers_survive_outlet_filter_cpu_gpu(tmp_path, topology):
    source = write_hot_layers(tmp_path/'hot_layers.h5')
    cases = [launch(tmp_path/backend, backend, n=32, steps=1, z_intervals=16, topology=topology,
                    field_source=source, h1_variant=VARIANT, diagnostics=True, check_stages=True)
             for backend in ['cpu', 'gpu']]
    ranks = int(np.prod(topology)); report = {}; plane_errors = []
    for case, backend in zip(cases, ['cpu', 'gpu']):
        rows = []; coefficients = []
        for stage in range(1, 4):
            rank_theta = []
            for rank in range(ranks):
                suffix = f'step00000000.rk{stage:02d}.rank{rank:08d}.bin'
                q = physical_filter_state(case/'diagnostics'/f'state.pre_rhs.{suffix}')
                rows.append(admissibility(q))
                theta = rhs_state(case/'diagnostics'/f'state.outlet_filter_theta.{suffix}', 'fp64')[..., 0]
                assert theta.min() >= 0 and theta.max() <= 1
                coefficients.append(float(theta.min())); rank_theta.append(theta)
            if ranks == 2 and topology[0] == 1:
                axis = topology.index(2)
                plane_errors.append(float(abs(np.take(rank_theta[0], -1, axis=axis)
                                               -np.take(rank_theta[1], 0, axis=axis)).max()))
        assert min(coefficients) < 1, 'the filter limiter must actually activate'
        assert all(row['invalid_nodes'] == 0 for row in rows)
        assert (case/'run.log').read_text().count(f'H1_STAGE_STATE_PASS {backend.upper()}') == 3
        report[backend] = dict(pre_rhs=rows, min_filter_coefficient=min(coefficients))
    states = [fields(case, 1)['q'] for case in cases]
    q_error = float(np.max(abs(states[1]-states[0])/(1+abs(states[0]))))
    report.update(max_scaled_final_q_error=q_error, max_shared_theta_jump=max(plane_errors, default=0))
    (tmp_path/'positive_filter.json').write_text(json.dumps(report, indent=2)+'\n')
    assert q_error <= 2e-10, report
    assert max(plane_errors, default=0) == 0, report


def test_admissible_filter_leaves_legal_candidates_unchanged(tmp_path):
    for backend in ['cpu', 'gpu']:
        original = launch(tmp_path/f'{backend}_original', backend, n=32, steps=2, z_intervals=16,
                          h1_variant='admissible')
        revised = launch(tmp_path/f'{backend}_revised', backend, n=32, steps=2, z_intervals=16,
                         h1_variant='admissible-filter', diagnostics=True)
        np.testing.assert_array_equal(fields(original, 2)['q'], fields(revised, 2)['q'])
        for stage in range(1, 4):
            theta = rhs_state(revised/'diagnostics'/
                f'state.outlet_filter_theta.step00000000.rk{stage:02d}.rank00000000.bin', 'fp64')
            assert np.all(theta == 1)


def test_repeated_filter_and_low_boundary_temperature_ordering(tmp_path):
    source = write_hot_layers(tmp_path/'hot_layers.h5')
    cases = []
    for backend in ['cpu', 'gpu']:
        case = launch(tmp_path/backend, backend, n=32, steps=2, z_intervals=16, topology=(1, 2, 1),
            field_source=source, h1_variant='admissible-filter', diagnostics=True,
            validation_steps=(0, 1), check_stages=True)
        assert (case/'run.log').read_text().count(f'H1_STAGE_STATE_PASS {backend.upper()}') == 6
        for step in [0, 1]:
            for stage in [1, 2, 3]:
                theta = []
                for rank in [0, 1]:
                    suffix = f'step{step:08d}.rk{stage:02d}.rank{rank:08d}.bin'
                    physical_filter_state(case/'diagnostics'/f'state.pre_rhs.{suffix}')
                    theta.append(rhs_state(case/'diagnostics'/f'state.outlet_filter_theta.{suffix}', 'fp64'))
                np.testing.assert_array_equal(theta[0][:, -1], theta[1][:, 0])
        cases.append(case)
    reference, actual = [fields(case, 2) for case in cases]
    errors = {name: float(np.max(abs(actual[name]-reference[name])/(1+abs(reference[name]))))
              for name in ['q', 'primitive_cache']}
    (tmp_path/'two_step_errors.json').write_text(json.dumps(errors, indent=2)+'\n')
    assert max(errors.values()) <= 2e-10, errors
