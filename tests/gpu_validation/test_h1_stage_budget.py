"""Bounded native verification of the opt-in H1 complete-stage budget."""
import json
import os

import h5py
import numpy as np
import pytest

from run_native_precision_tgv_gate import rhs_state, physical_state
from test_grcbc_oblique_shock import EXE, DT, launch
from test_h1_low_order_audit import SOURCE, admissibility, write_three_dimensional_input
from test_sbli_joint_nscbc import fields

pytestmark = pytest.mark.skipif(not EXE, reason='Set ASTR_SBLI_GRCBC_EXE to a root-CMake test executable')
CAPTURED_VARIANT = os.environ.get('ASTR_H1_BUDGET_VARIANT', 'admissible')


def check_theta(case, ranks, stages=3):
    errors = []
    for stage in range(1, stages+1):
        for rank in range(ranks):
            theta = rhs_state(case/'diagnostics'/
                f'state.budget_theta.step00000000.rk{stage:02d}.rank{rank:08d}.bin', 'fp64')
            assert theta.shape[-1] == 7
            assert theta.min() >= 0 and theta.max() <= 1
            for axis in range(3):
                a, b = [slice(None)]*3, [slice(None)]*3
                a[axis] = slice(None, -1); b[axis] = slice(1, None)
                left = theta[tuple(a)+(2*axis+1,)]
                right = theta[tuple(b)+(2*axis,)]
                errors.append(float(abs(left-right).max()))
    assert max(errors) == 0, errors
    return max(errors)


def test_captured_state_is_repaired_by_conservative_stage_budget(tmp_path):
    if not SOURCE.is_file():
        pytest.skip('Captured positive pre-failure state is required')
    reports = {}; cases = []
    for backend in ['cpu', 'gpu']:
        case = launch(tmp_path/backend, backend, n=192, steps=1, topology=(1, 1, 1),
            field_source=SOURCE.resolve(), diagnostics=True, h1_variant=CAPTURED_VARIANT, check_stages=True)
        cases.append(case)
        q = physical_state(case/'diagnostics/state.pre_rhs.step00000000.rk01.rank00000000.bin', 'fp64')
        rates = {label: rhs_state(case/'diagnostics'/
                 f'state.{label}.step00000000.rk01.rank00000000.bin', 'fp64')*192**2*64
                 for label in ['full', 'budget_low', 'limited']}
        reports[backend] = {label: admissibility(q+DT*rate) for label, rate in rates.items()}
        assert reports[backend]['full']['invalid_nodes'] > 0
        assert reports[backend]['budget_low']['invalid_nodes'] == 0
        assert reports[backend]['limited']['invalid_nodes'] == 0
        assert (case/'run.log').read_text().count(f'H1_STAGE_STATE_PASS {backend.upper()}') == 3
        check_theta(case, 1)
    state = [fields(case, 1)['q'] for case in cases]
    error = float(np.max(abs(state[1]-state[0])/(1+abs(state[0]))))
    (tmp_path/'captured_budget.json').write_text(json.dumps(dict(updates=reports,
        max_scaled_final_q_error=error), indent=2)+'\n')
    assert error <= 2e-10, error


def test_open_boundary_seventh_budget_actually_limits(tmp_path):
    # An admissible hot interior layer is outside the low normal stencil but
    # inside D6. Cold supersonic top outflow isolates the boundary residual.
    temperature = np.full((17, 33, 33), .01)
    temperature[:, 30, 8:24] = 10.
    q = np.empty(temperature.shape+(5,))
    q[..., 0] = 1; q[..., 1] = 1; q[..., 2] = .2; q[..., 3] = 0
    q[..., 4] = temperature/(5.6*.4)+.5*(1+.2**2)
    source = tmp_path/'positive_hot_layer.h5'
    with h5py.File(source, 'w') as file:
        for m in range(5):
            file[f'q{m+1:04d}'] = q[..., m]
    reports = {}; states = []
    for backend in ['cpu', 'gpu']:
        case = launch(tmp_path/backend, backend, n=32, steps=1, z_intervals=16, field_source=source,
                      h1_variant='admissible', diagnostics=True, check_stages=True)
        suffix = 'step00000000.rk01.rank00000000.bin'
        theta = rhs_state(case/'diagnostics'/f'state.budget_theta.{suffix}', 'fp64')
        reports[backend] = float(theta[:, -1, :, 6].min())
        assert reports[backend] < 1, reports
        check_theta(case, 1)
        states.append(fields(case, 1)['q'])
    error = float(np.max(abs(states[1]-states[0])/(1+abs(states[0]))))
    (tmp_path/'seventh_budget.json').write_text(json.dumps(dict(min_top_boundary_coefficients=reports,
        max_scaled_q_error=error), indent=2)+'\n')
    assert error <= 2e-10, error


@pytest.mark.parametrize('topology', [(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2)])
def test_three_dimensional_budget_cpu_gpu_and_shared_faces(tmp_path, topology):
    source = write_three_dimensional_input(tmp_path/'input.h5')
    cases = [launch(tmp_path/backend, backend, n=32, steps=2, topology=topology, z_intervals=16,
                    field_source=source, h1_variant=CAPTURED_VARIANT, diagnostics=True, check_stages=True)
             for backend in ['cpu', 'gpu']]
    ranks = int(np.prod(topology)); errors = []; initial_errors = []; jumps = []
    initial_q_errors = []
    for rank in range(ranks):
        name = f'state.pre_rhs.step00000000.rk01.rank{rank:08d}.bin'
        a, b = [physical_state(case/'diagnostics'/name, 'fp64') for case in cases]
        initial_q_errors.append(float(abs(a-b).max()))
    for case in cases:
        assert (case/'run.log').read_text().count('H1_STAGE_STATE_PASS ') == 6
        check_theta(case, ranks)
    for stage in range(1, 4):
        for rank in range(ranks):
            for label in ['budget_low', 'budget_residual', 'limited']:
                name = f'state.{label}.step00000000.rk{stage:02d}.rank{rank:08d}.bin'
                a, b = [rhs_state(case/'diagnostics'/name, 'fp64') for case in cases]
                error=float(abs(a-b).max()*32**2*128)
                errors.append(error)
                if stage==1: initial_errors.append(error)
        if ranks == 2:
            axis = topology.index(2)
            for case in cases:
                theta = [rhs_state(case/'diagnostics'/
                    f'state.budget_theta.step00000000.rk{stage:02d}.rank{rank:08d}.bin', 'fp64')
                    for rank in range(2)]
                jumps.append(float(abs(np.take(theta[0], -1, axis=axis)
                                       -np.take(theta[1], 0, axis=axis)).max()))
    state = [fields(case, 2)['q'] for case in cases]
    q_error = float(np.max(abs(state[1]-state[0])/(1+abs(state[0]))))
    report = dict(max_trajectory_rate_error=max(errors),max_common_initial_rate_error=max(initial_errors),
                  max_common_initial_q_error=max(initial_q_errors),max_scaled_q_error=q_error,
                  max_shared_theta_jump=max(jumps, default=0))
    (tmp_path/'budget_parity.json').write_text(json.dumps(report, indent=2)+'\n')
    assert max(initial_errors) <= 2e-10 and q_error <= 2e-10, report
    assert max(initial_q_errors) <= 2e-10, report
    assert max(jumps, default=0) == 0, report
