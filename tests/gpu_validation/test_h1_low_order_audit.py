"""Read-only native LF/characteristic audit, not admission of a flux limiter."""
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from run_native_precision_tgv_gate import physical_state, rhs_state
from test_grcbc_oblique_shock import EXE, DT, launch
from test_sbli_joint_nscbc import fields

pytestmark = pytest.mark.skipif(not EXE, reason='Set ASTR_SBLI_GRCBC_EXE to a root-CMake test executable')
SOURCE = Path(__file__).parent / 'out/h1_spanwise_repair_20261031/failure224_input.h5'
SUFFIX = 'step00000000.rk01.rank00000000.bin'


def admissibility(q):
    internal = q[..., 4] - .5 * np.sum(q[..., 1:4] * (q[..., 1:4] / q[..., 0, None]), axis=-1)
    primitive = q[..., 4] - .5 * q[..., 0] * np.sum((q[..., 1:4] / q[..., 0, None])**2, axis=-1)
    valid = (np.all(np.isfinite(q), axis=-1) & (q[..., 0] > 0) & np.isfinite(internal)
             & np.isfinite(primitive) & (internal > 0) & (primitive > 0))
    return dict(min_density=float(q[..., 0].min()), min_internal=float(internal.min()),
                min_primitive_internal=float(primitive.min()), invalid_nodes=int((~valid).sum()),
                min_internal_ijk=list(map(int, np.unravel_index(np.argmin(internal), internal.shape))))


def write_three_dimensional_input(source):
    # Vary all three coordinates so an extruded field cannot hide a z-halo error.
    z, y, x = np.meshgrid(np.linspace(0, .125, 17), np.linspace(0, 1, 33),
                          np.linspace(0, 1, 33), indexing='ij')
    phase = 2*np.pi*z/.125
    density = 1+.005*np.sin(2*np.pi*x)*np.sin(np.pi*y)*np.cos(phase)
    velocity = np.stack([1+.01*np.cos(2*np.pi*x)*np.sin(phase),
                         .005*np.sin(np.pi*y)*np.sin(phase), .003*np.sin(phase)], axis=-1)
    energy = 1/(5.6*.4)+.5*density*np.sum(velocity**2, axis=-1)
    state = np.concatenate([density[..., None], density[..., None]*velocity, energy[..., None]], axis=-1)
    with h5py.File(source, 'w') as file:
        for m in range(5):
            file[f'q{m+1:04d}'] = state[..., m]
    return source


@pytest.mark.parametrize('topology', [(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2)])
def test_low_order_audit_is_read_only_and_matches_cpu(tmp_path, topology):
    source = write_three_dimensional_input(tmp_path/'three_dimensional_input.h5')
    cases = []
    for backend in ['cpu', 'gpu']:
        reference = launch(tmp_path/f'{backend}_off', backend, n=32, steps=2, topology=topology,
                           z_intervals=16, h1_variant='continuous', field_source=source)
        audit = launch(tmp_path/f'{backend}_on', backend, n=32, steps=2, topology=topology,
                       z_intervals=16, h1_variant='continuous', diagnostics=True, low_order_audit=True,
                       field_source=source)
        np.testing.assert_array_equal(fields(reference, 2)['q'], fields(audit, 2)['q'])
        cases.append(audit)
    errors = []
    for rank in range(int(np.prod(topology))):
        for label in ['lf_raw', 'lf_complete']:
            name = f'state.{label}.step00000000.rk01.rank{rank:08d}.bin'
            a, b = [rhs_state(case/'diagnostics'/name, 'fp64') for case in cases]
            # Cartesian J=(1/32)^2*(.125/16), independent of the MPI partition.
            errors.append(float(abs(a-b).max() * 32**2 * 128))
    shared_errors = []
    if int(np.prod(topology)) == 2:
        axis = topology.index(2)
        for case in cases:
            for label in ['lf_raw', 'lf_complete']:
                rates = [rhs_state(case/'diagnostics'/
                         f'state.{label}.step00000000.rk01.rank{rank:08d}.bin', 'fp64')*32**2*128
                         for rank in [0, 1]]
                shared_errors.append(float(abs(np.take(rates[0], -1, axis=axis)
                                               -np.take(rates[1], 0, axis=axis)).max()))
    (tmp_path/'audit_parity.json').write_text(json.dumps(dict(max_rate_error=max(errors),
        max_shared_rate_jump=max(shared_errors, default=0)), indent=2)+'\n')
    assert max(errors) <= 2e-10, errors
    assert max(shared_errors, default=0) <= 2e-10, shared_errors


def test_captured_positive_state_low_order_budget(tmp_path):
    if not SOURCE.is_file():
        pytest.skip('Captured positive pre-failure state is required')
    reports = {}; snapshots = {}
    jacobian = 1/192**2/64
    for backend in ['cpu', 'gpu']:
        case = tmp_path/backend
        # The actual high-order update remains the known-red J2 counterexample.
        with pytest.raises(AssertionError):
            launch(case, backend, n=192, steps=1, field_source=SOURCE.resolve(),
                   h1_variant='continuous', diagnostics=True, check_stages=True, low_order_audit=True)
        log = (case/'run.log').read_text()
        assert f'H1_STAGE_STATE_FAILURE {backend.upper()}' in log, log[-3000:]
        q = physical_state(case/'diagnostics'/f'state.pre_rhs.{SUFFIX}', 'fp64')
        rates = {label: rhs_state(case/'diagnostics'/f'state.{label}.{SUFFIX}', 'fp64')/jacobian
                 for label in ['full', 'lf_raw', 'lf_complete']}
        updates = {label: q+DT*rate for label, rate in rates.items()}
        reports[backend] = dict(input=admissibility(q),
            updates={label: admissibility(value) for label, value in updates.items()},
            old_failure_point={label: admissibility(value[57:58, 191:192, 0:1])
                               for label, value in updates.items()})
        snapshots[backend] = (q, rates)
    q_error = float(abs(snapshots['cpu'][0]-snapshots['gpu'][0]).max())
    errors = {label: float(abs(snapshots['cpu'][1][label]-snapshots['gpu'][1][label]).max())
              for label in ['lf_raw', 'lf_complete']}
    report = dict(scope='FP64 Cartesian inviscid frozen first RK; private outputs only', dt=DT,
                  q_error=q_error, rate_errors=errors, backends=reports)
    (tmp_path/'low_order_budget.json').write_text(json.dumps(report, indent=2)+'\n')
    assert q_error <= 2e-10 and max(errors.values()) <= 2e-10, report
    for backend in reports.values():
        assert backend['updates']['full']['invalid_nodes'] > 0, report
        assert backend['updates']['lf_complete']['invalid_nodes'] == 0, report
