"""Bounded native ASTR integration of the approved HB6 coefficients, not production admission."""
import json

import h5py
import numpy as np
import pytest

from compare_flowstate import read_flowstate
from run_native_precision_tgv_gate import physical_state, rhs_state
from run_output_restart_validation import compare_fields
from test_high_order_wall_runtime import DT, EXE, GRID, check_completed_wall_diagnostics, defined_extras, launch
from review_high_order_local_filter_audit import reference_state

pytestmark = pytest.mark.skipif(not EXE, reason='Set ASTR_HB1B_EXE to the root-CMake FP64 CUDA executable')


def compare_checkpoints(a, b, normal_axis=2):
    errors = {}
    with h5py.File(a/'state.h5') as x, h5py.File(b/'state.h5') as y:
        for name in [f'q{j:04d}' for j in range(1, 12)]:
            p, q = x[name][:], y[name][:]
            assert np.isfinite(p).all() and np.isfinite(q).all()
            errors[name] = float(np.max(abs(p-q)/(1+abs(p))))
        for rank, (p, q) in enumerate(zip(defined_extras(x, normal_axis), defined_extras(y, normal_axis), strict=True)):
            assert np.isfinite(p).all() and np.isfinite(q).all()
            errors[f'extras_rank{rank}'] = float(np.max(abs(p-q)/(1+abs(p))))
    assert max(errors.values()) <= 2e-10, errors
    return errors


@pytest.mark.parametrize('kind', [41, 42, 421])
def test_native_first_filter_mapping_spatial_order(launch, kind):
    import mpmath as mp
    samples = []
    for n in (32, 64, 128):
        with mp.workdps(80):
            reference = np.array([[float(x) for x in reference_state(mp.mpf(j)/n, kind, 'wall-manufactured')[0]]
                                  for j in range(n+1)])
        strip = np.r_[0:8, n-7:n+1]
        row = dict(intervals=n)
        state = {}
        for backend in ('cpu', 'gpu'):
            case, _ = launch(backend, steps=1, grid=(8,n,8), manufactured=True,
                             wall_type=kind, filtering=True)
            state[backend] = physical_state(case/'diagnostics/state.pre_rhs.step00000000.rk01.rank00000000.bin', 'fp64')
            row[backend] = np.max(abs(state[backend][0,strip,0,:]-reference[strip]), axis=0).tolist()
        row['backend_scaled'] = float(np.max(abs(state['cpu']-state['gpu'])/(1+abs(state['cpu']))))
        assert row['backend_scaled'] <= 2e-10, row
        samples.append(row)
    orders = {}
    for backend in ('cpu', 'gpu'):
        errors = np.array([row[backend] for row in samples])
        orders[backend] = np.log2(errors[:-1]/errors[1:]).tolist()
    report = dict(bctype=kind, phase='real first stage after filtering and boundary restoration, before RHS',
                  source='existing wall manufactured fields and const2=.126, const6=2.5',
                  samples=samples, orders=orders)
    (case.parent/f'filter_mapping_convergence_bc{kind}.json').write_text(json.dumps(report, indent=2)+'\n')
    for backend in ('cpu', 'gpu'):
        assert np.min(orders[backend]) >= 5.5, report


@pytest.mark.parametrize('kind', [41, 42, 421])
@pytest.mark.parametrize('topology', [(1,1,1), (2,1,1), (1,2,1), (1,1,2), (2,2,1), (1,4,1)])
def test_filtered_backend_fields_rhs_and_statistics(launch, kind, topology):
    cpu, a = launch('cpu', topology, wall_type=kind, filtering=True)
    gpu, b = launch('gpu', topology, wall_type=kind, filtering=True)
    errors = compare_checkpoints(a, b)
    for rank in range(int(np.prod(topology))):
        for stage in (1, 2, 3):
            suffix = f'step00000000.rk{stage:02d}.rank{rank:08d}.bin'
            for phase in ('conv', 'full'):
                p = rhs_state(cpu/'diagnostics'/f'state.{phase}.{suffix}', 'fp64')
                q = rhs_state(gpu/'diagnostics'/f'state.{phase}.{suffix}', 'fp64')
                errors[f'{phase}_rk{stage}_rank{rank}'] = float(np.max(abs(p-q)/(1+abs(p))))
    ha, sa = read_flowstate(cpu)
    hb, sb = read_flowstate(gpu)
    assert ha == hb and sa.shape == sb.shape
    errors['statistics'] = float(np.max(abs(sa-sb)/(1+abs(sa))))
    (gpu/'comparison.json').write_text(json.dumps(errors, indent=2)+'\n')
    assert max(errors.values()) <= 2e-10, errors


@pytest.mark.parametrize('kind', [41, 42, 421])
def test_filtered_storage_paths(launch, kind):
    scalar, a = launch('gpu', wall_type=kind, filtering=True)
    full, b = launch('gpu', wall_type=kind, filtering=True, workspace='full')
    errors = compare_checkpoints(a, b)
    (full/'storage_comparison.json').write_text(json.dumps(errors, indent=2)+'\n')


@pytest.mark.parametrize('kind,axis', [(41,1), (41,3), (42,1)])
@pytest.mark.parametrize('workspace', ['scalar', 'full'])
def test_filtered_rotated_axis(launch, kind, axis, workspace):
    topology = tuple(2 if j == axis-1 else 1 for j in range(3))
    _, a = launch('cpu', topology, wall_type=kind, normal_axis=axis, filtering=True)
    gpu, b = launch('gpu', topology, wall_type=kind, normal_axis=axis, filtering=True, workspace=workspace)
    errors = compare_checkpoints(a, b, normal_axis=axis)
    (gpu/'rotation_comparison.json').write_text(json.dumps(errors, indent=2)+'\n')


@pytest.mark.parametrize('kind', [41, 42, 421])
def test_filtered_existing_minimum_boundary_rank(launch, kind):
    physical = {}
    grid = (8,14 if kind == 41 else 16,8)
    for backend in ('cpu', 'gpu'):
        for topology in ((1,1,1), (1,2,1)):
            _, checkpoint = launch(backend, topology, wall_type=kind, grid=grid, filtering=True)
            with h5py.File(checkpoint/'state.h5') as f:
                physical[backend, topology] = np.stack([f[f'q{j:04d}'][:] for j in range(1,12)])
    reference = physical['cpu', (1,1,1)]
    errors = {str(key): float(np.max(abs(value-reference)/(1+abs(reference))))
              for key, value in physical.items()}
    assert max(errors.values()) <= 2e-10, errors


@pytest.mark.parametrize('kind', [42, 421])
@pytest.mark.parametrize('backend', ['cpu', 'gpu'])
def test_filtered_adiabatic_old_seven_interval_rejection_is_unchanged(launch, kind, backend):
    launch(backend, (1,2,1), wall_type=kind, grid=(8,14,8), filtering=True,
           reject='Adiabatic Hermite7 wall requires seven strictly interior normal nodes')


@pytest.mark.parametrize('kind', [41, 42, 421])
@pytest.mark.parametrize('normal_intervals', [8, 9])
def test_filtered_global_short_domain(launch, kind, normal_intervals):
    _, a = launch('cpu', wall_type=kind, grid=(8,normal_intervals,8), filtering=True)
    _, b = launch('gpu', wall_type=kind, grid=(8,normal_intervals,8), filtering=True)
    compare_checkpoints(a, b)


@pytest.mark.parametrize('backend', ['cpu','gpu'])
def test_filtered_41_global_seven_interval_template_is_not_selected(launch, backend):
    launch(backend, wall_type=41, grid=(8,7,8), filtering=True,
           reject='High-order wall filter requires nine global normal nodes')


@pytest.mark.parametrize('kind', [41, 42, 421])
def test_filtered_wall_diagnostics(launch, kind, record_property):
    def filtered(*args, **kwargs):
        return launch(*args, **kwargs, filtering=True)
    check_completed_wall_diagnostics(filtered, (1,2,1), 'hermite7', record_property, wall_type=kind)


@pytest.mark.parametrize('kind', [41, 42, 421])
@pytest.mark.parametrize('backend', ['cpu', 'gpu'])
def test_filtered_wall_diagnostics_are_read_only(launch, kind, backend):
    _, plain = launch(backend, (1,2,1), steps=2, wall_type=kind, filtering=True)
    _, sampled = launch(backend, (1,2,1), steps=2, wall_type=kind, filtering=True, wall_samples=True)
    compare_fields(plain/'state.h5', sampled/'state.h5')


@pytest.mark.parametrize('kind', [41, 42, 421])
@pytest.mark.parametrize('backend', ['cpu', 'gpu'])
def test_filtered_qsave_and_rk_phase(launch, kind, backend):
    case, _ = launch(backend, wall_type=kind, filtering=True)
    with h5py.File(case/'outdat/new/resources/geometry.h5') as f:
        jacobian = f['q0004'][:].transpose(2,1,0)
    saved = None
    errors = {}
    for stage, a, b in ((1, 0., 1.), (2, .75, .25), (3, 1/3, 2/3)):
        suffix = f'step00000000.rk{stage:02d}.rank00000000.bin'
        before = physical_state(case/'diagnostics'/f'state.pre_rhs.{suffix}', 'fp64')
        after = physical_state(case/'diagnostics'/f'state.post_update.{suffix}', 'fp64')
        rhs = rhs_state(case/'diagnostics'/f'state.full.{suffix}', 'fp64')
        if stage == 1:
            saved = before.copy()
        expected = a*saved+b*before+b*DT*rhs/jacobian[...,None]
        errors[f'density_rk{stage}'] = float(np.max(abs(after[...,0]-expected[...,0])))
        errors[f'interior_rk{stage}'] = float(np.max(abs(after[:,1:-1,:,:]-expected[:,1:-1,:,:])))
    (case/'rk_phase.json').write_text(json.dumps(errors, indent=2)+'\n')
    assert max(errors.values()) < 2e-12, errors


@pytest.mark.parametrize('kind', [41, 42, 421])
def test_filtered_physical_quadrature_budget(launch, kind):
    def integral(q):
        return np.trapezoid(q[:-1,:,:-1,:], dx=1/GRID[1], axis=1).sum(axis=(0,1))*\
            (2*np.pi/GRID[0])*(2*np.pi/GRID[2])
    records = {}
    for backend in ('cpu', 'gpu'):
        case, _ = launch(backend, steps=2, wall_type=kind, filtering=True)
        stages = []
        for stage in (1,2,3):
            suffix = f'step00000000.rk{stage:02d}.rank00000000.bin'
            states = {label: physical_state(case/'diagnostics'/f'state.{label}.{suffix}', 'fp64')
                      for label in ('pre_filter','post_filter','pre_rhs','post_update')}
            values = {label: integral(q) for label, q in states.items()}
            filter_delta = values['post_filter']-values['pre_filter']
            boundary_delta = values['pre_rhs']-values['post_filter']
            stages.append(dict(stage=stage, before_filter=values['pre_filter'].tolist(),
                               filter_increment=filter_delta.tolist(),
                               boundary_and_reconciliation_increment=boundary_delta.tolist(),
                               rk_increment=(values['post_update']-values['pre_rhs']).tolist(),
                               telescope_residual=(filter_delta+boundary_delta-
                                   (values['pre_rhs']-values['pre_filter'])).tolist()))
        records[backend] = stages
    errors = []
    for cpu, gpu in zip(records['cpu'], records['gpu'], strict=True):
        scale = 1+abs(np.array(cpu['before_filter']))
        for label in ('filter_increment','boundary_and_reconciliation_increment','rk_increment'):
            errors.append(float(np.max(abs(np.array(cpu[label])-gpu[label])/scale)))
    report = dict(bctype=kind, stages=records, backend_maximum_scaled_difference=max(errors),
                  quadrature='unique periodic x/z nodes, trapezoidal y; physical diagnostic, not exact FD conservation norm')
    (case.parent/f'filter_budget_bc{kind}.json').write_text(json.dumps(report, indent=2)+'\n')
    assert max(errors) <= 2e-10, report


@pytest.mark.parametrize('kind', [41, 42, 421])
@pytest.mark.parametrize('backend', ['cpu','gpu'])
def test_filtered_validation_snapshots_are_read_only(launch, kind, backend):
    _, diagnostic = launch(backend, steps=2, wall_type=kind, filtering=True)
    plain, checkpoint = launch(backend, steps=2, wall_type=kind, filtering=True, dump_validation=False)
    compare_fields(diagnostic/'state.h5', checkpoint/'state.h5')
    assert (diagnostic/'control.bin').read_bytes() == (checkpoint/'control.bin').read_bytes()
    assert not list((plain/'diagnostics').glob('state.*.bin'))


@pytest.mark.parametrize('kind', [41, 42, 421])
@pytest.mark.parametrize('backend', ['cpu', 'gpu'])
def test_filtered_exact_restart(launch, kind, backend):
    _, whole = launch(backend, (2,1,1), 12, wall_type=kind, filtering=True)
    _, part = launch(backend, (2,1,1), 5, wall_type=kind, filtering=True)
    _, resumed = launch(backend, (2,1,1), 12, restore=part, wall_type=kind, filtering=True)
    compare_fields(whole/'state.h5', resumed/'state.h5')
    assert (whole/'control.bin').read_bytes() == (resumed/'control.bin').read_bytes()


def test_filtered_algorithm_identity_is_not_unfiltered(launch):
    _, part = launch('cpu', (2,1,1), 5, wall_type=42, filtering=True)
    assert b'channel-adiabatic-p6t7-v1-f6loc1' in (part/'control.bin').read_bytes()
    launch('cpu', (2,1,1), 12, restore=part, wall_type=42,
           reject='boundary/driver configuration mismatch')


@pytest.mark.parametrize('kind', [41, 42, 421])
@pytest.mark.parametrize('workspace', ['scalar', 'full'])
def test_filtered_device_memory_safety(launch, kind, workspace):
    launch('gpu', (1,2,1), steps=4, wall_type=kind, filtering=True,
           workspace=workspace, sanitizer=True)
