"""Approved two-flowthrough HCW numerical window, not a grid/physics validation."""
import json
import os

import h5py
import numpy as np
import pytest

from compare_flowstate import read_flowstate
from test_high_order_boundary_runtime import EXE, H1_VARIANT, launch_hcw
from run_adaptive_tgv_gate import STEP

AUTO_CFL=float(os.environ['ASTR_H1_HCW_AUTO_CFL']) if 'ASTR_H1_HCW_AUTO_CFL' in os.environ else None

pytestmark = pytest.mark.skipif(
    not EXE or H1_VARIANT not in ('admissible-filter','inlet-compatible') or os.environ.get('ASTR_H1_HCW_LONG') != '1',
    reason='Explicit executable, current H1 candidate and approved long-window selection required')


def state_report(case, steps):
    checkpoint = case/f'outdat/new/checkpoints/step{steps:012d}'
    with h5py.File(checkpoint/'state.h5') as file:
        fields = {f'q{i:04d}': file[f'q{i:04d}'][...] for i in range(1, 12)}
    identity=b'sbli-grcbc-h1cv' if H1_VARIANT=='inlet-compatible' else b'sbli-grcbc-h1pv'
    assert identity in (checkpoint/'control.bin').read_bytes()
    assert all(np.isfinite(value).all() for value in fields.values())
    assert fields['q0001'].min()>0
    # q6..q11 are the exact-restart rho/u/v/w/p/T cache.
    assert fields['q0006'].min()>0 and fields['q0010'].min()>0 and fields['q0011'].min()>0
    q = np.stack([fields[f'q{i:04d}'] for i in range(1, 6)], axis=-1)
    internal = q[..., 4]-.5*np.sum(q[..., 1:4]*(q[..., 1:4]/q[..., :1]), axis=-1)
    assert internal.min()>0
    unique = q[:-1]
    fluctuation = unique-unique.mean(axis=0, keepdims=True)
    return fields, dict(rho_min=float(q[..., 0].min()), pressure_min=float(fields['q0010'].min()),
                        temperature_min=float(fields['q0011'].min()),
                        internal_energy_min=float(internal.min()),
                        top_three_fluctuation_l2=float(np.linalg.norm(fluctuation[:, -3:])),
                        top_mean_density_max=float(unique[:, -3:, :, 0].mean(axis=0).max()))


def test_hcw_early_trajectory_gate(tmp_path):
    """Catch the observed first archived failure before attempting either long window."""
    states, statistics, clocks = [], [], []
    for backend in ('cpu', 'gpu'):
        launch_hcw(tmp_path/backend, backend, steps=420,automatic_cfl=AUTO_CFL)
        state, _ = state_report(tmp_path/backend, 420)
        states.append(state)
        statistics.append(read_flowstate(tmp_path/backend))
        clocks.append(np.array([[int(match[1]),*map(float,match.groups()[1:])]
            for match in STEP.finditer((tmp_path/backend/'run.log').read_text())]))
    errors = {name: float(np.max(abs(states[1][name]-value)/(1+abs(value))))
              for name, value in states[0].items()}
    assert statistics[0][0]==statistics[1][0]
    np.testing.assert_array_equal(statistics[0][1][:,0],statistics[1][1][:,0])
    statistics_error=float(abs(statistics[1][1]-statistics[0][1]).max())
    (tmp_path/'early_trajectory.json').write_text(json.dumps(dict(
        variant=H1_VARIANT,complete_steps=420, threshold=2e-10, scaled_field_errors=errors,
        mode='auto' if AUTO_CFL is not None else 'fixed',cfl_target=AUTO_CFL,
        maximum_backend_clock_difference=float(abs(clocks[1][:,1:]-clocks[0][:,1:]).max()),
        clocks=[dict(backend=backend,end_time=float(clock[-1,2]),
            dt_min=float(clock[:,3].min()),dt_max=float(clock[:,3].max()))
            for backend,clock in zip(('cpu','gpu'),clocks)],
        maximum_statistics_absolute_difference=statistics_error), indent=2)+'\n')
    np.testing.assert_allclose(clocks[0][:,1:],clocks[1][:,1:],rtol=0,atol=1e-12)
    assert max(errors.values()) <= 2e-10, errors
    assert statistics_error<=2e-10,statistics_error


@pytest.mark.parametrize('dt_scale,steps', [(1., 3360), (.5, 6720)], ids=['original_dt', 'half_dt'])
def test_hcw_two_flowthrough_window(tmp_path, dt_scale, steps):
    fields, stats, cases = [], [], []
    report = dict(scope='64x48x5, NP2 x; numerical robustness, not physical/grid convergence',
                  dt_scale=dt_scale, complete_steps=steps, cases=cases)
    for backend in ('cpu', 'gpu'):
        case = tmp_path/backend
        manifest = launch_hcw(case, backend, steps=steps, dt_scale=dt_scale,
                              checkpoint_interval=steps//8, volume_interval=steps//8)
        state, diagnostics = state_report(case, steps)
        names, data = read_flowstate(case)
        assert np.isfinite(data).all()
        fields.append(state)
        stats.append((names, data))
        dt = manifest['run_plan']['dt_nondimensional']*dt_scale
        length = manifest['geometry']['outlet_x_m']-manifest['geometry']['inlet_x_m']
        cases.append(dict(backend=backend, dt=dt, time_nondimensional=steps*dt,
                          flowthroughs=steps*dt/length, diagnostics=diagnostics,
                          statistics_columns=names, final_statistics=data[-1].tolist()))
        (tmp_path/'window.json').write_text(json.dumps(report, indent=2)+'\n')
    errors = {name: float(np.max(abs(fields[1][name]-expected)/(1+abs(expected))))
              for name, expected in fields[0].items()}
    report['final_scaled_field_errors'] = errors
    assert stats[0][0] == stats[1][0]
    np.testing.assert_array_equal(stats[0][1][:, 0], stats[1][1][:, 0])
    report['maximum_statistics_absolute_difference'] = float(abs(stats[1][1]-stats[0][1]).max())
    reference_files = sorted((tmp_path/'cpu/outdat/new/fields').glob('segment*/step*/data.h5'))
    actual_files = sorted((tmp_path/'gpu/outdat/new/fields').glob('segment*/step*/data.h5'))
    assert [p.relative_to(tmp_path/'cpu') for p in reference_files] == [
        p.relative_to(tmp_path/'gpu') for p in actual_files]
    assert reference_files
    sample_errors = []
    for reference, actual in zip(reference_files, actual_files):
        with h5py.File(reference) as a, h5py.File(actual) as b:
            assert all(np.isfinite(file[name][...]).all()
                       for file in (a, b) for name in ('density', 'velocity', 'pressure', 'temperature'))
            error = max(float(np.max(abs(b[name][...]-a[name][...])/(1+abs(a[name][...]))))
                        for name in ('density', 'velocity', 'pressure', 'temperature'))
            assert np.isfinite(error)
            sample_errors.append(dict(step=reference.parent.name, scaled_primitive_error=error))
    report['same_phase_samples'] = sample_errors
    (tmp_path/'window.json').write_text(json.dumps(report, indent=2)+'\n')
    assert max(errors.values()) <= 2e-10, report
    assert max(item['scaled_primitive_error'] for item in sample_errors) <= 2e-10, report


def test_hcw_halfstep_comparison(tmp_path):
    runs = {json.loads(path.read_text())['dt_scale']: path.parent
            for path in tmp_path.parent.glob('*/window.json') if not path.parent.is_symlink()}
    assert set(runs) == {1., .5}, 'Both approved windows must pass before time-step comparison'
    cases = []
    for backend in ('cpu', 'gpu'):
        coarse, _ = state_report(runs[1.]/backend, 3360)
        fine, _ = state_report(runs[.5]/backend, 6720)
        names, a = read_flowstate(runs[1.]/backend)
        other_names, b = read_flowstate(runs[.5]/backend)
        assert names == other_names
        # flowstate is sampled before the last RK advance, unlike the completed
        # checkpoint. Compare equal-time samples, not the two final log rows.
        aligned = b[::2]
        np.testing.assert_allclose(a[:, 1], aligned[:, 1], rtol=0, atol=1e-12)
        cases.append(dict(backend=backend,
            final_scaled_field_change={name: float(np.max(abs(fine[name]-value)/(1+abs(value))))
                                       for name, value in coarse.items()},
            last_common_statistics_time=float(a[-1, 1]),
            last_common_statistics_change=dict(zip(names[2:], (aligned[-1, 2:]-a[-1, 2:]).tolist())),
            statistics_maximum_change_at_common_times=dict(zip(
                names[2:], abs(aligned[:, 2:]-a[:, 2:]).max(axis=0).tolist()))))
    (tmp_path/'halfstep_comparison.json').write_text(json.dumps(dict(
        scope='time-step sensitivity only; no unapproved error tolerance or physical convergence claim',
        cases=cases), indent=2)+'\n')
