#!/usr/bin/env python3
"""Attribute matched-checkpoint differences; no flow integration or physical pass."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import h5py

from air5_radau_reference import Air5RadauReference
from check_air5_c5_normal_shock import primitive_metrics
from check_air5_c5_frozen_transport import read_rhs_snapshot
from check_air5_c5_hbl import _active_array
from compare_q_validation_snapshots import compare_snapshot_sets

COMPONENTS = ('rho', 'rho_u', 'rho_v', 'rho_w', 'rho_E',
              'rho_N2', 'rho_O2', 'rho_N', 'rho_O', 'rho_NO', 'rho_ev')


def path(case, label, stage, rank, step=1000):
    return case/'validation'/f'air5.{label}.step{step:08d}.rk{stage:02d}.rank{rank:08d}.bin'


def rhs(case, label, stage, rank):
    snapshot = read_rhs_snapshot(path(case, label, stage, rank))
    im, jm, km, nq = snapshot.header
    return snapshot.values.reshape((im+1, jm+1, km+1, nq), order='F')


def scalar(filename):
    with filename.open('rb') as stream:
        header = np.fromfile(stream, np.int32, 4)
        values = np.fromfile(stream, np.float64)
    if len(header) != 4 or min(header) < 0:
        raise ValueError(f'invalid scalar header: {filename}')
    im, jm, km, hm = header
    shape = (im+2*hm+1, jm+2*hm+1, km+2*hm+1)
    if values.size != np.prod(shape) or not np.isfinite(values).all():
        raise ValueError(f'invalid scalar payload: {filename}')
    field = values.reshape(shape, order='F')
    return field[hm:hm+im+1, hm:hm+jm+1, hm:hm+km+1]


def difference(a, b):
    """Report per-component changes, not a mixed-unit aggregate norm."""
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('incompatible/nonfinite diagnostic fields')
    return {name: dict(max_abs=float(abs(b[..., c]-a[..., c]).max()),
                      reference_max_abs=float(abs(a[..., c]).max()),
                      rms_difference=float(np.sqrt(np.mean((b[..., c]-a[..., c])**2))))
            for c, name in enumerate(COMPONENTS)}


def state_gate(case, thermo):
    bounds = [np.array([thermo.species_vibrational_energy(s, t) for s in range(5)])
              for t in thermo.temperature_bounds]
    result = dict(snapshots=0, min_temperature=float('inf'), max_temperature=0.,
                  min_pressure=float('inf'), max_pressure=0., mass_closure=0.)
    for label in ('post_chemistry', 'pre_rhs', 'post_update', 'post_transport'):
        files = sorted((case/'validation').glob(f'air5.{label}.*.bin'))
        if not files:
            raise ValueError(f'missing state snapshots: {case}/{label}')
        for filename in files:
            q = _active_array(filename)
            rho, species, temp, pressure, _ = primitive_metrics(q, thermo)
            closure = float(np.max(abs(species.sum(axis=-1)-rho)/rho))
            if (rho.min() <= 0 or species.min() < 0 or temp.min() < 1000 or
                temp.max() > thermo.temperature_bounds[1] or
                pressure.min() < thermo.pressure_bounds[0] or
                pressure.max() > thermo.pressure_bounds[1] or closure > 2e-12 or
                np.any(q[..., 10] < species@bounds[0]) or np.any(q[..., 10] > species@bounds[1])):
                raise ValueError(f'accepted-state domain/positivity failure: {filename}')
            result['snapshots'] += 1
            result['mass_closure'] = max(result['mass_closure'], closure)
            for name, array in [('temperature', temp), ('pressure', pressure)]:
                result['min_'+name] = min(result['min_'+name], float(array.min()))
                result['max_'+name] = max(result['max_'+name], float(array.max()))
    return result


def cartesian_jacobian(case):
    spacings = []
    with h5py.File(case/'datin/grid.h5') as grid:
        for name, axis in [('x', 2), ('y', 1), ('z', 0)]:
            values = grid[name][()]
            delta = np.diff(values, axis=axis)
            step = float(delta.flat[0])
            if step <= 0 or not np.allclose(delta, step, rtol=1e-11, atol=1e-16):
                raise ValueError('diagnostic requires a uniform Cartesian grid')
            for other in set(range(3))-{axis}:
                if np.any(np.diff(values, axis=other) != 0):
                    raise ValueError('diagnostic does not support skewed grids')
            spacings.append(step)
    return float(np.prod(spacings))


def check_window_contract(coarse, fine):
    for key in ('checkpoint_sha256', 'executable_sha256', 'start_step', 'start_time',
                'baseline', 'topology', 'convection_limiter', 'diffusion_limiter'):
        if coarse[key] != fine[key]:
            raise ValueError(f'incompatible matched window: {key}')
    for key, default in (('compensation', 'off'), ('compensation_restart', 'restore')):
        if coarse.get(key, default) != fine.get(key, default):
            raise ValueError(f'incompatible matched window: {key}')
    if (coarse['topology'] != '2,1,1' or coarse['updates'] < 1 or
        fine['updates'] != 2*coarse['updates'] or coarse['dt'] != 2*fine['dt']):
        raise ValueError('expected NP2 dt/dt2 matched update counts')
    times = [c['start_time']+c['dt']*c['updates'] for c in (coarse, fine)]
    if not np.isfinite(times).all() or not np.isclose(*times, rtol=0, atol=1e-18):
        raise ValueError('endpoint times differ')
    for c, time in zip((coarse, fine), times):
        if not np.isclose(c['target_time'], time, rtol=0, atol=1e-18):
            raise ValueError('inconsistent target time')
    return times[0]


def matched_window(root, names=('gpu_dt2', 'gpu_dt1'), report_path=None):
    contracts = [json.loads((root/name/'contract.json').read_text()) for name in names]
    end_time = check_window_contract(*contracts)
    thermo = Air5RadauReference(Path(__file__).resolve().parents[2]/'chemMech/air5_kimjo12.json')
    report = dict(status='matched-window-diagnostic-not-physical-pass', cases=names, target_time=end_time,
                  updates=[c['updates'] for c in contracts], states={}, ranks=[])
    for name in names:
        if json.loads((root/name/'result.json').read_text())['status'] != 'bounded-replay-completed-not-physical-pass':
            raise ValueError('cannot compare incomplete replay')
        case = root/name/'gpu'
        report['states'][name] = state_gate(case, thermo)
        cfl = [float(v) for v in re.findall(r'current CFL:\s+(\S+)', (case/'run.log').read_text())]
        if not cfl or not np.isfinite(cfl).all() or not 0 <= max(cfl) < 1:
            raise ValueError('CFL gate failed')
        report['states'][name]['max_cfl'] = max(cfl)
    for rank in range(2):
        ends, starts = [], []
        for name, c in zip(names, contracts):
            case = root/name/'gpu'
            starts.append(_active_array(path(case, 'pre_chemistry', 1, rank, c['start_step'])))
            ends.append(_active_array(path(case, 'post_chemistry', 2, rank, c['start_step']+c['updates']-1)))
        np.testing.assert_array_equal(*starts)
        a, b = ends
        velocity = abs(b[..., 1:4]/b[..., :1]-a[..., 1:4]/a[..., :1])
        ua = float(abs(starts[0][..., 1]/starts[0][..., 0]).max())
        qdiff = difference(a, b)
        entry = dict(rank=rank, endpoint_q=qdiff,
            max_velocity_difference_m_s=velocity.max(axis=(0, 1, 2)).tolist(),
            max_u_difference_over_reference_velocity=float(velocity[..., 0].max()/ua),
            max_u_local_ijk=list(map(int, np.unravel_index(np.argmax(velocity[..., 0]), velocity.shape[:3]))),
            primitive_max_abs={}, final_coefficients={})
        for label, aa, bb in zip(('temperature_K', 'pressure_Pa'), primitive_metrics(a, thermo)[2:4],
                                 primitive_metrics(b, thermo)[2:4]):
            entry['primitive_max_abs'][label] = float(abs(bb-aa).max())
        for name, c in zip(names, contracts):
            step = c['start_step']+c['updates']-1
            entry['final_coefficients'][name] = {channel: min(float(scalar(path(root/name/'gpu',
                f'diffusion_{channel}_ratio', stage, rank, step)).min()) for stage in range(1, 4))
                for channel in ('species', 'energy')}
        report['ranks'].append(entry)
    output = report_path or root/'matched_window.json'
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2))
    return report


def run(root):
    cases = {key: root/key/backend for key, backend in
             [('cpu_dt2', 'cpu'), ('gpu_dt2', 'gpu'), ('gpu_dt1', 'gpu')]}
    contracts = [json.loads((root/key/'contract.json').read_text()) for key in cases]
    if len({c['checkpoint_sha256'] for c in contracts}) != 1:
        raise ValueError('replays must start from the identical checkpoint')
    if len({c['executable_sha256'] for c in contracts}) != 1:
        raise ValueError('replays must use the identical executable')
    if len({c.get('convection_limiter', 'full_state') for c in contracts}) != 1:
        raise ValueError('replay comparisons must use the same convection limiter')
    if len({c.get('diffusion_limiter', 'full_state') for c in contracts}) != 1:
        raise ValueError('replay comparisons must use the same diffusion limiter')
    if (any(c['start_step'] != 1000 or c['topology'] != '2,1,1' for c in contracts) or
        [c['updates'] for c in contracts] != [1, 1, 2] or
        contracts[0]['dt'] != contracts[1]['dt'] or
        contracts[1]['dt'] != 2*contracts[2]['dt']):
        raise ValueError('expected NP2 step1000 CPU/GPU full-step and GPU two-half-step replays')
    if max(c['target_time'] for c in contracts)-min(c['target_time'] for c in contracts)>1e-20:
        raise ValueError('endpoint physical times differ')
    comparison = compare_snapshot_sets(cases['cpu_dt2']/'validation/air5',
        cases['gpu_dt2']/'validation/air5',
        ('post_chemistry', 'pre_rhs', 'post_update', 'post_transport'),
        atol=1e-9, rtol=1e-10, active_only=True)
    if not comparison.passed:
        raise ValueError('CPU/GPU state equivalence failed; inspect before attribution')
    thermo = Air5RadauReference(Path(__file__).resolve().parents[2]/'chemMech/air5_kimjo12.json')
    jacobian = cartesian_jacobian(cases['gpu_dt2'])
    result = dict(status='operator-diagnostic-not-physical-pass',
        rhs_convention='metric-weighted conservative RHS; not divided by Jacobian',
        state_gates={name:state_gate(case, thermo) for name, case in cases.items()},
        cpu_gpu=dict(files=comparison.file_count, max_abs=comparison.max_abs,
                     max_scaled=comparison.max_scaled), operators=[], timestep=[])
    for rank in range(2):
        for stage in range(1, 4):
            case = cases['gpu_dt2']
            raw, conv, full = [rhs(case, label, stage, rank) for label in ('conv_raw', 'conv', 'full')]
            theta = scalar(path(case, 'convection_ratio', stage, rank))
            diffusion_theta = scalar(path(case, 'diffusion_ratio', stage, rank))
            interior = (slice(1 if rank == 0 else 0, -1 if rank == 1 else None), slice(1, -1), slice(0, -1))
            active = theta[interior]
            entry = dict(rank=rank, stage=stage, min_convection_ratio=float(active.min()),
                convection_limited_points=int(np.count_nonzero(active < 1-1e-14)),
                min_diffusion_ratio=float(diffusion_theta[interior].min()),
                diffusion_limited_points=int(np.count_nonzero(diffusion_theta[interior] < 1-1e-14)),
                limiter_rhs_change=difference(raw[interior], conv[interior]),
                diffusion_rhs_change=difference(conv[interior], full[interior]), stations=[])
            if stage == 1:
                q = _active_array(path(case, 'pre_rhs', stage, rank))
                trial = q + contracts[1]['dt']/jacobian*raw
                negative = np.any(trial[..., 5:10] < 0, axis=-1)[interior]
                limited = active < 1-1e-14
                species_trial = trial[..., 5:10]
                minimum_index = np.unravel_index(np.argmin(species_trial), species_trial.shape)
                before = float(q[minimum_index[:3]+(minimum_index[3]+5,)])
                entry['raw_euler_species_diagnostic'] = dict(
                    scope='unaccepted diagnostic trial; never used to advance ASTR',
                    negative_points=int(np.count_nonzero(negative)),
                    limited_points_with_nonnegative_raw_species=int(np.count_nonzero(limited & ~negative)),
                    minimum_local_ijk_species0=list(map(int, minimum_index)),
                    minimum_location_species_density_before=before,
                    minimum_trial_species_density=float(trial[interior][..., 5:10].min()))
            for ix in ([1, 2, 4, 8, 16, 31] if rank == 0 else [1, 16, 31]):
                ys = np.flatnonzero(theta[ix].min(axis=-1) < 1-1e-14)
                entry['stations'].append(dict(local_ix=ix,
                    limited_y=ys.tolist(), min_ratio=float(theta[ix].min()),
                    near_wall_limiter_change=difference(raw[ix, 1:7, :-1], conv[ix, 1:7, :-1]),
                    near_wall_raw_rhs=raw[ix, :7, 0].tolist(),
                    near_wall_limiter_rhs=(conv-raw)[ix, :7, 0].tolist(),
                    near_wall_diffusion_rhs=(full-conv)[ix, :7, 0].tolist()))
            result['operators'].append(entry)
        a = _active_array(path(cases['gpu_dt2'], 'post_chemistry', 2, rank))
        b = _active_array(path(cases['gpu_dt1'], 'post_chemistry', 2, rank, step=1001))
        pre_a = _active_array(path(cases['gpu_dt2'], 'pre_rhs', 1, rank))
        pre_b = _active_array(path(cases['gpu_dt1'], 'pre_rhs', 1, rank))
        theta_a = scalar(path(cases['gpu_dt2'], 'convection_ratio', 1, rank))
        theta_b = scalar(path(cases['gpu_dt1'], 'convection_ratio', 1, rank))
        result['timestep'].append(dict(rank=rank, endpoint_q=difference(a, b),
            endpoint_u_max_abs=float(abs(a[..., 1]/a[..., 0]-b[..., 1]/b[..., 0]).max()),
            endpoint_u_max_local_ijk=list(map(int, np.unravel_index(
                np.argmax(abs(a[..., 1]/a[..., 0]-b[..., 1]/b[..., 0])), a.shape[:3]))),
            first_stage_input_q=difference(pre_a, pre_b),
            first_stage_theta_min=[float(theta_a.min()), float(theta_b.min())],
            first_stage_theta_max_abs=float(abs(theta_a-theta_b).max()),
            first_stage_rhs={label:difference(rhs(cases['gpu_dt2'], label, 1, rank),
                                               rhs(cases['gpu_dt1'], label, 1, rank))
                             for label in ('conv_raw', 'conv', 'full')}))
    (root/'operator_diagnosis.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'], cpu_gpu=result['cpu_gpu'],
                         report=str(root/'operator_diagnosis.json')), indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--matched-window', action='store_true')
    parser.add_argument('--window-cases', nargs=2, default=('gpu_dt2', 'gpu_dt1'))
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.matched_window:
        matched_window(args.root, args.window_cases, args.report)
    else:
        if args.report is not None or tuple(args.window_cases) != ('gpu_dt2', 'gpu_dt1'):
            parser.error('--report/--window-cases require --matched-window')
        run(args.root)
