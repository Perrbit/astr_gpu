#!/usr/bin/env python3
"""Locate H1 spanwise jumps in native ASTR snapshots; no flow time stepping."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from check_air5_numq11_shock_tube import read_sensor_mask
from run_native_precision_tgv_gate import physical_state, rhs_state, component_report
from test_sbli_joint_nscbc import fields


def snapshot(case, label, step=0, stage=1, rank=0):
    return case / 'diagnostics' / f'state.{label}.step{step:08d}.rk{stage:02d}.rank{rank:08d}.bin'


def rhs_point(path, point):
    # Directional working arrays have undefined, later-overwritten physical faces.
    # Only inspect the interior diagnostic point, not those unwritten entries.
    with path.open('rb') as stream:
        header = np.fromfile(stream, np.int32, 4)
    shape = tuple(header[:3] + 1) + (int(header[3]),)
    value = np.memmap(path, dtype=np.float64, mode='r', offset=16, shape=shape, order='F')[point]
    if not np.all(np.isfinite(value)):
        raise ValueError(f'nonfinite interior directional RHS: {path}')
    return np.array(value)


def spanwise_norm(q):
    fluctuation = q[:-1] - q[:-1].mean(axis=0, keepdims=True)
    return float(np.linalg.norm(fluctuation[:, -3:]))


def analyze(root):
    with h5py.File(root / 'trajectory_large/datin/grid.flatplate.h5') as grid:
        jacobian = float((grid['x'][0, 0, 1] - grid['x'][0, 0, 0]) *
                         (grid['y'][0, 1, 0] - grid['y'][0, 0, 0]) *
                         (grid['z'][1, 0, 0] - grid['z'][0, 0, 0]))
    onset = root / 'onset_large'
    stages = []
    for path in sorted((onset / 'diagnostics').glob('state.pre_rhs.*.rank00000000.bin')):
        step = int(path.name.split('.step')[1].split('.')[0])
        stage = int(path.name.split('.rk')[1].split('.')[0])
        pieces = [physical_state(snapshot(onset, 'pre_rhs', step, stage, rank), 'fp64')
                  for rank in range(2)]
        q = np.concatenate([pieces[0][:-1], pieces[1]], axis=0)
        sensor, mask = read_sensor_mask(snapshot(onset, 'sensor', step, stage))
        direction = [rhs_point(snapshot(onset, 'conv_after_' + axis, step, stage), (72, 191, slice(0, 8), 4)) /
                     jacobian for axis in 'xyz']
        full = rhs_state(snapshot(onset, 'full', step, stage), 'fp64') / jacobian
        conv = rhs_state(snapshot(onset, 'conv', step, stage), 'fp64') / jacobian
        stages.append(dict(step=step, stage=stage,
            top_three_l2=spanwise_norm(q.transpose(2, 1, 0, 3)),
            top_three_mask_disagreements=int(np.count_nonzero(np.ptp(mask[:, -3:, :8], axis=2))),
            raw_sensor_source=sensor[71, 191, :8].tolist(),
            target_mask=mask[72, 191, :8].tolist(),
            energy_rhs_spanwise_range=dict(x=float(np.ptp(direction[0])),
                y=float(np.ptp(direction[1] - direction[0])),
                z=float(np.ptp(direction[2] - direction[1])),
                full=float(np.ptp(full[72, 191, :8, 4]))),
            target_boundary_correction_max=float(abs((full - conv)[72, 191, :8]).max())))

    trajectory = []
    for path in sorted((root / 'trajectory_large/outdat/new/fields').glob('segment*/step*/data.h5')):
        if '.tmp' in str(path):
            continue
        with h5py.File(path) as frame:
            rho, u, p = frame['density'][:], frame['velocity'][:], frame['pressure'][:]
            q = np.concatenate([rho[..., None], rho[..., None] * u,
                (p / .4 + .5 * rho * np.sum(u * u, axis=-1))[..., None]], axis=-1)
        trajectory.append(dict(step=int(path.parent.name[4:]), top_three_l2=spanwise_norm(q)))

    probes = {}
    for family, labels in [('planar', ['z0', 'z1']), ('threshold', ['below', 'above'])]:
        states, rates = {}, {}
        rows = []
        for backend in ['cpu', 'gpu']:
            states[backend], rates[backend] = [], []
            for label in labels:
                case = root / f'{family}_{label}_{backend}'
                q = physical_state(snapshot(case, 'pre_rhs'), 'fp64')
                rhs = rhs_state(snapshot(case, 'full'), 'fp64') / jacobian
                sensor, mask = read_sensor_mask(snapshot(case, 'sensor'))
                states[backend].append(q)
                rates[backend].append(rhs)
                rows.append(dict(backend=backend, label=label,
                    sensor=float(sensor[71, 191, 0]), target_mask=int(mask[72, 191, 0]),
                    energy_rhs=float(rhs[72, 191, 0, 4])))
        probes[family] = dict(samples=rows, differences={backend: dict(
            input_linf=float(abs(states[backend][1] - states[backend][0]).max()),
            energy_rhs_jump=float((rates[backend][1] - rates[backend][0])[72, 191, 0, 4]))
            for backend in ['cpu', 'gpu']},
            cpu_gpu_input_max_scaled=max(row['max_scaled'] for pair in zip(states['cpu'], states['gpu'])
                                         for row in component_report(*pair)),
            cpu_gpu_full_rhs_max_abs=max(float(abs(a - b).max()) for a, b in zip(rates['cpu'], rates['gpu'])))

    np1 = fields(root / 'np1_large', 128)['q']
    np2 = fields(root / 'reproduction/gpu_0.0001', 128)['q']
    return dict(
        scope='192x192x8 Ma2 inviscid diagnostic, not HCW physical or robustness acceptance',
        norm='unweighted discrete L2 of all five q deviations from the unique-z mean, top three y rows',
        rhs='raw native weighted RHS divided by the Cartesian cell Jacobian; not dt-scaled',
        axes='q snapshots x,y,z,component; HDF fields z,y,x,component; all reported indices are zero-based',
        point=[72, 191], sensor_source=[71, 191], jacobian=jacobian,
        onset_stages=stages, trajectory_from_primitive_archive=trajectory,
        native_single_rank_probes=probes,
        topology_replay=dict(np1_top_three_l2=spanwise_norm(np1), np2_top_three_l2=spanwise_norm(np2),
            np1_np2_q_linf=float(abs(np1 - np2).max()),
            interpretation='same large amplification without an MPI interface; not topology equivalence acceptance'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    result = analyze(args.root)
    output = args.root / 'onset_summary.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(output)
    print(json.dumps(result['native_single_rank_probes']['threshold'], indent=2))
