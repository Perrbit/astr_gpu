#!/usr/bin/env python3
"""Read immutable HCW trajectory and frozen-state evidence; never advance flow."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from run_h1_limiter_factorial import frames
from run_native_precision_tgv_gate import physical_state, rhs_state


def difference(a, b):
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    error = abs(b-a)/(1+abs(a))
    point = np.unravel_index(error.argmax(), error.shape)
    return dict(scaled_max=float(error[point]), absolute_max=float(abs(b-a).max()),
                index=[int(i) for i in point],
                index_order='z,y,x,component' if a.ndim == 4 else 'z,y,x')


def analyze(root):
    cpu = root/'long_windows/test_hcw_two_flowthrough_windo0/cpu'
    gpu = root/'gpu_early_prefix'
    with h5py.File(cpu/'datin/flowini3d.h5') as a, h5py.File(gpu/'datin/flowini3d.h5') as b:
        initial_equal = all(np.array_equal(a[name][...], b[name][...]) for name in a)
    assert initial_equal
    report = dict(scope='stopped numerical acceptance and targeted diagnosis; no physical/performance claim',
                  threshold=2e-10, initial_primitive_files_bitwise_equal=initial_equal,
                  cpu_case=str(cpu), gpu_case=str(gpu), samples=[], checkpoints=[])
    for actual in sorted((gpu/'outdat/new/fields').glob('segment*/step*/data.h5')):
        reference = cpu/'outdat/new/fields'/actual.relative_to(gpu/'outdat/new/fields')
        with h5py.File(reference) as a, h5py.File(actual) as b:
            np.testing.assert_array_equal(a['metadata'][...], b['metadata'][...])
            assert a.attrs['phase'] == b.attrs['phase'] == b'completed_step'
            metrics = {name: difference(a[name][...], b[name][...])
                       for name in ('density', 'velocity', 'pressure', 'temperature')}
            delta = b['density'][...][:-1]-a['density'][...][:-1]
            mean_delta = delta.mean(axis=0)
        report['samples'].append(dict(step=int(actual.parent.name[4:]), fields=metrics,
                                      density_mean_difference_maxabs=float(abs(mean_delta).max()),
                                      density_fluctuation_difference_maxabs=float(abs(delta-mean_delta).max())))
    for step in (840, 1260):
        path = Path(f'outdat/new/checkpoints/step{step:012d}/state.h5')
        with h5py.File(cpu/path) as a, h5py.File(gpu/path) as b:
            metrics = {f'q{i:04d}': difference(a[f'q{i:04d}'][...], b[f'q{i:04d}'][...])
                       for i in range(1, 12)}
        report['checkpoints'].append(dict(step=step, fields=metrics))
    report['trajectory_diagnostics'] = dict(cpu=frames(cpu), gpu=frames(gpu))
    frozen = [root/'frozen1260_cpu', root/'frozen1260_gpu']
    with h5py.File(frozen[0]/'outdat/new/resources/geometry.h5') as geometry:
        jacobian = geometry['q0004'][...].transpose(2, 1, 0)
    common = []
    for rank in (0, 1):
        suffix = f'step00000000.rk01.rank{rank:08d}.bin'
        q = [physical_state(case/'diagnostics'/f'state.pre_rhs.{suffix}', 'fp64') for case in frozen]
        local_j = jacobian[rank*32:rank*32+33]
        row = dict(rank=rank, common_initial_q_maxabs=float(abs(q[1]-q[0]).max()), physical_rhs_maxabs={})
        for name in ('budget_low_conv', 'budget_common', 'budget_low', 'full', 'limited'):
            rates = [rhs_state(case/'diagnostics'/f'state.{name}.{suffix}', 'fp64') for case in frozen]
            row['physical_rhs_maxabs'][name] = float((abs(rates[1]-rates[0])/local_j[..., None]).max())
        common.append(row)
    with h5py.File(frozen[0]/'outdat/new/checkpoints/step000000000002/state.h5') as a, \
         h5py.File(frozen[1]/'outdat/new/checkpoints/step000000000002/state.h5') as b:
        final = {f'q{i:04d}': difference(a[f'q{i:04d}'][...], b[f'q{i:04d}'][...]) for i in range(1, 12)}
    report['frozen1260_replay'] = dict(scope='same diagnostic initial primitives, two native steps, not exact restart',
                                     first_stage=common, final_fields=final)
    half = root/'half_dt_diagnostic_prefix'
    half_reports = []
    for step in (420, 840):
        a = next((gpu/'outdat/new/fields').glob(f'segment*/step{step:012d}/data.h5'))
        b = next((half/'outdat/new/fields').glob(f'segment*/step{2*step:012d}/data.h5'))
        with h5py.File(a) as original, h5py.File(b) as reduced:
            x = original['density'][...][:-1]; y = reduced['density'][...][:-1]
            half_reports.append(dict(original_steps=step, half_dt_steps=2*step,
                density_change=difference(x, y),
                original_density_span_linf=float(abs(x-x.mean(axis=0)).max()),
                half_dt_density_span_linf=float(abs(y-y.mean(axis=0)).max())))
    report['half_dt_prefix'] = dict(scope='targeted sensitivity diagnosis, not the 6720-step gate; '
        'the per-stage outlet filter has twice as many applications per physical time', cases=half_reports)
    report['trajectory_gate_passed'] = all(item['scaled_max'] <= report['threshold']
                                           for row in report['samples'] for item in row['fields'].values())
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--check-trajectory', action='store_true')
    args = parser.parse_args()
    report = analyze(args.root.resolve())
    output = args.root/'diagnosis.json'
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(output)
    print(json.dumps(dict(trajectory_gate_passed=report['trajectory_gate_passed'],
        density_scaled_errors={row['step']: row['fields']['density']['scaled_max'] for row in report['samples']},
        frozen_full_rhs_maxabs=max(row['physical_rhs_maxabs']['full']
                                  for row in report['frozen1260_replay']['first_stage']),
        frozen_final_scaled_error=max(item['scaled_max']
                                      for item in report['frozen1260_replay']['final_fields'].values())), indent=2))
    if args.check_trajectory:
        assert report['trajectory_gate_passed'], 'Current HCW trajectory exceeds the unchanged 2e-10 gate'
