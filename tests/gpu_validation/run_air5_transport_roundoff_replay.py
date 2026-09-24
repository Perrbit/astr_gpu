#!/usr/bin/env python3
"""Replay the AIR5 RK cancellation failure through ASTR CPU and GPU."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from air5_radau_reference import Air5RadauReference
from check_air5_mach4_error_replay import run as diagnose, state_gate, scalar
from compare_q_validation_snapshots import compare_snapshot_sets
from run_air5_sbli_preflight import ROOT


def run(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    thermo = Air5RadauReference(ROOT/'chemMech/air5_kimjo12.json')
    report = dict(status='running', diffusion_limiter=args.diffusion_limiter, states={})
    try:
        # Run the previously failing CPU two-half-step case first.
        for name, backend, dt, updates in [('cpu_dt1', 'cpu', 1e-9, 2),
                ('gpu_dt1', 'gpu', 1e-9, 2), ('cpu_dt2', 'cpu', 2e-9, 1),
                ('gpu_dt2', 'gpu', 2e-9, 1)]:
            command = [sys.executable, str(ROOT/'tests/gpu_validation/run_air5_sbli_domain_replay.py'),
                '--baseline', str(args.baseline.resolve()), '--output', str(root/name),
                '--executable', str(args.executable.resolve()), '--backend', backend,
                '--dt', str(dt), '--updates', str(updates), '--snapshot-step', '1000',
                '--convection-limiter', 'species_budget', '--diffusion-limiter', args.diffusion_limiter]
            if updates == 2:
                command += ['--snapshot-step-secondary', '1001']
            subprocess.run(command, check=True)
            report['states'][name] = state_gate(root/name/backend, thermo)
            if args.diffusion_limiter == 'layered':
                coefficients = {}
                for channel in ('species', 'energy'):
                    files = sorted((root/name/backend/'validation').glob(
                        f'air5.diffusion_{channel}_ratio.*.bin'))
                    if len(files) != 6*updates:
                        raise ValueError(f'incomplete layered coefficient snapshots: {name}/{channel}')
                    minimum, limited = 1., 0
                    for filename in files:
                        values = scalar(filename)
                        if np.any((values < 0) | (values > 1)):
                            raise ValueError(f'invalid layered coefficients: {filename}')
                        minimum = min(minimum, float(values.min()))
                        limited += int(np.count_nonzero(values < 1-1e-14))
                        if backend == 'gpu':
                            cpu_file = root/name.replace('gpu_', 'cpu_', 1)/'cpu/validation'/filename.name
                            np.testing.assert_allclose(values, scalar(cpu_file), atol=1e-10, rtol=1e-10)
                    coefficients[channel] = dict(files=len(files), minimum=minimum,
                        limited_node_stage_samples_including_shared_nodes=limited)
                report['states'][name]['layered_coefficients'] = coefficients
            if backend == 'gpu':
                suffix = name.removeprefix('gpu_')
                result = compare_snapshot_sets(root/('cpu_'+suffix)/'cpu/validation/air5',
                    root/name/'gpu/validation/air5',
                    ('post_chemistry', 'pre_rhs', 'post_update', 'post_transport'),
                    atol=1e-9, rtol=1e-10, active_only=True)
                if not result.passed:
                    raise ValueError(f'CPU/GPU phase comparison failed: {suffix}')
                report['comparison_'+suffix] = dict(files=result.file_count,
                    max_abs=result.max_abs, max_scaled=result.max_scaled)
        diagnosis = diagnose(root)
        report['endpoint_u_max_abs_by_rank'] = [x['endpoint_u_max_abs'] for x in diagnosis['timestep']]
        report['status'] = 'roundoff-replay-pass-not-physical-pass'
    except Exception as exc:
        report.update(status='failed', error=str(exc))
        raise
    finally:
        (root/'roundoff_summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--diffusion-limiter', choices=('full_state', 'layered'), default='full_state')
    parser.add_argument('--executable', type=Path,
        default=ROOT/'tests/gpu_validation/out/c4_cuda_build_4/bin/astr')
    run(parser.parse_args())
