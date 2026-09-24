#!/usr/bin/env python3
"""Bounded ASTR replays for the optional species-budget convection limiter."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from air5_radau_reference import Air5RadauReference
from check_air5_mach4_error_replay import run as diagnose, state_gate
from compare_q_validation_snapshots import compare_snapshot_sets
from run_air5_sbli_domain_replay import run as replay
from run_air5_sbli_preflight import ROOT


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    thermo = Air5RadauReference(ROOT/'chemMech/air5_kimjo12.json')
    report = dict(status='running-not-physical-pass', state_gates={})
    for name, backend, dt, updates, mode in [
        ('gpu_legacy', 'gpu', 2e-9, 1, 'full_state'),
        ('gpu_dt2', 'gpu', 2e-9, 1, 'species_budget'),
        ('cpu_dt2', 'cpu', 2e-9, 1, 'species_budget'),
        ('gpu_dt1', 'gpu', 1e-9, 2, 'species_budget'),
    ]:
        replay(SimpleNamespace(baseline=args.baseline, output=out/name,
            executable=args.executable, dt=dt, updates=updates, backend=backend,
            convection_limiter=mode, snapshot_step=1000,
            snapshot_step_secondary=1001 if updates==2 else None))
        report['state_gates'][name] = state_gate(out/name/backend, thermo)
        if name == 'gpu_legacy':
            result = compare_snapshot_sets(args.reference/'validation/air5',
                out/name/backend/'validation/air5',
                ('pre_rhs', 'post_update', 'post_transport', 'post_chemistry'),
                atol=1e-9, rtol=1e-10, active_only=True)
            report['legacy_regression'] = dict(passed=result.passed,
                files=result.file_count, max_abs=result.max_abs, max_scaled=result.max_scaled)
            if not result.passed:
                raise ValueError('default full-state regression failed')
        (out/'probe_summary.json').write_text(json.dumps(report, indent=2)+'\n')
    diagnostic = diagnose(out)
    report.update(status='bounded-candidate-pass-not-promoted', cpu_gpu=diagnostic['cpu_gpu'],
                  endpoint_u_max_abs_by_rank=[row['endpoint_u_max_abs'] for row in diagnostic['timestep']])
    (out/'probe_summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True,
                        help='archived full-state one-step restart case with validation snapshots')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--executable', type=Path,
                        default=ROOT/'tests/gpu_validation/out/c4_cuda_build_4/bin/astr')
    run(parser.parse_args())
