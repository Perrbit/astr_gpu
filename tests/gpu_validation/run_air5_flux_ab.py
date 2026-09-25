#!/usr/bin/env python3
"""Interleaved unprofiled A/B timing of frozen AIR5 executable candidates."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

import h5py

from run_air5_performance_diagnosis import step_samples
from run_air5_sbli_preflight import ROOT


def run(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    checkpoint = args.baseline.resolve()/'outdat/flowfield.h5'
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with h5py.File(checkpoint) as f:
        step, time = int(f['nstep'][()].item()), float(f['time'][()].item())
    if any(s % 1000000 == 0 for s in range(step, step+4)):
        raise ValueError('window includes a checkpoint event')
    versions = {'baseline': args.reference.resolve(), 'candidate': args.candidate.resolve()}
    modes = {'baseline': ('off', 'baseline'), 'candidate': ('off', 'baseline')}
    if args.chemistry_candidates:
        versions = dict(baseline=args.reference.resolve(), reuse=args.candidate.resolve(),
                        packed=args.candidate.resolve(), combined=args.candidate.resolve())
        modes = dict(baseline=('off', 'baseline'), reuse=('chemistry', 'baseline'),
                     packed=('off', 'packed'), combined=('chemistry', 'packed'))
    report = dict(status='running', checkpoint_sha256=checkpoint_hash,
                  warmup_steps=1, measured_steps=3, rounds=3, dt=2e-9, cases={},
                  executable_sha256={k: hashlib.sha256(v.read_bytes()).hexdigest()
                                     for k, v in versions.items()})
    report['options'] = modes
    try:
        for ranks in (1, 2):
            for repeat in range(3):
                order = list(versions) if repeat % 2 == 0 else list(reversed(versions))
                for version in order:
                    label = f'{version}_np{ranks}'
                    out = root/f'{label}_r{repeat+1}'
                    command = [sys.executable, str(ROOT/'tests/gpu_validation/run_air5_sbli_domain_replay.py'),
                        '--baseline', str(args.baseline.resolve()), '--output', str(out),
                        '--executable', str(versions[version]), '--backend', 'gpu', '--np', str(ranks),
                        '--dt', '2e-9', '--updates', '4', '--checkpoint-interval', '1000000',
                        '--complete-step-timing', '--convection-limiter', 'symmetric_species',
                        '--diffusion-limiter', 'layered', '--primitive-reuse', modes[version][0],
                        '--chemistry-reductions', modes[version][1]]
                    with (root/f'{label}_r{repeat+1}.log').open('w') as log:
                        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                    result = json.loads((out/'result.json').read_text())
                    if result['status'] != 'bounded-replay-completed-not-physical-pass':
                        raise ValueError('replay did not complete')
                    if result['executable_sha256'] != report['executable_sha256'][version]:
                        raise ValueError('executable changed during A/B')
                    case = out/'gpu'
                    with h5py.File(case/'outdat/flowfield.h5') as f:
                        if int(f['nstep'][()].item()) != step or float(f['time'][()].item()) != time:
                            raise ValueError('advanced field written inside timing window')
                    if hashlib.sha256((case/'bakup/flowfield.h5').read_bytes()).hexdigest() != checkpoint_hash:
                        raise ValueError('input checkpoint changed')
                    log = (case/'run.log').read_text()
                    if log.count('<< ro') != 1 or list((case/'validation').glob('*.bin')):
                        raise ValueError('unexpected field output')
                    if any(p.name != 'flowfield.h5' for p in (case/'outdat').rglob('*.h5')):
                        raise ValueError('unexpected HDF5 output')
                    samples = step_samples(log, ranks, step, 4)
                    report['cases'].setdefault(label, []).append(dict(samples=samples,
                        mean_seconds=statistics.mean(samples[1:]), contract=result))
                    print(label, repeat+1, statistics.mean(samples[1:]), flush=True)
        report['statistics'] = {}
        for label, rounds in report['cases'].items():
            values = [r['mean_seconds'] for r in rounds]
            report['statistics'][label] = dict(median=statistics.median(values),
                minimum=min(values), maximum=max(values), stdev=statistics.stdev(values))
        report['status'] = 'timing-pass-not-physical-acceptance'
    except Exception as exc:
        report.update(status='failed', error=str(exc))
        raise
    finally:
        (root/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'reference', 'candidate', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--chemistry-candidates', action='store_true',
                        help='four-way baseline/reuse/packed/combined comparison')
    run(parser.parse_args())
