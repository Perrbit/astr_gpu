#!/usr/bin/env python3
"""Unprofiled complete-step timings for the fixed AIR5 restart workload."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

import h5py

from run_air5_sbli_preflight import ROOT


def step_samples(text, ranks, start, updates):
    records = {}
    for line in text.splitlines():
        if not line.startswith('ASTR_COMPLETE_STEP_TIMING '):
            continue
        parts = line.split()
        if len(parts) != 4:
            raise ValueError('malformed complete-step timing record')
        step, rank = map(int, parts[1:3])
        duration = float(parts[3])
        if not start <= step < start+updates or not 0 <= rank < ranks:
            raise ValueError('unexpected timing step/rank')
        if not math.isfinite(duration) or duration <= 0 or (step, rank) in records:
            raise ValueError('invalid or duplicate timing')
        records[step, rank] = duration
    if len(records) != ranks*updates:
        raise ValueError('incomplete timing records')
    return [max(records[step, rank] for rank in range(ranks)) for step in range(start, start+updates)]


def run(args):
    if args.repeats < 3 or args.warmup < 1 or args.steps < 1:
        raise ValueError('require >=3 repeats, >=1 warmup and >=1 measured step')
    args.output.mkdir(parents=True, exist_ok=True)
    baseline = args.baseline.resolve()
    checkpoint = baseline/'outdat/flowfield.h5'
    original_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with h5py.File(checkpoint) as h5:
        start = int(h5['nstep'][()].item())
        start_time = float(h5['time'][()].item())
    updates = args.warmup+args.steps
    if any(step % 1000000 == 0 for step in range(start, start+updates)):
        raise ValueError('timing window includes a checkpoint event')
    summary = dict(status='running', cases={}, warmup=args.warmup, measured_steps=args.steps,
                   repeats=args.repeats, dt=2e-9, checkpoint_sha256=original_hash,
                   scope='Mach4 flat plate without incident shock; complete step, not transport RK only')
    try:
        for label in args.cases:
            backend, n = label.split('_np')
            ranks = int(n)
            rounds = []
            for repeat in range(args.repeats):
                output = args.output/f'{label}_r{repeat+1}'
                command = [sys.executable, str(ROOT/'tests/gpu_validation/run_air5_sbli_domain_replay.py'),
                           '--baseline', str(baseline), '--output', str(output), '--dt', '2e-9',
                           '--updates', str(updates), '--timeout-seconds', '1800',
                           '--backend', backend, '--np', str(ranks), '--checkpoint-interval', '1000000',
                           '--convection-limiter', 'symmetric_species', '--diffusion-limiter', 'layered',
                           '--complete-step-timing', '--executable', str(args.executable.resolve())]
                subprocess.run(command, check=True)
                result = json.loads((output/'result.json').read_text())
                if result['status'] != 'bounded-replay-completed-not-physical-pass':
                    raise ValueError('invalid benchmark state/completion')
                case = output/backend
                # flowinit backs up and rewrites the initial checkpoint before steploop.
                # That startup output is excluded, but no advanced checkpoint is allowed.
                if hashlib.sha256((case/'bakup/flowfield.h5').read_bytes()).hexdigest() != original_hash:
                    raise ValueError('startup checkpoint backup does not match the input')
                with h5py.File(case/'outdat/flowfield.h5') as h5:
                    if int(h5['nstep'][()].item()) != start or float(h5['time'][()].item()) != start_time:
                        raise ValueError('advanced checkpoint was written during timing')
                if list((case/'validation').glob('*.bin')):
                    raise ValueError('large validation snapshots appeared during timing')
                log = (case/'run.log').read_text()
                if log.count('<< ro') != 1:
                    raise ValueError('expected only the initial checkpoint write')
                if any(p.name != 'flowfield.h5' for p in (case/'outdat').rglob('*.h5')):
                    raise ValueError('unexpected HDF5 output appeared during timing')
                samples = step_samples(log, ranks, start, updates)
                rounds.append(dict(all_steps=samples, measured=samples[args.warmup:],
                                   seconds_per_step=statistics.mean(samples[args.warmup:])))
                summary['cases'][label] = dict(rounds=rounds)
                print(label, repeat+1, rounds[-1]['seconds_per_step'], flush=True)
            values = [r['seconds_per_step'] for r in rounds]
            summary['cases'][label].update(median_seconds=statistics.median(values),
                min_seconds=min(values), max_seconds=max(values), stdev_seconds=statistics.stdev(values))
        if all(k in summary['cases'] for k in ('cpu_np1', 'gpu_np1', 'gpu_np2')):
            cpu, gpu1, gpu2 = [summary['cases'][k]['median_seconds'] for k in ('cpu_np1', 'gpu_np1', 'gpu_np2')]
            summary.update(gpu1_speedup_over_cpu=cpu/gpu1, gpu2_speedup_over_cpu=cpu/gpu2,
                           two_gpu_speedup=gpu1/gpu2, two_gpu_efficiency=gpu1/(2*gpu2))
        summary['status'] = 'timing-collected-not-full-diagnosis'
    except Exception as exc:
        summary.update(status='failed', error=str(exc))
        raise
    finally:
        (args.output/'timing_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--executable', type=Path, default=ROOT/'build_gpu_probe/bin/astr')
    parser.add_argument('--cases', nargs='+', choices=('cpu_np1', 'gpu_np1', 'gpu_np2'),
                        default=['cpu_np1', 'gpu_np1', 'gpu_np2'])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--steps', type=int, default=3)
    run(parser.parse_args())
