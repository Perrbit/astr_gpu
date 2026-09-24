#!/usr/bin/env python3
"""Bounded checkpoint replay for domain failures; never a physical acceptance gate."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import time

import h5py

from run_air5_sbli_preflight import ROOT, environment, set_value


def run(args):
    if not math.isfinite(args.dt) or args.dt <= 0 or args.updates < 1:
        raise ValueError('require positive dt and updates')
    baseline = args.baseline.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    backend = getattr(args, 'backend', 'gpu')
    case = out/backend
    shutil.copytree(baseline/'datin', case/'datin')
    (case/'outdat').mkdir()
    (case/'validation').mkdir()
    for name in ('flowfield.h5', 'auxiliary.txt'):
        shutil.copy2(baseline/'outdat'/name, case/'outdat'/name)
    with h5py.File(case/'outdat/flowfield.h5') as checkpoint:
        start_step = int(checkpoint['nstep'][()].item())
        start_time = float(checkpoint['time'][()].item())
    exe = out/'astr'
    shutil.copy2(args.executable.resolve(), exe)
    maximum = start_step + args.updates - 1
    set_value(case/'datin/input.air5_c4', 'lrestar', 't')
    set_value(case/'datin/input.air5_c4',
              'nondimen,diffterm,lfilter,lreadgrid,lfftz,limmbou,ltimrpt,lcomb',
              'f,t,f,f,f,f,t,t,' + ('t' if backend == 'gpu' else 'f'))
    set_value(case/'datin/controller', 'deltat', f'{args.dt:.17e}')
    set_value(case/'datin/controller',
              'maxstep,feqchkpt,feqwsequ,feqslice,feqlist,feqavg',
              f'{maximum},10,1000000,1000000,1,1000000')
    env = environment('2,1,1')
    limiter = getattr(args, 'convection_limiter', 'full_state')
    env['ASTR_AIR5_CONVECTION_LIMITER'] = limiter
    if args.snapshot_step is not None:
        env.update(ASTR_VALIDATION_RHS_PREFIX='validation/air5',
                   ASTR_VALIDATION_RHS_STEP=str(args.snapshot_step))
    secondary = getattr(args, 'snapshot_step_secondary', None)
    if secondary is not None:
        if args.snapshot_step is None:
            raise ValueError('secondary snapshot requires a primary snapshot step')
        env['ASTR_VALIDATION_RHS_STEP_SECONDARY'] = str(secondary)
    contract = dict(start_step=start_step, start_time=start_time, dt=args.dt,
                    updates=args.updates, target_time=start_time+args.dt*args.updates,
                    baseline=str(baseline), executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                    checkpoint_sha256=hashlib.sha256((case/'outdat/flowfield.h5').read_bytes()).hexdigest(),
                    topology='2,1,1', backend=backend, convection_limiter=limiter,
                    snapshot_step=args.snapshot_step,
                    snapshot_step_secondary=secondary)
    (out/'contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    command = ['timeout', '--kill-after=10s', '300s', shutil.which('mpirun'),
               '--oversubscribe', '-np', '2', str(exe), 'run', 'datin/input.air5_c4']
    start = time.monotonic()
    with (case/'run.log').open('w') as log:
        process = subprocess.run(command, cwd=case, env=env, stdout=log, stderr=subprocess.STDOUT)
    log = (case/'run.log').read_text()
    failure = re.search(r'AIR5_DIFFUSION_STATE_FAILURE rank/step/stage=(\d+)\s+(\d+)\s+(\d+)', log)
    result = dict(returncode=process.returncode, elapsed_seconds=time.monotonic()-start,
                  status='unexpected-failure', **contract)
    if process.returncode == 4 and failure:
        rank, step, stage = map(int, failure.groups())
        result.update(status='domain-failure-not-pass', rank=rank, step=step, stage=stage,
                      failed_step_start_time=start_time+(step-start_step)*args.dt,
                      diagnostics=[line for line in log.splitlines() if 'AIR5_DIFFUSION_STATE_FAILURE' in line])
    elif process.returncode == 0 and 'The job is done!' in log:
        result['status'] = 'bounded-replay-completed-not-physical-pass'
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    if result['status'] == 'unexpected-failure':
        raise RuntimeError(f'unexpected replay failure; inspect {case / "run.log"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dt', type=float, required=True)
    parser.add_argument('--updates', type=int, required=True)
    parser.add_argument('--snapshot-step', type=int)
    parser.add_argument('--snapshot-step-secondary', type=int)
    parser.add_argument('--backend', choices=('cpu', 'gpu'), default='gpu')
    parser.add_argument('--convection-limiter', choices=('full_state', 'species_budget'),
                        default='full_state')
    parser.add_argument('--executable', type=Path,
                        default=ROOT/'tests/gpu_validation/out/c4_cuda_build_4/bin/astr')
    run(parser.parse_args())
