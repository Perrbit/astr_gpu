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
    timeout_seconds = getattr(args, 'timeout_seconds', 300)
    if timeout_seconds < 1:
        raise ValueError('require positive timeout seconds')
    baseline = args.baseline.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    backend = getattr(args, 'backend', 'gpu')
    ranks = getattr(args, 'np', 2)
    checkpoint_interval = getattr(args, 'checkpoint_interval', 10)
    if ranks not in (1, 2) or checkpoint_interval < 1:
        raise ValueError('require one/two ranks and a positive checkpoint interval')
    memcheck = getattr(args, 'memcheck', False)
    nsys_profile = getattr(args, 'nsys', False)
    ncu_kernel = getattr(args, 'ncu_kernel', None)
    if ncu_kernel and (backend != 'gpu' or ranks != 1 or memcheck or nsys_profile):
        raise ValueError('Nsight Compute requires a separate single-GPU run')
    if nsys_profile and (backend != 'gpu' or memcheck):
        raise ValueError('Nsight requires GPU and a separate run from memcheck')
    if memcheck and backend != 'gpu':
        raise ValueError('--memcheck requires the GPU backend')
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
              f'{maximum},{checkpoint_interval},1000000,1000000,1,1000000')
    env = environment(f'{ranks},1,1')
    if getattr(args, 'complete_step_timing', False):
        env['ASTR_COMPLETE_STEP_TIMING'] = '1'
    limiter = getattr(args, 'convection_limiter', 'full_state')
    env['ASTR_AIR5_CONVECTION_LIMITER'] = limiter
    diffusion_limiter = getattr(args, 'diffusion_limiter', 'full_state')
    env['ASTR_AIR5_DIFFUSION_LIMITER'] = diffusion_limiter
    if getattr(args, 'symmetric_face_probe', False):
        if limiter != 'symmetric_species' or args.snapshot_step is None:
            raise ValueError('symmetric face probe requires the symmetric limiter and snapshots')
        env['ASTR_AIR5_SYMMETRIC_FACE_PROBE'] = '1'
    convection_probe = getattr(args, 'convection_probe_node', None)
    if convection_probe is not None:
        if limiter in ('consistent_species', 'symmetric_species'):
            raise ValueError('single-pass convection probe cannot describe the selected limiter')
        values = [int(value) for value in convection_probe.split(',')]
        if backend != 'cpu' or args.snapshot_step is None or len(values) != 4 or min(values) < 0:
            raise ValueError('convection probe requires CPU snapshots and nonnegative rank,i,j,k')
        if values[0] >= ranks:
            raise ValueError('convection probe rank is outside replay')
        env['ASTR_AIR5_CONVECTION_PROBE_NODE'] = convection_probe
    probe_node = getattr(args, 'diffusion_probe_node', None)
    if probe_node is not None:
        if diffusion_limiter == 'layered':
            raise ValueError('the full-state diffusion probe cannot describe layered fluxes')
        values = [int(value) for value in probe_node.split(',')]
        if backend != 'cpu' or args.snapshot_step is None or len(values) != 4 or min(values) < 0:
            raise ValueError('diffusion probe requires CPU snapshots and nonnegative rank,i,j,k')
        if values[0] >= ranks:
            raise ValueError('diffusion probe rank is outside replay')
        env['ASTR_AIR5_DIFFUSION_PROBE_NODE'] = probe_node
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
                    topology=f'{ranks},1,1', backend=backend, convection_limiter=limiter,
                    complete_step_timing=getattr(args, 'complete_step_timing', False),
                    checkpoint_interval=checkpoint_interval,
                    memcheck=memcheck,
                    nsys_profile=nsys_profile,
                    ncu_kernel=ncu_kernel,
                    symmetric_face_probe=getattr(args, 'symmetric_face_probe', False),
                    diffusion_limiter=diffusion_limiter,
                    diffusion_probe_node=probe_node,
                    convection_probe_node=convection_probe,
                    snapshot_step=args.snapshot_step,
                    snapshot_step_secondary=secondary, timeout_seconds=timeout_seconds)
    (out/'contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    instrument = []
    if memcheck:
        sanitizer = shutil.which('compute-sanitizer')
        if sanitizer is None:
            raise RuntimeError('compute-sanitizer is required for --memcheck')
        instrument = [sanitizer, '--tool', 'memcheck', '--error-exitcode', '99',
                      '--log-file', 'validation/memcheck.%p.log']
    if ncu_kernel:
        profiler = shutil.which('ncu')
        if profiler is None:
            raise RuntimeError('ncu is required for --ncu-kernel')
        instrument = [profiler, '--set', 'full', '--kernel-name', f'regex:{ncu_kernel}',
                      '--launch-count', '1', '--export', str(out/'kernel')]
    command = [shutil.which('mpirun'), '--oversubscribe', '-np', str(ranks),
               *instrument, str(exe), 'run', 'datin/input.air5_c4']
    if nsys_profile:
        profiler = shutil.which('nsys')
        if profiler is None:
            raise RuntimeError('nsys is required for --nsys')
        command = [profiler, 'profile', '--trace=cuda,mpi', '--mpi-impl=openmpi',
                   '--sample=none', '--cpuctxsw=none', '--force-overwrite=false',
                   '--output', str(out/'profile'), *command]
    command = ['timeout', '--kill-after=10s', f'{timeout_seconds}s', *command]
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
        if memcheck:
            logs = sorted((case/'validation').glob('memcheck.*.log'))
            clean = len(logs) == ranks and all('ERROR SUMMARY: 0 errors' in p.read_text() for p in logs)
            result['memcheck_rank_logs'] = len(logs)
            result['memcheck_clean'] = clean
            if not clean:
                result['status'] = 'unexpected-failure'
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
    parser.add_argument('--timeout-seconds', type=int, default=300)
    parser.add_argument('--snapshot-step', type=int)
    parser.add_argument('--snapshot-step-secondary', type=int)
    parser.add_argument('--backend', choices=('cpu', 'gpu'), default='gpu')
    parser.add_argument('--np', type=int, choices=(1, 2), default=2)
    parser.add_argument('--checkpoint-interval', type=int, default=10)
    parser.add_argument('--complete-step-timing', action='store_true')
    parser.add_argument('--memcheck', action='store_true')
    parser.add_argument('--nsys', action='store_true', help='separate CUDA/MPI timeline; not timing evidence')
    parser.add_argument('--ncu-kernel', help='single-GPU kernel regex; profile one matching launch separately')
    parser.add_argument('--symmetric-face-probe', action='store_true')
    parser.add_argument('--diffusion-probe-node', help='CPU read-only probe: rank,i,j,k')
    parser.add_argument('--convection-probe-node', help='CPU read-only convection probe: rank,i,j,k')
    parser.add_argument('--convection-limiter',
                        choices=('full_state', 'species_budget', 'consistent_species', 'symmetric_species'),
                        default='full_state')
    parser.add_argument('--diffusion-limiter', choices=('full_state', 'layered'), default='full_state')
    parser.add_argument('--executable', type=Path,
                        default=ROOT/'tests/gpu_validation/out/c4_cuda_build_4/bin/astr')
    run(parser.parse_args())
