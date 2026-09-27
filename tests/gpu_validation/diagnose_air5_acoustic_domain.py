#!/usr/bin/env python3
"""Short paired-domain diagnostic; not an acoustic acceptance gate."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from check_air5_c5_hbl import _active_array
from run_air5_characteristic_acoustic import prepare
from run_air5_sbli_preflight import ROOT, environment, set_value


def check_zero_species(case):
    files = sorted((case/'validation').glob('air5.pre_rhs.*.bin'))
    files += sorted((case/'validation').glob('air5.post_update.*.bin'))
    if not files:
        raise ValueError('missing sampled states for zero-species check')
    for path in files:
        state = _active_array(path)
        if not np.all(np.isfinite(state)):
            raise ValueError(f'nonfinite sampled state: {path}')
        if np.any(state[..., 7:10] != 0.):
            raise ValueError(f'absent species created: {path}')
    return dict(passed=True, snapshots=len(files), criterion='N/O/NO exactly zero')


def run(output, updates, sample, gpu, limiter, diagnostic_period, require_zero=False):
    if updates<2 or not 0<=sample<updates-1 or diagnostic_period<0:
        raise ValueError('invalid diagnostic update/sample configuration')
    output.mkdir(parents=True, exist_ok=False)
    for name, extended in [('baseline', False), ('extended', True)]:
        case = output/name
        meta = prepare(case, ROOT/'build_gpu_probe/bin/astr', extended=extended,
                       use_gpu=gpu, short_updates=updates)
        meta['environment']['ASTR_VALIDATION_RHS_STEP_SECONDARY'] = str(sample)
        meta['environment']['ASTR_AIR5_CONVECTION_LIMITER'] = limiter
        meta['scope'] = 'paired-domain diagnosis only; no physical admission'
        (case/'gate.json').write_text(json.dumps(meta, indent=2)+'\n')
        if diagnostic_period:
            set_value(case/'datin/controller',
                      'maxstep,feqchkpt,feqwsequ,feqslice,feqlist,feqavg',
                      f'{updates-1},{diagnostic_period},10,50,50,50')
        env = environment('1,1,1')
        env.update(meta['environment'])
        print(f'Running {case}', flush=True)
        with (case/'run.log').open('w') as log:
            subprocess.run([shutil.which('mpirun'), '-np', '1', meta['executable'],
                            'run', 'datin/input.air5_c4'], cwd=case, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
        text = (case/'run.log').read_text()
        if 'The job is done!' not in text or 'ieee_invalid' in text.lower():
            raise RuntimeError(f'unclean diagnostic completion: {case}')
        if require_zero:
            result = check_zero_species(case)
            (case/'zero_species.json').write_text(json.dumps(result, indent=2)+'\n')
            print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=100)
    parser.add_argument('--sample', type=int, default=30)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--limiter', choices=['full_state', 'symmetric_species'], default='full_state')
    parser.add_argument('--diagnostic-period', type=int, default=0)
    parser.add_argument('--require-zero-species', action='store_true')
    args = parser.parse_args()
    run(args.output.resolve(), args.updates, args.sample, not args.cpu,
        args.limiter, args.diagnostic_period, args.require_zero_species)
