#!/usr/bin/env python3
"""Compare native small-grid q snapshots with CP9 instrumentation off/on."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from compare_q_validation_snapshots import read_q_snapshot,comparison_values


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    case=args.case.resolve()
    launch=json.loads((case/'launch.json').read_text())
    results=[]
    for label in ('reference','instrumented'):
        if args.check_only:
            break
        env={k:v for k,v in os.environ.items() if not k.startswith(('ASTR_','OMPI_MCA_','UCX_'))}
        env.update(launch['environment'])
        env.update(ASTR_VALIDATION_RHS_PREFIX=f'diagnostics/{label}',ASTR_VALIDATION_RHS_STEP='3')
        command=launch['command'].copy()
        if label=='reference':
            command[-3]=str(args.reference.resolve(strict=True))
            env.pop('ASTR_GPU_PHASE_TIMING',None)
            env.pop('ASTR_VALIDATION_RK_STATE',None)
        (case/'diagnostics').mkdir(exist_ok=True)
        with (case/f'{label}.log').open('w') as stream:
            subprocess.run(command,cwd=case,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    files=sorted(path for path in (case/'diagnostics').glob('reference.*.bin')
                 if path.name.startswith(('reference.pre_rhs.step','reference.post_update.step')))
    if not files:
        raise ValueError('missing native state snapshots')
    for reference in files:
        candidate=reference.with_name(reference.name.replace('reference.','instrumented.',1))
        a=read_q_snapshot(reference)
        b=read_q_snapshot(candidate)
        if a.header!=b.header or not np.array_equal(comparison_values(a,True),comparison_values(b,True)):
            raise ValueError(f'instrumentation changed authoritative snapshot: {reference.name}')
        results.append(reference.name)
    (case/'transparency.json').write_text(json.dumps(dict(status='pass',bitwise_snapshots=results),indent=2)+'\n')
    print(f'PASS: {len(results)} bitwise q snapshots',flush=True)


if __name__=='__main__':
    main()
