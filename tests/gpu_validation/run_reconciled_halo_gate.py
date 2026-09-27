#!/usr/bin/env python3
"""Single-species TGV regression for reconciled CPU/GPU solution halos."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from check_air5_c5_hbl import _phase_files
from check_reconciled_solution_halos import check
from compare_q_validation_snapshots import read_q_snapshot
from run_air5_sbli_preflight import ROOT, environment


def active(path):
    s = read_q_snapshot(path)
    im, jm, km, hm, nq = s.header
    q = s.values.reshape((im+2*hm+1, jm+2*hm+1, km+2*hm+1, nq), order='F')
    return q[hm:hm+im+1, hm:hm+jm+1, hm:hm+km+1]


def run(output, executable, mpiexec):
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for topology in ('1,1,1', '2,1,1', '1,2,1', '1,1,2'):
        reference = None
        variants = [('cpu', False, 'pageable'), ('gpu', True, 'pageable')]
        if topology == '2,1,1':
            variants.append(('pipeline', True, 'pinned-pipeline'))
        for name, gpu, transport in variants:
            case = output/topology.replace(',', 'x')/name
            subprocess.run([sys.executable, str(ROOT/'tests/gpu_validation/prepare_tgv_case.py'),
                '--src-case', str(ROOT/'examples/Taylor_Green_Vortex'), '--dst-case', str(case),
                '--use-gpu', 't' if gpu else 'f', '--grid', '32,32,32', '--maxstep', '2',
                '--feqchkpt', '2', '--lfilter', 't', '--diffterm', 't', '--scheme', '643e'], check=True)
            (case/'validation').mkdir()
            env = environment(topology)
            env.update(ASTR_VALIDATION_RHS_PREFIX='validation/tgv', ASTR_VALIDATION_RHS_STEP='2',
                       ASTR_GPU_HALO_TRANSPORT=transport, ASTR_GPU_SYNC_MODE='explicit')
            np_ = str(int(np.prod([int(n) for n in topology.split(',')])))
            with (case/'run.log').open('w') as log:
                subprocess.run([str(mpiexec), '--oversubscribe', '-np', np_, str(executable),
                                'run', 'datin/input.tgv'], cwd=case, env=env, stdout=log,
                               stderr=subprocess.STDOUT, timeout=180, check=True)
            halos = [check(case, 2, stage, (0, 1, 2), 'tgv') for stage in (1, 2, 3)]
            if not all(row['passed'] for row in halos):
                raise ValueError(f'{case}: stale RHS-input halo')
            states = {}
            for stage in (1, 2, 3):
                for rank, path in _phase_files(case/'validation/tgv', 'post_update', stage, 2).items():
                    states[stage, rank] = active(path)
            if not states or not all(np.isfinite(q).all() for q in states.values()):
                raise ValueError(f'{case}: missing or nonfinite state')
            if reference is None:
                reference = states
            if states.keys() != reference.keys():
                raise ValueError('phase/rank mismatch')
            error = max(float(np.max(abs(q-reference[key])/np.maximum(1., abs(reference[key]))))
                        for key, q in states.items())
            if error > 1e-10:
                raise ValueError(f'{case}: CPU/GPU mismatch {error}')
            results.append(dict(topology=topology, variant=name, halo_checks=halos, max_scaled=error))
            (output/'result.json').write_text(json.dumps(dict(passed=False, completed=results), indent=2)+'\n')
    (output/'result.json').write_text(json.dumps(dict(passed=True, completed=results), indent=2)+'\n')
    print('reconciled TGV halo gate passed:', len(results), 'runs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--executable', type=Path, default=ROOT/'build_gpu_probe/bin/astr')
    parser.add_argument('--mpiexec', type=Path, required=True)
    args = parser.parse_args()
    run(args.output.resolve(), args.executable.resolve(), args.mpiexec.absolute())
