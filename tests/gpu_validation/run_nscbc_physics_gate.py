#!/usr/bin/env python3
"""Bounded analytic checks of production GPU NSCBC kernels, not CPU equivalence."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from compare_q_validation_snapshots import read_q_snapshot


ROOT = Path(__file__).resolve().parents[2]
MODES = ('sign', 'owner', 'transverse_y', 'transverse_z', 'inlet_entropy',
         'inlet_sup', 'inlet_sub', 'inlet_out', 'out_sub', 'out_sup',
         'out_tangent', 'top_sub', 'top_sup', 'top_in', 'out_sup_gradient',
         'inlet_out_gradient', 'top_sup_gradient', 'inlet_reverse', 'inlet_grazing',
         'out_grazing', 'top_subin', 'top_grazing', 'inlet_uniform', 'out_uniform', 'top_uniform', 'joint')


def short_case_gate(args, out, env, report):
    """Six production RK steps and an x-slab comparison of the same state."""
    final_states = []
    for np_rank in (1, 2):
        case = out / f'openshock_np{np_rank}'
        subprocess.run([sys.executable, str(ROOT / 'tests/gpu_validation/prepare_tgv_case.py'),
                        '--src-case', str(ROOT / 'examples/Shuosher'), '--dst-case', str(case),
                        '--input-name', 'input.shuosher', '--flowtype', 'openshock',
                        '--homogeneous', 'f,t,t', '--bctype', '12;22,10.333333333333333;1;1;1;1',
                        '--use-gpu', 't', '--maxstep', '5', '--feqlist', '1',
                        '--lfilter', 'f', '--diffterm', 'f', '--conschm', '543e', '--difschm', '643e',
                        '--recon-schem', '3', '--lchardecomp', 'f', '--grid', '64,8,8',
                        '--deltat', '1.d-5'], check=True, stdout=subprocess.DEVNULL)
        (case / 'datin/input.output').write_text(
            "&output\n buffer_bytes=4096, host_budget_bytes=67108864, device_budget_bytes=67108864\n/\n"
            '&checkpoint\n enabled=.false.\n/\n'
            '&volume\n enabled=.false.\n/\n&slices\n enabled=.false.\n/\n')
        run_env = dict(env, ASTR_FORCE_MPI_TOPOLOGY=f'{np_rank},1,1',
                       ASTR_VALIDATION_NSCBC_RK='1',
                       ASTR_VALIDATION_RHS_PREFIX=str(case / 'state'), ASTR_VALIDATION_RHS_STEP='5')
        command = [args.mpirun, '-np', str(np_rank)]
        if args.memcheck:
            command += ['compute-sanitizer', '--tool', 'memcheck', '--error-exitcode', '99']
        command += [str(args.exe.resolve()), 'run', 'datin/input.shuosher']
        result = subprocess.run(command, cwd=case, env=run_env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
        (case / 'run.log').write_text(result.stdout)
        entry = {'np': np_rank, 'mode': 'production_six_steps', 'returncode': result.returncode, 'pass': False}
        report['checks'].append(entry)
        if result.returncode:
            raise ValueError(f'Production short case failed: {case}')
        if args.memcheck and result.stdout.count('ERROR SUMMARY: 0 errors') != np_rank:
            raise ValueError(f'Missing per-rank clean memcheck results: {case}')
        pieces = []
        for rank in range(np_rank):
            snapshot = read_q_snapshot(case / f'state.post_update.step00000005.rk03.rank{rank:08d}.bin')
            im, jm, km, hm, nq = snapshot.header
            q = snapshot.values.reshape((im + 2*hm + 1, jm + 2*hm + 1, km + 2*hm + 1, nq), order='F')
            q = q[hm:hm+im+1, hm:hm+jm+1, hm:hm+km+1, :]
            internal = q[..., 4] - .5*np.sum(q[..., 1:4]**2, axis=-1)/q[..., 0]
            if not np.all(np.isfinite(q)) or np.min(q[..., 0]) <= 0 or np.min(internal) <= 0:
                raise ValueError(f'Invalid physical-node state: NP{np_rank} rank{rank}')
            entry['min_density'] = min(entry.get('min_density', float('inf')), float(np.min(q[..., 0])))
            entry['min_internal_energy'] = min(entry.get('min_internal_energy', float('inf')), float(np.min(internal)))
            pieces.append(q if rank == np_rank-1 else q[:-1])
        final_states.append(np.concatenate(pieces, axis=0))
        entry['pass'] = True
    error = float(np.max(np.abs(final_states[0] - final_states[1])))
    report['checks'].append({'mode': 'production_np1_np2_x', 'max_abs': error, 'pass': error <= 2e-10})
    if error > 2e-10:
        raise ValueError(f'NP1/NP2 final-state mismatch: {error}')
    print(f'PASS production six steps NP1/2 max_abs={error:.16e}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, default=ROOT / 'build_gpu_probe/bin/astr')
    parser.add_argument('--mpirun', default='mpirun')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--modes', nargs='+', choices=MODES, default=MODES)
    parser.add_argument('--np', nargs='+', type=int, choices=(1, 2), default=(1, 2))
    parser.add_argument('--memcheck', action='store_true')
    parser.add_argument('--backflow-rejection', action='store_true')
    parser.add_argument('--short-case', action='store_true')
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {'scope': 'production CUDA kernels; analytic states; no CPU oracle',
              'status': 'running', 'checks': []}
    env = dict(os.environ, OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,vader,tcp',
               OMPI_MCA_osc='pt2pt', OMPI_MCA_opal_cuda_support='false',
               OMPI_MCA_coll='^hcoll,ucc,cuda', OMPI_MCA_coll_hcoll_enable='0',
               OMPI_MCA_coll_ucc_enable='0')
    checks = [(np, mode) for np in args.np for mode in args.modes]
    if args.backflow_rejection:
        checks.extend((np, 'out_backflow') for np in args.np)
    for np, mode in checks:
        command = [args.mpirun, '-np', str(np)]
        if args.memcheck:
            command += ['compute-sanitizer', '--tool', 'memcheck', '--error-exitcode', '99']
        command += [str(args.exe.resolve()), 'test', 'bcns']
        env['ASTR_NSCBC_PROBE'] = mode
        entry = {'np': np, 'mode': mode, 'command': command, 'pass': False}
        report['checks'].append(entry)
        try:
            result = subprocess.run(command, cwd=out, env=env, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
            log = result.stdout
            entry['returncode'] = result.returncode
            if mode == 'out_backflow':
                entry['pass'] = result.returncode != 0 and 'bc22 backflow requires a full target state' in log
            else:
                entry['pass'] = result.returncode == 0 and f'NSCBC_PHYSICS_PASS {mode}' in log
            if args.memcheck:
                entry['pass'] &= log.count('ERROR SUMMARY: 0 errors') == np
            (out / f'np{np}_{mode}.log').write_text(log)
        except subprocess.TimeoutExpired as exc:
            entry['error'] = '120-second probe timeout'
            (out / f'np{np}_{mode}.log').write_bytes(exc.stdout or b'')
        if not entry['pass']:
            report['status'] = 'failed'
            (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
            raise SystemExit(f'NSCBC gate failed: NP={np} {mode}; see {out}')
        print(f'PASS NP={np} {mode}', flush=True)
    if args.short_case:
        try:
            short_case_gate(args, out, env, report)
        except (ValueError, subprocess.SubprocessError) as exc:
            report['status'] = 'failed'
            report['error'] = str(exc)
            (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
            raise SystemExit(str(exc)) from exc
    report['status'] = 'passed'
    (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
