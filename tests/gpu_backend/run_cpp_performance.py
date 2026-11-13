"""Pure completed-RK windows of ordinary ASTR, without large output/readbacks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np

from run_cpp_tgv_gate import prepare_case, compare, logged_steps


def summarize(directory, ranks, warmup, window_steps, windows):
    text = (directory/'run.log').read_text()
    count = warmup+window_steps*windows
    rows = re.findall(r'^ASTR_GPU_RANK_RK_TIMING\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)',
                      text, re.MULTILINE)
    times = np.full((ranks, count), np.nan)
    for rank, step, preparation, integration, total in rows:
        rank, step = int(rank), int(step)
        if not (0 <= rank < ranks and 0 <= step < count) or np.isfinite(times[rank, step]):
            raise AssertionError('duplicate or invalid per-rank RK timing identity')
        times[rank, step] = float(total)
        if abs(float(preparation)+float(integration)-float(total)) > 1e-10*max(1, float(total)):
            raise AssertionError('RK timing components do not sum to the total')
    if not np.isfinite(times).all() or not (times > 0).all():
        raise AssertionError('incomplete positive RK timing samples')
    # Maximum accumulated rank time in each window, not the sum of per-step
    # maxima (which can belong to different ranks).
    per_rank = times[:, warmup:].reshape(ranks, windows, window_steps).sum(axis=2)
    samples = per_rank.max(axis=0)/window_steps
    result = dict(window_seconds_per_step=samples.tolist(), rank_window_seconds=per_rank.tolist(),
                  median_seconds_per_step=float(np.median(samples)),
                  best_seconds_per_step=float(np.min(samples)))
    halo = re.findall(r'^ASTR_CPP_HALO_TIMING\s+(\d+)\s+(\d+)\s+(.+)$', text, re.MULTILINE)
    if halo:
        entries = np.full((ranks, count, 6), np.nan)
        for rank, step, values in halo:
            entries[int(rank), int(step)] = list(map(float, values.split()))
        if not np.isfinite(entries).all() or (entries < 0).any():
            raise AssertionError('incomplete nonnegative halo timing/copy counters')
        result['halo_rank_mean_per_step'] = entries[:, warmup:].mean(axis=1).tolist()
    if list((directory/'outdat/new').rglob('*.h5')) or list((directory/'outdat').glob('*.bin')):
        raise AssertionError('field/checkpoint/test-readback output present in timing run')
    logged_steps(directory, count)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--old-binary', type=Path)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--mpiexec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--grid', type=int, default=128)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--window-steps', type=int, default=5)
    parser.add_argument('--windows', type=int, default=5)
    parser.add_argument('--transport', choices=['pinned', 'device'], default='pinned')
    parser.add_argument('--workspace', choices=['scalar', 'full'], default='scalar')
    parser.add_argument('--sync', choices=['explicit', 'dependency'], default='dependency')
    parser.add_argument('--include-quad', action='store_true')
    args = parser.parse_args()
    if args.windows < 1 or args.warmup < 1 or args.window_steps < 1:
        parser.error('positive measured windows, warmup and window length are required')
    if args.old_binary and args.transport != 'pinned':
        parser.error('this old/new baseline uses the common pinned transport')
    if args.old_binary and args.sync != 'explicit':
        parser.error('frozen CUDA Fortran pinned transport requires explicit synchronization; select --sync explicit')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = args.reference.resolve()/'evidence/standard_tgv_explicit10/cpu_np1_keep_x/datin'
    environment = os.environ.copy()
    for name in list(environment):
        if name.startswith('ASTR_VALIDATION_'):
            del environment[name]
    environment.update(OMP_NUM_THREADS='1', OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,vader,tcp',
                       OMPI_MCA_osc='pt2pt', OMPI_MCA_coll='^cuda,hcoll,ucc', OMPI_MCA_opal_cuda_support='0',
                       ASTR_GPU_FILTER_WORKSPACE=args.workspace, ASTR_GPU_HALO_TRANSPORT=args.transport,
                       ASTR_GPU_SYNC_MODE=args.sync, ASTR_GPU_RK_TIMING='1', ASTR_GPU_RANK_RK_TIMING='1',
                       ASTR_GPU_STATISTICS_INTERVAL='1', ASTR_GPU_BENCHMARK_NO_FIELD_IO='1')
    if args.transport == 'device':
        environment.update(OMPI_MCA_pml='ucx', OMPI_MCA_opal_cuda_support='1')
    topology = [(1, (1, 1, 1)), (2, (2, 1, 1)), (2, (1, 2, 1)), (2, (1, 1, 2))]
    if args.include_quad:
        topology.append((4, (2, 2, 1)))
    binaries = {'cpp': args.binary.resolve()}
    if args.old_binary:
        binaries['cuda_fortran'] = args.old_binary.resolve()
    report = dict(grid=args.grid, dt=2e-4, warmup_steps=args.warmup, window_steps=args.window_steps,
                  windows=args.windows, transport=args.transport, sync=args.sync, filter_workspace=args.workspace,
                  scalar_statistics_interval=1, results={})
    report['executables'] = {b: dict(path=str(p), sha256=hashlib.file_digest(p.open('rb'), 'sha256').hexdigest())
                             for b, p in binaries.items()}
    for ranks, layout in topology:
        label = f'np{ranks}_'+''.join(map(str, layout))
        for backend, binary in binaries.items():
            directory = root/(backend+'_'+label)
            prepare_case(directory, source, True, True, False, grid=args.grid,
                         steps=args.warmup+args.window_steps*args.windows, dt=2e-4)
            env = environment | {'ASTR_FORCE_MPI_TOPOLOGY': ','.join(map(str, layout))}
            command = [str(args.mpiexec.resolve()), '--bind-to', 'none', '-n', str(ranks), str(binary), 'run']
            with (directory/'run.log').open('w') as log:
                subprocess.run(command, cwd=directory, env=env, stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            result = summarize(directory, ranks, args.warmup, args.window_steps, args.windows)
            result['command'] = command
            report['results'][backend+'_'+label] = result
            if backend == 'cuda_fortran':
                cpp = root/('cpp_'+label)
                result['scalar_comparison'] = compare(np.loadtxt(directory/'flowstate.dat', skiprows=1),
                                                      np.loadtxt(cpp/'flowstate.dat', skiprows=1))
                result['step_comparison'] = compare(logged_steps(directory, args.warmup+args.window_steps*args.windows),
                                                   logged_steps(cpp, args.warmup+args.window_steps*args.windows))
                result['old_over_new_median'] = result['median_seconds_per_step']/report['results']['cpp_'+label]['median_seconds_per_step']
            (root/'report.json').write_text(json.dumps(report, indent=2)+'\n')
            print(backend, label, result['median_seconds_per_step'], 's/step', flush=True)
    for backend in binaries:
        first = report['results'][backend+'_np1_111']['median_seconds_per_step']
        for ranks, layout in topology:
            item = report['results'][backend+f'_np{ranks}_'+''.join(map(str, layout))]
            item['strong_speedup'] = first/item['median_seconds_per_step']
            item['strong_efficiency'] = item['strong_speedup']/ranks
    (root/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
