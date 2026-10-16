#!/usr/bin/env python3
"""Bounded, same-topology M12 MP7-LD/positivity/NSCBC measurements."""

import argparse
import hashlib
import json
from math import isfinite, prod
import os
from pathlib import Path
import re
import shutil
from statistics import median
import struct
import subprocess
import tarfile

import h5py
import numpy as np

from compare_flowstate import read_flowstate
from insitu_cfl_gate import CflGate
from insitu_resource_monitor import run_monitored
from prepare_m12_local_case import REFINED_INTERVALS, prepare
from run_m12_boundary_compare import configuration, environment

ROOT = Path(__file__).resolve().parents[2]
SDK = Path('/opt/nvidia/hpc_sdk/Linux_x86_64/26.1')


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def freeze(directory, executable):
    directory.mkdir(parents=True, exist_ok=False)
    (directory/'bin').mkdir()
    shutil.copy2(executable, directory/'bin/astr')
    with tarfile.open(directory/'source.tar.gz', 'w:gz') as archive:
        for name in ('src', 'src_gpu', 'user_define_module', 'CMakeLists.txt'):
            archive.add(ROOT/name, arcname=name)
    for name in ('CMakeCache.txt', 'src/CMakeFiles/astr.dir/flags.make', 'src/CMakeFiles/astr.dir/link.txt'):
        source = executable.parents[1]/name
        if source.exists():
            shutil.copy2(source, directory/source.name)
    cache = directory/'CMakeCache.txt'
    if cache.exists():
        for line in cache.read_text().splitlines():
            if line.startswith('ASTR_USERDEFINE_SOURCE:FILEPATH='):
                shutil.copy2(Path(line.split('=', 1)[1]), directory/'build_userdefine.F90')
    for name, command in (
            ('git-status.txt', ['git', 'status', '--short']),
            ('git-diff.patch', ['git', 'diff', '--', 'src', 'src_gpu', 'CMakeLists.txt']),
            ('libraries.txt', ['ldd', str(executable)]),
            ('gpu.csv', ['nvidia-smi', '--query-gpu=index,uuid,name,driver_version,memory.total', '--format=csv'])):
        result = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
        (directory/name).write_text(result.stdout)
    return directory/'bin/astr'


def timing_window(text, first, stop, ranks):
    """Take max rank of window sums, not the sum of per-step maxima."""
    if first < 0 or stop <= first or ranks < 1:
        raise ValueError('invalid timing window')
    rk, complete, limited = {}, {}, {}
    pending_limiter = None
    for line in text.splitlines():
        fields = line.split()
        if fields and fields[0] == 'ASTR_GPU_RANK_RK_TIMING':
            if len(fields) != 6:
                raise ValueError('malformed rank RK timing')
            rank, step = map(int, fields[1:3])
            values = list(map(float, fields[3:]))
            if not all(isfinite(v) and v >= 0 for v in values):
                raise ValueError('invalid RK timing')
            if abs(values[0]+values[1]-values[2]) > 1e-10:
                raise ValueError('RK prepare/integration total differs')
            if (step, rank) in rk:
                raise ValueError('duplicate RK timing')
            rk[step, rank] = values
        elif fields and fields[0] == 'ASTR_COMPLETE_STEP_TIMING':
            if len(fields) != 4:
                raise ValueError('malformed complete-step timing')
            step, rank = map(int, fields[1:3])
            value = float(fields[3])
            if not isfinite(value) or value < 0 or (step, rank) in complete:
                raise ValueError('invalid or duplicate complete-step timing')
            complete[step, rank] = value
        elif 'ASTR_MP_POSITIVITY step=' in line:
            match = re.search(r'step=\s*(\d+)\s+stage=\s*(\d+)', line)
            if not match or pending_limiter is not None:
                raise ValueError('malformed limiter record')
            pending_limiter = (int(match[1]), int(match[2]))
        # NVHPC list-directed output can wrap one PRINT across two lines.
        if 'limited_rank_nodes=' in line:
            match = re.search(r'limited_rank_nodes=\s*(\d+)', line)
            if not match or pending_limiter is None:
                raise ValueError('unpaired limiter count')
            if pending_limiter in limited:
                raise ValueError('duplicate limiter record')
            limited[pending_limiter] = int(match[1])
            pending_limiter = None
    if pending_limiter is not None:
        raise ValueError('incomplete limiter record')
    expected = {(step, rank) for step in range(first, stop) for rank in range(ranks)}
    if not expected.issubset(rk) or not expected.issubset(complete):
        raise ValueError('missing rank/window timing')
    if not {(s, a) for s in range(first, stop) for a in (1, 2, 3)}.issubset(limited):
        raise ValueError('missing limiter stage records')
    totals = {str(rank): dict(
        prepare_seconds=sum(rk[s, rank][0] for s in range(first, stop)),
        integration_seconds=sum(rk[s, rank][1] for s in range(first, stop)),
        rk_seconds=sum(rk[s, rank][2] for s in range(first, stop)),
        complete_seconds=sum(complete[s, rank] for s in range(first, stop)))
        for rank in range(ranks)}
    count = sum(limited[s, a] for s in range(first, stop) for a in (1, 2, 3))
    return dict(first_step=first, stop_step_exclusive=stop, measured_steps=stop-first,
                per_rank=totals,
                rk_seconds_per_step=max(v['rk_seconds'] for v in totals.values())/(stop-first),
                complete_seconds_per_step=max(v['complete_seconds'] for v in totals.values())/(stop-first),
                limited_rank_node_stage_events=count,
                limiter_window='active' if count else 'inactive',
                limiter_counts={str(s): [limited[s, a] for a in (1, 2, 3)] for s in range(first, stop)})


def runtime(topology, transport='pinned', sync='explicit'):
    env = environment(topology)
    env.update(ASTR_GPU_MP_POSITIVITY='flux', ASTR_GPU_HALO_TRANSPORT=transport,
               ASTR_GPU_SYNC_MODE=sync, ASTR_GPU_RK_TIMING='1',
               ASTR_GPU_RANK_RK_TIMING='1', ASTR_COMPLETE_STEP_TIMING='1')
    if transport == 'device-aware':
        env['OMPI_MCA_opal_cuda_support'] = 'true'
        env.update(UCX_MEMTYPE_CACHE='n', UCX_CUDA_COPY_ENABLE_FABRIC='no',
                   UCX_CUDA_COPY_DMABUF='no', UCX_CUDA_IPC_ENABLE_MNNVL='no',
                   UCX_TLS='self,sm,cuda_copy,cuda_ipc')
    libraries = [SDK/'compilers/lib', SDK/'cuda/lib64', SDK/'comm_libs/hpcx/lib',
                 Path('/opt/hdf5-1.14.6/lib')]
    env['LD_LIBRARY_PATH'] = ':'.join(map(str, libraries))+':'+env.get('LD_LIBRARY_PATH', '')
    env['PATH'] = str(SDK/'compilers/bin')+':'+env.get('PATH', '')
    return env


def mpi_command(mpiexec, ranks, transport):
    command = [str(mpiexec), '--bind-to', 'none', '-np', str(ranks)]
    if transport == 'device-aware':
        return command+['--mca', 'pml', 'ucx']
    return command+['--mca', 'pml', 'ob1', '--mca', 'btl', 'self,vader,tcp']


def configure_case(case, steps, checkpoint_interval=None, restore=None):
    # Use the existing output override contract to restore without new writes.
    interval = checkpoint_interval or (1000000000 if restore else None)
    configuration(case, 'nscbc', REFINED_INTERVALS, steps, archive=False,
                  checkpoint_interval=interval, restore=restore)
    if restore:
        path = case/'datin/input.output'
        text = path.read_text().replace("restart_output='saved'", "restart_output='override'")
        if checkpoint_interval is None:
            text = text.replace('&checkpoint\n enabled=t', '&checkpoint\n enabled=f')
        path.write_text(text)


def compare_state(reference, candidate):
    """Compare checkpoint state/phase/cache payloads bitwise, including halos."""
    report = {}
    with h5py.File(reference, 'r') as a, h5py.File(candidate, 'r') as b:
        if set(a) != set(b):
            raise ValueError('checkpoint dataset names differ')
        for name in a:
            x, y = a[name], b[name]
            if x.shape != y.shape or x.dtype != y.dtype:
                raise ValueError('checkpoint shape/type differs: '+name)
            chunk = 131072 if x.ndim == 1 else 8
            for start in range(0, x.shape[0] if x.shape else 1, chunk):
                left = x[start:start+chunk] if x.shape else x[()]
                right = y[start:start+chunk] if y.shape else y[()]
                if np.asarray(left).tobytes() != np.asarray(right).tobytes():
                    raise ValueError('checkpoint payload not bitwise equal: '+name)
            report[name] = 'bitwise-equal'
        q = [a[f'q{i:04d}'] for i in range(1, 6)]
        minimum = [float('inf'), float('inf')]
        for start in range(0, q[0].shape[0], 8):
            values = [v[start:start+8] for v in q]
            if not all(np.isfinite(v).all() for v in values):
                raise ValueError('nonfinite authoritative checkpoint')
            rho, mx, my, mz, energy = values
            internal = energy-.5*(mx*mx+my*my+mz*mz)/rho
            minimum[0] = min(minimum[0], float(rho.min()))
            minimum[1] = min(minimum[1], float(internal.min()))
        if min(minimum) <= 0:
            raise ValueError('non-positive checkpoint density/internal energy')
        if 'q0011' in a:
            for start in range(0, a['q0011'].shape[0], 8):
                temperature = a['q0011'][start:start+8]
                if not np.isfinite(temperature).all() or np.any(temperature <= 0):
                    raise ValueError('non-positive/nonfinite temperature cache')
        report['minimum_density_internal_energy'] = minimum
    return report


def wall_rng(path):
    data = path.read_bytes()
    offset = max(data.rfind(b'ASTRWR01'), data.rfind(b'ASTRWR02'))
    if offset < 0:
        raise ValueError('missing legacy wall RNG trailer')
    ranks, packet = struct.unpack_from('<ii', data, offset+8)
    if ranks < 1 or packet < 8 or packet > 4096:
        raise ValueError('invalid RNG trailer')
    size = 8+8+2*16384+64+4*ranks*packet+(8 if data[offset:offset+8] == b'ASTRWR01' else 0)
    if len(data)-offset != size:
        raise ValueError('incorrect RNG trailer length')
    return data[offset:]


def compare_stages(reference, candidate, step, ranks):
    from run_m12_boundary_compare import active_state

    names = {p.name for p in (reference/'diagnostics').glob(f'state.*.step{step:08d}.*.bin')}
    other = {p.name for p in (candidate/'diagnostics').glob(f'state.*.step{step:08d}.*.bin')}
    required = {f'state.{label}.step{step:08d}.rk{stage:02d}.rank{rank:08d}.bin'
                for label in ('pre_rhs', 'post_update', 'boundary', 'conv', 'full')
                for stage in (1, 2, 3) for rank in range(ranks)}
    if names != other or not required.issubset(names):
        raise ValueError('missing or unmatched stage snapshots')
    minimum = [float('inf'), float('inf')]
    for name in sorted(names):
        a, b = reference/'diagnostics'/name, candidate/'diagnostics'/name
        if a.stat().st_size != b.stat().st_size or digest(a) != digest(b):
            raise ValueError('stage payload differs: '+name)
        if '.post_update.' in name:
            q = active_state(a)
            rho = q[..., 0]
            if not np.isfinite(q).all() or np.any(rho <= 0):
                raise ValueError('nonfinite/non-positive accepted stage')
            internal = q[..., 4]-.5*np.sum(q[..., 1:4]**2, axis=-1)/rho
            if not np.isfinite(internal).all() or np.any(internal <= 0):
                raise ValueError('non-positive accepted stage internal energy')
            minimum[0] = min(minimum[0], float(rho.min()))
            minimum[1] = min(minimum[1], float(internal.min()))
    return dict(status='bitwise-equal', files=len(names), accepted_stages=3*ranks,
                minimum_density_internal_energy=minimum)


def compare_runs(reference, candidate, step):
    a = json.loads((reference/'provenance.json').read_text())
    b = json.loads((candidate/'provenance.json').read_text())
    for key in ('intervals', 'topology', 'first_step', 'stop_step', 'warmup_steps', 'input_sha256'):
        if a[key] != b[key]:
            raise ValueError('unmatched comparison: '+key)
    batch = Path(f'outdat/new/checkpoints/step{step:012d}')
    state = compare_state(reference/batch/'state.h5', candidate/batch/'state.h5')
    if wall_rng(reference/batch/'control.bin') != wall_rng(candidate/batch/'control.bin'):
        raise ValueError('wall RNG state differs')
    header, x = read_flowstate(reference)
    other, y = read_flowstate(candidate)
    if header != other or x.tobytes() != y.tobytes():
        raise ValueError('statistics differ')
    counts = [json.loads((p/'receipt.json').read_text())['timing']['limiter_counts']
              for p in (reference, candidate)]
    if counts[0] != counts[1]:
        raise ValueError('per-step/stage limiter event counts differ')
    return dict(status='bitwise-equal', state=state, rng='bitwise-equal',
                statistics='bitwise-equal', limiter_counts='identical')


def performance_summary(references, candidates):
    if len(references) != 5 or len(candidates) != 5:
        raise ValueError('formal characterization requires five paired rounds')
    receipts = [json.loads((p/'receipt.json').read_text()) for p in [*references, *candidates]]
    base = receipts[0]
    for value in receipts:
        if value['timing_scope'] != 'no-field-io-no-profiling' or value['status'] != 'passed':
            raise ValueError('diagnostic/profiling/incomplete run is not formal timing')
        for key in ('intervals', 'topology', 'first_step', 'stop_step', 'warmup_steps', 'input_sha256'):
            if value['provenance'][key] != base['provenance'][key]:
                raise ValueError('unmatched timing workload: '+key)
        if value['timing']['limiter_counts'] != base['timing']['limiter_counts']:
            raise ValueError('limiter activation workload differs')
    report = {}
    for metric in ('rk_seconds_per_step', 'complete_seconds_per_step'):
        a = [r['timing'][metric] for r in receipts[:5]]
        b = [r['timing'][metric] for r in receipts[5:]]
        ma, mb = median(a), median(b)
        da, db = median(abs(x-ma) for x in a)/ma, median(abs(x-mb) for x in b)/mb
        report[metric] = dict(reference=a, candidate=b, reference_median=ma,
                              candidate_median=mb, speedup=ma/mb, reduction=1-mb/ma,
                              reference_mad_fraction=da, candidate_mad_fraction=db,
                              faster_rounds=sum(y<x for x, y in zip(a, b)),
                              reference_range=[min(a), max(a)], candidate_range=[min(b), max(b)])
    rk = report['rk_seconds_per_step']
    complete = report['complete_seconds_per_step']
    report['measurement_quality'] = ('stable' if max(
        rk['reference_mad_fraction'], rk['candidate_mad_fraction'],
        complete['reference_mad_fraction'], complete['candidate_mad_fraction']) <= .03
        else 'variable-do-not-claim-stable-small-gains')
    report['measured_rk_reduction'] = rk['candidate_median'] < rk['reference_median']
    report['measured_complete_step_reduction'] = complete['candidate_median'] < complete['reference_median']
    report['optional_retention'] = 'requires-theoretical-rationale-and-separate-correctness-evidence-no-minimum-speedup'
    report['numerical_gate'] = 'requires-separate-bitwise-state-RNG-limiter-and-sanitizer-evidence'
    return report


def run(args):
    topology = tuple(map(int, args.topology.split(',')))
    if len(topology) != 3 or min(topology) <= 0 or prod(topology) > 2:
        raise ValueError('local performance driver allows one or two ranks')
    case = args.output.resolve()
    executable = args.executable.resolve(strict=True)
    cache = executable.parents[1]/'CMakeCache.txt'
    if cache.exists():
        if 'userdefine_hbl.F90' not in cache.read_text():
            raise ValueError('M12 requires its explicit ASTR_USERDEFINE_SOURCE build entry')
    if args.freeze:
        executable = freeze(args.freeze.resolve(), executable)
    first = 0
    if args.restore:
        args.restore = args.restore.resolve(strict=True)
        with h5py.File(args.restore/'state.h5') as checkpoint:
            first = int(checkpoint['identity'][8])
    if args.steps <= first+args.warmup:
        raise ValueError('no measured steps after warmup')
    if args.stage_snapshots is not None and (not args.diagnostic or
                                            not first <= args.stage_snapshots < args.steps):
        raise ValueError('stage snapshots require a diagnostic step inside the run')
    if args.checkpoint_every and not args.diagnostic:
        raise ValueError('checkpoints are preparation/diagnostic only, not formal timing')
    if args.sanitizer_suppressions and not args.memcheck:
        raise ValueError('sanitizer suppressions require memcheck')
    prepare(args.source, case, 'gpu', REFINED_INTERVALS)
    configure_case(case, args.steps, args.checkpoint_every, args.restore)
    env = runtime(args.topology, args.transport, args.sync)
    env['ASTR_GPU_MP_FACE_LAYOUT'] = args.face_layout
    if args.split_cache == 'on' and args.face_layout != 'x-contiguous':
        raise ValueError('split cache candidate requires x-contiguous face layout')
    env['ASTR_GPU_MP_SPLIT_CACHE'] = args.split_cache
    if args.stage_snapshots is not None:
        env.update(ASTR_VALIDATION_RHS_PREFIX='diagnostics/state',
                   ASTR_VALIDATION_RHS_STEP=str(args.stage_snapshots))
    if args.verify_faces:
        if not args.diagnostic or args.face_layout != 'x-contiguous':
            raise ValueError('face verification is an x-contiguous diagnostic, not formal timing')
        env['ASTR_GPU_MP_FACE_LAYOUT_VERIFY'] = '1'
    command = mpi_command(args.mpiexec, prod(topology), args.transport)
    if args.profile:
        command += ['nsys', 'profile', '--trace=cuda,mpi,nvtx,osrt', '--sample=none',
                    '--cpuctxsw=none', '--force-overwrite=false',
                    '-o', str(case/'timeline.%p')]
    if args.memcheck:
        command += ['compute-sanitizer', '--tool', 'memcheck', '--error-exitcode', '99',
                    '--log-file', str(case/'memcheck.%p.log')]
        if args.sanitizer_suppressions:
            args.sanitizer_suppressions = args.sanitizer_suppressions.resolve(strict=True)
            command += ['--suppressions', str(args.sanitizer_suppressions)]
    if args.ncu:
        command += ['ncu', '--kernel-name', 'regex:mp_faces.*kernel', '--launch-count', '3',
                    '--section', 'SpeedOfLight', '--section', 'MemoryWorkloadAnalysis',
                    '--section', 'LaunchStats', '--section', 'Occupancy',
                    '-o', str(case/'kernels.%p')]
    command += [str(executable), 'run', 'datin/input.dat']
    provenance = dict(executable=str(executable), executable_sha256=digest(executable),
                      intervals=REFINED_INTERVALS, topology=topology, first_step=first,
                      stop_step=args.steps, warmup_steps=args.warmup,
                      diagnostic=args.diagnostic, profiling=args.profile or args.ncu, memcheck=args.memcheck,
                      sanitizer_suppressions=(dict(path=str(args.sanitizer_suppressions),
                                                  sha256=digest(args.sanitizer_suppressions))
                                              if args.sanitizer_suppressions else None),
                      checkpoint_every=args.checkpoint_every, restore=str(args.restore or ''),
                      resource_baseline=(dict(path=str(args.resource_baseline.resolve()),
                                              sha256=digest(args.resource_baseline))
                                         if args.resource_baseline else None),
                      command=command, runtime={k:v for k,v in env.items()
                                               if k.startswith(('ASTR_', 'OMPI_MCA_', 'UCX_'))},
                      input_sha256={p.name:digest(p) for p in (case/'datin').iterdir() if p.is_file()})
    (case/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    gate = CflGate(case/'run.log', 1., .02, first, args.steps)
    baseline = json.loads(args.resource_baseline.read_text()) if args.resource_baseline else None
    with (case/'run.log').open('wb') as log:
        run_monitored(command, case, env, log, case/'resources.json',
                      baseline=baseline, timeout_seconds=args.timeout, check_progress=gate.check)
    collect_existing(case)


def collect_existing(case):
    provenance = json.loads((case/'provenance.json').read_text())
    resources = json.loads((case/'resources.json').read_text())
    if resources['status'] != 'passed-sampled-observation-not-allocation-proof':
        raise ValueError('cannot collect a failed solver run')
    first, stop = provenance['first_step'], provenance['stop_step']
    ranks = prod(provenance['topology'])
    gate = CflGate(case/'run.log', 1., .02, first, stop)
    gate.check(final=True)
    if provenance['memcheck']:
        logs = list(case.glob('memcheck.*.log'))
        if len(logs) != ranks or any('ERROR SUMMARY: 0 errors' not in p.read_text() for p in logs):
            raise ValueError('unclean memcheck')
    header, statistics = read_flowstate(case)
    expected = np.arange(max(first, 1), stop)
    if (len(statistics) != len(expected) or not np.isfinite(statistics).all()
            or not np.array_equal(statistics[:, 0], expected)
            or not np.allclose(statistics[:, 1], expected*.02, rtol=0, atol=2e-10)):
        raise ValueError('incomplete/nonfinite statistics')
    hdf5 = [str(p.relative_to(case)) for p in (case/'outdat').rglob('*.h5')]
    if not provenance['diagnostic'] and hdf5:
        raise ValueError('unexpected field/checkpoint output in timing run')
    result = dict(provenance=provenance, status='passed', max_cfl=gate.maximum,
                  resources=resources, hdf5_outputs=hdf5,
                  timing=timing_window((case/'run.log').read_text(), first+provenance['warmup_steps'],
                                       stop, ranks),
                  timing_scope='diagnostic-not-formal' if any(provenance[k] for k in ('diagnostic','profiling','memcheck'))
                               else 'no-field-io-no-profiling',
                  statistics_last=dict(zip(header, statistics[-1].tolist())) if len(statistics) else {})
    (case/'receipt.json').write_text(json.dumps(result, indent=2)+'\n')
    display = {k: result[k] for k in ('status', 'max_cfl', 'timing_scope')}
    display['timing'] = {k:v for k,v in result['timing'].items() if k not in ('limiter_counts', 'per_rank')}
    print(json.dumps(display, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT.parent/'M12_C13')
    parser.add_argument('--executable', type=Path, default=ROOT/'build_m12_perf/bin/astr')
    parser.add_argument('--mpiexec', type=Path, default=SDK/'comm_libs/hpcx/bin/mpiexec')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=120)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--topology', default='2,1,1')
    parser.add_argument('--transport', choices=('pinned', 'device-aware', 'pinned-pipeline'), default='pinned')
    parser.add_argument('--sync', choices=('explicit', 'dependency'), default='explicit')
    parser.add_argument('--face-layout', choices=('directional', 'x-contiguous'), default='directional')
    parser.add_argument('--verify-faces', action='store_true')
    parser.add_argument('--split-cache', choices=('off', 'on'), default='off')
    parser.add_argument('--checkpoint-every', type=int)
    parser.add_argument('--restore', type=Path)
    parser.add_argument('--diagnostic', action='store_true')
    parser.add_argument('--stage-snapshots', type=int)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--memcheck', action='store_true')
    parser.add_argument('--sanitizer-suppressions', type=Path)
    parser.add_argument('--ncu', action='store_true')
    parser.add_argument('--timeout', type=int, default=3600)
    parser.add_argument('--freeze', type=Path)
    parser.add_argument('--resource-baseline', type=Path)
    parser.add_argument('--compare', type=Path, nargs=2)
    parser.add_argument('--reference-rounds', type=Path, nargs=5)
    parser.add_argument('--candidate-rounds', type=Path, nargs=5)
    parser.add_argument('--collect-existing', action='store_true')
    args = parser.parse_args()
    if args.reference_rounds or args.candidate_rounds:
        if not args.reference_rounds or not args.candidate_rounds:
            parser.error('both reference and candidate rounds are required')
        report = performance_summary(args.reference_rounds, args.candidate_rounds)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2))
    elif args.collect_existing:
        collect_existing(args.output.resolve())
    elif args.compare:
        report = compare_runs(*args.compare, args.steps)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2))
    else:
        run(args)
