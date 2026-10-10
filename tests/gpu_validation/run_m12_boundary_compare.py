#!/usr/bin/env python3
"""Matched M12 boundary runs and same-topology restart checks; no rendering."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import h5py
import numpy as np

from prepare_m12_local_case import REFINED_INTERVALS, prepare, replace_value
from insitu_cfl_gate import CflGate
from insitu_resource_monitor import run_monitored
from compare_flowstate import read_flowstate
from compare_q_validation_snapshots import read_q_snapshot
from run_output_restart_validation import compare_fields, archive_schedule_payload

ROOT = Path(__file__).resolve().parents[2]


def configuration(case, kind, shape, steps, archive=True, checkpoint_interval=None, restore=None):
    if steps <= 0 or (checkpoint_interval is not None and checkpoint_interval <= 0):
        raise ValueError('step counts and checkpoint intervals must be positive')
    if restore is not None and checkpoint_interval is None:
        raise ValueError('restart requires an explicit checkpoint schedule')
    (case/'outdat/new').mkdir()
    path = case / 'datin/input.dat'
    lines = path.read_text().splitlines()
    index = next(i for i, s in enumerate(lines) if s.startswith('# bctype')) + 1
    if kind == 'nscbc':
        lines[index+1] = '22,1.d0'
        lines[index+3] = '52'
    path.write_text('\n'.join(lines)+'\n')
    path = case / 'datin/controller'
    lines = path.read_text().splitlines()
    # feqchkpt controls CFL reporting; input.output controls actual checkpoints.
    replace_value(lines, 'maxstep,feqchkpt', f'{steps-1},1,999999,999999,1,999999')
    path.write_text('\n'.join(lines)+'\n')
    (case/'datin/input.output').write_text(f"""&output
 directory='outdat/new',restore_directory='{restore or ''}',restart_output='saved',
 host_budget_bytes=1073741824,device_budget_bytes=67108864,
 buffer_bytes=8388608,device_reserve_bytes=1073741824
/
&checkpoint
 enabled={'t' if checkpoint_interval else 'f'},mode='steps',
 interval_steps={checkpoint_interval or 1000000000},keep=2,initial_frame=f
/
&volume
 enabled={'t' if archive else 'f'},interval_steps={steps if archive else 1000000000},initial_frame=f,final_frame=t
/
&slices
 enabled={'t' if archive else 'f'},interval_steps=100,initial_frame=t,final_frame=t,
 i_indices={shape[0]},j_indices=0,{shape[1]},k_indices={shape[2]//2}
/
""")


def environment(topology, diagnostics=False):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('ASTR_', 'OMPI_MCA_', 'UCX_'))}
    env.update(ASTR_FORCE_MPI_TOPOLOGY=topology, ASTR_WALL_BLOWING_MODE='legacy_random',
               ASTR_GPU_SYNC_MODE='explicit', ASTR_GPU_PRECISION_MODE='fp64',
               ASTR_GPU_HALO_TRANSPORT='pinned', ASTR_NSCBC_FARFIELD_MODE='compatibility',
               OMPI_MCA_coll='^hcoll,ucc,cuda', OMPI_MCA_coll_hcoll_enable='0',
               OMPI_MCA_coll_ucc_enable='0', OMPI_MCA_opal_cuda_support='false',
               OMPI_MCA_osc='pt2pt')
    if diagnostics:
        env.update(ASTR_VALIDATION_RHS_PREFIX='diagnostics/state',
                   ASTR_VALIDATION_RHS_STEP='9')
    return env


def validate_final_volume(case, shape, steps):
    path = case/'outdat/new/fields/segment00000000'/f'step{steps:012d}'/'data.h5'
    ranges = {}
    with h5py.File(path, 'r') as archive:
        expected = tuple(n+1 for n in reversed(shape))
        for name in ('density', 'pressure', 'temperature', 'velocity_x', 'velocity_y', 'velocity_z'):
            values = archive[name][...]
            if values.shape != expected or not np.isfinite(values).all():
                raise ValueError(f'invalid final field: {path}/{name}')
            lower, upper = float(values.min()), float(values.max())
            if name in ('density', 'pressure', 'temperature') and lower <= 0:
                raise ValueError(f'non-positive final field: {path}/{name}')
            ranges[name] = [lower, upper]
    return ranges


def run_case(args, label, kind, shape, steps, ranks=2, diagnostics=False, memcheck=False,
             checkpoint_interval=None, restore=None, continuation_probe=False, archive=True):
    case = args.output / label
    prepare(args.source, case, 'gpu', shape)
    configuration(case, kind, shape, steps, archive, checkpoint_interval, restore)
    first_step = 0
    if restore is not None:
        with h5py.File(restore/'state.h5') as state:
            first_step = int(state['identity'][8])
        for name in ('grid.2d', 'flowini2d.h5', 'inlet.prof', 'wallbs.dat'):
            (case/'datin'/name).unlink()
    if diagnostics:
        path = case/'datin/wallbs.dat'
        lines = path.read_text().splitlines()
        lines[2] = '0.0, 0.0, 20.0, 40.0, 60.0'
        path.write_text('\n'.join(lines)+'\n')
    topology = f'{ranks},1,1'
    env = environment(topology, diagnostics)
    if continuation_probe:
        env.update(ASTR_VALIDATION_RHS_PREFIX='diagnostics/state',
                   ASTR_VALIDATION_RHS_STEP='0', ASTR_VALIDATION_RHS_STEP_END=str(steps-1))
    provenance = dict(boundary=kind, shape=shape, steps=steps, topology=topology,
                      wall_blowing='zero amplitude for partition check' if diagnostics else 'original legacy_random',
                      checkpoint_interval=checkpoint_interval, restore=str(restore) if restore else None,
                      first_step=first_step, executable=str(args.executable),
                      executable_sha256=hashlib.sha256(args.executable.read_bytes()).hexdigest(),
                      runtime=env, input_sha256={
                          p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((case/'datin').iterdir()) if p.is_file()})
    # Store only the chosen runtime controls, not credentials from the parent shell.
    provenance['runtime'] = {k: v for k, v in env.items() if k.startswith(('ASTR_', 'OMPI_MCA_'))}
    (case/'comparison.json').write_text(json.dumps(provenance, indent=2)+'\n')
    command = [str(args.mpiexec), '--mca', 'pml', 'ob1', '--mca', 'btl', 'self,vader,tcp',
               '--bind-to', 'none', '-np', str(ranks)]
    if memcheck:
        command += ['compute-sanitizer', '--tool', 'memcheck', '--error-exitcode', '99',
                    '--log-file', str(case/'memcheck.%p.log')]
    command += [str(args.executable), 'run', 'datin/input.dat']
    gate = CflGate(case/'run.log', 1., .02, first_step, steps)
    started = time.monotonic()
    with (case/'run.log').open('wb') as log:
        resources = run_monitored(command, case, env, log, case/'resources.json',
                                  device_reserve_bytes=1024**3,
                                  timeout_seconds=args.timeout,
                                  check_progress=gate.check)
    gate.check(final=True)
    if memcheck:
        reports = list(case.glob('memcheck.*.log'))
        if len(reports) != ranks or any('ERROR SUMMARY: 0 errors' not in p.read_text() for p in reports):
            raise ValueError(f'unclean per-rank memcheck: {case}')
    header, statistics = read_flowstate(case)
    # The existing statistics writer samples pre-RK states 1 .. steps-1.
    sample_steps = np.arange(max(1, first_step), steps)
    if (len(statistics) != len(sample_steps) or not np.isfinite(statistics).all()
            or not np.array_equal(statistics[:, 0], sample_steps)
            or not np.allclose(statistics[:, 1], .02*sample_steps, rtol=0, atol=2e-10)):
        raise ValueError(f'incomplete or nonfinite statistics: {case}')
    result = dict(case=str(case), boundary=kind, shape=shape, steps=steps,
                  topology=topology, max_cfl=gate.maximum,
                  wall_seconds=time.monotonic()-started, resources=resources,
                  statistics_last=dict(zip(header, statistics[-1].tolist())),
                  final_field_ranges=validate_final_volume(case, shape, steps) if archive else None)
    result['hdf5_files'] = [str(p.relative_to(case)) for p in (case/'outdat/new').rglob('*.h5')]
    (case/'receipt.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)
    return result


def restart_gate(args, kind, ranks):
    """Use the real random wall and delete original resources before restore."""
    shape = (64,32,24)
    prefix = f'{kind}_np{ranks}'
    paths = {}
    for label, steps, restore, interval in (
            ('continuous', 10, None, 5), ('first', 5, None, 5),
            ('resumed', 10, args.output/(prefix+'_first')/'outdat/new/checkpoints/step000000000005', 5),
            ('no_checkpoint', 10, None, None)):
        result = run_case(args, prefix+'_'+label, kind, shape, steps, ranks,
                          checkpoint_interval=interval, restore=restore,
                          continuation_probe=True, archive=False)
        paths[label] = Path(result['case'])
    reference, first, resumed, plain = (paths[key] for key in
                                        ('continuous', 'first', 'resumed', 'no_checkpoint'))
    final = 'outdat/new/checkpoints/step000000000010'
    datasets = compare_fields(reference/final/'state.h5', resumed/final/'state.h5')
    for name in ('control.bin', 'insitu_control.bin'):
        if (reference/final/name).read_bytes() != (resumed/final/name).read_bytes():
            raise ValueError('exact restart control differs: '+name)
    if archive_schedule_payload(reference/final/'archives.bin') != \
            archive_schedule_payload(resumed/final/'archives.bin'):
        raise ValueError('exact restart archive schedule differs')
    for candidate, count in ((reference,30*ranks),(first,15*ranks),
                             (resumed,15*ranks),(plain,30*ranks)):
        if len(list((candidate/'diagnostics').glob('state.post_update.*.bin'))) != count:
            raise ValueError('missing three-stage restart snapshots: '+str(candidate))
        snapshots = [p for p in sorted((candidate/'diagnostics').glob('state.*.bin'))
                     if not (candidate == resumed and '.initialized.' in p.name)]
        if not snapshots:
            raise ValueError('missing restart stage diagnostics')
        for path in snapshots:
            if path.read_bytes() != (reference/'diagnostics'/path.name).read_bytes():
                raise ValueError('restart or checkpoint observer changed stage: '+str(path))
        for path in (candidate/'diagnostics').glob('state.post_update.*.bin'):
            q = active_state(path)
            energy = q[...,4] - .5*np.sum(q[...,1:4]**2,axis=-1)/q[...,0]
            if not np.isfinite(q).all() or np.min(q[...,0]) <= 0 or np.min(energy) <= 0:
                raise ValueError('invalid physical restart stage: '+str(path))
    columns, a = read_flowstate(reference)
    headers, b = read_flowstate(first)
    other, c = read_flowstate(resumed)
    plain_headers, d = read_flowstate(plain)
    if columns != headers or columns != other or columns != plain_headers or \
            a.tobytes() != np.concatenate((b,c)).tobytes() or a.tobytes() != d.tobytes():
        raise ValueError('restart or checkpoint observer changed statistics sequence')
    return dict(boundary=kind,ranks=ranks,datasets=datasets,exact_state=True,
                exact_stage_bytes=True,exact_rng_control=True,exact_statistics=True,
                checkpoint_observer_unchanged=True,frozen_resources_only=True)


def active_state(path):
    snapshot = read_q_snapshot(path)
    im, jm, km, hm, nq = snapshot.header
    q = snapshot.values.reshape((im+2*hm+1, jm+2*hm+1, km+2*hm+1, nq), order='F')
    return q[hm:hm+im+1, hm:hm+jm+1, hm:hm+km+1, :]


def partition_check(a, b):
    def joined(case):
        paths = sorted(Path(case).glob('diagnostics/state.post_update.step00000009.rk03.rank*.bin'))
        pieces = [active_state(p) for p in paths]
        if not pieces:
            raise ValueError('missing final RK diagnostic')
        for q in pieces:
            e = q[..., 4]-.5*np.sum(q[..., 1:4]**2, axis=-1)/q[..., 0]
            if not np.isfinite(q).all() or np.min(q[..., 0]) <= 0 or np.min(e) <= 0:
                raise ValueError('invalid physical-node state')
        return np.concatenate([q[:-1] for q in pieces[:-1]]+[pieces[-1]], axis=0)
    error = float(np.max(abs(joined(a)-joined(b))))
    if error > 2e-10:
        raise ValueError(f'NSCBC NP1/2 field mismatch: {error}')
    return error


def compare_pair(directory, steps):
    cases = [Path(directory)/kind for kind in ('extrapolation', 'nscbc')]
    for name in ('grid.2d', 'flowini2d.h5', 'inlet.prof', 'wallbs.dat', 'controller', 'input.output'):
        if (cases[0]/'datin'/name).read_bytes() != (cases[1]/'datin'/name).read_bytes():
            raise ValueError(f'unmatched comparison input: {name}')
    columns, a = read_flowstate(cases[0])
    other_columns, b = read_flowstate(cases[1])
    if columns != other_columns or a.shape != b.shape or not np.array_equal(a[:, :2], b[:, :2]):
        raise ValueError('unmatched statistics sampling identities')
    comparison = dict(scope='boundary sensitivity, differences are not equivalence errors',
                      complete_steps=steps, final_field_time=.02*steps,
                      statistics_last_time=float(a[-1, 1]), fields={}, statistics={})
    for i, name in enumerate(columns[2:], 2):
        comparison['statistics'][name] = dict(max_abs_difference=float(np.max(abs(a[:, i]-b[:, i]))),
            final_extrapolation=float(a[-1, i]), final_nscbc=float(b[-1, i]))
    paths = [case/'outdat/new/fields/segment00000000'/f'step{steps:012d}'/'data.h5' for case in cases]
    with h5py.File(paths[0]) as fa, h5py.File(paths[1]) as fb:
        metadata = fa['metadata'][...]
        if (metadata.shape != (11,) or metadata[1] != steps
                or not np.array_equal(metadata, fb['metadata'][...])):
            raise ValueError('unmatched completed-field identities')
        for archive in (fa, fb):
            phase = archive.attrs['phase']
            if isinstance(phase, bytes):
                phase = phase.decode('ascii')
            if phase != 'completed_step':
                raise ValueError('comparison field is not a complete RK state')
        comparison['final_field_time'] = float(metadata[2:3].view(np.float64)[0])
        for name in ('density', 'velocity_x', 'velocity_y', 'velocity_z', 'pressure', 'temperature'):
            left, right = fa[name][...], fb[name][...]
            if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
                raise ValueError(f'invalid matched final fields: {name}')
            difference = abs(left-right)
            kji = np.unravel_index(np.argmax(difference), difference.shape)
            comparison['fields'][name] = dict(max_abs_difference=float(difference[kji]),
                rms_difference=float(np.sqrt(np.mean(difference**2))),
                global_ijk=[int(index) for index in reversed(kji)])
    path = Path(directory)/'comparison.json'
    with path.open('x') as stream:
        json.dump(comparison, stream, indent=2)
        stream.write('\n')
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('/home/dell/workspace/M12_C13'))
    parser.add_argument('--executable', type=Path, default=ROOT/'build_m12_gpu/bin/astr')
    parser.add_argument('--mpiexec', type=Path,
                        default=Path('/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpiexec'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=('smoke', 'preflight', 'medium', 'restart'), required=True)
    parser.add_argument('--mesh', choices=('medium', 'refined'), default='medium',
                        help='Refined restores original x nodes around the wall forcing band')
    parser.add_argument('--steps', type=int, default=2000, help='Total completed steps for medium runs')
    parser.add_argument('--checkpoint-interval', type=int, help='Completed-step checkpoint interval; retain latest two')
    parser.add_argument('--order', nargs='+', choices=('nscbc', 'extrapolation'),
                        default=['nscbc', 'extrapolation'])
    parser.add_argument('--timeout', type=float, default=43200.)
    args = parser.parse_args()
    if args.steps <= 0 or len(set(args.order)) != len(args.order):
        parser.error('require positive steps and unique boundary selections')
    if args.checkpoint_interval is not None and args.checkpoint_interval <= 0:
        parser.error('checkpoint interval must be positive')
    if args.mesh == 'refined' and args.stage not in ('preflight', 'medium'):
        parser.error('refined mesh is only available for preflight or medium runs')
    args.output = args.output.resolve()
    args.executable = args.executable.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    # Keep exact restart usable even if the development build is later rebuilt.
    (args.output/'bin').mkdir()
    shutil.copy2(args.executable,args.output/'bin/astr')
    args.executable = args.output/'bin/astr'
    report = dict(status='running', stage=args.stage, mesh=args.mesh, runs=[],
                  scope='boundary/mesh sensitivity, not DNS resolution or steady-state validation')
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        if args.stage == 'restart':
            for kind, ranks in (('nscbc',1), ('nscbc',2), ('extrapolation',2)):
                report['runs'].append(restart_gate(args,kind,ranks))
        elif args.stage == 'smoke':
            a = run_case(args, 'nscbc_np1', 'nscbc', (64,32,24), 10, 1, True)
            report['runs'].append(a)
            b = run_case(args, 'nscbc_np2', 'nscbc', (64,32,24), 10, 2, True, True)
            report['runs'].append(b)
            report['partition_max_abs'] = partition_check(a['case'], b['case'])
        else:
            steps = 10 if args.stage == 'preflight' else args.steps
            shape = REFINED_INTERVALS if args.mesh == 'refined' else (450,130,96)
            for kind in args.order:
                report['runs'].append(run_case(args, kind, kind, shape, steps,
                                               checkpoint_interval=args.checkpoint_interval))
                (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
            if set(args.order) == {'nscbc', 'extrapolation'}:
                report['comparison'] = compare_pair(args.output, steps)
        report['status'] = 'passed'
    except BaseException as exc:
        report.update(status='failed', error=str(exc))
        raise
    finally:
        (args.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
