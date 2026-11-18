"""Native FP32 time-series inlet: original references, ring history and exact restart."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import h5py
import numpy as np

from run_cpp_tgv_gate import namelist, logged_steps, compare, cfl_log
from run_cpp_multi_tgv_gate import raw_field
from run_cpp_insitu_gate import compare_hdf


def physical_phase(path, phase):
    if phase in ('pre_rhs', 'pre_rhs_primitives', 'post_update'):
        h, field = raw_field(path, primitive=phase == 'pre_rhs_primitives', precision='fp32')
        region = tuple(slice(h[3], h[3]+n+1) for n in h[:3])
        value = field[region]
        if phase != 'pre_rhs_primitives':
            q = value.astype(np.float64)
            internal = q[..., 4]-.5*np.sum(q[..., 1:4]**2, axis=-1)/q[..., 0]
            if not (q[..., 0] > 0).all() or not (internal > 0).all():
                raise AssertionError('native inflow physical density/internal energy')
    else:
        with path.open('rb') as stream:
            h = np.fromfile(stream, np.int32, 4)
            value = np.fromfile(stream, np.float32).reshape((*h[:3]+1, h[3]), order='F')
    if not np.isfinite(value).all():
        raise AssertionError('nonfinite native inflow physical phase')
    return value


def stage_differences(actual, reference, ranks):
    result = {}
    for phase in ('pre_rhs', 'pre_rhs_primitives', 'post_source', 'post_update'):
        paths = sorted((reference/'outdat').glob(f'rhs.{phase}.*.bin'))
        if len(paths) != 12*ranks:
            raise AssertionError('native inflow stage inventory')
        maximum = None
        for path in paths:
            x, y = physical_phase(actual/'outdat'/path.name, phase), physical_phase(path, phase)
            error = np.asarray(compare(x.astype(np.float64), y.astype(np.float64), limit=None))
            maximum = error if maximum is None else np.maximum(maximum, error)
        result[phase] = maximum.tolist()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('original', 'binary', 'fixture', 'output', 'mpiexec'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--reuse-original', type=Path, help='explicit completed CPU/old-GPU runs from this runner')
    parser.add_argument('--automatic', action='store_true')
    parser.add_argument('--memcheck', action='store_true')
    args = parser.parse_args()
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=False)
    meta = json.loads((args.fixture/'reference.json').read_text())
    ranks = int(np.prod(meta['topology'])); dt = meta['dt']
    if ranks != 2 or meta['topology'] != [1, 2, 1] or meta['steps'] != 4 or meta.get('automatic', False):
        parser.error('select the existing four-step NP2-y fixed-dt dynamic-inlet fixture')
    for name, binary in (('original_astr', args.original), ('cpp_astr', args.binary)):
        shutil.copy2(binary, root/name)
    shutil.copy2(__file__, root/'runner.py')
    env = {key: value for key, value in os.environ.items() if not key.startswith('ASTR_')}
    env.update(OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,vader,tcp', OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc', OMPI_MCA_opal_cuda_support='0',
        ASTR_FORCE_MPI_TOPOLOGY=','.join(map(str, meta['topology'])), ASTR_GPU_HALO_TRANSPORT='pinned',
        ASTR_GPU_SYNC_MODE='explicit', ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',
        ASTR_VALIDATION_RHS_STEP='0', ASTR_VALIDATION_RHS_STEP_END='4')
    report = dict(precision='fp32', topology=meta['topology'], dt=dt, interval=meta['interval'], automatic=args.automatic,
        binaries={name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ('original_astr', 'cpp_astr')},
        numerical_policy='D10: report cross-implementation errors; finite/positive and same-backend exact continuation required',
        backends={})
    if args.reuse_original:
        reused = json.loads((args.reuse_original/'progress.json').read_text())
        if (reused['binaries']['original_astr'] != report['binaries']['original_astr']
                or any(reused[key] != report[key] for key in ('precision', 'topology', 'dt', 'interval', 'automatic'))):
            raise AssertionError('original reuse identity differs')
        report['original_reuse_directory'] = str(args.reuse_original.resolve())

    def run(backend, name, steps=4, restore=None, writing=True):
        if args.reuse_original and backend in ('cpu', 'gpu'):
            return args.reuse_original.resolve()/backend/name
        case = root/backend/name; case.mkdir(parents=True)
        source = args.fixture/('cpu' if backend == 'cpu' else 'gpu')
        for folder in ('datin', 'inflow'):
            shutil.copytree(source/folder, case/folder, copy_function=shutil.copyfile)
            (case/folder).chmod(0o755)
        (case/'outdat/new').mkdir(parents=True)
        timestep = (dict(mode='auto', initial_dt=dt, max_dt=1.6e-5, cfl_target=.5, growth_factor=1.1)
                    if args.automatic else dict(mode='fixed', fixed_dt=dt))
        (case/'datin/input.run').write_text(namelist(dict(run_control=dict(max_steps=steps),
            timestep=timestep, logging=dict(list_steps=1),
            repair=dict(crashfix=False), statistics=dict(enabled=False))))
        (case/'datin/input.output').write_text(namelist(dict(output=dict(directory='outdat/new', exact_restart=True,
            restore_directory=str(restore or ''), checkpoint_statistics=False,
            host_budget_bytes=-1, device_budget_bytes=-1, device_reserve_bytes=0),
            checkpoint=dict(enabled=writing, mode='steps', interval_steps=2, keep=2, initial_frame=False),
            volume=dict(enabled=False), slices=dict(enabled=False))))
        binary = root/('cpp_astr' if backend == 'cpp' else 'original_astr')
        command = [str(args.mpiexec.resolve()), '--bind-to', 'none', '-n', str(ranks)]
        if args.memcheck and backend == 'cpp' and name == 'full':
            command += ['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer', '--tool', 'memcheck',
                '--leak-check', 'full', '--error-exitcode', '99', '--log-file', str(case/'memcheck.%p.log')]
        with (case/'run.log').open('w') as log:
            subprocess.run(command+[str(binary), 'run'], cwd=case, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, check=True)
        if backend == 'cpp' and 'ASTR_GPU_CPP dynamic_inflow device_bytes=' not in (case/'run.log').read_text():
            raise AssertionError('C++ dynamic device inlet was not selected')
        return case

    completed = {}
    for backend in ('cpu', 'gpu', 'cpp'):
        full = run(backend, 'full'); completed[backend] = full
        with h5py.File(full/'outdat/new/checkpoints/step000000000004/state.h5') as stream:
            if stream['q0001'].dtype != np.dtype('float32') or stream['numerical_kind_bytes'][0] != 4:
                raise AssertionError('not native FP32 authoritative state')
        if not np.isfinite(cfl_log(full)).all():
            raise AssertionError('nonfinite native inflow CFL')
        row = dict(stages_to_cpu=stage_differences(full, completed['cpu'], ranks))
        quiet = run(backend, 'output-off', writing=False)
        for path in (full/'outdat').glob('rhs.post_update.*.bin'):
            if path.read_bytes() != (quiet/'outdat'/path.name).read_bytes():
                raise AssertionError('checkpoint output changed conservative words/halos')
        if not np.array_equal(logged_steps(full), logged_steps(quiet)):
            raise AssertionError('checkpoint output changed physical clocks')
        prefix = run(backend, 'prefix', steps=2)
        tail = run(backend, 'tail', restore=prefix/'outdat/new/checkpoints/step000000000002')
        final = Path('outdat/new/checkpoints/step000000000004')
        for name in ('state.h5', 'inflow.h5'):
            compare_hdf(full/final/name, tail/final/name, exact=True)
        if (full/final/'control.bin').read_bytes() != (tail/final/'control.bin').read_bytes():
            raise AssertionError('native inflow restart control/history bits')
        if not np.array_equal(logged_steps(full)[2:], logged_steps(tail, 2, 3)):
            raise AssertionError('native inflow restart clocks')
        a = np.loadtxt(full/'flowstate.dat', skiprows=1, ndmin=2)
        b = np.loadtxt(tail/'flowstate.dat', skiprows=1, ndmin=2)
        if not np.array_equal(b, a[np.isin(a[:, 0], b[:, 0])]):
            raise AssertionError('native inflow restart scalar sequence')
        with h5py.File(full/final/'inflow.h5') as stream:
            metadata = stream['metadata'][...].reshape(-1)
            if metadata[2] <= 3:
                raise AssertionError('native inflow ring did not advance')
            if stream['q0001'].dtype != np.dtype('float32'):
                raise AssertionError('native inflow ring payload is not FP32')
            row['inflow_metadata'] = metadata.tolist()
        if args.memcheck and backend == 'cpp':
            logs = list(full.glob('memcheck.*.log'))
            if len(logs) != ranks or any('ERROR SUMMARY: 0 errors' not in p.read_text() or
                    'LEAK SUMMARY: 0 bytes leaked' not in p.read_text() for p in logs):
                raise AssertionError('native inflow per-rank memory evidence')
        row.update(output_q_halo_clock_words_equal=True, exact_state_inflow_control_scalars=True)
        report['backends'][backend] = row
        (root/'progress.json').write_text(json.dumps(report, indent=2)+'\n')
    report['cpp_to_original_gpu'] = stage_differences(completed['cpp'], completed['gpu'], ranks)
    report['memcheck'] = args.memcheck
    (root/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
