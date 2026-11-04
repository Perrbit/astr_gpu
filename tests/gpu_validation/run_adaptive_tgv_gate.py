"""Real ASTR short runs for adaptive complete-step clocks and exact restart."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from time import sleep

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/input'))
from convert_legacy_input import convert_case, namelist
STEP = re.compile(r'ASTR_TIMESTEP complete_step=(\d+) time_start=\s*(\S+)'
                  r' time_end=\s*(\S+) dt_used=\s*(\S+) dt_next=\s*(\S+)')


def compare_state(first, second, exact, details=None, max_scaled_error=2e-10):
    maximum = 0.0
    with h5py.File(first) as a, h5py.File(second) as b:
        if exact and set(a) != set(b):
            raise AssertionError('exact checkpoint dataset identities differ')
        for state in (a, b):
            rho = state['q0001'][:]
            internal = state['q0005'][:] - sum(state[f'q{q:04d}'][:].astype(float)**2
                                              for q in (2, 3, 4))/(2*rho)
            if not (rho > 0).all() or not (internal > 0).all():
                raise AssertionError('nonpositive physical density/internal energy')
        for name in a:
            if not isinstance(a[name], h5py.Dataset) or name not in b:
                continue
            x, y = a[name][:], b[name][:]
            if not exact and not name.startswith('q0'):
                continue
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise AssertionError(f'nonfinite {name}')
            difference = np.abs(x.astype(float)-y.astype(float))
            error = float(np.max(difference))
            scaled = error/max(1.0, float(np.max(np.abs(x))))
            if details is not None:
                location = list(map(int,np.unravel_index(np.argmax(difference),x.shape)))
                if 'maximum_absolute_error' not in details or error > details['maximum_absolute_error']:
                    details.update(maximum_absolute_error=error, worst_absolute_dataset=name,
                                   worst_absolute_hdf5_index=location)
                if 'scaled_field_error' not in details or scaled > details['scaled_field_error']:
                    details.update(scaled_field_error=scaled, worst_scaled_dataset=name,
                                   worst_scaled_hdf5_index=location)
            maximum = max(maximum, scaled)
            if exact and not np.array_equal(x, y):
                raise AssertionError(f'exact restart differs in {name}: {error}')
    if not exact and max_scaled_error is not None and maximum > max_scaled_error:
        raise AssertionError(f'CPU/GPU scaled field difference: {maximum}')
    return maximum


def run(args, backend, ranks, label, restore=None, stop_time=None, checkpoint=True, default_path=False,
        runtime_packets=None,replay_library=None,clock_dtype=np.float64,axis='x'):
    directory = args.output / label
    if args.analyze_only:
        return directory, read_clocks(directory, restore, stop_time,clock_dtype)
    (directory/'datin').mkdir(parents=True)
    (directory/'outdat/new').mkdir(parents=True)
    text = (ROOT/'examples/GPU_Quickstart/tgv/datin/input.dat').read_text()
    size = getattr(args, 'size', 32)
    text = text.replace('32,32,32', f'{size},{size},{size}')
    text = text.replace('t,t,t,f,f,f,t,f,t', 't,t,t,f,f,f,f,f,'+('t' if backend == 'gpu' else 'f'))
    text = text.replace('643e,643e,rk3', f'{args.scheme},{args.scheme},rk3')
    (directory/'datin/input.case').write_text(namelist(convert_case(text)))
    stop = 'max_steps=12' if stop_time is None else f'stop_time={stop_time}'
    initial_dt = '0.001' if stop_time is None else '0'
    list_steps = 0 if stop_time is None else 1
    (directory/'datin/input.run').write_text(
        f'&run_control {stop}, reload_steps=100 /\n'
        f'&timestep mode="auto", initial_dt={initial_dt} /\n&logging list_steps={list_steps} /\n')
    (directory/'datin/input.output').write_text(
        f'&output\n directory="outdat/new", restore_directory="{restore or ""}", exact_restart=t,\n'
        ' host_budget_bytes=-1, device_budget_bytes=-1, buffer_bytes=1048576\n/\n'
        f'&checkpoint\n enabled={"t" if checkpoint else "f"}, interval_steps=5, keep=2\n/\n'
        '&volume\n enabled=f\n/\n&slices\n enabled=f\n/\n')
    env = os.environ.copy()
    topology=[1,1,1]
    topology['xyz'.index(axis)]=ranks
    env.update(OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,vader,tcp', OMPI_MCA_osc='pt2pt',
               OMPI_MCA_coll='^cuda,hcoll,ucc', OMPI_MCA_opal_cuda_support='0',
               ASTR_FORCE_MPI_TOPOLOGY=','.join(map(str,topology)), ASTR_GPU_SYNC_MODE='explicit',
               ASTR_COMPLETE_STEP_TIMING='1', ASTR_INSITU_TIMING='1')
    if runtime_packets is not None:
        replay=directory/'control_replay'
        replay.mkdir()
        for completed, text in runtime_packets.items():
            (replay/f'controller{completed:012d}').write_text(text)
        env.update(LD_PRELOAD=str(replay_library.resolve()),
                   ASTR_CONTROLLER_TEST_FILE=str((directory/'datin/input.run').resolve()),
                   ASTR_CONTROLLER_TEST_REPLAY=str(replay.resolve()),
                   ASTR_CONTROLLER_TEST_START_STEP='5' if restore else '0')
    with (directory/'run.log').open('w') as log:
        with subprocess.Popen([str(args.mpiexec), '-n', str(ranks), str(getattr(args, backend)),
                               'run', *([] if default_path else ['datin/input.case'])], cwd=directory, env=env, stdout=log,
                              stderr=subprocess.STDOUT) as process:
            source = directory/'outdat/new/checkpoints/step000000000005'
            saved = directory/'captured/checkpoints/step000000000005'
            while process.poll() is None:
                if checkpoint and restore is None and source.is_dir() and not saved.exists():
                    # Preserve immutable step 5 before keep=2 legitimately retires it.
                    shutil.copytree(source, saved)
                    shutil.copytree(directory/'outdat/new/resources', directory/'captured/resources')
                sleep(0.001)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, process.args)
    return directory, read_clocks(directory, restore, stop_time,clock_dtype)


def read_clocks(directory, restore, stop_time,clock_dtype=np.float64):
    log = (directory/'run.log').read_text()
    if 'The job is done!' not in log:
        raise AssertionError(f'incomplete run: {directory}')
    records = [[int(m[1]), *map(float, m.groups()[1:])] for m in STEP.finditer(log)]
    if not records:
        raise AssertionError(f'missing complete-step clocks: {directory}')
    expected_first = 6 if restore else 1
    expected_last = 12 if stop_time is None else records[-1][0]
    if [r[0] for r in records] != list(range(expected_first, expected_last+1)):
        raise AssertionError('extra or missing completed steps')
    for _, start, end, used, next_dt in records:
        if not np.isfinite([start, end, used, next_dt]).all() or min(used, next_dt) <= 0:
            raise AssertionError('invalid timestep clock')
        expected_end=float(clock_dtype(clock_dtype(start)+clock_dtype(used)))
        if abs(end-expected_end) > 32*np.finfo(float).eps*max(1, abs(end)):
            raise AssertionError('clock did not use the completed dt')
        growth_bound=float(clock_dtype(clock_dtype(used)*clock_dtype(1.1)))
        if next_dt > growth_bound*(1+32*np.finfo(float).eps):
            raise AssertionError('growth limit not respected')
    if stop_time is not None:
        if not records[-1][2] >= stop_time > records[-1][1]:
            raise AssertionError('physical stop did not use the first crossing complete step')
        if records[-1][2] == stop_time:
            raise AssertionError('test unexpectedly landed on the stop target; no clipping proof')
        bounds = re.findall(r'ASTR_CFL directional/local_sum/upper_bound=([^\n]+)', log)
        if bounds:
            if len(bounds) != len(records):
                raise AssertionError('missing detailed CFL cross-check')
            for record, bound in zip(records, bounds, strict=True):
                used, next_dt = record[3:]
                upper = float(bound.split()[-1])
                expected = float(min(clock_dtype(clock_dtype(0.5)*clock_dtype(used)/clock_dtype(upper)),
                                     clock_dtype(clock_dtype(1.1)*clock_dtype(used))))
                tolerance=64*np.finfo(float).eps*max(1,expected)
                if clock_dtype is np.float32:
                    tolerance=64*np.finfo(np.float32).eps*abs(expected)
                if abs(next_dt-expected) > tolerance:
                    raise AssertionError('fast wave scan differs from detailed CFL collector')
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu', type=Path, required=True)
    parser.add_argument('--gpu', type=Path, required=True)
    parser.add_argument('--mpiexec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scheme', choices=['643e', '643c'], default='643e')
    parser.add_argument('--analyze-only', action='store_true')
    parser.add_argument('--size', type=int, default=32)
    parser.add_argument('--ranks',nargs='+',type=int,default=[1,2])
    parser.add_argument('--axis',choices=('x','y','z'),default='x')
    parser.add_argument('--skip-time-stop',action='store_true')
    parser.add_argument('--native-fp32',action='store_true',help='Require FP32 fields; report cross-backend errors without the FP64 cutoff')
    args = parser.parse_args()
    for name in ('cpu', 'gpu', 'mpiexec', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    if not args.analyze_only:
        args.output.mkdir(parents=True, exist_ok=False)
    report = {'scheme': args.scheme, 'grid_intervals': args.size, 'decomposition_axis':args.axis,'cases': {}}
    clock_dtype=np.float32 if args.native_fp32 else np.float64
    finals = {};end_times={}
    for ranks in args.ranks:
        for backend in ('cpu', 'gpu'):
            label = f'{backend}_np{ranks}'
            continuous, clocks = run(args, backend, ranks, label,clock_dtype=clock_dtype,axis=args.axis)
            seed = continuous/'captured/checkpoints/step000000000005'
            resumed, resumed_clocks = run(args, backend, ranks, label+'_resumed', restore=seed,clock_dtype=clock_dtype,axis=args.axis)
            if resumed_clocks != clocks[5:]:
                raise AssertionError(f'{label}: exact clock history differs')
            final = Path('outdat/new/checkpoints/step000000000012/state.h5')
            with h5py.File(continuous/final) as state:
                if state['q0001'].shape != (args.size+1,)*3:
                    raise AssertionError('stored grid differs from --size')
                expected_dtype=np.dtype('float32' if args.native_fp32 else 'float64')
                if state['q0001'].dtype!=expected_dtype:
                    raise AssertionError('stored field precision differs from the requested validation')
            exact_error = compare_state(continuous/final, resumed/final, True)
            for name in ('control.bin', 'archives.bin', 'insitu_control.bin'):
                if (continuous/final.parent/name).read_bytes() != (resumed/final.parent/name).read_bytes():
                    raise AssertionError(f'{label}: exact {name} differs')
            finals[backend, ranks] = continuous/final
            end_times[backend,ranks] = clocks[-1][2]
            report['cases'][label] = {'steps': len(clocks), 'exact_scaled_error': exact_error,
                                     'exact_all_datasets_and_controls': True, 'last_clock': clocks[-1]}
            log = (continuous/'run.log').read_text()
            timing = re.search(r'ASTR_RUN_CONTROL_TIMING choices=(\d+) polls=(\d+)'
                               r' initial_selection_max_seconds=\s*(\S+) next_selection_max_seconds=\s*(\S+)'
                               r' reload_max_seconds=\s*(\S+)', log)
            if timing:
                report['cases'][label]['control_timing'] = dict(choices=int(timing[1]), polls=int(timing[2]),
                    initial_selection_seconds=float(timing[3]),next_selection_seconds=float(timing[4]),
                    reload_seconds=float(timing[5]))
            elif not args.analyze_only:
                raise AssertionError('new run is missing separate control timing')
        comparison = {}
        compare_state(finals['cpu', ranks], finals['gpu', ranks], False, comparison,
                      max_scaled_error=None if args.native_fp32 else 2e-10)
        phase_difference = abs(end_times['cpu',ranks]-end_times['gpu',ranks])
        comparison['actual_time_difference'] = phase_difference
        phase_tolerance=64*np.finfo(float).eps*max(1.0,abs(end_times['cpu',ranks]))
        if args.native_fp32:
            phase_tolerance=64*np.finfo(np.float32).eps*abs(end_times['cpu',ranks])
        if phase_difference > phase_tolerance:
            raise AssertionError('CPU/GPU fields are not at matching physical time to rounding precision')
        report['cases'][f'cpu_gpu_np{ranks}'] = comparison
    if not args.skip_time_stop:
        for backend in ('cpu', 'gpu'):
            _, clocks = run(args, backend, 1, backend+'_time_stop', stop_time=0.0025, checkpoint=False,clock_dtype=clock_dtype)
            report['cases'][backend+'_time_stop'] = {'last_clock': clocks[-1]}
    report['status'] = 'passed'
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
