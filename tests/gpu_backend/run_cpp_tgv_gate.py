"""Ordinary ASTR executable: frozen FP64 TGV fields, scalars and step control.

Only case preparation and comparisons live here. All time advancement is ASTR.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/input'))
from convert_legacy_input import convert_case, convert_controller, namelist


def compare(actual, reference):
    if actual.shape != reference.shape or not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise AssertionError('field/statistic layout or finiteness')
    errors = [float(np.max(abs(actual[..., n]-reference[..., n])) /
                    max(1., float(np.max(abs(reference[..., n]))))) for n in range(actual.shape[-1])]
    if max(errors) > 2e-10:
        raise AssertionError(f'scaled comparison exceeds 2e-10: {errors}')
    return errors


def checkpoint(path):
    with h5py.File(path) as source:
        return np.stack([source[f'q{n:04}'][...].transpose(2, 1, 0) for n in range(1, 6)], axis=-1)


def snapshot(path):
    with path.open('rb') as source:
        im, jm, km, hm, nq = np.fromfile(source, np.int32, 5)
        data = np.fromfile(source, np.float64)
    shape = (im+2*hm+1, jm+2*hm+1, km+2*hm+1, nq)
    if data.size != np.prod(shape):
        raise AssertionError('terminal q snapshot size')
    return data.reshape(shape, order='F')[hm:hm+im+1, hm:hm+jm+1, hm:hm+km+1]


def logged_steps(directory, count=4, first_step=1):
    pattern = re.compile(r'ASTR_TIMESTEP complete_step=(\d+)\s+time_start=\s*(\S+)\s+time_end=\s*(\S+)'
                         r'\s+dt_used=\s*(\S+)\s+dt_next=\s*(\S+)')
    rows = [tuple(map(float, match.groups())) for match in pattern.finditer((directory/'run.log').read_text())]
    if len(rows) != count or [row[0] for row in rows] != list(range(first_step, first_step+count)):
        raise AssertionError(f'{count} complete timestep records required')
    return np.asarray(rows)


def cfl_log(directory, count=4):
    prefix = 'ASTR_CFL directional/local_sum/upper_bound='
    rows = [list(map(float, line.split('=', 1)[1].split()))
            for line in (directory/'run.log').read_text().splitlines() if line.startswith(prefix)]
    if len(rows) != count:
        raise AssertionError(f'{count} complete-state CFL diagnostics required')
    return np.asarray(rows)


def prepare_case(directory, original, gpu, cpp, automatic, grid=16, steps=4, dt=.001, max_dt=.002):
    (directory/'datin').mkdir(parents=True, exist_ok=False)
    (directory/'outdat/new').mkdir(parents=True)
    groups = convert_case((original/'input.tgv').read_text())
    groups['case_options']['use_gpu'] = gpu
    dimensions = (grid, grid, grid) if isinstance(grid, int) else grid
    groups['mesh'].update(zip(('ia', 'ja', 'ka'), dimensions))
    groups['numerics'].update(convection_form='standard', viscous_form='standard', filter_scheme='explicit10')
    groups['precision_options'] = {'gpu_precision_mode': 'fp64'}
    (directory/'datin/input.case').write_text(namelist(groups))
    controls = convert_controller((original/'controller').read_text())
    controls['run_control']['max_steps'] = steps
    controls['timestep'] = (dict(mode='auto', cfl_target=.5, initial_dt=dt, max_dt=max_dt, growth_factor=1.1)
                            if automatic else dict(mode='fixed', fixed_dt=dt))
    (directory/'datin/input.run').write_text(namelist(controls))
    outputs = dict(output=dict(directory='outdat/new', exact_restart=not cpp, host_budget_bytes=-1,
                               device_budget_bytes=-1, device_reserve_bytes=0),
                   checkpoint=dict(enabled=not cpp, mode='steps', interval_steps=steps, keep=2),
                   volume=dict(enabled=False), slices=dict(enabled=False))
    (directory/'datin/input.output').write_text(namelist(outputs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--mpiexec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['fixed', 'automatic', 'all'], default='all')
    parser.add_argument('--sanitizer', choices=['memcheck', 'initcheck'])
    parser.add_argument('--grid', type=int, default=16)
    parser.add_argument('--steps', type=int, default=4)
    parser.add_argument('--dt', type=float, default=.001)
    parser.add_argument('--max-dt', type=float, default=.002)
    parser.add_argument('--fresh-reference', action='store_true')
    options = parser.parse_args()
    if (options.grid, options.steps, options.dt) != (16, 4, .001) and not options.fresh_reference:
        parser.error('changed grid/steps/dt requires --fresh-reference from the frozen binaries')
    options.output = options.output.resolve()
    options.output.mkdir(parents=True, exist_ok=False)
    frozen = options.reference.resolve()
    evidence = frozen/'evidence/standard_tgv_explicit10'
    original = evidence/'cpu_np1_keep_x/datin'
    environment = os.environ.copy()
    environment.update(OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,vader,tcp', OMPI_MCA_osc='pt2pt',
                       OMPI_MCA_coll='^cuda,hcoll,ucc', OMPI_MCA_opal_cuda_support='0',
                       ASTR_GPU_FILTER_WORKSPACE='full', ASTR_CFL_PROFILE_Y='1',
                       ASTR_FORCE_MPI_TOPOLOGY='1,1,1', ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',
                       ASTR_VALIDATION_RHS_STEP=str(options.steps))
    report = {'reference': str(frozen), 'binary': str(options.binary.resolve()), 'cases': {},
              'grid': options.grid, 'steps': options.steps, 'fixed_dt': options.dt, 'max_dt': options.max_dt}
    final = f'outdat/new/checkpoints/step{options.steps:012d}/state.h5'

    def run(name, binary, gpu, cpp, automatic, sync='explicit'):
        directory = options.output/name
        prepare_case(directory, original, gpu, cpp, automatic, options.grid, options.steps, options.dt, options.max_dt)
        env = environment | {'ASTR_GPU_SYNC_MODE': sync}
        application = [str(binary.resolve()), 'run']
        if cpp and options.sanitizer:
            application = ['compute-sanitizer', '--tool', options.sanitizer, '--error-exitcode', '1',
                           '--log-file', str(directory/'sanitizer.log'),
                           *(['--leak-check', 'full'] if options.sanitizer == 'memcheck' else []), *application]
        command = [str(options.mpiexec), '--bind-to', 'none', '-n', '1', *application]
        with (directory/'run.log').open('w') as log:
            subprocess.run(command, cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        if cpp and 'ASTR_GPU_CPP rank=0 backend=cuda' not in (directory/'run.log').read_text():
            raise AssertionError('ordinary ASTR did not report the C++ backend')
        q = (snapshot(directory/f'outdat/rhs.cpp_final.step{options.steps:08d}.rk01.rank00000000.bin') if cpp else
             checkpoint(directory/final))
        rho = q[..., 0]
        pressure = .4*(q[..., 4]-.5*np.sum(q[..., 1:4]**2, axis=-1)/rho)
        if not np.isfinite(q).all() or not (rho > 0).all() or not (pressure > 0).all():
            raise AssertionError('nonpositive/nonfinite physical state')
        return directory, q

    if options.mode in ('fixed', 'all'):
        fixed_references = {}
        for backend, binary, gpu in [('cpu', frozen/'bin/astr_cpu', False),
                                     ('gpu', frozen/'bin/astr_cuda_fortran', True)]:
            fixed_references[backend] = (run(backend+'_fixed_reference', binary, gpu, False, False)
                                         if options.fresh_reference else
                                         (evidence/f'{backend}_np1_keep_x',
                                          checkpoint(evidence/f'{backend}_np1_keep_x'/final)))
        report['cases']['fixed_reference_crosscheck'] = compare(fixed_references['cpu'][1], fixed_references['gpu'][1])
        for sync in ('dependency', 'explicit'):
            directory, q = run('cpp_fixed_'+sync, options.binary, True, True, False, sync)
            result = {}
            for backend in ('cpu', 'gpu'):
                reference, qref = fixed_references[backend]
                result[backend] = dict(
                    q=compare(q, qref),
                    scalars=compare(np.loadtxt(directory/'flowstate.dat', skiprows=1),
                                    np.loadtxt(reference/'flowstate.dat', skiprows=1)),
                    steps=compare(logged_steps(directory, options.steps), logged_steps(reference, options.steps)))
            report['cases'][directory.name] = result
            print(directory.name, result, flush=True)
        directory, q = run('cpu_in_cpp_build', options.binary, False, False, False)
        report['cases'][directory.name] = compare(q, fixed_references['cpu'][1])
        print(directory.name, report['cases'][directory.name], flush=True)
    if options.mode in ('automatic', 'all'):
        states = {}
        for backend, binary, gpu, cpp in [('cpu', frozen/'bin/astr_cpu', False, False),
                                         ('old_gpu', frozen/'bin/astr_cuda_fortran', True, False),
                                         ('cpp', options.binary, True, True)]:
            if backend == 'cpp':
                report['cases']['automatic_reference_crosscheck'] = compare(states['cpu'][1], states['old_gpu'][1])
            states[backend] = run(backend+'_automatic', binary, gpu, cpp, True, 'dependency' if cpp else 'explicit')
        directory, q = states['cpp']
        for backend in ('cpu', 'old_gpu'):
            reference, qref = states[backend]
            result = dict(q=compare(q, qref), steps=compare(logged_steps(directory, options.steps), logged_steps(reference, options.steps)),
                          cfl=compare(cfl_log(directory, options.steps), cfl_log(reference, options.steps)),
                          scalars=compare(np.loadtxt(directory/'flowstate.dat', skiprows=1),
                                          np.loadtxt(reference/'flowstate.dat', skiprows=1)))
            report['cases']['automatic_vs_'+backend] = result
            print('automatic_vs_'+backend, result, flush=True)
    (options.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
