"""Exercise ordinary C++ ASTR checkpoint events and continuation.

Only ASTR advances the state. Python prepares inputs and compares the existing
HDF5 protocol, including raw halo/cache words for same-environment exact runs.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

import h5py
import numpy as np

from run_cpp_tgv_gate import prepare_case, namelist, checkpoint, compare, snapshot, logged_steps


def datasets(path):
    values = {}
    with h5py.File(path) as source:
        source.visititems(lambda name, obj: values.update({name: obj[...]})
                          if isinstance(obj, h5py.Dataset) else None)
    return values


def exact_state(actual, expected):
    a, b = datasets(actual), datasets(expected)
    if a.keys() != b.keys():
        raise AssertionError('checkpoint dataset identities differ')
    for name in a:
        if a[name].shape != b[name].shape or a[name].dtype != b[name].dtype:
            raise AssertionError(f'checkpoint dataset layout: {name}')
        if a[name].tobytes() != b[name].tobytes():
            raise AssertionError(f'checkpoint dataset words differ: {name}')
    return len(a)


def restored_statistics(actual, reference):
    resumed = np.loadtxt(actual/'flowstate.dat', skiprows=1, ndmin=2)
    baseline = np.loadtxt(reference/'flowstate.dat', skiprows=1, ndmin=2)
    if resumed.shape[1] != 5 or resumed[:, 0].tolist() != [2., 3.]:
        raise AssertionError('restored TGV scalar sampling identities')
    selected = baseline[np.isin(baseline[:, 0], resumed[:, 0])]
    if not np.array_equal(resumed[:, :2], selected[:, :2]):
        raise AssertionError('restored TGV scalar step/time phase')
    return compare(resumed, selected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--mpiexec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ranks', type=int, choices=[1, 2], default=1)
    parser.add_argument('--mode', choices=['exact', 'nonexact', 'all'], default='all')
    parser.add_argument('--sync', choices=['dependency', 'explicit'], default='explicit')
    parser.add_argument('--memcheck', action='store_true')
    parser.add_argument('--automatic-reference', type=Path)
    parser.add_argument('--workspace', choices=['full', 'scalar', 'default'], default='full')
    parser.add_argument('--restore-source', type=Path)
    parser.add_argument('--backend', choices=['cpp', 'cpu', 'cuda-fortran'], default='cpp')
    parser.add_argument('--transport', choices=['pinned', 'device'], default='pinned')
    parser.add_argument('--automatic', action='store_true')
    parser.add_argument('--keep', type=int, choices=[1, 2], default=2)
    parser.add_argument('--no-test-observations', action='store_true')
    args = parser.parse_args()
    if args.automatic_reference is not None and args.mode != 'exact':
        parser.error('automatic history continuity is an exact-restart test; select --mode exact')
    if args.restore_source is not None and args.mode != 'nonexact':
        parser.error('an external step-two checkpoint is checked here as non-exact continuation')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    frozen = args.reference.resolve()/'evidence/standard_tgv_explicit10'
    original = frozen/'cpu_np1_keep_x/datin'
    env = os.environ.copy()
    for name in list(env):
        if name.startswith('ASTR_VALIDATION_'):
            del env[name]
    env.update(OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,vader,tcp', OMPI_MCA_osc='pt2pt',
               OMPI_MCA_coll='^cuda,hcoll,ucc', OMPI_MCA_opal_cuda_support='0',
               ASTR_GPU_FILTER_WORKSPACE=args.workspace, ASTR_GPU_HALO_TRANSPORT=args.transport,
               ASTR_GPU_SYNC_MODE=args.sync, ASTR_CFL_PROFILE_Y='1',
               ASTR_FORCE_MPI_TOPOLOGY='1,1,1' if args.ranks == 1 else '2,1,1',
               ASTR_VALIDATION_RHS_PREFIX='outdat/rhs', ASTR_VALIDATION_RHS_STEP='4')
    if args.workspace == 'default':
        del env['ASTR_GPU_FILTER_WORKSPACE']
    if args.no_test_observations:
        del env['ASTR_VALIDATION_RHS_PREFIX'], env['ASTR_VALIDATION_RHS_STEP']
    if args.transport == 'device':
        env.update(OMPI_MCA_pml='ucx', OMPI_MCA_opal_cuda_support='1')
    automatic = args.automatic or args.automatic_reference is not None
    if args.automatic and args.mode != 'nonexact':
        parser.error('--automatic without a frozen continuous reference selects --mode nonexact')
    references = {b: (frozen/f'{b}_np1_keep_x' if args.automatic_reference is None else
                      args.automatic_reference/('cpu_automatic' if b == 'cpu' else 'old_gpu_automatic')) /
                      'outdat/new/checkpoints/step000000000004/state.h5' for b in ('cpu', 'gpu')}
    reference_fields = {b: checkpoint(path) for b, path in references.items()}
    report = {'binary': str(args.binary.resolve()), 'ranks': args.ranks, 'cases': {},
              'reference_pair': compare(reference_fields['cpu'], reference_fields['gpu'])}

    def run(name, exact, restore=None, checkpoints=True, binary=None, backend=None, steps=4):
        binary = args.binary if binary is None else binary
        backend = args.backend if backend is None else backend
        directory = root/name
        prepare_case(directory, original, backend != 'cpu', backend == 'cpp', automatic, steps=steps)
        options = dict(directory='outdat/new', exact_restart=exact, host_budget_bytes=-1,
                       device_budget_bytes=-1, device_reserve_bytes=0)
        if restore is not None:
            options['restore_directory'] = str(restore)
        output = dict(output=options,
                      checkpoint=dict(enabled=checkpoints, mode='steps', interval_steps=2, keep=args.keep),
                      volume=dict(enabled=False), slices=dict(enabled=False))
        (directory/'datin/input.output').write_text(namelist(output))
        application = [str(binary.resolve()), 'run']
        if args.memcheck:
            application = ['compute-sanitizer', '--tool', 'memcheck', '--error-exitcode', '1',
                           '--leak-check', 'full', '--log-file',
                           str(directory/'sanitizer_rank%q{OMPI_COMM_WORLD_RANK}.log'), *application]
        command = [str(args.mpiexec.resolve()), '--bind-to', 'none', '-n', str(args.ranks), *application]
        run_env = env if backend == 'cpp' else env | {'ASTR_GPU_SYNC_MODE': 'explicit'}
        with (directory/'run.log').open('w') as log:
            subprocess.run(command, cwd=directory, env=run_env, stdin=subprocess.DEVNULL,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        if args.memcheck:
            for rank in range(args.ranks):
                text = (directory/f'sanitizer_rank{rank}.log').read_text()
                if 'ERROR SUMMARY: 0 errors' not in text or 'LEAK SUMMARY: 0 bytes leaked' not in text:
                    raise AssertionError('checkpoint memcheck/leak check failed')
        if checkpoints:
            control = directory/f'outdat/new/checkpoints/step{steps:012d}/control.bin'
            with control.open('rb') as stream:
                stream.read(8)
                contract = np.fromfile(stream, np.int64, 8)
            if contract.size != 8 or bool(int(contract[6]) & 1) != (args.workspace == 'full'):
                raise AssertionError('filter storage was not recorded in checkpoint identity')
            retained = sorted(p.name for p in control.parents[1].glob('step*') if p.is_dir())
            first = 2 if restore is None else 4
            expected = [f'step{step:012d}' for step in range(first, steps+1, 2)][-args.keep:]
            if retained != expected:
                raise AssertionError(f'checkpoint retention: {retained} != {expected}')
        return directory

    def state(directory, step=4):
        return directory/f'outdat/new/checkpoints/step{step:012d}/state.h5'

    plain = (run('output_off', True, checkpoints=False)
             if args.backend == 'cpp' and not args.no_test_observations else None)
    for mode in ('exact', 'nonexact') if args.mode == 'all' else (args.mode,):
        continuous = run(mode+'_continuous', mode == 'exact')
        if args.restore_source is not None:
            source = args.restore_source.resolve()
        elif args.keep == 1:
            source = state(run(mode+'_prefix', mode == 'exact', steps=2), 2).parent
        else:
            source = state(continuous, 2).parent
        resumed = run(mode+'_resumed', mode == 'exact', source)
        if args.automatic:
            # Non-exact restoration deliberately starts a new controller
            # history. Compare independent ASTR restorations at that phase,
            # not the continuous trajectory at a different physical time.
            restored_references = {
                b: run(mode+'_'+b+'_restored', False, source,
                       binary=args.reference.resolve()/'bin'/exe, backend=backend)
                for b, exe, backend in [('cpu', 'astr_cpu', 'cpu'), ('gpu', 'astr_cuda_fortran', 'cuda-fortran')]}
            reference_fields = {b: checkpoint(state(d)) for b, d in restored_references.items()}
            reference = reference_fields['cpu']
        else:
            reference = checkpoint(state(continuous))
        candidate = checkpoint(state(resumed))
        pressure = .4*(candidate[..., 4]-.5*np.sum(candidate[..., 1:4]**2, axis=-1)/candidate[..., 0])
        if not (candidate[..., 0] > 0).all() or not (pressure > 0).all():
            raise AssertionError('nonpositive checkpoint physical state')
        identity_reference = restored_references['cpu'] if args.automatic else continuous
        with h5py.File(state(identity_reference)) as left, h5py.File(state(resumed)) as right:
            if not np.array_equal(left['identity'][...], right['identity'][...]):
                raise AssertionError('restored checkpoint step/clock/precision identity')
            if mode == 'nonexact':
                if int(right['identity'][12]) != 11 or 'q0006' in right or right['rank_extras'].size:
                    raise AssertionError('non-exact checkpoint retained primitive caches or redundant rank data')
                if (resumed/'outdat/new/resources/geometry.h5').exists():
                    raise AssertionError('non-exact checkpoint retained the exact geometry resource')
        item = {'continuation': compare(candidate, reference), 'frozen': {}, 'restored_from': str(source)}
        expected_rows = (logged_steps(restored_references['cpu'], 2, first_step=3) if args.automatic
                         else logged_steps(continuous)[2:])
        item['resumed_step_sequence'] = compare(logged_steps(resumed, 2, first_step=3), expected_rows)
        item['restored_statistics'] = restored_statistics(resumed, identity_reference)
        if mode == 'exact':
            item['identical_datasets'] = exact_state(state(resumed), state(continuous))
        for backend in ('cpu', 'gpu'):
            item['frozen'][backend] = compare(candidate, reference_fields[backend])
            if args.automatic:
                item['restored_'+backend+'_steps'] = compare(logged_steps(resumed, 2, first_step=3),
                    logged_steps(restored_references[backend], 2, first_step=3))
            if args.automatic_reference is not None:
                item['automatic_'+backend] = compare(logged_steps(continuous), logged_steps(references[backend].parents[4]))
        if plain is not None:
            for rank in range(args.ranks):
                filename = f'outdat/rhs.cpp_final.step00000004.rk01.rank{rank:08d}.bin'
                a, b = snapshot(continuous/filename), snapshot(plain/filename)
                if not np.array_equal(a.view(np.uint64), b.view(np.uint64)):
                    raise AssertionError('checkpoint output changed physical q words')
            item['output_toggle_physical_q_identical'] = True
        # The resumed log intentionally only has steps 3/4; its terminal
        # step/clock comes from the checkpoint identity, not log row count.
        report['cases'][mode] = item
        (root/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(mode, 'checkpoint and continuation passed',
              '; output-on/off physical q bits equal' if plain is not None else '', flush=True)


if __name__ == '__main__':
    main()
