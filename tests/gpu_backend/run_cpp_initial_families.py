"""Existing HIT/vortex/Sod initializers through the ordinary C++ flow backend."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from run_cpp_tgv_gate import prepare_case, namelist, compare, logged_steps, cfl_log
from run_cpp_characteristic_gate import read_phase
from run_cpp_insitu_gate import compare_hdf

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = {
    'hit': dict(topology=[1, 1, 2], scheme='643e', mach=.1, viscous=True, filtering=True),
    '2dvort': dict(topology=[1, 2, 1], scheme='643e', mach=.5, viscous=False, filtering=True),
    'sod': dict(topology=[2, 1, 1], scheme='543e', mach=3., viscous=False, filtering=False),
}


def stages(current, reference):
    result = {}
    for phase in ('pre_rhs', 'pre_rhs_primitives', 'post_source', 'post_update'):
        paths = sorted((reference/'outdat').glob(f'rhs.{phase}.*.bin'))
        if len(paths) != 24:
            raise AssertionError('four RK3 steps on two ranks are required')
        maximum = 0.
        for path in paths:
            a, b = read_phase(current/'outdat'/path.name, phase), read_phase(path, phase)
            maximum = max(maximum, max(compare(a, b)))
            if phase in ('pre_rhs', 'post_update'):
                internal = a[..., 4]-.5*np.sum(a[..., 1:4]**2, axis=-1)/a[..., 0]
                if not (a[..., 0] > 0).all() or not (internal > 0).all():
                    raise AssertionError('nonpositive initial-family physical state')
            elif phase == 'pre_rhs_primitives' and any(not (a[..., c] > 0).all() for c in (0, 4, 5)):
                raise AssertionError('nonpositive initial-family primitive density/pressure/temperature')
        result[phase] = maximum
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('binary', 'template', 'output', 'mpiexec'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--stage', choices=('reference', 'candidate'), required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--family', choices=tuple(FAMILIES), action='append')
    parser.add_argument('--memcheck', action='store_true')
    parser.add_argument('--checkpoint', action='store_true')
    args = parser.parse_args()
    if (args.stage == 'candidate') != (args.reference is not None):
        parser.error('candidate requires --reference; reference stage must not receive it')
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.binary, root/'tested_astr'); shutil.copy2(__file__, root/'runner.py')
    report = dict(binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        stage=args.stage, precision='fp64', formal_output=args.checkpoint, checkpoint_tested=args.checkpoint, families={})
    families = args.family or list(FAMILIES)

    def run(family, backend, observed=True, name=None, steps=4, restore=None, writing=None):
        config = FAMILIES[family]; case = root/family/(name or backend)
        prepare_case(case, args.template.resolve(), backend != 'cpu', args.stage == 'candidate', False,
            grid=16, steps=steps, dt=1e-5, numerics=dict(conschm=config['scheme'], difschm='643e',
                recon_schem=3, lchardecomp=False, lfilter=config['filtering'],
                filter_scheme='explicit10', convection_form='standard', viscous_form='standard'),
            physics=dict(diffterm=config['viscous'], mach=config['mach']),
            case_options=dict(flowtype=family),
            mesh=dict(lihomo=family != 'sod', ljhomo=True, lkhomo=True))
        # Retain the named fixture's model groups; only the existing Sod end states differ.
        if family == 'sod':
            text = (case/'datin/input.case').read_text()
            old = '  bctype = 1, 1, 1, 1, 1, 1'
            if text.count(old) != 1:
                raise AssertionError('Sod fixture boundary declaration')
            (case/'datin/input.case').write_text(text.replace(old, '  bctype = 50, 50, 1, 1, 1, 1'))
        if family == 'hit' and restore is None:
            if args.stage == 'candidate' or backend == 'gpu':
                source = (args.reference/family/'gpu/datin/velocity.h5' if args.stage == 'candidate'
                          else root/family/'cpu/datin/velocity.h5')
                shutil.copyfile(source, case/'datin/velocity.h5')
            else:
                subprocess.run([sys.executable, str(ROOT/'tests/gpu_validation/generate_hit_velocity.py'),
                    '--output', str(case/'datin/velocity.h5'), '--grid', '16,16,16', '--amplitude', '.05'], check=True)
        (case/'datin/input.output').write_text(namelist(dict(output=dict(directory='outdat/new', exact_restart=True,
            restore_directory=str(restore or ''), checkpoint_statistics=False,
            host_budget_bytes=-1, device_budget_bytes=-1, device_reserve_bytes=0),
            checkpoint=dict(enabled=args.checkpoint if writing is None else writing,
                mode='steps', interval_steps=2, keep=2, initial_frame=False),
            volume=dict(enabled=False), slices=dict(enabled=False))))
        env = {key: value for key, value in os.environ.items() if not key.startswith('ASTR_')}
        env.update(OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,vader,tcp', OMPI_MCA_osc='pt2pt',
            OMPI_MCA_coll='^cuda,hcoll,ucc', OMPI_MCA_opal_cuda_support='0',
            ASTR_GPU_HALO_TRANSPORT='pinned', ASTR_GPU_SYNC_MODE='explicit',
            ASTR_FORCE_MPI_TOPOLOGY=','.join(map(str, config['topology'])),
            ASTR_VALIDATION_RHS_PREFIX='outdat/rhs', ASTR_VALIDATION_RHS_STEP='0' if observed else '4',
            ASTR_VALIDATION_RHS_STEP_END='4')
        command = [str(args.mpiexec.resolve()), '--bind-to', 'none', '-n', '2']
        if args.memcheck and args.stage == 'candidate' and observed and name is None:
            command += ['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer', '--tool', 'memcheck',
                '--leak-check', 'full', '--error-exitcode', '99', '--log-file', str(case/'memcheck.%p.log')]
        with (case/'run.log').open('w') as log:
            subprocess.run(command+[str(root/'tested_astr'), 'run'], cwd=case, env=env,
                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, check=True)
        if args.stage == 'candidate' and (case/'run.log').read_text().count('backend=cuda') != 2:
            raise AssertionError('actual C++ two-device initializer consumer')
        return case

    def continuation(family, backend, full):
        prefix = run(family, backend, name=backend+'_prefix', steps=2)
        tail = run(family, backend, name=backend+'_tail', restore=prefix/'outdat/new/checkpoints/step000000000002')
        final = Path('outdat/new/checkpoints/step000000000004')
        compare_hdf(full/final/'state.h5', tail/final/'state.h5', exact=True)
        if (full/final/'control.bin').read_bytes() != (tail/final/'control.bin').read_bytes():
            raise AssertionError('initializer exact restart controls')
        if not np.array_equal(logged_steps(full)[2:], logged_steps(tail, 2, 3)):
            raise AssertionError('initializer exact restart clocks')
        a = np.loadtxt(full/'flowstate.dat', skiprows=1, ndmin=2)
        b = np.loadtxt(tail/'flowstate.dat', skiprows=1, ndmin=2)
        if not np.array_equal(b, a[np.isin(a[:, 0], b[:, 0])]):
            raise AssertionError('initializer exact restart scalar sequence')
        if family == 'hit':
            if (tail/'datin/velocity.h5').exists():
                raise AssertionError('HIT restart test must use the frozen resource, not a local velocity file')
            if (prefix/'datin/velocity.h5').read_bytes() != (tail/'outdat/new/resources/velocity.h5').read_bytes():
                raise AssertionError('HIT frozen velocity resource differs')
            if backend == 'cpu':
                a = np.loadtxt(full/'turbstats.dat', skiprows=1, ndmin=2)
                b = np.loadtxt(tail/'turbstats.dat', skiprows=1, ndmin=2)
                if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.array_equal(b, a[np.isin(a[:, 0], b[:, 0])]):
                    raise AssertionError('repaired CPU turbulence report must be finite and exact on continuation')
        return dict(exact_state_control_scalars=True, frozen_hit_resource_tested=family == 'hit')

    for family in families:
        if args.stage == 'reference':
            cpu = run(family, 'cpu'); gpu = run(family, 'gpu')
            row = dict(cpu_old_gpu=stages(gpu, cpu))
            row['scalar_scaled'] = compare(np.loadtxt(gpu/'flowstate.dat', skiprows=1, ndmin=2),
                np.loadtxt(cpu/'flowstate.dat', skiprows=1, ndmin=2))
            if not np.array_equal(logged_steps(cpu), logged_steps(gpu)):
                raise AssertionError('original initializer clocks differ')
            if args.checkpoint:
                row['continuation'] = {backend: continuation(family, backend, full)
                                       for backend, full in (('cpu', cpu), ('gpu', gpu))}
        else:
            current = run(family, 'cpp'); row = {}
            for backend in ('cpu', 'gpu'):
                original = args.reference/family/backend
                row[backend] = dict(stages=stages(current, original),
                    scalars=compare(np.loadtxt(current/'flowstate.dat', skiprows=1, ndmin=2),
                                    np.loadtxt(original/'flowstate.dat', skiprows=1, ndmin=2)))
                if not np.array_equal(logged_steps(current), logged_steps(original)):
                    raise AssertionError('C++ initializer clocks differ')
                compare(cfl_log(current), cfl_log(original))
            quiet = run(family, 'quiet', observed=False, writing=False)
            for rank in range(2):
                path = f'outdat/rhs.cpp_final.step00000004.rk01.rank{rank:08d}.bin'
                if (current/path).read_bytes() != (quiet/path).read_bytes():
                    raise AssertionError('initializer observation changed q/halo words')
            if not np.array_equal(logged_steps(current), logged_steps(quiet)):
                raise AssertionError('initializer observation changed clock')
            if args.memcheck:
                logs = list(current.glob('memcheck.*.log'))
                if len(logs) != 2 or any('ERROR SUMMARY: 0 errors' not in p.read_text() or
                    'LEAK SUMMARY: 0 bytes leaked' not in p.read_text() for p in logs):
                    raise AssertionError('initializer per-rank memory evidence')
            row.update(observation_full_q_halo_clock_words_equal=True, memcheck=args.memcheck)
            if args.checkpoint:
                row['continuation'] = continuation(family, 'cpp', current)
        report['families'][family] = dict(config=FAMILIES[family], **row)
        (root/'progress.json').write_text(json.dumps(report, indent=2)+'\n')
    (root/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
