#!/usr/bin/env python3
"""Restart and device-memory gates before the bounded air5 SBLI long run."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import h5py

from prepare_air5_c4_case import replace_after_marker
from prepare_air5_sbli_case import prepare
from compare_q_validation_snapshots import compare_snapshot_sets

ROOT = Path(__file__).resolve().parents[2]


def environment(topology):
    env = {k: v for k, v in os.environ.items() if not k.startswith('ASTR_')}
    env.update(ASTR_FORCE_MPI_TOPOLOGY=topology, ASTR_GPU_SYNC_MODE='explicit',
        ASTR_GPU_PRECISION_MODE='fp64', ASTR_AIR5_C4_CONSERVATION='f',
        ASTR_AIR5_SOURCE_MODE='coupled', ASTR_GPU_FILTER_WORKSPACE='full',
        OMPI_MCA_coll='^hcoll,ucc', OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,vader,tcp',
        OMPI_MCA_osc='pt2pt', OMPI_MCA_opal_cuda_support='0', UCX_MEMTYPE_CACHE='n',
        CUDA_VISIBLE_DEVICES='0,1')
    return env


def set_value(path, marker, value):
    lines = path.read_text().splitlines()
    replace_after_marker(lines, marker, value)
    path.write_text('\n'.join(lines)+'\n')


def run_case(case, exe, np_, stage, log_name, sanitizer=False, expected_failure=None):
    env = environment(f'{np_},1,1')
    env.update(ASTR_VALIDATION_RHS_PREFIX='validation/air5', ASTR_VALIDATION_RHS_STEP=str(stage))
    command = [shutil.which('mpirun'), '--oversubscribe', '-np', str(np_)]
    if sanitizer:
        command += [shutil.which('compute-sanitizer'), '--tool', 'memcheck',
                    '--leak-check', 'full', '--error-exitcode', '99']
    command += [str(exe), 'run', 'datin/input.air5_c4']
    with (case/log_name).open('w') as log:
        result = subprocess.run(['timeout', '--kill-after=10s', '900s', *command], cwd=case,
                                env=env, stdout=log, stderr=subprocess.STDOUT)
    text = (case/log_name).read_text()
    if expected_failure:
        if result.returncode != 4 or expected_failure not in text:
            raise ValueError(f'expected diffusion-domain failure was not reproduced: {result.returncode}')
        return text
    result.check_returncode()
    if 'The job is done!' not in text:
        raise ValueError('missing normal completion')
    if sanitizer and ('ERROR SUMMARY: 0 errors' not in text or
                      'LEAK SUMMARY: 0 bytes leaked' not in text):
        raise ValueError('memcheck did not report zero errors and leaks')
    return text


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    exe = args.executable.resolve()
    if args.mode == 'failure-replay':
        if args.baseline is None:
            raise ValueError('failure replay requires a baseline case directory')
        case = out/('gpu' if args.use_gpu == 't' else 'cpu')
        shutil.copytree(args.baseline/'datin', case/'datin')
        (case/'outdat').mkdir()
        (case/'validation').mkdir()
        for name in ('flowfield.h5','auxiliary.txt'):
            shutil.copy2(args.baseline/'outdat'/name, case/'outdat'/name)
        set_value(case/'datin/input.air5_c4', 'lrestar', 't')
        set_value(case/'datin/input.air5_c4',
                  'nondimen,diffterm,lfilter,lreadgrid,lfftz,limmbou,ltimrpt,lcomb',
                  f'f,t,f,f,f,f,t,t,{args.use_gpu}')
        set_value(case/'datin/controller', 'maxstep,feqchkpt,feqwsequ,feqslice,feqlist,feqavg',
                  '750,50,1000000,1000000,1,1000000')
        marker = ('AIR5_DIFFUSION_STATE_FAILURE' if args.use_gpu == 't' else
                  'fixed air5 CPU diffusion failed, status=4')
        text = run_case(case, exe, 2, 741, 'failure_replay.log', expected_failure=marker)
        result = dict(status='expected-failure-reproduced-not-pass',
                      details=[line for line in text.splitlines() if marker in line])
    elif args.mode == 'restart':
        continuous, split = out/'continuous', out/'restart'
        for case, maximum in ((continuous, 2), (split, 1)):
            prepare(case, '31,31,7', maximum, '5.d-10', 't')
        run_case(continuous, exe, 2, 2, 'continuous.log')
        set_value(split/'datin/controller', 'maxstep,feqchkpt,feqwsequ,feqslice,feqlist,feqavg',
                  '1,1,10,50,1,50')
        run_case(split, exe, 2, 999999, 'fresh.log')
        with h5py.File(split/'outdat/flowfield.h5') as stream:
            if 'tv' not in stream or int(stream['nstep'][()].item()) != 1:
                raise ValueError('restart checkpoint missing Tv or wrong phase')
        set_value(split/'datin/input.air5_c4', 'lrestar', 't')
        set_value(split/'datin/controller', 'maxstep,feqchkpt,feqwsequ,feqslice,feqlist,feqavg',
                  '2,2,10,50,1,50')
        text = run_case(split, exe, 2, 2, 'restart.log')
        if 'checkpoint file read' not in text:
            raise ValueError('restart was not used')
        comparison = compare_snapshot_sets(continuous/'validation/air5', split/'validation/air5',
            ('post_chemistry','pre_rhs','post_update','post_transport'),
            1e-9, 1e-10, active_only=True, scaled_tol=5e-9)
        if not comparison.passed:
            raise ValueError(f'restart comparison failed: {comparison}')
        result = dict(status='restart-pass', files=comparison.file_count,
                      max_scaled=comparison.max_scaled, max_abs=comparison.max_abs)
    else:
        case = out/'gpu'
        prepare(case, '31,31,7', 1, '5.d-10', 't')
        run_case(case, exe, 1, 1, 'memcheck.log', sanitizer=True)
        result = dict(status='memcheck-pass', errors=0, leaked_bytes=0)
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('restart','memcheck','failure-replay'), required=True)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--use-gpu', choices=('t','f'), default='t', help='Failure replay only')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--executable', type=Path,
        default=ROOT/'tests/gpu_validation/out/c4_cuda_build_4/bin/astr')
    run(parser.parse_args())
