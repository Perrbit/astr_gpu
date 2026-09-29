#!/usr/bin/env python3
"""Check full/scalar and split restarts against a completed scalar boundary case."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import h5py
import numpy as np

from prepare_tgv_case import set_controller_steps, set_restart
from compare_flowstate import read_flowstate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=Path, required=True)
    parser.add_argument('--case', choices=('channel', 'curve'), required=True)
    args = parser.parse_args()
    reference, out, exe = args.reference.resolve(), args.output.resolve(), args.gpu.resolve()
    helpers = Path(__file__).resolve().parent
    name = 'input.chl' if args.case == 'channel' else 'input.flatplate'
    topology = '2,1,1' if args.case == 'channel' else '1,2,1'
    with h5py.File(reference/'outdat/flowfield.h5') as handle:
        assert handle['nstep'][()].item() == 100
        final_time = handle['time'][()].item()
    out.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items() if not k.startswith('ASTR_')}
    env.update(ASTR_FORCE_MPI_TOPOLOGY=topology, ASTR_GPU_SYNC_MODE='explicit',
               ASTR_GPU_HALO_TRANSPORT='pinned', ASTR_GPU_PRECISION_MODE='fp64')
    if args.case == 'channel':
        env.update(ASTR_CHANNEL_FORCE_MODE='fixed', ASTR_CHANNEL_FORCE_FIXED='1.d-4')
    result = dict(status='running', reference=str(reference), topology=topology,
                  gpu_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(), completed=[])

    def save():
        (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')

    def run(command, log, cwd, runtime):
        with log.open('w') as stream:
            subprocess.run(command, cwd=cwd, env=runtime, stdout=stream,
                           stderr=subprocess.STDOUT, check=True, timeout=300)

    def launch(case, label, workspace, step):
        runtime = dict(env, ASTR_GPU_FILTER_WORKSPACE=workspace)
        log = case/(label+'.log')
        run(['mpirun', '-np', '2', str(exe), 'run', 'datin/'+name], log, case, runtime)
        if 'The job is done!' not in log.read_text():
            raise ValueError(f'no normal completion: {log}')
        with h5py.File(case/'outdat/flowfield.h5') as handle:
            if handle['nstep'][()].item() != step:
                raise ValueError('checkpoint step mismatch')
            if step == 100 and abs(handle['time'][()].item()-final_time) > 1e-14:
                raise ValueError('checkpoint time mismatch')

    save()
    try:
        for mode in ('full', 'restart'):
            case = out/mode
            shutil.copytree(reference/'datin', case/'datin')
            for directory in ('bakup', 'outdat', 'islice', 'jslice', 'kslice', 'monitor', 'testout'):
                (case/directory).mkdir()
            set_restart(case/'datin'/name, 'f')
            step = 50 if mode == 'restart' else 100
            set_controller_steps(case/'datin/controller', step, step, 1)
            launch(case, 'fresh', 'scalar' if mode == 'restart' else 'full', step)
            if mode == 'restart':
                shutil.copytree(case/'outdat', case/'checkpoint_step50')
                set_restart(case/'datin'/name, 't')
                set_controller_steps(case/'datin/controller', 100, 100, 1)
                launch(case, 'restart', 'scalar', 100)
                if 'checkpoint file read' not in (case/'restart.log').read_text():
                    raise ValueError('restart state was not loaded')
            run([sys.executable, str(helpers/'compare_flowfield_h5.py'),
                 '--cpu', str(reference), '--gpu', str(case),
                 '--atol', '0' if mode == 'full' else '1e-10', '--rtol', '0',
                 '--report', str(out/(mode+'_field.txt'))],
                out/(mode+'_compare.log'), helpers, env)
            statistics_reference = reference
            if mode == 'restart':
                header, baseline = read_flowstate(reference)
                restart_header, restarted = read_flowstate(case)
                if header != restart_header or header[0] != 'nstep':
                    raise ValueError('restart statistics header mismatch')
                # Channel replaces its text log; the flatplate path appends it.
                expected_steps = np.arange(50 if args.case == 'channel' else 1, 101)
                if not np.array_equal(restarted[:, 0], expected_steps):
                    raise ValueError('restart statistics step sequence mismatch')
                selected = baseline[np.isin(baseline[:, 0], expected_steps)]
                if not np.array_equal(selected[:, 0], expected_steps):
                    raise ValueError('reference statistics missing restart window')
                statistics_reference = out/'reference_restart_statistics.dat'
                np.savetxt(statistics_reference, selected, header=' '.join(header), comments='')
            run([sys.executable, str(helpers/'compare_flowstate.py'),
                 '--cpu', str(statistics_reference), '--gpu', str(case),
                 '--atol', '1e-10', '--rtol', '1e-10',
                 '--report', str(out/(mode+'_statistics.txt'))],
                out/(mode+'_statistics.log'), helpers, env)
            result['completed'].append(mode)
            save()
        result['status'] = 'passed-bounded-regression-not-physical-validation'
    except BaseException as error:
        result.update(status='failed', error=str(error))
        save()
        raise
    save()
    print(result['status'])


if __name__ == '__main__':
    main()
