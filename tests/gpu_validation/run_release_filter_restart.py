#!/usr/bin/env python3
"""Bounded FP64 TGV filter/restart checks using caller-supplied release binaries."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np

from prepare_tgv_case import set_restart, set_controller_steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu', type=Path, required=True)
    parser.add_argument('--gpu', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    helpers = root/'tests/gpu_validation'
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = {k:v for k,v in os.environ.items() if not k.startswith('ASTR_')}
    env.update(ASTR_FORCE_MPI_TOPOLOGY='2,1,1', ASTR_GPU_SYNC_MODE='explicit',
               ASTR_GPU_HALO_TRANSPORT='pinned', ASTR_GPU_PRECISION_MODE='fp64')
    result = dict(status='running', grid=[32,32,32], topology=[2,1,1], dt=1e-3,
        final_step=100, restart_step=50, completed=[],
        executable_sha256={k:hashlib.sha256(p.read_bytes()).hexdigest()
                           for k,p in [('cpu',args.cpu),('gpu',args.gpu)]})

    def save():
        (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')

    def command(cmd, log, cwd=root, runtime=env):
        with log.open('w') as stream:
            subprocess.run(cmd, cwd=cwd, env=runtime, stdout=stream,
                           stderr=subprocess.STDOUT, timeout=180, check=True)

    def launch(case, exe, workspace, label):
        runtime = dict(env)
        if case.name == 'cpu':
            runtime['ASTR_VALIDATION_RK_SNAPSHOT'] = 'outdat/rk_complete_snapshot.h5'
        if workspace:
            runtime['ASTR_GPU_FILTER_WORKSPACE'] = workspace
        log = case/(label+'.log')
        command(['mpirun','-np','2',str(exe.resolve()),'run','datin/input.tgv'],
                log, case, runtime)
        if 'The job is done!' not in log.read_text():
            raise ValueError(f'no normal completion: {log}')

    def compare(a, b, name, atol, rtol):
        command([sys.executable,str(helpers/'compare_flowfield_h5.py'),
            '--cpu',str(a), '--gpu',str(b), '--report',str(out/(name+'.txt')),
            '--atol',str(atol),'--rtol',str(rtol)], out/(name+'.log'))

    save()
    try:
        for name, gpu, workspace, steps in [('cpu',False,None,100),
                ('full',True,'full',100), ('default',True,None,100),
                ('restart',True,None,50)]:
            case = out/name
            command([sys.executable,str(helpers/'prepare_tgv_case.py'),
                '--src-case',str(root/'examples/Taylor_Green_Vortex'),
                '--dst-case',str(case), '--use-gpu','t' if gpu else 'f',
                '--grid','32,32,32','--maxstep',str(steps),'--feqchkpt',str(steps),
                '--deltat','1.d-3','--lfilter','t','--diffterm','t','--scheme','643e'],
                out/(name+'_prepare.log'))
            launch(case, args.gpu if gpu else args.cpu, workspace, 'fresh')
            if name == 'restart':
                with h5py.File(case/'outdat/flowfield.h5') as h:
                    if int(h['nstep'][()].item()) != 50:
                        raise ValueError('wrong restart checkpoint step')
                set_restart(case/'datin/input.tgv','t')
                set_controller_steps(case/'datin/controller',100,100,1)
                launch(case,args.gpu,None,'restart')
                if 'checkpoint file read' not in (case/'restart.log').read_text():
                    raise ValueError('restart was not loaded')
            with h5py.File(case/'outdat/flowfield.h5') as h:
                if int(h['nstep'][()].item()) != 100 or not np.isclose(h['time'][()].item(),.1,rtol=0,atol=1e-14):
                    raise ValueError('wrong final checkpoint clock')
            result['completed'].append(name)
            save()
        compare(out/'cpu/outdat/rk_complete_snapshot.h5',out/'default','cpu_default',1e-10,1e-10)
        compare(out/'full',out/'default','full_default',0,0)
        compare(out/'default',out/'restart','continuous_restart',1e-10,0)
        command([sys.executable,str(helpers/'compare_flowstate.py'),
            '--cpu',str(out/'cpu'),'--gpu',str(out/'default'),
            '--report',str(out/'statistics.txt'),'--atol','1e-10','--rtol','1e-10'],
            out/'statistics.log')
        result['status']='passed-bounded-regression-not-physical-validation'
    except BaseException as error:
        result.update(status='failed', error=str(error))
        save()
        raise
    save()
    print(result['status'])


if __name__ == '__main__':
    main()
