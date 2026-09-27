#!/usr/bin/env python3
"""Matched-endpoint source-active refinement using the ASTR flow integrator."""
import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from air5_radau_reference import Air5RadauReference
from check_air5_mach4_error_replay import state_gate
from check_air5_mass_closure_stages import metrics
from run_air5_characteristic_acoustic import prepare, RHO
from run_air5_characteristic_backend_gate import fields
from run_air5_characteristic_restart_gate import launch
from run_air5_sbli_preflight import ROOT, environment, set_value


def refinement(coarse, medium, fine, scales):
    if coarse.shape != medium.shape or coarse.shape != fine.shape:
        raise ValueError('refinement field shape mismatch')
    if not all(np.isfinite(q).all() for q in (coarse, medium, fine)):
        raise ValueError('nonfinite refinement state')
    axes = tuple(range(coarse.ndim-1))
    first = np.max(abs(coarse-medium), axis=axes)/scales
    second = np.max(abs(medium-fine), axis=axes)/scales
    floor = 100*np.finfo(float).eps
    resolved = first > floor
    passed = bool(np.any(resolved) and np.all(second[resolved] < first[resolved])
                  and np.all(second[~resolved] <= floor))
    return dict(passed=passed, coarse_medium=first.tolist(), medium_fine=second.tolist(),
                resolved_components=(np.flatnonzero(resolved)+1).tolist(),
                roundoff_floor=floor, formal_order_claim=False)


def run(output, executable, mpi_launcher=None, source_mode='coupled'):
    output.mkdir(parents=True, exist_ok=False)
    mpi_launcher=str(mpi_launcher or shutil.which('mpirun'))
    version=subprocess.run([mpi_launcher,'--version'],check=True,capture_output=True,text=True).stdout
    contract = dict(endpoint_seconds=60e-9, steps=[12,24,48],
                    mpi_launcher=mpi_launcher, mpi_version=version.strip(),
                    timesteps=[5e-9,2.5e-9,1.25e-9], topology='1,2,1',
                    source_mode=source_mode, viscosity=True, filter=False,
                    convection_limiter='symmetric_species', diffusion_limiter='layered',
                    criterion='componentwise decreasing successive differences above 100 epsilon; '
                              'otherwise both differences below the roundoff floor',
                    scope=('short-window refinement, not physical SBLI admission' if source_mode=='coupled'
                           else 'V-T-only diagnostic control, not coupled-source acceptance'))
    (output/'contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    thermo = Air5RadauReference(ROOT/'chemMech/air5_kimjo12.json')
    final, source, states, endpoints = [], [], [], []
    scales = None
    for name, steps, dt in zip(('coarse','medium','fine'),contract['steps'],contract['timesteps']):
        case = output/name
        meta = prepare(case, executable, ny=128, nz=16, pulse_y=.0096,
                       short_updates=steps, use_gpu=True, temperature0=3000., tv0=1500.)
        set_value(case/'datin/input.air5_c4',
                  'nondimen,diffterm,lfilter,lreadgrid,lfftz,limmbou,ltimrpt,lcomb',
                  'f,t,f,f,f,f,t,t,t')
        set_value(case/'datin/controller', 'deltat', f'{dt:.17e}')
        set_value(case/'datin/controller', 'maxstep,feqchkpt,feqwsequ,feqslice,feqlist,feqavg',
                  f'{steps-1},{steps-1},1000000,1000000,1000000,1000000')
        env = environment(contract['topology'])
        env.update(meta['environment'], ASTR_AIR5_SOURCE_MODE=source_mode,
                   ASTR_AIR5_CONVECTION_LIMITER=contract['convection_limiter'],
                   ASTR_AIR5_DIFFUSION_LIMITER=contract['diffusion_limiter'])
        env.pop('ASTR_AIR5_ACOUSTIC_PROBE', None)
        meta.update(dt=dt, source_mode=source_mode, viscous=True, topology=contract['topology'],
                    sampling_phase='post_chemistry half 2 at matched complete-step endpoint',
                    environment={k:v for k,v in env.items() if k.startswith('ASTR_')})
        (case/'gate.json').write_text(json.dumps(meta, indent=2)+'\n')
        try:
            launch(case, executable, contract['topology'], env, 'run.log', mpi_launcher=mpi_launcher)
        except subprocess.CalledProcessError as error:
            failure=dict(passed=False, contract=contract, failed_case=name,
                         returncode=error.returncode, command=error.cmd,
                         completed_cases=len(final), log=str(case/'run.log'))
            (output/'result.json').write_text(json.dumps(failure,indent=2)+'\n')
            raise
        records=re.findall(r'ASTR_CFL complete_step=(\d+) state_time=\s*(\S+)',
                           (case/'run.log').read_text())
        if not records or int(records[-1][0])!=steps-1:
            raise ValueError('missing endpoint CFL record')
        endpoint=float(records[-1][1])
        if abs(endpoint-contract['endpoint_seconds'])>1e-14*contract['endpoint_seconds']:
            raise ValueError('physical endpoint mismatch')
        endpoints.append(endpoint)
        states.append(state_gate(case, thermo))
        data = fields(case, (steps-1,))
        for phase, q in data.items():
            row = metrics(q, (0,0,0))
            if row['transport_sum_violations'] or row['min_species_density'] < 0:
                raise ValueError(f'{name}/{phase}: composition failure')
        q = data[f'{steps-1}:post_chemistry:2']
        source.append(float(abs(q[1:-1,-1,1:-1,10]-
                          data[f'{steps-1}:post_transport:1'][1:-1,-1,1:-1,10]).max()))
        if source[-1] == 0:
            raise ValueError('source-inactive dynamic top')
        final.append(q)
        if scales is None:
            scales=np.array([RHO,*([RHO*meta['sound_speed']]*3),meta['pressure'],
                             *([RHO]*5),meta['pressure']])
    report=dict(contract=contract, scales=scales.tolist(), states=states, endpoints=endpoints,
                top_ev_second_half_max_abs=source,
                volume=refinement(*final,scales),
                top=refinement(*(q[:,-1,:,:] for q in final),scales))
    report['passed']=report['volume']['passed'] and report['top']['passed']
    (output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
    if not report['passed']:
        raise SystemExit('source refinement gate failed; do not advance downstream cases')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--executable',type=Path,default=ROOT/'build_gpu_probe/bin/astr')
    parser.add_argument('--mpiexec',type=Path)
    parser.add_argument('--source-mode',choices=('coupled','vt'),default='coupled')
    args=parser.parse_args()
    run(args.output.resolve(),args.executable.resolve(),args.mpiexec,args.source_mode)
