"""Matched-time TGV observations for fixed/automatic stepping and stage filtering."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np

from named_fixture import prepare_named_run
from run_adaptive_tgv_gate import STEP,compare_state

ROOT=Path(__file__).resolve().parents[2]
AUTO_STEPS=5
FIXED_STEPS=12
INITIAL_DT=1e-3


def endpoint():
    time=0.;dt=INITIAL_DT
    for _ in range(AUTO_STEPS):
        time+=dt
        dt*=1.1
    return time


def run(args,backend,mode,filtering):
    case=args.output/f'{backend}_{mode}_filter{int(filtering)}'
    steps=AUTO_STEPS if mode=='auto' else FIXED_STEPS
    subprocess.run([sys.executable,str(ROOT/'tests/gpu_validation/prepare_tgv_case.py'),
        '--src-case',str(ROOT/'examples/Taylor_Green_Vortex'),'--dst-case',str(case),
        '--grid','32,32,32','--use-gpu','t' if backend=='gpu' else 'f',
        '--scheme','643e','--diffterm','t','--lfilter','t' if filtering else 'f',
        '--maxstep',str(steps-1),'--feqchkpt','1','--feqlist','1','--deltat','1.d-3'],
        check=True)
    env={k:v for k,v in os.environ.items() if not k.startswith(('ASTR_','OMPI_MCA_','UCX_'))}
    env.update(ASTR_FORCE_MPI_TOPOLOGY='2,1,1',ASTR_GPU_SYNC_MODE='explicit',
        ASTR_GPU_HALO_TRANSPORT='pinned',OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',
        OMPI_MCA_osc='pt2pt',OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0')
    primary=prepare_named_run(case,'datin/input.tgv',env)
    timestep='mode="auto",initial_dt=0.001' if mode=='auto' else f'mode="fixed",fixed_dt={endpoint()/FIXED_STEPS:.17g}'
    (case/'datin/input.run').write_text(f'&run_control max_steps={steps} /\n&timestep {timestep} /\n&logging list_steps=0 /\n')
    (case/'outdat/new').mkdir(parents=True)
    (case/'datin/input.output').write_text('&output\n directory="outdat/new",exact_restart=t,'
        'host_budget_bytes=-1,device_budget_bytes=-1\n/\n'
        '&checkpoint\n enabled=t,interval_steps=1000000,keep=1\n/\n'
        '&volume\n enabled=f\n/\n&slices\n enabled=f\n/\n')
    with (case/'run.log').open('w') as log:
        subprocess.run([str(args.mpiexec),'-n','2',str(getattr(args,backend)),'run',primary],
            cwd=case,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    clocks=[(int(m[1]),*[float(m[i]) for i in range(2,6)]) for m in STEP.finditer((case/'run.log').read_text())]
    if [r[0] for r in clocks]!=list(range(1,steps+1)):
        raise AssertionError(f'{case}: incomplete update sequence')
    tolerance=64*np.finfo(float).eps*endpoint()
    if abs(clocks[-1][2]-endpoint())>tolerance:
        raise AssertionError(f'{case}: endpoints are not matched to rounding precision')
    path=case/f'outdat/new/checkpoints/step{steps:012d}/state.h5'
    with h5py.File(path) as state:
        if int(state['identity'][8])!=steps:
            raise AssertionError('checkpoint is not the complete-step endpoint')
        values=np.stack([state[f'q{n:04d}'][:-1,:-1,:-1] for n in range(1,6)],axis=-1)
    return path,values,dict(steps=steps,time=clocks[-1][2],clocks=clocks,
        planned_stage_filter_calls=3*steps if filtering else 0,
        mean_density=float(np.mean(values[...,0])),mean_total_energy=float(np.mean(values[...,4])))


def difference(a,b):
    delta=abs(a-b)
    axes=tuple(range(delta.ndim-1))
    return dict(component_max_abs=np.max(delta,axis=axes).tolist(),
        component_max_scaled=(np.max(delta,axis=axes)/np.maximum(1.,np.max(abs(a),axis=axes))).tolist())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cpu','gpu','mpiexec','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    for name in ('cpu','gpu','mpiexec','output'):
        setattr(args,name,getattr(args,name).resolve())
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',target_time=endpoint(),cases={},backend_comparisons={},observations={})
    fields={}
    for filtering in (False,True):
        for mode in ('auto','fixed'):
            paths={}
            for backend in ('cpu','gpu'):
                paths[backend],fields[backend,mode,filtering],record=run(args,backend,mode,filtering)
                report['cases'][f'{backend}_{mode}_filter{int(filtering)}']=record
            errors={}
            compare_state(paths['cpu'],paths['gpu'],False,errors)
            report['backend_comparisons'][f'{mode}_filter{int(filtering)}']=errors
        report['observations'][f'auto_minus_fixed_filter{int(filtering)}']=difference(
            fields['gpu','auto',filtering],fields['gpu','fixed',filtering])
    for mode in ('auto','fixed'):
        report['observations'][f'filter_on_minus_off_{mode}']=difference(
            fields['gpu',mode,True],fields['gpu',mode,False])
    report.update(status='passed',scope='short matched-time comparison; no equality gate between different schedules; '
        'stage filter counts follow the three-stage call schedule, not profiler measurements; '
        'does not establish long-time filter equivalence or temporal convergence')
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['backend_comparisons'],indent=2))


if __name__=='__main__':
    main()
