"""Freeze existing time-series inlet stages, including a ring-window rollover."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests/gpu_validation'))
from prepare_upwind_components import prepare_flat
from prepare_ldc_reference import compare_stages
from run_cpp_tgv_gate import compare,snapshot,logged_steps,namelist,convert_case


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('reference','output','mpiexec'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--ranks',type=int,choices=[1,2],default=1)
    parser.add_argument('--axis',choices=list('xyz'),default='x')
    parser.add_argument('--automatic',action='store_true')
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    topology=[1,1,1];topology['xyz'.index(args.axis)]=args.ranks;dt=8e-6
    env={k:v for k,v in os.environ.items() if not k.startswith('ASTR_')}
    env.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',ASTR_GPU_SYNC_MODE='explicit',
        ASTR_GPU_HALO_TRANSPORT='pinned',ASTR_FORCE_MPI_TOPOLOGY=','.join(map(str,topology)),
        ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',ASTR_VALIDATION_RHS_STEP='0',ASTR_VALIDATION_RHS_STEP_END='4')
    for backend,binary in [('cpu','astr_cpu'),('gpu','astr_cuda_fortran')]:
        case=root/backend;prepare_flat(case,backend=='gpu')
        groups=convert_case((case/'datin/input.flatplate').read_text())
        groups['numerics'].update(conschm='543e',difschm='643e',convection_form='standard',
            viscous_form='standard',recon_schem=3,lchardecomp=False,lfilter=False)
        groups['boundary'].update(bctype=[11,21,41,51,1,1],turbinf='intp')
        groups['precision_options']=dict(gpu_precision_mode='fp64',gpu_mp_positivity='off')
        groups['boundary_options']=dict(high_order_wall_mode='off')
        (case/'datin/input.case').write_text(namelist(groups))
        timestep=(dict(mode='auto',initial_dt=dt,max_dt=1.6e-5,cfl_target=.5,growth_factor=1.1)
            if args.automatic else dict(mode='fixed',fixed_dt=dt))
        (case/'datin/input.run').write_text(namelist(dict(run_control=dict(max_steps=4),
            timestep=timestep,logging=dict(list_steps=1),repair=dict(crashfix=False),statistics=dict(enabled=False))))
        (case/'datin/input.output').write_text(namelist(dict(output=dict(directory='outdat/new',
            host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0),checkpoint=dict(enabled=False),
            volume=dict(enabled=False),slices=dict(enabled=False))))
        subprocess.run([sys.executable,str(ROOT/'tests/gpu_validation/generate_dynamic_inflow_slices.py'),
            '--output',str(case/'inflow'),'--jm','16','--km','16','--count','12','--delta-time','1e-5',
            '--temporal-mode','nonpolynomial'],check=True)
        with (case/'run.log').open('w') as log:
            subprocess.run([str(args.mpiexec.resolve()),'--bind-to','none','-n',str(args.ranks),
                str((args.reference/'bin'/binary).resolve()),'run'],cwd=case,env=env,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
    report=dict(topology=topology,dt=dt,steps=4,interval=1e-5,automatic=args.automatic,
        binaries={name:hashlib.sha256((args.reference/'bin'/name).read_bytes()).hexdigest() for name in ('astr_cpu','astr_cuda_fortran')},
        stages=compare_stages(root/'cpu',root/'gpu',ranks=args.ranks),updated={})
    for rank in range(args.ranks):
        paths=sorted((root/'cpu/outdat').glob(f'rhs.post_update.*.rank{rank:08d}.bin'))
        if len(paths)!=12:raise AssertionError('missing dynamic-inflow stages')
        report['updated'][str(rank)]=max(max(compare(snapshot(root/'gpu/outdat'/p.name),snapshot(p))) for p in paths)
    report['scalars']=compare(np.loadtxt(root/'gpu/flowstate.dat',skiprows=1,ndmin=2),np.loadtxt(root/'cpu/flowstate.dat',skiprows=1,ndmin=2))
    if not np.array_equal(logged_steps(root/'cpu'),logged_steps(root/'gpu')):raise AssertionError('dynamic clock')
    if 'islice00004.h5' not in (root/'gpu/run.log').read_text():raise AssertionError('reference did not roll inflow window')
    (root/'reference.json').write_text(json.dumps(report,indent=2)+'\n');shutil.copyfile(__file__,root/'preparation.py')
    for p in root.rglob('*'):p.chmod(0o555 if p.is_dir() else 0o444)
    root.chmod(0o555);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
