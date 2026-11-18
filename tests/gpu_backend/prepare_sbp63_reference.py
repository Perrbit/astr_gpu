"""Freeze actual original SBP63 weak-wall states and independent operator checks."""
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

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests/gpu_validation'))
from run_keep_sbp63_wall_gate import frozen_reference,complete_step_budget
from prepare_ldc_reference import compare_stages
from run_cpp_tgv_gate import compare,checkpoint,logged_steps,namelist


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('reference','output','mpiexec'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--perturbed-wall',action='store_true')
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    cases={};report=dict(scope='original CPU/old GPU SBP63 with independent tensor-operator and RK budget checks',
        grid=[16,16,16],steps=4,dt=1e-5,perturbed_wall=args.perturbed_wall,
        transport=dict(prandtl=.72,sutherland_k=110.3,reference_temperature_k=273.15),cases={},binaries={})
    env={k:v for k,v in os.environ.items() if not k.startswith('ASTR_')}
    env.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',ASTR_FORCE_MPI_TOPOLOGY='1,1,1',
        ASTR_GPU_SYNC_MODE='explicit',ASTR_GPU_HALO_TRANSPORT='pinned',ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',
        ASTR_VALIDATION_RHS_STEP='0',ASTR_VALIDATION_RHS_STEP_END='4',ASTR_VALIDATION_GRADIENTS='1')
    for backend,name in [('cpu','astr_cpu'),('gpu','astr_cuda_fortran')]:
        case=root/backend;cases[backend]=case
        subprocess.run([sys.executable,str(ROOT/'examples/KEEP_SBP63_Walls/prepare.py'),
            '--backend',backend,'--intervals','16','--output',str(case)],check=True)
        if args.perturbed_wall:
            # Excite every weak SAT term in a private operator fixture, not the shipped example.
            with h5py.File(case/'datin/grid.h5') as f:
                x,y,z=(f[n][:] for n in 'xyz')
            with h5py.File(case/'datin/flowini3d.h5','r+') as f:
                f['ro'][...]=f['ro'][:]+.003*np.cos(x)*np.cos(z)*np.cos(y)
                f['t'][...]=f['t'][:]+.005*np.cos(x)*np.cos(z)*np.cos(y)
                f['u1'][...]=f['u1'][:]+.002*np.cos(x)*np.cos(z)*np.cos(y)**2
                f['u2'][...]=f['u2'][:]+.001*np.sin(x)*np.cos(z)*np.cos(y)**2
                f['u3'][...]=f['u3'][:]+.0005*np.sin(z)*np.cos(y)**2
        (case/'datin/input.output').write_text(namelist(dict(output=dict(directory='outdat/new',exact_restart=True,
            checkpoint_statistics=False,host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0),
            checkpoint=dict(enabled=True,mode='steps',interval_steps=2,keep=2,initial_frame=False),
            volume=dict(enabled=False),slices=dict(enabled=False))))
        binary=(args.reference/'bin'/name).resolve()
        with (case/'run.log').open('w') as log:
            subprocess.run([str(args.mpiexec.resolve()),'--bind-to','none','-n','1',str(binary),'run'],
                cwd=case,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        if 's63sat-v1' not in (case/'run.log').read_text():raise AssertionError('original SBP wall path not selected')
        report['cases'][backend]=dict(independent={str(s):frozen_reference(case,16,s,sutherland_k=110.3) for s in (1,2,3)},
            complete_step_budget=complete_step_budget(case,16))
        report['binaries'][name]=hashlib.sha256(binary.read_bytes()).hexdigest()
    final='outdat/new/checkpoints/step000000000004/state.h5'
    report.update(stages=compare_stages(cases['cpu'],cases['gpu']),
        terminal=compare(checkpoint(cases['gpu']/final),checkpoint(cases['cpu']/final)),
        scalars=compare(np.loadtxt(cases['gpu']/'flowstate.dat',skiprows=1,ndmin=2),
            np.loadtxt(cases['cpu']/'flowstate.dat',skiprows=1,ndmin=2)))
    if not np.array_equal(logged_steps(cases['cpu']),logged_steps(cases['gpu'])):raise AssertionError('SBP reference clock')
    report['complete_step_clock_equal']=True
    (root/'reference.json').write_text(json.dumps(report,indent=2)+'\n');shutil.copyfile(__file__,root/'preparation.py')
    for p in root.rglob('*'):p.chmod(0o555 if p.is_dir() else 0o444)
    root.chmod(0o555)
    print(json.dumps(dict(stages=report['stages'],terminal=report['terminal'],complete_step_clock_equal=True),indent=2))


if __name__=='__main__':main()
