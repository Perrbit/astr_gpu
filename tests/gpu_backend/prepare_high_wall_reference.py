"""Freeze original Hermite7 wall trajectories before new C++ consumers."""
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
from test_high_order_wall_runtime import prepare
from named_fixture import prepare_named_run,namelist
from prepare_ldc_reference import compare_stages
from run_cpp_tgv_gate import compare,checkpoint,logged_steps


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','output','mpiexec'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--wall-type',type=int,choices=[41,42,421],default=41)
    parser.add_argument('--force-fixed',type=float,default=0.)
    parser.add_argument('--wall-axis',choices=list('xyz'),default='y')
    parser.add_argument('--filter',action='store_true')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    cases={}
    for name,binary in [('cpu','astr_cpu'),('gpu','astr_cuda_fortran')]:
        case=out/name;cases[name]=case
        prepare(case,name,4,None,grid=(16,16,16),wall_type=args.wall_type,filtering=args.filter,
            normal_axis='xyz'.index(args.wall_axis)+1)
        env=os.environ|dict(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
            OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',ASTR_FORCE_MPI_TOPOLOGY='1,1,1',
            ASTR_GPU_SYNC_MODE='explicit',ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',ASTR_VALIDATION_RHS_STEP='0',
            ASTR_VALIDATION_RHS_STEP_END='4',ASTR_VALIDATION_GRADIENTS='1',ASTR_HIGH_ORDER_WALL_MODE='hermite7',
            ASTR_GPU_MP_POSITIVITY='off',ASTR_CHANNEL_FORCE_MODE='fixed',ASTR_CHANNEL_FORCE_FIXED=str(args.force_fixed))
        prepare_named_run(case,'datin/input.chl',env)
        (case/'datin/input.run').write_text(namelist(dict(run_control=dict(max_steps=4),
            timestep=dict(mode='fixed',fixed_dt=1e-5),logging=dict(list_steps=1),
            repair=dict(crashfix=False),statistics=dict(enabled=False))))
        (case/'datin/input.output').write_text(namelist(dict(output=dict(directory='outdat/new',exact_restart=True,
            checkpoint_statistics=False,host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0),
            checkpoint=dict(enabled=True,mode='steps',interval_steps=2,keep=2,initial_frame=False),
            volume=dict(enabled=False),slices=dict(enabled=False))))
        with (case/'run.log').open('w') as log:
            subprocess.run([str(args.mpiexec.resolve()),'--bind-to','none','-n','1',
                str((args.reference/'bin'/binary).resolve()),'run'],cwd=case,env=env,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        text=(case/'run.log').read_text()
        if 'hermite7' not in (case/'datin/input.case').read_text() or not any(
            marker in text for marker in ('p6t7-v1','wall-h7-v1','Cartesian Hermite7 wall axis=')):
            raise AssertionError('actual Hermite7 wall path was not selected')
        if args.filter and 'high-order wall filter: local-f6-v1' not in text:
            raise AssertionError('actual local-f6 wall filter was not selected')
    final=Path('outdat/new/checkpoints/step000000000004/state.h5')
    report=dict(scope='original CPU/old GPU reference only; no new C++ high-wall implementation',
        wall_type=args.wall_type,wall_axis=args.wall_axis,filtering=args.filter,force_fixed=args.force_fixed,grid=[16,16,16],steps=4,dt=1e-5,
        stages=compare_stages(cases['cpu'],cases['gpu']),
        terminal=compare(checkpoint(cases['gpu']/final),checkpoint(cases['cpu']/final)),
        scalars=compare(np.loadtxt(cases['gpu']/'flowstate.dat',skiprows=1,ndmin=2),
            np.loadtxt(cases['cpu']/'flowstate.dat',skiprows=1,ndmin=2)),
        binaries={name:hashlib.sha256((args.reference/'bin'/name).read_bytes()).hexdigest()
            for name in ('astr_cpu','astr_cuda_fortran')})
    if not np.array_equal(logged_steps(cases['cpu']),logged_steps(cases['gpu'])):raise AssertionError('high wall reference clock')
    report['complete_step_clock_equal']=True
    (out/'reference.json').write_text(json.dumps(report,indent=2)+'\n');shutil.copyfile(__file__,out/'preparation.py')
    for path in out.rglob('*'):path.chmod(0o555 if path.is_dir() else 0o444)
    out.chmod(0o555);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
