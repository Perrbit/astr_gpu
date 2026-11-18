"""Independent stationary ideal-gas state through the actual high-wall main."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests/gpu_validation'))
from test_high_order_wall_runtime import prepare,initial_fields
from named_fixture import prepare_named_run,namelist
from run_cpp_tgv_gate import snapshot


def uniform(grid,manufactured,wall_type):
    coords,fields=initial_fields(grid,manufactured,wall_type)
    for name,value in fields.items():
        value[...] = 1. if name in ('ro','t') else 0.
    if wall_type==421:
        fields['u1'][...]=.1;fields['u3'][...]=.02
    return coords,fields


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','output','mpiexec'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--wall-type',type=int,choices=[41,42,421],default=41)
    parser.add_argument('--filter',action='store_true')
    args=parser.parse_args();case=args.output.resolve();case.mkdir(parents=True,exist_ok=False)
    prepare(case,'gpu',1,None,grid=(16,16,16),initial_condition=uniform,filtering=args.filter,wall_type=args.wall_type)
    env=os.environ|dict(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',ASTR_FORCE_MPI_TOPOLOGY='1,1,1',
        ASTR_GPU_SYNC_MODE='explicit',ASTR_HIGH_ORDER_WALL_MODE='hermite7',ASTR_GPU_MP_POSITIVITY='off',
        ASTR_CHANNEL_FORCE_MODE='fixed',ASTR_CHANNEL_FORCE_FIXED='0',ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',
        ASTR_VALIDATION_RHS_STEP='0',ASTR_VALIDATION_RHS_STEP_END='1')
    prepare_named_run(case,'datin/input.chl',env)
    (case/'datin/input.run').write_text(namelist(dict(run_control=dict(max_steps=1),
        timestep=dict(mode='fixed',fixed_dt=1e-5),logging=dict(list_steps=1),repair=dict(crashfix=False),statistics=dict(enabled=False))))
    (case/'datin/input.output').write_text(namelist(dict(output=dict(directory='outdat/new',exact_restart=True,
        host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0),checkpoint=dict(enabled=False),
        volume=dict(enabled=False),slices=dict(enabled=False))))
    with (case/'run.log').open('w') as log:
        subprocess.run([str(args.mpiexec.resolve()),'--bind-to','none','-n','1',str(args.binary.resolve()),'run'],
            cwd=case,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
    if 'ASTR_GPU_CPP high_wall_axis=' not in (case/'run.log').read_text():raise AssertionError('high-wall path not selected')
    actual=snapshot(case/'outdat/rhs.cpp_final.step00000001.rk01.rank00000000.bin')
    expected=np.array([1.,0.,0.,0.,1/(1.4*.3**2)/(1.4-1)])
    if args.wall_type==421:
        expected[1]=.1;expected[3]=.02;expected[4]+=.5*(.1**2+.02**2)
    field_error=np.max(abs(actual-expected),axis=(0,1,2))
    if not np.isfinite(actual).all() or max(field_error)>2e-10:raise AssertionError('uniform static state changed')
    rhs_error={}
    for phase in ('conv','post_source'):
        files=sorted((case/'outdat').glob('rhs.'+phase+'.*.bin'))
        if len(files)!=3:raise AssertionError('three uniform RK residuals required')
        maximum=0.
        for path in files:
            with path.open('rb') as f:
                header=np.fromfile(f,np.int32,4);value=np.fromfile(f,np.float64)
            if value.size!=int(np.prod(header[:3]+1))*5 or not np.isfinite(value).all():raise AssertionError('uniform residual layout/finite')
            maximum=max(maximum,float(np.max(abs(value))))
        if maximum>2e-10:raise AssertionError('uniform residual is not zero to FP64 gate')
        rhs_error[phase]=maximum
    report=dict(scope='one actual uniform high-wall RK step; independent stationary state',
        wall_type=args.wall_type,filtering=args.filter,
        max_field_absolute_per_variable=field_error.tolist(),max_rhs_absolute=rhs_error,
        binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest())
    (case/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
