"""Freeze the shipped LDC UDF and existing multi-physical-axis operators."""
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
from run_keep_simple_boundary_gate import prepare,ldc_wall_targets
from run_cpp_tgv_gate import compare,checkpoint,logged_steps
from run_cpp_multi_tgv_gate import raw_field


def compare_stages(cpu,gpu,ranks=1):
    result={}
    for phase,primitive in [('pre_rhs',False),('pre_rhs_primitives',True),('post_source',False)]:
        maximum=0.
        paths=sorted((cpu/'outdat').glob('rhs.'+phase+'.*.bin'))
        if len(paths)!=12*ranks:raise AssertionError('missing reference stages for the selected rank count')
        for path in paths:
            arrays=[]
            for source in (path,gpu/'outdat'/path.name):
                if phase=='post_source':
                    with source.open('rb') as f:
                        header=np.fromfile(f,np.int32,4)
                        value=np.fromfile(f,np.float64).reshape(tuple(header[:3]+1)+(header[3],),order='F')
                else:
                    header,value=raw_field(source,primitive=primitive);h=header[3]
                    value=value[tuple(slice(h,h+n+1) for n in header[:3])+(slice(None),)]
                arrays.append(value)
            maximum=max(maximum,max(compare(*arrays)))
        result[phase]=maximum
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','output','mpiexec'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--variant',choices=['explicit-filter','compact-me6-filter'],default='explicit-filter')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    compact=args.variant=='compact-me6-filter';cases={}
    env=os.environ|dict(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',ASTR_FORCE_MPI_TOPOLOGY='1,1,1',
        ASTR_GPU_SYNC_MODE='explicit',ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',ASTR_VALIDATION_RHS_STEP='0',
        ASTR_VALIDATION_RHS_STEP_END='4',ASTR_GPU_FILTER_WORKSPACE='scalar',ASTR_GPU_COMPACT_TRANSPORT='pinned')
    for name,binary in [('cpu','astr_cpu'),('gpu','astr_cuda_fortran')]:
        case=out/(name+'_y');cases[name]=case
        prepare(case,'ldc',name,scheme='643c' if compact else '643e',diffusion='643c' if compact else '643e',
            filtering='compact10' if compact else 'explicit10',steps=4,convection='keep',
            viscosity='me6_opti' if compact else 'standard')
        with (case/'run.log').open('w') as log:
            subprocess.run([str(args.mpiexec.resolve()),'--bind-to','none','-n','1',str((args.reference/'bin'/binary).resolve()),'run'],
                cwd=case,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
    final=Path('outdat/new/checkpoints/step000000000004/state.h5')
    report=dict(variant=args.variant,stages=compare_stages(cases['cpu'],cases['gpu']),
        terminal=compare(checkpoint(cases['gpu']/final),checkpoint(cases['cpu']/final)),
        scalars=compare(np.loadtxt(cases['gpu']/'flowstate.dat',skiprows=1,ndmin=2),np.loadtxt(cases['cpu']/'flowstate.dat',skiprows=1,ndmin=2)),
        walls={name:ldc_wall_targets(case) for name,case in cases.items()},
        binaries={name:hashlib.sha256((args.reference/'bin'/name).read_bytes()).hexdigest() for name in ('astr_cpu','astr_cuda_fortran')})
    if not np.array_equal(logged_steps(cases['cpu']),logged_steps(cases['gpu'])):raise AssertionError('LDC reference clock')
    for targets in report['walls'].values():
        if any(value>2e-10 for stage in targets.values() for value in stage.values()):raise AssertionError('LDC fixed lid/side-wall ownership')
    (out/'reference.json').write_text(json.dumps(report,indent=2)+'\n');shutil.copyfile(__file__,out/'preparation.py')
    for path in out.rglob('*'):path.chmod(0o555 if path.is_dir() else 0o444)
    out.chmod(0o555);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
