"""Freeze actual original native-FP32 CPU/CUDA Fortran periodic explicit phases."""
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

from run_cpp_tgv_gate import prepare_case,namelist

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests/gpu_validation'))
from compare_q_validation_snapshots import read_q_snapshot
from run_native_precision_tgv_gate import component_report,physical_state,primitive_state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('binary','template','mpiexec','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--derivative',choices=('explicit','compact'),default='explicit')
    parser.add_argument('--convection',choices=('standard','keep'),default='standard')
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    binary=args.binary.resolve();shutil.copyfile(binary,root/'original_astr_native32')
    shutil.copyfile(__file__,root/'preparation.py')
    env=os.environ|dict(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',
        ASTR_FORCE_MPI_TOPOLOGY='1,1,1',ASTR_GPU_SYNC_MODE='explicit',ASTR_GPU_FILTER_WORKSPACE='scalar',
        ASTR_GPU_MP_POSITIVITY='off',ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',
        ASTR_VALIDATION_RHS_STEP='0',ASTR_VALIDATION_RHS_STEP_END='4',ASTR_VALIDATION_GRADIENTS='1',
        ASTR_GEOMETRY_DUMP='outdat/geometry.h5')
    report=dict(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),grid=[16,16,16],steps=4,
        initial_dt=1e-4,precision='fp32',cases={},scope='Original native FP32 explicit periodic numeric reference; not physical validation')
    report['derivative']=args.derivative
    report['convection']=args.convection
    report['scope']='Original native FP32 periodic numeric reference; not physical validation'
    scheme='643c' if args.derivative=='compact' else '643e'
    for filtering in (False,True):
        pair={}
        for gpu in (False,True):
            name=f'filter{int(filtering)}_'+('gpu' if gpu else 'cpu');case=root/name
            prepare_case(case,args.template.resolve(),gpu,True,False,steps=4,dt=1e-4,
                numerics=dict(conschm=scheme,difschm=scheme,lfilter=filtering,convection_form=args.convection,
                    filter_scheme='compact10' if args.derivative=='compact' else 'explicit10',
                    recon_schem=0,lchardecomp=False),physics=dict(diffterm=True))
            (case/'datin/input.output').write_text(namelist(dict(output=dict(directory='outdat/new',
                host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0),checkpoint=dict(enabled=False),
                volume=dict(enabled=False),slices=dict(enabled=False))))
            with (case/'run.log').open('w') as log:
                subprocess.run([str(args.mpiexec.resolve()),'--bind-to','none','-n','1',str(binary),'run'],
                    cwd=case,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
            suffix='.step00000003.rk03.rank00000000.bin'
            final=case/('outdat/rhs.post_update'+suffix)
            record=read_q_snapshot(final,precision='fp32')
            if record.values.dtype!=np.dtype('float32'):raise AssertionError('original native q is not FP32')
            pair[gpu]=physical_state(final,'fp32')
            for step in range(4):
                for stage in (1,2,3):
                    path=case/f'outdat/rhs.pre_rhs_primitives.step{step:08d}.rk{stage:02d}.rank00000000.bin'
                    primitive_state(path,'fp32')
            with h5py.File(case/'outdat/geometry.h5') as geometry:
                if any(geometry[key].dtype!=np.dtype('float32') for key in ('jacob','dxi11','dxi22','dxi33')):
                    raise AssertionError('original native geometry is not FP32')
            values=np.loadtxt(case/'flowstate.dat',skiprows=1,ndmin=2)
            if not np.isfinite(values).all():raise AssertionError('nonfinite original scalar samples')
            report['cases'][name]=dict(physical_nodes_positive=True,stage_count=12,geometry_dtype='float32')
        report['cases'][f'filter{int(filtering)}_cpu_gpu_report']=component_report(pair[False],pair[True])
    (root/'reference.json').write_text(json.dumps(report,indent=2)+'\n')
    for path in root.rglob('*'):path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
