"""Freeze the approved compact-mask CPU/CUDA-Fortran flat-plate reference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from prepare_upwind_components import prepare_flat
from run_cpp_tgv_gate import namelist,compare
from run_cpp_multi_tgv_gate import raw_field
from named_fixture import convert_case


def compare_stages(current,reference,ranks,reference_ranks):
    errors={}
    for phase,primitive,halo in [('pre_rhs',False,True),('pre_rhs_primitives',True,True),('post_source',False,False)]:
        maximum=0.
        for stage in (1,2,3):
            for rank in range(ranks):
                ref_rank=rank if ranks==reference_ranks else 0
                arrays=[];headers=[]
                for directory,r in ((current,rank),(reference,ref_rank)):
                    path=directory/f'outdat/rhs.{phase}.step00000000.rk{stage:02d}.rank{r:08d}.bin'
                    if halo:
                        header,value=raw_field(path,primitive=primitive);h=header[3]
                        value=value[tuple(slice(h,h+n+1) for n in header[:3])+(slice(None),)]
                    else:
                        with path.open('rb') as stream:
                            header=np.fromfile(stream,np.int32,4)
                            value=np.fromfile(stream,np.float64).reshape((*header[:3]+1,header[3]),order='F')
                    arrays.append(value);headers.append(header)
                actual,expected=arrays
                if ranks!=reference_ranks:
                    nz=headers[0][2];expected=expected[:,:,rank*nz:(rank+1)*nz+1,:]
                maximum=max(maximum,max(compare(actual,expected)))
        errors[phase]=maximum
    return errors


def main():
    parser=argparse.ArgumentParser()
    for name in ('reference','mpiexec','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    env=os.environ|dict(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',
        ASTR_GPU_SYNC_MODE='explicit',ASTR_GPU_HALO_TRANSPORT='pinned',
        ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',ASTR_VALIDATION_RHS_STEP='0',
        ASTR_VALIDATION_GRADIENTS='1',ASTR_SHOCK_SENSOR_DUMP='outdat/sensor.txt',
        ASTR_GEOMETRY_DUMP='outdat/geometry.h5',ASTR_GPU_MP_POSITIVITY='off')
    report={};cases={}
    for ranks in (1,2):
        for backend,name in [('cpu','astr_cpu'),('gpu','astr_cuda_fortran')]:
            case=root/f'{backend}_np{ranks}';prepare_flat(case,backend=='gpu')
            groups=convert_case((case/'datin/input.flatplate').read_text())
            groups['numerics'].update(difschm='643c',convection_form='standard',viscous_form='standard')
            groups['precision_options']=dict(gpu_precision_mode='fp64')
            (case/'datin/input.case').write_text(namelist(groups))
            (case/'datin/input.output').write_text(namelist(dict(output=dict(directory='outdat/new',
                host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0),
                checkpoint=dict(enabled=False),volume=dict(enabled=False),slices=dict(enabled=False))))
            binary=(args.reference/'bin'/name).resolve()
            with (case/'run.log').open('w') as log:
                subprocess.run([str(args.mpiexec.resolve()),'--bind-to','none','-n',str(ranks),str(binary),'run'],
                    cwd=case,env=env|dict(ASTR_FORCE_MPI_TOPOLOGY=f'1,1,{ranks}'),
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
            cases[backend,ranks]=case
            report[f'{backend}_np{ranks}']=dict(binary=str(binary),sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
        report[f'cpu_gpu_np{ranks}']=compare_stages(cases['gpu',ranks],cases['cpu',ranks],ranks,ranks)
    for backend in ('cpu','gpu'):
        report[f'{backend}_decomposition']=compare_stages(cases[backend,2],cases[backend,1],2,1)
    report['scope']='Existing 16-cubed stretched 743e/643c MP7-LD fixture, three RK stages, no filter or crashfix'
    (root/'reference.json').write_text(json.dumps(report,indent=2)+'\n')
    for path in root.rglob('*'):path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
