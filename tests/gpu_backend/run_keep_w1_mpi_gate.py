"""Check W1 x/y/z slab stencils against a serial three-dimensional frozen state."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import h5py
import numpy as np

from run_keep_f0_gate import checkpoint,compare,logged_steps,stage
from run_keep_w1_main_gate import prepare


def assemble(case,label,rk):
    layout=np.loadtxt(case/'datin/parallel.info',dtype=int,skiprows=3,ndmin=2)
    shape=tuple(np.max(layout[:,4:7]+layout[:,7:10],axis=0)+1)
    covered=np.zeros(shape,dtype=bool)
    out=None
    for record in layout:
        position=tuple(slice(start,start+local+1) for start,local in zip(record[7:],record[4:7]))
        part=stage(case,label,rk,record[0])
        if out is None:
            out=np.empty((*shape,part.shape[-1]),dtype=part.dtype)
        mask=covered[position]
        if np.any(mask):
            compare(out[position][mask],part[mask])
        out[position]=part
        covered[position]=True
    if not np.all(covered):
        raise AssertionError('rank snapshots do not cover the global field')
    return out


def viscous(case,rk):
    with h5py.File(case/'outdat/geometry.h5') as f:
        jac=f['jacob'][...].transpose(2,1,0)
    values=(assemble(case,'full',rk)-assemble(case,'conv',rk))/jac[...,None]
    return values


def run(binary,mpiexec,root,device_aware):
    root.mkdir(parents=True,exist_ok=False)
    env=os.environ | dict(ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',ASTR_VALIDATION_RHS_STEP='0',
                          ASTR_GEOMETRY_DUMP='outdat/geometry.h5',ASTR_GPU_SYNC_MODE='explicit',
                          ASTR_GPU_HALO_TRANSPORT='pinned',OMPI_MCA_opal_cuda_support='0',
                          OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
                          OMPI_MCA_coll='^cuda,hcoll,ucc')
    saved=Path('outdat/new/checkpoints')
    final=saved/'step000000000004/state.h5'

    def execute(name,n,backend,dims,restore=None,transport='pinned'):
        case=root/name
        prepare(case,n,backend,4,'keep_sbp63_pair',restore,mixed=True)
        command=[str(mpiexec),'-n',str(np.prod(dims)),str(binary),'run']
        options={'ASTR_FORCE_MPI_TOPOLOGY':','.join(map(str,dims))}
        if transport=='device':
            options.update(ASTR_GPU_HALO_TRANSPORT='device',OMPI_MCA_pml='ucx',OMPI_MCA_opal_cuda_support='1')
        with (case/'run.log').open('w') as file:
            process=subprocess.run(command,cwd=case,env=env|options,
                                   stdout=file,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        log=(case/'run.log').read_text()
        process.check_returncode()
        logged_steps(case,count=2 if restore else 4,first_step=3 if restore else 1)
        if 's63pair-v1' not in log:
            raise AssertionError('paired mode not executed')
        if backend=='gpu' and f'ASTR_GPU_CPP transport={transport}' not in log:
            raise AssertionError(f'W1 transport {transport} not executed')
        return case

    report={}
    for n,axes,backends in ((24,(0,1,2),('cpu','gpu')),(18,(1,),('cpu','gpu')),(20,(),('cpu',)),(21,(),('cpu',))):
        for backend in backends:
            serial=execute(f'n{n}-{backend}-single',n,backend,[1,1,1])
            layouts={}
            for axis in axes:
                dims=[1,1,1]
                dims[axis]=2
                layouts['xyz'[axis]]=dims
            if n==24 and backend=='cpu':
                layouts.update(xy=[2,2,1],xz=[2,1,2],y4=[1,4,1])
            if n==20:
                layouts['y4']=[1,4,1]
            if n==21:
                layouts['y3']=[1,3,1]
            for name,dims in layouts.items():
                tag=f'n{n}-{backend}-{name}'
                case=execute(tag,n,backend,dims)
                discrepancies=[]
                for rk in (1,2,3):
                    primitive=compare(assemble(case,'pre_rhs_primitives',rk),assemble(serial,'pre_rhs_primitives',rk))
                    flux=compare(viscous(case,rk),viscous(serial,rk))
                    discrepancies.append(dict(primitive=primitive,viscous=flux))
                record=dict(stages=discrepancies,terminal=compare(checkpoint(case/final),checkpoint(serial/final)))
                if (name,n) in (('y',24),('y4',20),('y4',24),('y3',21)):
                    restored=execute(f'{tag}-restore',n,backend,dims,case/saved/'step000000000002')
                    np.testing.assert_array_equal(checkpoint(restored/final).view(np.uint64),
                                                  checkpoint(case/final).view(np.uint64))
                    record['exact_restart']=True
                if device_aware and tag=='n24-gpu-y':
                    device=execute(f'{tag}-device',n,backend,dims,transport='device')
                    for rk in (1,2,3):
                        np.testing.assert_array_equal(viscous(device,rk).view(np.uint64),viscous(case,rk).view(np.uint64))
                    np.testing.assert_array_equal(checkpoint(device/final).view(np.uint64),
                                                  checkpoint(case/final).view(np.uint64))
                    record['device_matches_pinned']=True
                report[tag]=record
                (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
                print(tag,'RHS and terminal match serial',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('binary','mpiexec','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--device-aware',action='store_true',help='also compare one CUDA-aware y slab with pinned MPI')
    args=parser.parse_args()
    run(args.binary.resolve(),args.mpiexec.resolve(),args.output.resolve(),args.device_aware)
