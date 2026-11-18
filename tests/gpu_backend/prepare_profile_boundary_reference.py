"""Freeze original bc11/profile and bc51/y-upper states without changing BC math."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from prepare_upwind_components import prepare_flat
from run_cpp_tgv_gate import compare,namelist
from run_cpp_multi_tgv_gate import raw_field
from named_fixture import convert_case


def main():
    parser=argparse.ArgumentParser()
    for name in ('reference','mpiexec','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    data=[];report={}
    for backend,binary in [('cpu','astr_cpu'),('gpu','astr_cuda_fortran')]:
        case=root/backend;prepare_flat(case,backend=='gpu')
        groups=convert_case((case/'datin/input.flatplate').read_text())
        groups['numerics'].update(conschm='543e',recon_schem=3,lchardecomp=False,
            convection_form='standard',viscous_form='standard')
        groups['boundary']['bctype'][3]=51
        groups['precision_options']=dict(gpu_precision_mode='fp64')
        (case/'datin/input.case').write_text(namelist(groups))
        (case/'datin/input.output').write_text(namelist(dict(output=dict(directory='outdat/new',
            host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0),
            checkpoint=dict(enabled=False),volume=dict(enabled=False),slices=dict(enabled=False))))
        env=os.environ|dict(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
            OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',ASTR_FORCE_MPI_TOPOLOGY='1,1,1',
            ASTR_GPU_SYNC_MODE='explicit',ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',ASTR_VALIDATION_RHS_STEP='0')
        executable=(args.reference/'bin'/binary).resolve()
        with (case/'run.log').open('w') as log:
            subprocess.run([str(args.mpiexec.resolve()),'--bind-to','none','-n','1',str(executable),'run'],
                cwd=case,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        profile=np.loadtxt(case/'datin/inlet.prof',skiprows=4)
        stages=[]
        for stage in (1,2,3):
            suffix=f'.step00000000.rk{stage:02d}.rank00000000.bin'
            h,p=raw_field(case/('outdat/rhs.pre_rhs_primitives'+suffix),primitive=True)
            qh,q=raw_field(case/('outdat/rhs.pre_rhs'+suffix))
            if qh[:4]!=h or profile.shape!=(h[1]+1,4):raise AssertionError('boundary reference layout')
            halo=h[3];physical=tuple(slice(halo,halo+n+1) for n in h[:3])
            target=np.zeros((h[1]+1,h[2]+1,9),order='F')
            target[...,0]=profile[:,0,None];target[...,1]=profile[:,1,None]
            target[...,2]=profile[:,2,None];target[...,5]=profile[:,3,None]
            target[...,4]=target[...,0]*target[...,5]/(1.4*2.**2);target[...,6]=1
            # Other faces own the two corners after boucon; test the isolated
            # profile producer only on its nonintersecting interior rows.
            compare(p[halo,halo+1:halo+h[1],halo:halo+h[2]+1,:],target[1:-1,:,:6])
            with (root/f'{backend}.rk{stage:02d}.bin').open('xb') as dest:
                np.asarray(h,np.int32).tofile(dest)
                q.ravel(order='F').tofile(dest);p.ravel(order='F').tofile(dest);target.ravel(order='F').tofile(dest)
            stages.append(p[physical+(slice(None),)])
        data.append(stages)
        report[backend]=dict(binary=str(executable),sha256=hashlib.sha256(executable.read_bytes()).hexdigest())
    report['pair_error']=[compare(a,b) for a,b in zip(*data)]
    (root/'reference.json').write_text(json.dumps(report,indent=2)+'\n')
    for path in root.rglob('*'):path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
