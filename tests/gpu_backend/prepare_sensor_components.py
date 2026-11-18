"""Freeze original sensor inputs/results before introducing the C++ consumer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from run_cpp_tgv_gate import prepare_case, compare, namelist
from run_cpp_multi_tgv_gate import raw_field


def main():
    parser=argparse.ArgumentParser()
    for name in ('reference','template','mpiexec','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--characteristic',action='store_true')
    args=parser.parse_args()
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    report={}
    sensors=[];masks=[]
    for backend,binary in [('cpu','astr_cpu'),('gpu','astr_cuda_fortran')]:
        case=root/backend;executable=(args.reference/'bin'/binary).resolve()
        prepare_case(case,args.template.resolve(),backend=='gpu',False,False,steps=1,dt=1e-5,
            numerics=dict(conschm='543e',difschm='643e',lfilter=False,
                lchardecomp=args.characteristic,recon_schem=3,shkcrt=.001),physics=dict(diffterm=False,mach=3.),
            case_options=dict(flowtype='shuosher'))
        (case/'datin/input.output').write_text(namelist(dict(
            output=dict(directory='outdat/new',host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0),
            checkpoint=dict(enabled=False),
            volume=dict(enabled=False),slices=dict(enabled=False))))
        env=os.environ|dict(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
            OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',ASTR_FORCE_MPI_TOPOLOGY='1,1,1',
            ASTR_GPU_SYNC_MODE='explicit',ASTR_SHOCK_SENSOR_DUMP='outdat/sensor.txt',
            ASTR_GEOMETRY_DUMP='outdat/geometry.h5',
            ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',ASTR_VALIDATION_RHS_STEP='0',ASTR_VALIDATION_GRADIENTS='1')
        with (case/'run.log').open('w') as log:
            subprocess.run([str(args.mpiexec.resolve()),'--bind-to','none','-n','1',str(executable),'run'],
                cwd=case,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        entries=[];raws=[];flags=[]
        for stage in (1,2,3):
            suffix=f'.step00000000.rk{stage:02d}.rank00000000.bin'
            header,primitives=raw_field(case/('outdat/rhs.pre_rhs_primitives'+suffix),primitive=True)
            upper=np.asarray(header[:3]);nodes=int(np.prod(upper+1))
            with (case/('outdat/rhs.gradients'+suffix)).open('rb') as source:
                gh=np.fromfile(source,np.int32,5);gradient=np.fromfile(source,np.float64,nodes*9)
            if tuple(gh[:3])!=tuple(upper) or gradient.size!=nodes*9:
                raise AssertionError('frozen gradient shape')
            with (case/('outdat/rhs.sensor'+suffix)).open('rb') as source:
                sh=np.fromfile(source,np.int32,3)
                sensor=np.fromfile(source,np.float64,nodes)
                mask=np.fromfile(source,np.int8,nodes)
                extra=source.read()
            if tuple(sh)!=tuple(upper) or sensor.size!=nodes or mask.size!=nodes or extra:
                raise AssertionError('frozen sensor shape')
            if not np.isfinite(sensor).all() or np.any((mask!=0)&(mask!=1)):
                raise AssertionError('invalid frozen sensor/mask')
            with (root/f'{backend}.rk{stage:02d}.bin').open('xb') as target:
                np.asarray([*header[:4],0,0,0,0,0,0],np.int32).tofile(target)
                np.asarray([.001],np.float64).tofile(target)
                primitives[...,4].ravel(order='F').tofile(target)
                gradient.tofile(target);sensor.tofile(target);mask.tofile(target)
            raws.append(sensor);flags.append(mask)
            entries.append(dict(stage=stage,range=[float(sensor.min()),float(sensor.max())],active=int(mask.sum())))
        sensors.append(raws);masks.append(flags)
        report[backend]=dict(binary=str(executable),sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),stages=entries)
    report['pair_errors']=[compare(a[:,None],b[:,None]) for a,b in zip(*sensors)]
    if any(not np.array_equal(a,b) for a,b in zip(*masks)):
        raise AssertionError('frozen CPU/old GPU sensor masks differ')
    report['definition']='divergence squared, pressure curvature, halo-only exchange, -hm+1:hm expansion, strict threshold'
    report['characteristic']=args.characteristic
    (root/'reference.json').write_text(json.dumps(report,indent=2)+'\n')
    for item in root.rglob('*'):
        item.chmod(0o555 if item.is_dir() else 0o444)
    root.chmod(0o555)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
