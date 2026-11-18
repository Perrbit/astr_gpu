"""Freeze complete original upwind inputs/RHS, not a Python flow integrator."""
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

from run_cpp_tgv_gate import prepare_case,compare,namelist

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests/gpu_validation'))
from named_fixture import convert_case
from prepare_s1_flatplate_case import write_grid,write_profile,write_input


def prepare_flat(case,gpu):
    datin=case/'datin';datin.mkdir(parents=True)
    (case/'outdat').mkdir()
    y=write_grid(datin/'grid.flatplate.h5',16,16,16,0.,10.,.8,.25,0.,0.)
    write_profile(datin/'inlet.prof',y,.08,1.,True,False)
    write_input(datin/'input.flatplate',16,16,16,'t' if gpu else 'f','t','f','743e','t',
        .001,1000.,2.,273.15,1.,50,11,3,0,'prof',1)
    groups=convert_case((datin/'input.flatplate').read_text())
    groups['numerics'].update(convection_form='standard',viscous_form='standard')
    groups['precision_options']=dict(gpu_precision_mode='fp64')
    (datin/'input.case').write_text(namelist(groups))
    with h5py.File(datin/'grid.flatplate.h5') as grid:
        x,y,z=(grid[n][...].transpose(2,1,0) for n in 'xyz')
    velocity=(1-np.exp(-(y/.08)**2))*np.where(x>5.,.8,1.)
    fields=dict(ro=np.where(x>5.,1.25,1.)+0*y,t=np.ones_like(x),u1=velocity,
        u2=.002*np.sin(np.pi*y)*np.sin(2*np.pi*z/.25),
        u3=.003*np.sin(np.pi*y)*np.cos(2*np.pi*z/.25))
    with h5py.File(datin/'flowini3d.h5','x') as flow:
        for name,value in fields.items():flow[name]=value.transpose(2,1,0)
    (datin/'input.run').write_text(namelist(dict(run_control=dict(max_steps=1),
        timestep=dict(mode='fixed',fixed_dt=1e-6),logging=dict(list_steps=1),
        repair=dict(crashfix=False),statistics=dict(enabled=False))))


def main():
    parser=argparse.ArgumentParser()
    for name in ('template','mpiexec','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--native-binary',type=Path,help='original CPU/CUDA Fortran executable, selected by case input')
    parser.add_argument('--precision',choices=('fp64','fp32'),default='fp64')
    args=parser.parse_args()
    if (args.reference is None)==(args.native_binary is None):
        parser.error('select --reference or --native-binary')
    if args.precision=='fp32' and args.native_binary is None:
        parser.error('FP32 requires an explicitly selected original native binary')
    dtype=np.float32 if args.precision=='fp32' else np.float64
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    if args.native_binary:
        shutil.copy2(args.native_binary,root/'original_astr')

    def raw_field(path,primitive=False):
        with path.open('rb') as stream:
            header=tuple(int(v) for v in np.fromfile(stream,np.int32,4 if primitive else 5))
            values=np.fromfile(stream,dtype)
        im,jm,km,halo=header[:4]
        components=6 if primitive else header[4]
        shape=(im+2*halo+1,jm+2*halo+1,km+2*halo+1,components)
        if values.size!=np.prod(shape):raise AssertionError('original native field width/layout')
        return header,values.reshape(shape,order='F')

    report={}
    for family,reconstruction,characteristic,mach,weight,periodic in [
        ('mp7',3,False,3.,.3,[1,1,1]),('mp7_roe',3,True,3.,.3,[1,1,1]),
        ('mp7ld_flat',5,True,2.,.1,[0,0,1])]:
        paired=[];case_reports={}
        for backend,binary in [('cpu','astr_cpu'),('gpu','astr_cuda_fortran')]:
            case=root/(family+'_'+backend);gpu=backend=='gpu'
            if reconstruction==5:
                prepare_flat(case,gpu)
            else:
                prepare_case(case,args.template.resolve(),gpu,False,False,steps=1,dt=1e-5,
                    numerics=dict(conschm='543e',difschm='643e',lfilter=False,
                        lchardecomp=characteristic,recon_schem=3,shkcrt=.001),
                    physics=dict(diffterm=False,mach=mach),case_options=dict(flowtype='shuosher'))
            (case/'datin/input.output').write_text(namelist(dict(output=dict(directory='outdat/new',
                host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0),
                checkpoint=dict(enabled=False),volume=dict(enabled=False),slices=dict(enabled=False))))
            executable=(root/'original_astr' if args.native_binary else args.reference/'bin'/binary).resolve()
            env=os.environ|dict(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
                OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',ASTR_FORCE_MPI_TOPOLOGY='1,1,1',
                ASTR_GPU_SYNC_MODE='explicit',ASTR_SHOCK_SENSOR_DUMP='outdat/sensor.txt',
                ASTR_GEOMETRY_DUMP='outdat/geometry.h5',ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',
                ASTR_VALIDATION_RHS_STEP='0',ASTR_VALIDATION_GRADIENTS='1',ASTR_GPU_MP_POSITIVITY='off')
            with (case/'run.log').open('w') as log:
                subprocess.run([str(args.mpiexec.resolve()),'--bind-to','none','-n','1',str(executable),'run'],
                    cwd=case,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
            rhs_stages=[];entries=[]
            for stage in (1,2,3):
                suffix=f'.step00000000.rk{stage:02d}.rank00000000.bin'
                header,q=raw_field(case/('outdat/rhs.pre_rhs'+suffix))
                ph,p=raw_field(case/('outdat/rhs.pre_rhs_primitives'+suffix),primitive=True)
                if header[:4]!=ph:raise AssertionError('frozen q/cache phase layout')
                upper=np.asarray(header[:3]);halo=header[3];nodes=int(np.prod(upper+1))
                with (case/('outdat/rhs.sensor'+suffix)).open('rb') as source:
                    sh=np.fromfile(source,np.int32,3);sensor=np.fromfile(source,dtype,nodes)
                    mask=np.fromfile(source,np.int8,nodes)
                    if source.read(1):raise AssertionError('original sensor native payload width')
                with (case/('outdat/rhs.gradients'+suffix)).open('rb') as source:
                    gh=np.fromfile(source,np.int32,5);gradient=np.fromfile(source,dtype,nodes*9)
                    # Temperature gradient follows the velocity tensor in this protocol.
                    thermal=np.fromfile(source,dtype,nodes*3)
                    normal_count=int(gh[3]+gh[4])*int(upper[0]+1)*int(upper[2]+1)*3
                    normals=np.fromfile(source,dtype,normal_count)
                    if thermal.size!=nodes*3 or normals.size!=normal_count or source.read(1):
                        raise AssertionError('original gradient native payload width')
                with (case/('outdat/rhs.conv'+suffix)).open('rb') as source:
                    rh=np.fromfile(source,np.int32,4);rhs=np.fromfile(source,dtype).reshape((*upper+1,5),order='F')
                if tuple(sh)!=tuple(upper) or tuple(gh[:3])!=tuple(upper) or tuple(rh)!=(*upper,5):
                    raise AssertionError('frozen sensor/gradient/RHS layout')
                points=np.indices(q.shape[:3])-halo
                indices=tuple(points[a]%upper[a] if periodic[a] else np.clip(points[a],0,upper[a]) for a in range(3))
                metric=np.zeros((*q.shape[:3],3,3),dtype=dtype,order='F')
                physical=tuple(slice(halo,halo+n+1) for n in upper)
                with h5py.File(case/'outdat/geometry.h5') as geometry:
                    j=geometry['jacob'][...].transpose(2,1,0);jac=j[indices];jac[physical]=j
                    if j.dtype!=np.dtype(dtype):raise AssertionError('original geometry native precision')
                    for a in range(3):
                        for b in range(3):
                            values=geometry[f'dxi{a+1}{b+1}'][...].transpose(2,1,0)
                            metric[...,a,b]=values[indices];metric[physical+(a,b)]=values
                if not np.isfinite(rhs).all() or not np.isfinite(sensor).all():raise AssertionError('nonfinite frozen observation')
                with (root/f'{family}_{backend}.rk{stage:02d}.bin').open('xb') as target:
                    np.asarray([*upper,halo,reconstruction,int(characteristic),*periodic],np.int32).tofile(target)
                    np.asarray([1.4,mach,weight,.001],dtype).tofile(target)
                    for values in (q,p,metric,jac):values.ravel(order='F').tofile(target)
                    gradient.tofile(target);sensor.tofile(target);mask.tofile(target);rhs.ravel(order='F').tofile(target)
                rhs_stages.append(rhs);entries.append(dict(stage=stage,active_nodes=int(mask.sum()),nodes=nodes))
            paired.append(rhs_stages)
            case_reports[backend]=dict(binary=str(executable),sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),stages=entries)
        if args.precision=='fp64':
            case_reports['rhs_pair_error']=[compare(a,b) for a,b in zip(*paired)]
        else:
            case_reports['rhs_pair_error']=[(np.max(np.abs(np.asarray(a,np.float64)-b),axis=(0,1,2))/
                np.maximum(1.,np.max(np.abs(b),axis=(0,1,2)))).tolist() for a,b in zip(*paired)]
        report[family]=case_reports
    report['precision']=args.precision
    (root/'reference.json').write_text(json.dumps(report,indent=2)+'\n')
    for item in root.rglob('*'):item.chmod(0o555 if item.is_dir() else 0o444)
    root.chmod(0o555)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
