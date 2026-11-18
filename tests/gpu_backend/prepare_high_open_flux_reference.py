"""Freeze original production H1 gradients, masks and interior flux divergence."""
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

from run_cpp_tgv_gate import compare,namelist
from run_cpp_multi_tgv_gate import raw_field

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests/gpu_validation'))
from test_grcbc_oblique_shock import prepare
from named_fixture import prepare_named_run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cpu','old-gpu','mpiexec','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'bin').mkdir()
    report=dict(scope='16-cubed production H1 entropy/oblique-shock frozen states; one step, not a C++ main test',
        dt=1e-5,steps=1,schemes=['743e','643e'],candidate='none',binaries={},fixtures={})
    for backend,source in [('cpu',args.cpu),('gpu',args.old_gpu)]:
        binary=out/'bin'/backend;shutil.copy2(source.resolve(),binary)
        report['binaries'][backend]=dict(source=str(source.resolve()),sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    for initial,entry in [('entropy',2.),('shock',.25)]:
        packets=out/initial;packets.mkdir();paired=[];details={}
        for backend in ('cpu','gpu'):
            case=out/f'{initial}_{backend}'
            prepare(case,backend,16,1,initial=initial,entry_x=entry,z_intervals=16)
            env={k:v for k,v in os.environ.items() if not k.startswith('ASTR_')}
            env.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
                OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',OMP_NUM_THREADS='1',
                ASTR_GPU_SYNC_MODE='explicit',ASTR_GPU_HALO_TRANSPORT='pinned',ASTR_FORCE_MPI_TOPOLOGY='1,1,1',
                ASTR_NSCBC_FARFIELD_MODE='sbli_grcbc',ASTR_NSCBC_FARFIELD_SHOCK_X=str(entry),
                ASTR_NSCBC_FARFIELD_SHOCK_ANGLE_DEG='40',ASTR_NSCBC_FARFIELD_RHO='1',ASTR_NSCBC_FARFIELD_U='1',
                ASTR_NSCBC_FARFIELD_V='0',ASTR_NSCBC_FARFIELD_W='0',ASTR_NSCBC_FARFIELD_T='1',
                ASTR_GPU_MP_POSITIVITY='off',ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',ASTR_VALIDATION_RHS_STEP='0',
                ASTR_VALIDATION_GRADIENTS='1',ASTR_SHOCK_SENSOR_DUMP='outdat/sensor.txt',ASTR_GEOMETRY_DUMP='outdat/geometry.h5')
            prepare_named_run(case,'datin/input.flatplate',env,numerics=dict(convection_form='standard',viscous_form='standard',
                conschm='743e',difschm='643e',recon_schem=5,bfacmpld=.1,lchardecomp=True,lfilter=False))
            (case/'datin/input.run').write_text(namelist(dict(run_control=dict(max_steps=1),
                timestep=dict(mode='fixed',fixed_dt=1e-5),logging=dict(list_steps=1),
                repair=dict(crashfix=False),statistics=dict(enabled=False))))
            (case/'datin/input.output').write_text(namelist(dict(output=dict(directory='outdat/new',
                host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0),
                checkpoint=dict(enabled=False),volume=dict(enabled=False),slices=dict(enabled=False))))
            command=[str(args.mpiexec.resolve()),'--bind-to','none','-n','1',str(out/'bin'/backend),'run']
            with (case/'run.log').open('w') as log:
                subprocess.run(command,cwd=case,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            if 'sbli-grcbc-h1 high-order boundary chain' not in (case/'run.log').read_text():
                raise AssertionError('original H1 chain was not activated')
            values=[];stages=[]
            for stage in (1,2,3):
                suffix=f'.step00000000.rk{stage:02d}.rank00000000.bin'
                header,q=raw_field(case/('outdat/rhs.pre_rhs'+suffix))
                ph,p=raw_field(case/('outdat/rhs.pre_rhs_primitives'+suffix),primitive=True)
                if tuple(header[:4])!=tuple(ph):raise AssertionError('H1 prepared phase layout')
                upper=np.asarray(header[:3]);halo=header[3];nodes=int(np.prod(upper+1));shape=tuple(upper+1)
                with (case/('outdat/rhs.sensor'+suffix)).open('rb') as stream:
                    sh=np.fromfile(stream,np.int32,3);sensor=np.fromfile(stream,np.float64,nodes)
                    mask=np.fromfile(stream,np.int8,nodes)
                with (case/('outdat/rhs.gradients'+suffix)).open('rb') as stream:
                    gh=np.fromfile(stream,np.int32,5);gradient=np.fromfile(stream,np.float64,nodes*9)
                    temperature=np.fromfile(stream,np.float64,nodes*3)
                with (case/('outdat/rhs.conv'+suffix)).open('rb') as stream:
                    rh=np.fromfile(stream,np.int32,4);rhs=np.fromfile(stream,np.float64).reshape((*shape,5),order='F')
                if tuple(sh)!=tuple(upper) or tuple(gh[:3])!=tuple(upper) or tuple(rh)!=(*upper,5):
                    raise AssertionError('H1 gradient/sensor/convection layout')
                # Only the interior divergence is owned by the upwind-axis consumer.
                rhs[[0,-1],:,:,:]=0;rhs[:,[0,-1],:,:]=0
                points=np.indices(q.shape[:3])-halo
                indices=(np.clip(points[0],0,upper[0]),np.clip(points[1],0,upper[1]),points[2]%upper[2])
                metric=np.zeros((*q.shape[:3],3,3),order='F')
                with h5py.File(case/'outdat/geometry.h5') as geometry:
                    jac=geometry['jacob'][...].transpose(2,1,0)[indices]
                    for a in range(3):
                        for b in range(3):metric[...,a,b]=geometry[f'dxi{a+1}{b+1}'][...].transpose(2,1,0)[indices]
                packet=packets/f'mp7ld_flat_{backend}.rk{stage:02d}.bin'
                with packet.open('xb') as stream:
                    np.asarray([*upper,halo,5,1,0,0,1],np.int32).tofile(stream)
                    np.asarray([1.4,2.,.1,.001],np.float64).tofile(stream)
                    for array in (q,p,metric,jac,gradient,temperature,sensor,mask,rhs):array.ravel(order='F').tofile(stream)
                physical=tuple(slice(halo,halo+n+1) for n in upper)
                values.append(dict(q=q[physical],p=p[physical],gradient=gradient.reshape((*shape,3,3),order='F'),
                    temperature=temperature.reshape((*shape,3),order='F'),sensor=sensor.reshape(shape,order='F'),
                    mask=mask,rhs=rhs))
                stages.append(dict(stage=stage,marked=int(np.count_nonzero(mask)),nodes=nodes))
            paired.append(values);details[backend]=dict(stages=stages,command=command)
        differences=[]
        for cpu,gpu in zip(*paired):
            if not np.array_equal(cpu['mask'],gpu['mask']):raise AssertionError('original H1 mask differs')
            differences.append({key:max(compare(gpu[key][...,None] if key=='sensor' else gpu[key],
                cpu[key][...,None] if key=='sensor' else cpu[key])) for key in ('q','p','gradient','temperature','sensor','rhs')})
        report['fixtures'][initial]=dict(entry_x=entry,references=details,original_max_scaled=differences)
    shutil.copyfile(__file__,out/'preparation.py')
    report['source_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
        ('src/solver.F90','src/riemann.F90','src/flux.F90','src/boundary_high_order_body.inc',
         'src_gpu/solver_gpu.cuf','src_gpu/gradcal_gpu.cuf','src_gpu/boundary_gpu.cuf',
         'tests/gpu_validation/test_grcbc_oblique_shock.py','tests/gpu_backend/prepare_high_open_flux_reference.py')}
    (out/'reference.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['fixtures'],indent=2))


if __name__=='__main__':main()
