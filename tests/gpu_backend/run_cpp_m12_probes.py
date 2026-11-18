"""Existing 72 M12 probes: original references, sparse C++ supply and continuation."""
import argparse
import csv
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import h5py
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests/gpu_validation'))
sys.path.insert(0,str(ROOT/'scripts/input'))
from prepare_m12_local_case import prepare
from convert_legacy_input import convert_case,namelist,RealLiteral
from run_m12_hd_coverage import prepare_probe_map
from run_cpp_tgv_gate import checkpoint,compare,logged_steps
from run_cpp_restart_gate import exact_state
from run_output_restart_validation import archive_schedule_payload
from run_native_precision_restart_gate import check_state

FINAL=Path('outdat/new/checkpoints/step000000000004')
FIELDS=('u_Uinf','v_Uinf','w_Uinf','rho_rhoinf','p_pinf','T_Tinf')


def records(case):
    paths=list((case/'outdat/new/monitor_products').rglob('probes.csv'))
    if len(paths)!=1:raise AssertionError(('probe stream',paths))
    with paths[0].open() as stream:
        rows=list(csv.DictReader(stream))
    values=np.asarray([[float(row[key]) for key in ('step','time','probe',*FIELDS)] for row in rows])
    if not np.isfinite(values).all():raise AssertionError('probe values must be finite')
    return values


def independent(case):
    data=records(case);last=data[data[:,0]==4]
    if last.shape!=(72,9) or not np.array_equal(last[:,2],np.arange(1,73)):
        raise AssertionError('complete probe identity')
    indices=np.loadtxt(case/'datin/insitu_probes.dat',skiprows=1,usecols=(0,1,2),dtype=int)
    with h5py.File(case/FINAL/'state.h5') as h:
        q=np.asarray([[float(h[f'q{c:04d}'][k,j,i]) for c in range(1,6)] for i,j,k in indices])
    density=q[:,0];velocity=q[:,1:4]/density[:,None]
    energy=q[:,4]-.5*density*np.sum(velocity**2,axis=1)
    pressure=energy/(1/(1.4-1));pinf=1/(1.4*12**2)
    expected=np.column_stack((velocity,density,pressure/pinf,(pressure/density)*(1.4*12**2)))
    return compare(last[:,3:],expected)


def wall_records(case,steps=(1,2,3,4)):
    result={}
    for step in steps:
        paths=sorted((case/'outdat/new/wall_products').rglob(f'wall.step{step:012d}.rank*.csv'))
        if len(paths)!=2:raise AssertionError(('two wall partitions required',step,paths))
        values=np.concatenate([np.loadtxt(path,delimiter=',',skiprows=1,ndmin=2) for path in paths])
        if values.shape!=(65,4) or not np.array_equal(values[:,0],np.arange(65)) or not np.isfinite(values).all():
            raise AssertionError('unique ordered finite M12 wall curve')
        result[step]=values
    return result


def independent_wall(case):
    with h5py.File(case/FINAL/'state.h5') as state:
        q=np.stack([state[f'q{c:04d}'][...] for c in range(1,6)],axis=-1)
    with h5py.File(case/'outdat/new/resources/geometry.h5') as geometry:
        mx=geometry['q0005'][...];my=geometry['q0009'][...]
        x=geometry['q0001'][0,0,:];z=geometry['q0003'][:,0,:]
    rho=q[...,0];v=q[...,1:4]/rho[...,None]
    p=(q[...,4]-.5*rho*np.sum(v*v,axis=-1))/(1/(1.4-1));temperature=(p/rho)*(1.4*12**2)
    du=(-1.5*v[:,0,:,0]+2*v[:,1,:,0]-.5*v[:,2,:,0])*my[:,0,:]
    dt=(-1.5*temperature[:,0,:]+2*temperature[:,1,:]-.5*temperature[:,2,:])*my[:,0,:]
    normal=v[:,0,:,1];dv=np.empty_like(normal)
    for i in range(65):
        if i==0:ids=[0,1,2];weights=[-1.5,2.,-.5]
        elif i==64:ids=[62,63,64];weights=[.5,-2.,1.5]
        elif i in (1,63):ids=[i-1,i+1];weights=[-.5,.5]
        elif i in (2,62):ids=list(range(i-2,i+3));weights=[1/12,-2/3,0.,2/3,-1/12]
        else:ids=list(range(i-3,i+4));weights=[-1/60,.15,-.75,0.,.75,-.15,1/60]
        dv[:,i]=normal[:,ids]@np.asarray(weights)
    dv*=mx[:,0,:]
    s=110.3/594.3;t=temperature[:,0,:]
    mu=t*np.sqrt(t)*(1+s)/(t+s)/1.e4
    dz=np.diff(z,axis=0);weights=.5*(dz+np.roll(dz,1,axis=0))
    cf=2*mu*(du+dv);heat=((mu/.72)/(.4*12**2))*dt
    values=np.column_stack((np.arange(65),x,
        np.sum(weights*cf[:-1],axis=0)/np.sum(weights,axis=0),
        np.sum(weights*heat[:-1],axis=0)/np.sum(weights,axis=0)))
    return compare(wall_records(case,steps=(4,))[4],values)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','output','mpiexec'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--reference',type=Path)
    p.add_argument('--freeze-only',action='store_true')
    p.add_argument('--memcheck',action='store_true')
    p.add_argument('--wall',action='store_true',help='also exercise the existing spanwise-averaged wall product')
    p.add_argument('--transport',choices=('pinned','device'),default='pinned',help='solver halo transport')
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    binary=out/'astr';shutil.copy2(a.binary,binary);binary.chmod(0o555)
    env={k:v for k,v in os.environ.items() if not k.startswith(('ASTR_','OMPI_MCA_','UCX_'))}
    env.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',OMP_NUM_THREADS='1',
        ASTR_GPU_SYNC_MODE='explicit',ASTR_GPU_HALO_TRANSPORT='pinned',ASTR_FORCE_MPI_TOPOLOGY='2,1,1')
    if a.transport=='device':
        env.update(OMPI_MCA_pml='ucx',OMPI_MCA_opal_cuda_support='1',UCX_MEMTYPE_CACHE='n',ASTR_GPU_HALO_TRANSPORT='device')
    definition=dict(id='m12_probes',kind='m12_probes',enable_request=1,source='instantaneous',
        sampling=dict(mode='provider'),output=dict(channel='data',format='csv'),schedule=dict(mode='steps',interval=1))
    definitions=[definition]
    if a.wall:
        wall=deepcopy(definition);wall.update(id='m12_wall',kind='m12_wall');definitions.append(wall)
    def run(name,gpu=True,steps=4,restore=None,observe=True,instrument=False):
        case=out/name
        prepare(ROOT.parent/'M12_C13',case,'gpu' if gpu else 'cpu',intervals=(64,32,24))
        # Materialize the original UDF's z=90/ka*k extrusion for the generic root build.
        with h5py.File(case/'datin/grid.2d') as source,h5py.File(case/'datin/grid.m12.h5','x') as target:
            for key in ('x','y'):target[key]=np.repeat(source[key][...][None,:,:],25,axis=0)
            target['z']=np.broadcast_to(((90./24)*np.arange(25))[:,None,None],(25,33,65)).copy()
        with h5py.File(case/'datin/flowini2d.h5') as source,h5py.File(case/'datin/flowini3d.h5','x') as target:
            if np.any(source['u3'][...]!=0):raise AssertionError('original 2D initialization has nonzero spanwise velocity')
            for key in source:target[key]=np.repeat(source[key][...][None,:,:],25,axis=0)
        prepare_probe_map(case,(64,32,24));(case/'outdat/new').mkdir()
        groups=convert_case((case/'datin/input.dat').read_text(),model_environment={
            'ASTR_WALL_BLOWING_MODE':'legacy_random','ASTR_NSCBC_FARFIELD_MODE':'compatibility',
            'ASTR_GPU_PRECISION_MODE':'fp64','ASTR_GPU_MP_POSITIVITY':'off'})
        groups['boundary']['bctype']=[11,22,41,52,1,1]
        groups['boundary']['pout']=RealLiteral('1.d0')
        groups['mesh'].update(lreadgrid=True,gridfile='datin/grid.m12.h5')
        groups['initialization']['ninit']=3
        (case/'datin/input.case').write_text(namelist(groups))
        (case/'datin/input.run').write_text(namelist(dict(run_control=dict(max_steps=steps),
            timestep=dict(mode='fixed',fixed_dt=.02),logging=dict(list_steps=1),
            repair=dict(crashfix=False),statistics=dict(enabled=False))))
        (case/'datin/input.output').write_text(namelist(dict(output=dict(directory='outdat/new',
            restore_directory=str(restore or ''),exact_restart=True,checkpoint_statistics=True,
            host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0),
            checkpoint=dict(enabled=True,mode='steps',interval_steps=2,keep=2,initial_frame=False),
            volume=dict(enabled=False),slices=dict(enabled=False))))
        execution=dict(env)
        if observe:
            (case/'datin/input.insitu').write_text(namelist(dict(insitu_run=dict(enabled=True,
                statistics=True,render=False,products='catalog_stats',statistics_window=[0.,1.],
                processing_backend='host',derivative_backend='cpu',host_budget_bytes=-1,
                device_budget_bytes=-1,device_reserve_bytes=0,output_directory='outdat/render'))))
            (case/'input.products.jsonl').write_text('\n'.join(json.dumps(r) for r in
                [dict(catalog_schema='astr-insitu-products-v1'),*(definitions if restore is None else [])])+'\n')
            execution['ASTR_INSITU_CONFIG']='datin/input.insitu'
        command=[str(a.mpiexec.resolve()),'--bind-to','none','-n','2']
        if instrument:command+=['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer',
            '--tool','memcheck','--error-exitcode','99','--log-file',str(case/'memcheck.%p.log')]
        command +=[str(binary),'run','datin/input.case']
        with (case/'run.log').open('x') as log:
            subprocess.run(command,cwd=case,env=execution,stdout=log,stderr=subprocess.STDOUT,check=True)
        text=(case/'run.log').read_text()
        if 'The job is done!' not in text:raise AssertionError('incomplete M12 run')
        if instrument:
            logs=list(case.glob('memcheck.*.log'))
            if len(logs)!=2 or any('ERROR SUMMARY: 0 errors' not in x.read_text() for x in logs):
                raise AssertionError('M12 probe memory check')
        if observe:
            status=json.loads((case/'outdat/new/products.status.json').read_text())
            if len(status['products'])!=len(definitions) or any(item['state']!='active' for item in status['products']):
                raise AssertionError(status)
            data=records(case);steps_expected=range(1 if restore is None else 3,steps+1)
            if not np.array_equal(data[:,0],np.repeat(list(steps_expected),72)):
                raise AssertionError('M12 probe complete-step clock')
            first_step=1 if restore is None else 3
            if not np.array_equal(data[:,1],np.repeat(logged_steps(case,steps-first_step+1,first_step)[:,2],72)):
                raise AssertionError('M12 probe time differs from completed RK')
        return case
    report=dict(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),ranks=2,transport=a.transport)
    if a.freeze_only:
        cpu=run('cpu',gpu=False,observe=False);old=run('gpu')
        report['flow']=compare(checkpoint(old/FINAL/'state.h5'),checkpoint(cpu/FINAL/'state.h5'))
        report['independent']=independent(old)
        if a.wall:report['independent_wall']=independent_wall(old)
        (out/'reference.json').write_text(json.dumps(report,indent=2)+'\n')
        for path in out.rglob('*'):path.chmod(0o555 if path.is_dir() else 0o444)
        out.chmod(0o555);print('original CPU/GPU M12 probe reference frozen');return
    if a.reference is None:p.error('--reference required for the C++ gate')
    current=run('full',instrument=a.memcheck)
    if (current/'run.log').read_text().count('backend=cuda')!=2:raise AssertionError('actual C++ backend')
    previous=records(a.reference/'gpu');actual=records(current)
    if not np.array_equal(previous[:,:3],actual[:,:3]):raise AssertionError('reference probe identities')
    report['probes']=compare(actual[:,3:],previous[:,3:]);report['independent']=independent(current)
    report['flow']={kind:compare(checkpoint(current/FINAL/'state.h5'),
        checkpoint(a.reference/kind/FINAL/'state.h5')) for kind in ('cpu','gpu')}
    report['physical']=check_state(current/FINAL/'state.h5','fp64')
    receipts=re.findall(r'ASTR_RUNTIME_CPP_M12_PROBES rank=(\d+) step=(\d+) device_bytes=(\d+) '
        r'setup_index_h2d_bytes=(\d+) status_d2h_bytes=(\d+) values_d2h_bytes=(\d+)',(current/'run.log').read_text())
    if len(receipts)!=8 or {(int(r[0]),int(r[1])) for r in receipts}!={(r,s) for r in range(2) for s in range(1,5)}:
        raise AssertionError('missing or misphased per-rank probe receipts')
    if any(list(map(int,row[2:]))!=[4324,864,4,3456] for row in receipts):
        raise AssertionError('unexpected sparse probe allocations or transfers')
    report['transfer_receipts']=receipts
    if a.wall:
        report['walls']={step:compare(values,wall_records(a.reference/'gpu')[step])
            for step,values in wall_records(current).items()}
        report['independent_wall']=independent_wall(current)
        wall_receipts=re.findall(r'ASTR_RUNTIME_CPP_M12_WALL rank=(\d+) step=(\d+) device_bytes=(\d+) '
            r'setup_weights_h2d_bytes=(\d+) status_d2h_bytes=(\d+) curve_d2h_bytes=(\d+)',(current/'run.log').read_text())
        if len(wall_receipts)!=8 or {(int(r[0]),int(r[1])) for r in wall_receipts}!={(r,s) for r in range(2) for s in range(1,5)}:
            raise AssertionError('missing or misphased per-rank wall receipts')
        if any(list(map(int,row[2:]))!=[7132,6336,4,792] for row in wall_receipts):
            raise AssertionError('unexpected profile-wall allocations or transfers')
        report['wall_transfer_receipts']=wall_receipts
    quiet=run('quiet',observe=False)
    report['observation_isolation']=exact_state(current/FINAL/'state.h5',quiet/FINAL/'state.h5')
    prefix=run('prefix',steps=2);tail=run('tail',restore=prefix/'outdat/new/checkpoints/step000000000002')
    report['exact_restart']=exact_state(current/FINAL/'state.h5',tail/FINAL/'state.h5')
    for name in ('control.bin','insitu_control.bin'):
        if (current/FINAL/name).read_bytes()!=(tail/FINAL/name).read_bytes():raise AssertionError('exact '+name)
    if archive_schedule_payload(current/FINAL/'archives.bin')!=archive_schedule_payload(tail/FINAL/'archives.bin'):
        raise AssertionError('M12 product schedule restoration')
    if records(tail).tobytes()!=actual[actual[:,0]>2].tobytes():raise AssertionError('resumed M12 probe words')
    if a.wall:
        original_walls=wall_records(current)
        if any(value.tobytes()!=original_walls[step].tobytes() for step,value in wall_records(tail,(3,4)).items()):
            raise AssertionError('resumed M12 wall words')
    report['memcheck']=a.memcheck
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('C++ M12 probes, state isolation and exact continuation complete')


if __name__=='__main__':main()
