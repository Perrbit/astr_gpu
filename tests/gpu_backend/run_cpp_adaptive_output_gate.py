"""Existing TGV adaptive collector, output clocks and same-kind continuation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import h5py
import numpy as np

from run_cpp_tgv_gate import prepare_case, namelist, compare, checkpoint, logged_steps
from run_cpp_restart_gate import exact_state

ROOT=Path(__file__).resolve().parents[2]
MONITOR=re.compile(r'ASTR_AP_MONITOR step=(\d+) time=\s*(\S+) kinetic_energy=\s*(\S+) rates=(.*?) active=(.*?) windows=(.*?) wall_seconds=')
PRODUCT=re.compile(r'ASTR_AP_PRODUCT id=(\S+) step=(\d+)')


def monitor(case):
    rows=[];flags=[]
    for m in MONITOR.finditer((case/'run.log').read_text()):
        rows.append([int(m[1]),float(m[2]),float(m[3]),*map(float,m[4].split())])
        flags.append([m[5].split(),list(map(int,m[6].split()))])
    return np.asarray(rows),flags


def product_steps(case):
    return [list(m.groups()) for m in PRODUCT.finditer((case/'run.log').read_text())]


def energy_from_checkpoint(case,precision='fp64'):
    rows,_=monitor(case);errors={}
    for path in sorted((case/'outdat/new/checkpoints').glob('step*/state.h5')):
        step=int(path.parent.name.removeprefix('step'))
        q=checkpoint(path)[:16,:16,:16,:]
        if q.dtype!=np.dtype('f4' if precision=='fp32' else 'f8'):raise AssertionError('adaptive state storage kind')
        q=q.astype(np.float64)
        density=q[...,0];internal=q[...,4]-.5*np.sum(q[...,1:4]**2,axis=-1)/density
        if not np.isfinite(q).all() or not (density>0).all() or not (internal>0).all():
            raise AssertionError('nonphysical adaptive checkpoint state')
        # The fixture has unit reference density and velocity.
        value=.5*np.mean(np.sum(q[...,1:4]**2,axis=-1)/density)
        selected=rows[rows[:,0]==step]
        if selected.shape[0]!=1:raise AssertionError('checkpoint/monitor step identity')
        error=float(abs(selected[0,2]-value))
        if precision=='fp64' and error>2e-10:raise AssertionError('adaptive energy differs from same-phase stored q')
        errors[str(step)]=error
    return errors


def compare_observation(actual,reference,first=0,limit=2e-10):
    a,af=monitor(actual);b,bf=monitor(reference)
    take=a[:,0]>=first;a=a[take];af=[f for f,t in zip(af,take) if t]
    take=b[:,0]>=first;b=b[take];bf=[f for f,t in zip(bf,take) if t]
    if af!=bf or not np.array_equal(a[:,:2],b[:,:2]):raise AssertionError('adaptive sample clocks/event/window flags')
    return compare(a[:,2:],b[:,2:],limit=limit)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    selection=parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--original',type=Path,help='freeze the actual dual CPU/CUDA-Fortran binary')
    selection.add_argument('--binary',type=Path,help='test the new C++ binary against --reference')
    parser.add_argument('--reference',type=Path)
    for key in ('output','mpiexec'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--ranks',type=int,choices=(1,2),default=1)
    parser.add_argument('--axis',choices=('x','y','z'),default='y')
    parser.add_argument('--automatic',action='store_true')
    parser.add_argument('--transport',choices=('pinned','device'),default='pinned')
    parser.add_argument('--sync',choices=('explicit','dependency'),default='explicit')
    parser.add_argument('--memcheck',action='store_true')
    parser.add_argument('--analyze-only',action='store_true')
    parser.add_argument('--precision',choices=('fp64','fp32'),default='fp64')
    args=parser.parse_args()
    limit=None if args.precision=='fp32' else 2e-10
    if args.binary and args.reference is None:parser.error('--binary requires --reference')
    root=args.output.resolve()
    if not args.analyze_only:root.mkdir(parents=True,exist_ok=False)
    binary=args.original or args.binary
    if not args.analyze_only:shutil.copy2(binary,root/'tested_astr')
    binary=root/'tested_astr'
    topology=[1,1,1];topology['xyz'.index(args.axis)]=args.ranks
    env={k:v for k,v in os.environ.items() if not k.startswith('ASTR_')}
    env.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',
        ASTR_FORCE_MPI_TOPOLOGY=','.join(map(str,topology)),ASTR_GPU_HALO_TRANSPORT=args.transport,
        ASTR_GPU_SYNC_MODE=args.sync,ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',ASTR_VALIDATION_RHS_STEP='4')
    if args.transport=='device':env.update(OMPI_MCA_pml='ucx',OMPI_MCA_opal_cuda_support='1')

    def run(name,gpu=True,steps=4,restore=None,adaptive=True,writing=True):
        case=root/name
        if args.analyze_only:
            logged_steps(case,count=steps-(2 if restore else 0),first_step=3 if restore else 1)
            return case
        prepare_case(case,ROOT/'examples/Taylor_Green_Vortex/datin',gpu,False,args.automatic,
            steps=steps,dt=.001,max_dt=.002,numerics=dict(lfilter=False),physics=dict(reynolds=1.))
        config=dict(output=dict(directory='outdat/new',restore_directory=str(restore or ''),exact_restart=True,
            checkpoint_statistics=False,host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0),
            checkpoint=dict(enabled=writing,mode='steps',interval_steps=2,keep=2,initial_frame=False),
            volume=dict(enabled=writing,mode='steps',interval_steps=2,initial_frame=False,
                adaptive=adaptive,dense_interval_steps=1 if adaptive else 0,window_ids='burst' if adaptive else ''),
            slices=dict(enabled=False))
        if adaptive:
            config['adaptive_output']=dict(enabled=True,monitor_steps=1,event_ids='decay',s_ref=1.,t_ref=1.,
                r_on=.74,r_off=.723,hold_time=.005,window_ids='burst',window_modes='steps',
                **{'window_steps(:,1)':[1,4]})
        (case/'datin/input.output').write_text(namelist(config))
        command=[str(args.mpiexec.resolve()),'--bind-to','none','-n',str(args.ranks)]
        if args.memcheck and name=='cpp':
            command+=['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer',
                '--tool','memcheck','--leak-check','full','--error-exitcode','99',
                '--log-file',str(case/'memcheck.%p.log')]
        with (case/'run.log').open('w') as log:
            subprocess.run(command+[str(binary),'run'],cwd=case,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,check=True)
        logged_steps(case,count=steps-(2 if restore else 0),first_step=3 if restore else 1)
        if adaptive and monitor(case)[0][:,0].tolist()!=list(range(3 if restore else 0,steps+1)):
            raise AssertionError('adaptive monitor sample sequence')
        return case

    report=dict(topology=topology,automatic=args.automatic,transport=args.transport,sync=args.sync,
        original=bool(args.original),precision=args.precision,difference_limit=limit,
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    final=Path('outdat/new/checkpoints/step000000000004')
    if args.original:
        cpu=run('cpu',gpu=False);old=run('gpu')
        report['cpu_gpu_monitor_scaled']=compare_observation(old,cpu,limit=limit)
        report['cpu_gpu_state_scaled']=compare(checkpoint(old/final/'state.h5').astype(float),
            checkpoint(cpu/final/'state.h5').astype(float),limit=limit)
        if product_steps(old)!=product_steps(cpu):raise AssertionError('original CPU/GPU adaptive product schedule')
        report['products']=product_steps(old)
        report['independent_checkpoint_energy']={b:energy_from_checkpoint(root/b,args.precision) for b in ('cpu','gpu')}
        (root/'reference.json').write_text(json.dumps(report,indent=2)+'\n')
        shutil.copy2(__file__,root/'preparation.py')
        for p in root.rglob('*'):p.chmod(0o555 if p.is_dir() else 0o444)
        root.chmod(0o555)
    else:
        meta=json.loads((args.reference/'reference.json').read_text())
        if meta['topology']!=topology or meta['automatic']!=args.automatic:
            raise AssertionError('adaptive reference topology/control mismatch')
        current=run('cpp');report['references']={}
        report['independent_checkpoint_energy']=energy_from_checkpoint(current,args.precision)
        for backend in ('cpu','gpu'):
            ref=args.reference/backend
            report['references'][backend]=dict(monitor_scaled=compare_observation(current,ref,limit=limit),
                state_scaled=compare(checkpoint(current/final/'state.h5').astype(float),
                    checkpoint(ref/final/'state.h5').astype(float),limit=limit))
            if product_steps(current)!=product_steps(ref):raise AssertionError('C++ adaptive product schedule')
        prefix=run('prefix',steps=2)
        tail=run('tail',restore=prefix/'outdat/new/checkpoints/step000000000002')
        report['exact_state_datasets']=exact_state(current/final/'state.h5',tail/final/'state.h5')
        if (current/final/'control.bin').read_bytes()!=(tail/final/'control.bin').read_bytes():
            raise AssertionError('adaptive controller/history exact continuation')
        if any(compare_observation(tail,current,first=3,limit=None)):raise AssertionError('adaptive monitor exact continuation')
        if product_steps(prefix)+product_steps(tail)!=product_steps(current):raise AssertionError('adaptive restart product schedule')
        report['exact_restart_field_frames']={}
        for path in sorted((tail/'outdat/new/fields').glob('*/*/data.h5')):
            relative=path.relative_to(tail/'outdat/new/fields')
            report['exact_restart_field_frames'][str(relative)]=exact_state(path,current/'outdat/new/fields'/relative)
        for label,adaptive in (('monitor-off',False),('output-off',True)):
            quiet=run(label,adaptive=adaptive,writing=False)
            for rank in range(args.ranks):
                name=f'outdat/rhs.cpp_final.step00000004.rk01.rank{rank:08d}.bin'
                if (current/name).read_bytes()!=(quiet/name).read_bytes():
                    raise AssertionError(label+' changed complete q')
            if not np.array_equal(logged_steps(current),logged_steps(quiet)):raise AssertionError('observer changed step clock')
        report['monitor_output_full_q_halo_clock_bits']=True
        if args.memcheck:
            logs=list(current.glob('memcheck.*.log'))
            if len(logs)!=args.ranks or any('ERROR SUMMARY: 0 errors' not in p.read_text() or
                'LEAK SUMMARY: 0 bytes leaked' not in p.read_text() for p in logs):raise AssertionError('adaptive pinned memory evidence')
            report['memcheck_zero_errors_leaks']=True
        report['products']=product_steps(current)
        frames=sorted((current/'outdat/new/fields').glob('*/*/data.h5'))
        report['frame_fields_scaled']={}
        for path in frames:
            relative=path.relative_to(current/'outdat/new/fields')
            with h5py.File(path) as a,h5py.File(args.reference/'gpu/outdat/new/fields'/relative) as b:
                if not np.array_equal(a['metadata'][...],b['metadata'][...]):raise AssertionError('adaptive frame identity/time')
                report['frame_fields_scaled'][str(relative)]={}
                for name in ('density','velocity_x','velocity_y','velocity_z','pressure','temperature'):
                    if a[name].dtype!=b[name].dtype:raise AssertionError('adaptive frame storage kind')
                    report['frame_fields_scaled'][str(relative)][name]=max(compare(
                        a[name][...].astype(float)[...,None],b[name][...].astype(float)[...,None],limit=limit))
        (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
