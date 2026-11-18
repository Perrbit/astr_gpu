"""Existing FP5 native CURVE states, frozen gradients and real C++ continuation."""
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

from run_cpp_tgv_gate import namelist,convert_controller,logged_steps,cfl_log,compare
from run_cpp_restart_gate import exact_state
from check_geometry_face_halos import check as geometry_halos

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests/gpu_validation'))
from run_native_precision_tgv_gate import physical_state,primitive_state,component_report,cached_state_report
from run_native_precision_curve_gate import geometry_report,read_gradients


def geometry_difference(actual,reference):
    errors={}
    with h5py.File(actual) as a,h5py.File(reference) as b:
        if set(a)!=set(b):raise AssertionError('native CURVE geometry identities')
        for name in a:
            x,y=a[name][...],b[name][...]
            if x.dtype!=np.float32 or y.dtype!=np.float32 or x.shape!=y.shape:raise AssertionError('native CURVE geometry layout/kind')
            if not np.isfinite(x).all() or not np.isfinite(y).all():raise AssertionError('nonfinite CURVE geometry')
            errors[name]=float(np.max(abs(x.astype(float)-y.astype(float))))
    return errors


def validate_scalar_samples(actual,reference,clocks):
    if actual.shape!=reference.shape or not np.array_equal(actual[:,0],reference[:,0]):
        raise AssertionError('CURVE scalar sample identities')
    selected=clocks[np.isin(clocks[:,0],actual[:,0])]
    if len(selected)!=len(actual) or not np.array_equal(selected[:,0],actual[:,0]):
        raise AssertionError('CURVE statistic steps differ from complete-step log')
    # Both statistic formats round-trip FP32, but not their parsed FP64 tails.
    expected=selected[:,2].astype(np.float32).view(np.uint32)
    if any(not np.array_equal(table[:,1].astype(np.float32).view(np.uint32),expected)
        for table in (actual,reference)):
        raise AssertionError('CURVE statistic native time differs from complete-step log')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','reference','output','mpiexec'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--transport',choices=('pinned','device'),default='pinned')
    parser.add_argument('--sync',choices=('explicit','dependency'),default='explicit')
    parser.add_argument('--memcheck',action='store_true')
    execution=parser.add_mutually_exclusive_group()
    execution.add_argument('--analyze-only',action='store_true')
    execution.add_argument('--reuse-continuous',action='store_true',help='reuse the completed primary run, execute only missing planned lifecycle cases')
    args=parser.parse_args();meta=json.loads((args.reference/'reference.json').read_text())
    root=args.output.resolve()
    if not args.analyze_only and not args.reuse_continuous:
        root.mkdir(parents=True,exist_ok=False);shutil.copy2(args.binary,root/'tested_astr')
        shutil.copy2(__file__,root/'runner.py')
    ranks=int(np.prod(meta['topology']))
    partitions=np.loadtxt(args.reference/'cpu/datin/parallel.info',skiprows=3,dtype=int,ndmin=2)
    if len(partitions)!=ranks:raise AssertionError('reference partition count')
    env={k:v for k,v in os.environ.items() if not k.startswith(('ASTR_','OMPI_MCA_','UCX_'))}
    env.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',ASTR_GPU_HALO_TRANSPORT=args.transport,
        ASTR_GPU_SYNC_MODE=args.sync,ASTR_GPU_FILTER_WORKSPACE='scalar',ASTR_GEOMETRY_DUMP='geometry.h5',
        ASTR_FORCE_MPI_TOPOLOGY=','.join(map(str,meta['topology'])),ASTR_VALIDATION_RHS_PREFIX='state',
        ASTR_VALIDATION_RHS_STEP='0',ASTR_VALIDATION_RHS_STEP_END='4')
    if args.transport=='device':env.update(OMPI_MCA_pml='ucx',OMPI_MCA_opal_cuda_support='1')

    def run(name,steps=4,restore=None,observe=True,writing=True):
        case=root/name
        if args.analyze_only or (args.reuse_continuous and name=='cpp'):return case
        shutil.copytree(args.reference/'gpu/datin',case/'datin',copy_function=shutil.copyfile)
        (case/'datin').chmod(0o755);(case/'outdat/new').mkdir(parents=True)
        controls=convert_controller((case/'datin/controller').read_text());controls['run_control']['max_steps']=steps
        (case/'datin/input.run').write_text(namelist(controls))
        (case/'datin/input.output').write_text(namelist(dict(output=dict(directory='outdat/new',
            exact_restart=True,restore_directory=str(restore or ''),checkpoint_statistics=False,
            host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0),
            checkpoint=dict(enabled=writing,mode='steps',interval_steps=2,keep=2,initial_frame=False),
            volume=dict(enabled=writing,mode='steps',interval_steps=2,initial_frame=False),slices=dict(enabled=False))))
        execution=dict(env)
        if not observe:execution.update(ASTR_VALIDATION_RHS_STEP='4',ASTR_VALIDATION_RHS_STEP_END='4')
        command=[str(args.mpiexec.resolve()),'--bind-to','none','-n',str(ranks)]
        if args.memcheck and name=='cpp':command+=['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer',
            '--tool','memcheck','--leak-check','full','--error-exitcode','99','--log-file',str(case/'memcheck.%p.log')]
        with (case/'run.log').open('w') as log:
            subprocess.run(command+[str(root/'tested_astr'),'run'],cwd=case,env=execution,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        return case

    current=run('cpp');report=dict(precision='fp32',mapping=meta['mapping'],uniform=meta['uniform'],
        topology=meta['topology'],transport=args.transport,sync=args.sync,difference_limit=None,
        binary_sha256=hashlib.sha256((root/'tested_astr').read_bytes()).hexdigest(),references={})
    clocks=logged_steps(current);cfl=cfl_log(current)
    if not np.isfinite(cfl).all() or np.max(cfl[:,-1])>.5:raise AssertionError('native CURVE CFL exceeds original FP5 .5')
    report['geometry']=geometry_report(current,'fp32',meta['mapping'])
    report['all_RK_physical_states_checked']=0
    for label in ('pre_rhs','post_update'):
        paths=sorted(current.glob(f'state.{label}.*.bin'))
        if len(paths)!=12*ranks:raise AssertionError('C++ CURVE complete RK observation coverage')
        for path in paths:
            if not np.isfinite(physical_state(path,'fp32')).all():raise AssertionError('nonfinite native CURVE physical q')
        report['all_RK_physical_states_checked']+=len(paths)
    sides={int(row[0]):((True,True),(meta['mapping']!='y-wavy' or row[8]>0,
        meta['mapping']!='y-wavy' or row[8]+row[5]<32),(True,True)) for row in partitions}
    for backend in ('cpu','gpu'):
        ref=args.reference/backend;states={};primitives={};caches={}
        for label in ('initialized','pre_rhs','post_update'):
            paths=sorted(ref.glob(f'state.{label}.*.bin'))
            if len(paths)!=(1 if label=='initialized' else 6)*ranks:raise AssertionError('native CURVE reference RK coverage')
            for path in paths:states[path.name]=component_report(physical_state(path,'fp32'),physical_state(current/path.name,'fp32'))
        for step in (0,3):
            for stage in (1,2,3):
                for rank in range(ranks):
                    name=f'state.pre_rhs_primitives.step{step:08d}.rk{stage:02d}.rank{rank:08d}.bin'
                    p=primitive_state(current/name,'fp32',halo_sides=sides[rank])
                    primitives[name]=component_report(primitive_state(ref/name,'fp32',halo_sides=sides[rank]),p)
                    caches[name]=cached_state_report(physical_state(current/name.replace('pre_rhs_primitives','pre_rhs'),'fp32'),p,'fp32',.1)
        if not np.array_equal(clocks,logged_steps(ref)):raise AssertionError('native CURVE reference complete-step clock')
        a=np.loadtxt(current/'flowstate.dat',skiprows=1,ndmin=2)
        b=np.loadtxt(ref/'flowstate.dat',skiprows=1,ndmin=2)
        validate_scalar_samples(a,b,clocks)
        report['references'][backend]=dict(states=states,primitives=primitives,caches=caches,
            geometry_max_abs=geometry_difference(current/'geometry.h5',ref/'geometry.h5'),
            statistics_scaled=compare(a[:,2:],b[:,2:],limit=None),
            statistics_max_abs=np.max(abs(a[:,2:]-b[:,2:]),axis=0).tolist())
    quiet=run('quiet',observe=False,writing=False)
    for rank in range(ranks):
        name=f'state.cpp_final.step00000004.rk01.rank{rank:08d}.bin'
        if (current/name).read_bytes()!=(quiet/name).read_bytes():raise AssertionError('CURVE observation/output changed full q/halo words')
    if not np.array_equal(logged_steps(quiet),clocks):raise AssertionError('CURVE observation changed clock')
    prefix=run('prefix',steps=2);tail=run('tail',restore=prefix/'outdat/new/checkpoints/step000000000002')
    last=Path('outdat/new/checkpoints/step000000000004')
    report['exact_datasets']=exact_state(current/last/'state.h5',tail/last/'state.h5')
    report['geometry_halos']=geometry_halos(current/'outdat/new/resources/geometry.h5')
    if not report['geometry_halos']['owner_words_equal'] or not report['geometry_halos']['shared_owner_words_equal']:
        raise AssertionError('native metric halos/shared nodes differ from owner words')
    for name in ('control.bin','insitu_control.bin'):
        if (current/last/name).read_bytes()!=(tail/last/name).read_bytes():raise AssertionError('CURVE exact control '+name)
    if not np.array_equal(logged_steps(tail,count=2,first_step=3),clocks[2:]):raise AssertionError('CURVE exact clock')
    report['field_archive_phase_words']={}
    names=('density','velocity_x','velocity_y','velocity_z','pressure','temperature')
    for step in (2,4):
        frames=list((current/'outdat/new/fields').glob(f'*/step{step:012d}/data.h5'))
        if len(frames)!=1:raise AssertionError('CURVE field frame step identity')
        saved=current/f'outdat/new/checkpoints/step{step:012d}/state.h5'
        with h5py.File(frames[0]) as a,h5py.File(saved) as b:
            metadata=a['metadata'][...]
            words=clocks[step-1,2:5].astype(np.float32).view(np.int32).astype(np.int64)
            if metadata[1]!=step or not np.array_equal(metadata[2:5],words):raise AssertionError('CURVE frame complete-step time words')
            if not np.array_equal(a['density'][...],b['q0001'][...]):raise AssertionError('CURVE frame density phase')
            for c,name in enumerate(names,6):
                x,y=a[name][...],b[f'q{c:04d}'][...]
                if x.dtype!=np.float32 or y.dtype!=np.float32 or x.shape!=y.shape or x.tobytes()!=y.tobytes():
                    raise AssertionError('CURVE frame/cache words '+name)
        report['field_archive_phase_words'][str(step)]=True
    report['observation_output_full_q_clock_bits']=True
    if meta['uniform']:
        report['compatible_constant_drift']={}
        for rank in range(ranks):
            initial=current/f'state.initialized.step00000000.rk01.rank{rank:08d}.bin'
            final=current/f'state.cpp_final.step00000004.rk01.rank{rank:08d}.bin'
            report['compatible_constant_drift'][str(rank)]=component_report(physical_state(initial,'fp32'),physical_state(final,'fp32'))
    if args.memcheck:
        logs=list(current.glob('memcheck.*.log'))
        if len(logs)!=ranks or any('ERROR SUMMARY: 0 errors' not in p.read_text() or
            'LEAK SUMMARY: 0 bytes leaked' not in p.read_text() for p in logs):raise AssertionError('CURVE memory evidence')
        report['memcheck_zero_errors_leaks']=True
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='references'},indent=2))


if __name__=='__main__':main()
