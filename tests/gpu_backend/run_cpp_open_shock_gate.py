"""Bounded ordinary-main x-NSCBC path against the repaired CUDA Fortran solver."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from run_cpp_tgv_gate import ROOT,compare,namelist,logged_steps,convert_case
from run_cpp_multi_tgv_gate import layouts,raw_field
from run_cpp_insitu_gate import compare_hdf
from run_cpp_native_tgv import field as native_field


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','output','mpiexec'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--reference-results',type=Path);parser.add_argument('--freeze-only',action='store_true')
    parser.add_argument('--ranks',type=int,choices=[1,2],default=1)
    parser.add_argument('--axis',choices=list('xyz'),default='x')
    parser.add_argument('--transport',choices=['pinned','device'],default='pinned')
    parser.add_argument('--sync',choices=['explicit','dependency'],default='explicit')
    parser.add_argument('--memcheck',action='store_true')
    parser.add_argument('--checkpoint',action='store_true')
    parser.add_argument('--sponge-width',type=int,default=0)
    parser.add_argument('--transverse',type=int,default=8)
    parser.add_argument('--precision',choices=('fp64','fp32'),default='fp64')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    dtype=np.float32 if args.precision=='fp32' else np.float64
    def compare_data(a,b):
        return compare(a.astype(np.float64),b.astype(np.float64),limit=None if args.precision=='fp32' else 2e-10)
    def qbody(path):
        if args.precision=='fp32':return native_field(path)[2]
        header,values=raw_field(path);h=header[3]
        return values[tuple(slice(h,h+n+1) for n in header[:3])+(slice(None),)]
    topology=[1,1,1];topology['xyz'.index(args.axis)]=args.ranks
    env=os.environ|dict(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',
        ASTR_GPU_HALO_TRANSPORT=args.transport,ASTR_GPU_SYNC_MODE=args.sync,ASTR_GPU_MP_POSITIVITY='off',
        ASTR_FORCE_MPI_TOPOLOGY=','.join(map(str,topology)),
        ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',ASTR_VALIDATION_RHS_STEP='0',ASTR_VALIDATION_RHS_STEP_END='4')
    if args.transport=='device':env.update(OMPI_MCA_pml='ucx',OMPI_MCA_opal_cuda_support='1')
    def run(name,observe=True,steps=4,restore=None,writing=False):
        case=out/name;(case/'datin').mkdir(parents=True);(case/'outdat/new').mkdir(parents=True)
        groups=convert_case((ROOT/'examples/Shuosher/datin/input.shuosher').read_text())
        groups['case_options'].update(flowtype='openshock',use_gpu=True)
        groups['mesh'].update(ia=64,ja=args.transverse,ka=args.transverse,lihomo=False,ljhomo=True,lkhomo=True,lreadgrid=False)
        groups['physics'].update(diffterm=False)
        groups['numerics'].update(conschm='543e',difschm='643e',recon_schem=3,lchardecomp=False,lfilter=False,
            convection_form='standard',viscous_form='standard',filter_scheme='explicit10')
        groups['boundary'].update(bctype=[12,22,1,1,1,1],pout=10.333333333333333)
        groups['precision_options']=dict(gpu_precision_mode='fp64')
        groups['sponge'].update(spg_i0=0,spg_im=args.sponge_width,spg_j0=0,spg_jm=0,spg_k0=0,spg_km=0,spg_def='layer')
        (case/'datin/input.case').write_text(namelist(groups))
        (case/'datin/input.run').write_text(namelist(dict(run_control=dict(max_steps=steps),
            timestep=dict(mode='fixed',fixed_dt=1e-5),logging=dict(list_steps=1),repair=dict(crashfix=False),
            statistics=dict(enabled=False))))
        (case/'datin/input.output').write_text(namelist(dict(output=dict(directory='outdat/new',
            host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0,exact_restart=True,
            restore_directory=str(restore or ''),checkpoint_statistics=False),
            checkpoint=dict(enabled=writing,mode='steps',interval_steps=2,keep=2,initial_frame=False),
            volume=dict(enabled=False),slices=dict(enabled=False))))
        execution=dict(env)
        if not observe:
            execution={k:v for k,v in execution.items() if not k.startswith('ASTR_VALIDATION_RHS_')}
            execution['ASTR_VALIDATION_RHS_PREFIX']='outdat/terminal'
            execution['ASTR_VALIDATION_RHS_STEP']='4'
        with (case/'run.log').open('w') as log:
            command=[str(args.mpiexec.resolve()),'--bind-to','none','-n',str(args.ranks)]
            if args.memcheck and name=='cpp':
                command+=['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer','--tool','memcheck',
                    '--error-exitcode','99','--leak-check','full','--log-file',str(case/'memcheck.%p.log')]
            subprocess.run(command+[str(args.binary.resolve()),'run'],
                cwd=case,env=execution,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,check=True)
        return case
    current=run('old_gpu' if args.freeze_only else 'cpp',writing=args.checkpoint)
    report=dict(topology=topology,transport=args.transport,sync=args.sync,sponge_width=args.sponge_width,
        grid=[64,args.transverse,args.transverse],precision=args.precision,
        binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),formal_output_tested=False,exact_restart_tested=False)
    if args.freeze_only:
        for phase in ('post_source','post_update'):
            if len(list((current/'outdat').glob('rhs.'+phase+'.*.bin')))!=12*args.ranks:
                raise AssertionError('missing original matching phase')
        logged_steps(current)
        for path in sorted((current/'outdat').glob('rhs.post_update.*.bin')):
            if not np.isfinite(qbody(path)).all():raise AssertionError('original NSCBC state finiteness')
        (out/'reference.json').write_text(json.dumps(report,indent=2)+'\n')
        shutil.copyfile(__file__,out/'preparation.py')
        for p in out.rglob('*'):p.chmod(0o555 if p.is_dir() else 0o444)
        out.chmod(0o555);print(json.dumps(report,indent=2));return
    if args.reference_results is None:parser.error('--reference-results is required')
    if 'backend=cuda' not in (current/'run.log').read_text():raise AssertionError('native C++ path not selected')
    if args.memcheck:
        logs=list(current.glob('memcheck.*.log'))
        if len(logs)!=args.ranks or any('ERROR SUMMARY: 0 errors' not in p.read_text() for p in logs):
            raise AssertionError('missing clean per-rank main memory result')
    reference=args.reference_results/'old_gpu';report['phases']={}
    for phase in ('post_source','post_update'):
        maximum=np.zeros(5)
        paths=sorted((reference/'outdat').glob('rhs.'+phase+'.*.rank00000000.bin'))
        if len(paths)!=12:raise AssertionError('reference requires four NP=1 steps')
        for p in paths:
            if phase=='post_update':
                whole=qbody(p)
            else:
                with p.open('rb') as f:
                    header=np.fromfile(f,np.int32,4);whole=np.fromfile(f,dtype).reshape(tuple(header[:3]+1)+(5,),order='F')
            for rank,layout in layouts(current).items():
                path=current/'outdat'/p.name.replace('rank00000000',f'rank{rank:08d}')
                if phase=='post_update':
                    values=qbody(path)
                else:
                    with path.open('rb') as f:
                        header=np.fromfile(f,np.int32,4);values=np.fromfile(f,dtype).reshape(tuple(header[:3]+1)+(5,),order='F')
                subset=whole[tuple(slice(o,o+n+1) for o,n in zip(layout['origin'],layout['upper']))+(slice(None),)]
                maximum=np.maximum(maximum,compare_data(values,subset))
                if phase=='post_update':
                    internal=values[...,4]-.5*np.sum(values[...,1:4]**2,axis=-1)/values[...,0]
                    if np.min(values[...,0])<=0 or np.min(internal)<=0:raise AssertionError('nonphysical stage state')
        report['phases'][phase]=maximum.tolist() if args.precision=='fp32' else float(max(maximum))
    report['clock_error']=max(compare(logged_steps(current),logged_steps(reference)))
    scalars=np.loadtxt(current/'flowstate.dat',skiprows=1,ndmin=2)
    reference_scalars=np.loadtxt(reference/'flowstate.dat',skiprows=1,ndmin=2)
    report['scalar_error']=max(compare_data(scalars,reference_scalars))
    quiet=run('observation_off',False,writing=args.checkpoint)
    for rank in layouts(current):
        a=current/f'outdat/rhs.cpp_final.step00000004.rk01.rank{rank:08d}.bin'
        b=quiet/f'outdat/terminal.cpp_final.step00000004.rk01.rank{rank:08d}.bin'
        if a.read_bytes()!=b.read_bytes():raise AssertionError('phase observations changed final q')
    if not np.array_equal(logged_steps(current),logged_steps(quiet)):raise AssertionError('observations changed clock')
    report['observation_isolation']=True
    if args.checkpoint:
        off=run('checkpoint_off')
        for rank in layouts(current):
            name=f'outdat/rhs.cpp_final.step00000004.rk01.rank{rank:08d}.bin'
            if (current/name).read_bytes()!=(off/name).read_bytes():raise AssertionError('checkpoint changed q')
        if not np.array_equal(logged_steps(current),logged_steps(off)):raise AssertionError('checkpoint changed clock')
        prefix=run('prefix',steps=2,writing=True)
        tail=run('tail',restore=prefix/'outdat/new/checkpoints/step000000000002',writing=True)
        final=Path('outdat/new/checkpoints/step000000000004')
        compare_hdf(current/final/'state.h5',tail/final/'state.h5',exact=True)
        if (current/final/'control.bin').read_bytes()!=(tail/final/'control.bin').read_bytes():
            raise AssertionError('restored control differs')
        resumed=np.loadtxt(tail/'flowstate.dat',skiprows=1,ndmin=2)
        if not np.array_equal(resumed,scalars[np.isin(scalars[:,0],resumed[:,0])]):
            raise AssertionError('restored scalar words differ')
        report.update(formal_output_tested=True,exact_restart_tested=True,
            checkpoint_isolation=True,exact_state_control_scalars=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
