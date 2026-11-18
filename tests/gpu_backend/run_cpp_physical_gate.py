"""Ordinary C++ physical-boundary stages, channel scalars, output and restart."""
import argparse
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
from named_fixture import namelist
from run_cpp_tgv_gate import compare,checkpoint,logged_steps,snapshot
from run_cpp_multi_tgv_gate import raw_field,layouts
from run_cpp_insitu_gate import compare_hdf


def stages(current,reference,rhs_limit=2e-10,precision='fp64'):
    local=layouts(current);errors={}
    dtype=np.float32 if precision=='fp32' else np.float64
    for name,primitive,halo in [('pre_rhs',False,True),('pre_rhs_primitives',True,True),('post_source',False,False)]:
        previous=sorted((reference/'outdat').glob('rhs.'+name+'.*.bin'))
        if len(previous)!=12:
            raise AssertionError(('reference phase missing',name,len(previous)))
        maximum=np.zeros(6 if primitive else 5) if precision=='fp32' else 0.
        for path in previous:
            if halo:
                header,whole=raw_field(path,primitive=primitive,precision=precision);h=header[3]
                whole=whole[tuple(slice(h,h+n+1) for n in header[:3])+(slice(None),)]
            else:
                with path.open('rb') as f:
                    header=np.fromfile(f,np.int32,4)
                    whole=np.fromfile(f,dtype).reshape(tuple(header[:3]+1)+(header[3],),order='F')
            for rank,layout in local.items():
                actual=current/'outdat'/path.name.replace('rank00000000',f'rank{rank:08d}')
                if halo:
                    header,values=raw_field(actual,primitive=primitive,precision=precision);h=header[3]
                    values=values[tuple(slice(h,h+n+1) for n in header[:3])+(slice(None),)]
                else:
                    with actual.open('rb') as f:
                        header=np.fromfile(f,np.int32,4)
                        values=np.fromfile(f,dtype).reshape(tuple(header[:3]+1)+(header[3],),order='F')
                expected=whole[tuple(slice(o,o+n+1) for o,n in zip(layout['origin'],layout['upper']))+(slice(None),)]
                if precision=='fp32':
                    maximum=np.maximum(maximum,compare(values.astype(np.float64),expected.astype(np.float64),limit=None))
                else:
                    maximum=max(maximum,max(compare(values,expected,limit=rhs_limit if name=='post_source' else 2e-10)))
        errors[name]=maximum.tolist() if precision=='fp32' else maximum
    return errors


def main():
    parser=argparse.ArgumentParser()
    for name in ('binary','reference','output','mpiexec'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--ranks',type=int,default=1)
    parser.add_argument('--precision',choices=['fp64','fp32'],default='fp64')
    parser.add_argument('--sync',choices=['explicit','dependency'],default='dependency')
    parser.add_argument('--axis',choices=['x','y','z'],default='x')
    parser.add_argument('--wall-axis',choices=['x','y','z'],default='y')
    parser.add_argument('--transport',choices=['pinned','device'],default='pinned')
    parser.add_argument('--workspace',choices=['scalar','full'],default='scalar')
    parser.add_argument('--compact-transport',choices=['pinned','device-aware'],default='pinned')
    parser.add_argument('--memcheck',action='store_true')
    opt=parser.parse_args();output=opt.output.resolve();output.mkdir(parents=True,exist_ok=False)
    reference=opt.reference.resolve();original=reference/('gpu_'+opt.wall_axis)
    binary=output/'tested_astr';shutil.copy2(opt.binary.resolve(),binary)
    shutil.copy2(__file__,output/'runner.py')
    dtype=np.float32 if opt.precision=='fp32' else np.float64
    def compare_data(a,b):
        return compare(a.astype(np.float64),b.astype(np.float64),limit=None if opt.precision=='fp32' else 2e-10)
    topology=[1,1,1];topology['xyz'.index(opt.axis)]=opt.ranks
    env=os.environ|dict(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',ASTR_GPU_HALO_TRANSPORT=opt.transport,
        ASTR_GPU_FILTER_WORKSPACE=opt.workspace,ASTR_GPU_COMPACT_TRANSPORT=opt.compact_transport,
        ASTR_GPU_SYNC_MODE=opt.sync,
        ASTR_FORCE_MPI_TOPOLOGY=','.join(map(str,topology)),
        ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',ASTR_VALIDATION_RHS_STEP='0',ASTR_VALIDATION_RHS_STEP_END='4')
    if opt.transport=='device' or opt.compact_transport=='device-aware':
        env.update(OMPI_MCA_pml='ucx',OMPI_MCA_opal_cuda_support='1')
    def run(name,steps=4,restore=None,writing=True):
        case=output/name;case.mkdir()
        shutil.copytree(original/'datin',case/'datin',copy_function=shutil.copyfile)
        (case/'datin').chmod(0o755)
        (case/'outdat/new').mkdir(parents=True)
        (case/'datin/input.run').write_text(namelist(dict(run_control=dict(max_steps=steps),
            timestep=dict(mode='fixed',fixed_dt=1e-5),logging=dict(list_steps=1),
            repair=dict(crashfix=False),statistics=dict(enabled=False))))
        (case/'datin/input.output').write_text(namelist(dict(
            output=dict(directory='outdat/new',restore_directory=str(restore or ''),exact_restart=True,
                checkpoint_statistics=False,host_budget_bytes=-1,device_budget_bytes=-1,
                buffer_bytes=67108864,device_reserve_bytes=0),
            checkpoint=dict(enabled=writing,mode='steps',interval_steps=2,keep=2,initial_frame=False),
            volume=dict(enabled=False),slices=dict(enabled=False))))
        command=[str(opt.mpiexec.resolve()),'--bind-to','none','-n',str(opt.ranks)]
        if opt.memcheck and name=='continuous':
            command+=['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer','--tool','memcheck',
                '--error-exitcode','99','--leak-check','full','--log-file','memcheck.%p.log']
        command += [str(binary),'run']
        with (case/'run.log').open('w') as log:
            subprocess.run(command,cwd=case,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        if 'backend=cuda' not in (case/'run.log').read_text():
            raise AssertionError('ordinary C++ device path was not selected')
        return case
    current=run('continuous');final='outdat/new/checkpoints/step000000000004'
    with h5py.File(current/final/'state.h5') as state:
        if state['q0001'].dtype.itemsize!=np.dtype(dtype).itemsize:raise AssertionError('actual physical main numerical kind')
    report=dict(topology=topology,wall_axis=opt.wall_axis,transport=opt.transport,workspace=opt.workspace,
        precision=opt.precision,sync=opt.sync,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        compact_transport=opt.compact_transport,references={})
    for backend in ('cpu','gpu'):
        previous=reference/(backend+'_'+opt.wall_axis)
        with h5py.File(previous/final/'state.h5') as state:
            if state['q0001'].dtype.itemsize!=np.dtype(dtype).itemsize:raise AssertionError('actual physical reference numerical kind')
        report['references'][backend]=dict(stages=stages(current,previous,precision=opt.precision),
            terminal_by_variable=compare_data(checkpoint(current/final/'state.h5'),checkpoint(previous/final/'state.h5')),
            scalars_by_variable=compare_data(np.loadtxt(current/'flowstate.dat',skiprows=1,ndmin=2),
                                np.loadtxt(previous/'flowstate.dat',skiprows=1,ndmin=2)))
        report['references'][backend]['terminal']=max(report['references'][backend]['terminal_by_variable'])
        report['references'][backend]['scalars']=max(report['references'][backend]['scalars_by_variable'])
        if not np.array_equal(logged_steps(current),logged_steps(previous)):
            raise AssertionError('physical reference timestep sequence differs')
    quiet=run('without-checkpoint',writing=False)
    for rank,layout in layouts(current).items():
        suffix=f'outdat/rhs.cpp_final.step00000004.rk01.rank{rank:08d}.bin'
        _,x=raw_field(current/suffix,precision=opt.precision)
        _,y=raw_field(quiet/suffix,precision=opt.precision)
        if x.tobytes()!=y.tobytes():
            raise AssertionError('checkpoint output changed complete q words')
    if not np.array_equal(logged_steps(current),logged_steps(quiet)):
        raise AssertionError('checkpoint output changed timestep control')
    prefix=run('prefix',2);tail=run('tail',restore=prefix/'outdat/new/checkpoints/step000000000002')
    compare_hdf(current/final/'state.h5',tail/final/'state.h5',exact=True)
    for name in ('control.bin','archives.bin','insitu_control.bin'):
        if (current/final/name).read_bytes()!=(tail/final/name).read_bytes():
            raise AssertionError('restored timestep/output/channel-driver control words changed: '+name)
    for rank in layouts(current):
        suffix=f'outdat/rhs.cpp_final.step00000004.rk01.rank{rank:08d}.bin'
        _,x=raw_field(current/suffix,precision=opt.precision)
        _,y=raw_field(tail/suffix,precision=opt.precision)
        if x.tobytes()!=y.tobytes():raise AssertionError('restored complete q/halo words changed')
    full=np.loadtxt(current/'flowstate.dat',skiprows=1,ndmin=2);resumed=np.loadtxt(tail/'flowstate.dat',skiprows=1,ndmin=2)
    if not np.array_equal(resumed,full[np.isin(full[:,0],resumed[:,0])]):
        raise AssertionError('restored channel scalar/driver values changed')
    with h5py.File(current/final/'state.h5') as state:
        if not all(np.isfinite(state[name][...]).all() for name in state if state[name].dtype.kind=='f'):
            raise AssertionError('nonfinite serialized authoritative/cache words')
    q=checkpoint(current/final/'state.h5').astype(np.float64)
    if not (q[...,0]>0).all() or not (q[...,4]-.5*np.sum(q[...,1:4]**2,axis=-1)/q[...,0]>0).all():
        raise AssertionError('physical main density/internal energy is not positive')
    if opt.memcheck:
        logs=list(current.glob('memcheck.*.log'))
        if len(logs)!=opt.ranks or any('ERROR SUMMARY: 0 errors' not in p.read_text() or
                'LEAK SUMMARY: 0 bytes leaked' not in p.read_text() for p in logs):
            raise AssertionError('physical main Memcheck errors/leaks')
    report.update(on_off_state_and_clock_bits=True,exact_restart=dict(state_bits=True,control_bits=True,scalars_and_driver_bits=True))
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
