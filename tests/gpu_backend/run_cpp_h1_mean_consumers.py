"""Existing H1/random-wall fixture with the original main statistics lifecycle."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from run_cpp_tgv_gate import checkpoint,compare,logged_steps,namelist
from run_cpp_restart_gate import exact_state
from run_cpp_insitu_gate import compare_hdf
from run_cpp_legacy_statistics_gate import compact_statistics
from prepare_legacy_statistics_reference import sampling_isolation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('binary','original','fixture','output','mpiexec'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--ranks',type=int,choices=(1,2),default=2)
    parser.add_argument('--axis',choices=('x','y','z'),default='x')
    parser.add_argument('--memcheck',action='store_true')
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    binaries={}
    for key,path in (('cpp',args.binary),('original',args.original)):
        binary=root/(key+'_astr');shutil.copy2(path,binary);binary.chmod(0o555);binaries[key]=binary
    topology=[1,1,1];topology['xyz'.index(args.axis)]=args.ranks
    env={k:v for k,v in os.environ.items() if not k.startswith(('ASTR_','OMPI_MCA_','UCX_'))}
    env.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',OMP_NUM_THREADS='1',
        ASTR_GPU_SYNC_MODE='explicit',ASTR_GPU_HALO_TRANSPORT='pinned',ASTR_GPU_COMPACT_TRANSPORT='pinned',
        ASTR_FORCE_MPI_TOPOLOGY=','.join(map(str,topology)))
    final=Path('outdat/new/checkpoints/step000000000004')
    def run(name,implementation,gpu=True,steps=4,restore=None,sampling=True,instrument=False):
        case=root/name
        shutil.copytree(args.fixture/('gpu' if gpu else 'cpu')/'datin',case/'datin',copy_function=shutil.copyfile)
        (case/'datin').chmod(0o755);(case/'outdat/new').mkdir(parents=True)
        (case/'datin/input.run').write_text(namelist(dict(run_control=dict(max_steps=steps),
            timestep=dict(mode='fixed',fixed_dt=1e-6),logging=dict(list_steps=1),repair=dict(crashfix=False),
            statistics=dict(enabled=sampling,sample_clock='steps',sample_steps=1,write_clock='steps',write_steps=2))))
        (case/'datin/input.output').write_text(namelist(dict(output=dict(directory='outdat/new',
            restore_directory=str(restore or ''),exact_restart=True,checkpoint_statistics=True,
            host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0),
            checkpoint=dict(enabled=True,mode='steps',interval_steps=2,keep=2,initial_frame=False),
            volume=dict(enabled=False),slices=dict(enabled=False))))
        command=[str(args.mpiexec.resolve()),'--bind-to','none','-n',str(args.ranks)]
        if instrument:
            command+=['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer',
                '--tool','memcheck','--leak-check','full','--error-exitcode','99',
                '--log-file',str(case/'memcheck.%p.log')]
        command +=[str(binaries[implementation]),'run']
        with (case/'run.log').open('x') as log:
            subprocess.run(command,cwd=case,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,check=True)
        text=(case/'run.log').read_text()
        if 'sbli-grcbc-h1 high-order boundary chain' not in text:raise AssertionError('existing H1 chain inactive')
        if implementation=='cpp' and text.count('backend=cuda')!=args.ranks:raise AssertionError('C++ backend inactive')
        q=checkpoint(case/f'outdat/new/checkpoints/step{steps:012d}/state.h5')
        internal=q[...,4]-.5*np.sum(q[...,1:4]**2,axis=-1)/q[...,0]
        if not np.isfinite(q).all() or not (q[...,0]>0).all() or not (internal>0).all():
            raise AssertionError('nonfinite/nonpositive H1 field')
        return case
    cpu=run('cpu', 'original',gpu=False)
    old=run('old_gpu','original');current=run('cpp','cpp',instrument=args.memcheck)
    report=dict(topology=topology,binary_sha256={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in binaries.items()},
        physical_fields={},main_statistics={})
    for name,reference in (('cpu',cpu),('old_gpu',old)):
        report['physical_fields'][name]=compare(checkpoint(current/final/'state.h5'),checkpoint(reference/final/'state.h5'))
        if logged_steps(current).tolist()!=logged_steps(reference).tolist():raise AssertionError('H1 statistics clock')
    for step in (2,4):
        path=Path(f'outdat/new/checkpoints/step{step:012d}/statistics.h5')
        actual_meta,actual=compact_statistics(current/path);old_meta,expected=compact_statistics(old/path)
        if actual_meta.tobytes()!=old_meta.tobytes():raise AssertionError('original/new main sampling metadata')
        report['main_statistics'][str(step)]=compare(actual,expected)
    for label,implementation,full,gpu in (('cpu','original',cpu,False),('original','original',old,True),('cpp','cpp',current,True)):
        quiet=run(label+'_sampling_off',implementation,gpu=gpu,sampling=False)
        isolation=sampling_isolation(full/final/'state.h5',quiet/final/'state.h5')
        if logged_steps(full).tolist()!=logged_steps(quiet).tolist():raise AssertionError('sampling changed clock')
        prefix=run(label+'_prefix',implementation,gpu=gpu,steps=2)
        tail=run(label+'_tail',implementation,gpu=gpu,restore=prefix/'outdat/new/checkpoints/step000000000002')
        exact_state(full/final/'state.h5',tail/final/'state.h5')
        compare_hdf(full/final/'statistics.h5',tail/final/'statistics.h5',exact=True)
        for name in ('control.bin','insitu_control.bin'):
            if (full/final/name).read_bytes()!=(tail/final/name).read_bytes():raise AssertionError('H1 statistics/RNG control '+name)
        expected=np.loadtxt(full/'flowstate.dat',skiprows=1,ndmin=2)
        resumed=np.loadtxt(tail/'flowstate.dat',skiprows=1,ndmin=2)
        if not np.array_equal(resumed,expected[np.isin(expected[:,0],resumed[:,0])]):raise AssertionError('H1 resumed scalars')
        report[label]=dict(sampling_isolation=isolation,exact_state_history_rng_controls_scalars=True)
    if args.memcheck:
        logs=list(current.glob('memcheck.*.log'))
        if len(logs)!=args.ranks or any('ERROR SUMMARY: 0 errors' not in p.read_text() or
            'LEAK SUMMARY: 0 bytes leaked' not in p.read_text() for p in logs):raise AssertionError('H1 means Memcheck')
    report['memcheck']=args.memcheck
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('H1 main statistics: old/new history, sampling isolation and exact RNG/state continuation checked')


if __name__=='__main__':main()
