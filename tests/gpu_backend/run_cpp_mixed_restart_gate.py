"""LP1: same-mode viscous-storage checkpoints through the ordinary C++ solver."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from prepare_mixed_viscous_reference import prepare_mixed_case
from run_cpp_tgv_gate import namelist, logged_steps
from run_cpp_multi_tgv_gate import raw_field
from run_cpp_restart_gate import exact_state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('binary','template','mpiexec','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--ranks',type=int,choices=(1,2),default=1)
    parser.add_argument('--axis',choices=list('xyz'),default='x')
    parser.add_argument('--transport',choices=('pinned','device'),default='pinned')
    parser.add_argument('--sync',choices=('explicit','dependency'),default='explicit')
    parser.add_argument('--workspace',choices=('scalar','full'),default='scalar')
    parser.add_argument('--fixture',choices=('tgv','plate'),default='tgv')
    parser.add_argument('--filter',choices=('none','explicit10'),default='none')
    parser.add_argument('--automatic',action='store_true')
    parser.add_argument('--memcheck',action='store_true')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    binary=args.binary.resolve();shutil.copyfile(__file__,out/'runner.py');shutil.copyfile(binary,out/'tested_astr')
    topology=[1,1,1];topology['xyz'.index(args.axis)]=args.ranks
    env=os.environ|dict(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',
        ASTR_FORCE_MPI_TOPOLOGY=','.join(map(str,topology)),ASTR_GPU_HALO_TRANSPORT=args.transport,
        ASTR_GPU_SYNC_MODE=args.sync,ASTR_GPU_FILTER_WORKSPACE=args.workspace,
        ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',ASTR_VALIDATION_RHS_STEP='4',ASTR_VALIDATION_RHS_STEP_END='4')
    if args.transport=='device':env.update(OMPI_MCA_pml='ucx',OMPI_MCA_opal_cuda_support='1')

    def run(name,mixed,steps=4,restore=None,writing=True,reject=False):
        case=out/name
        prepare_mixed_case(case,args.template.resolve(),True,mixed,args.automatic,args.filter,args.fixture,steps=steps)
        (case/'outdat/new').mkdir(exist_ok=True)
        (case/'datin/input.output').write_text(namelist(dict(output=dict(directory='outdat/new',
            exact_restart=True,restore_directory=str(restore or ''),checkpoint_statistics=False,
            host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0),
            checkpoint=dict(enabled=writing,mode='steps',interval_steps=2,keep=2,initial_frame=False),
            volume=dict(enabled=False),slices=dict(enabled=False))))
        command=[str(args.mpiexec.resolve()),'--bind-to','none','-n',str(args.ranks)]
        if args.memcheck and name=='mixed_tail':
            command+=['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer',
                '--tool','memcheck','--leak-check','full','--error-exitcode','99',
                '--log-file',str(case/'memcheck.%p.log')]
        with (case/'run.log').open('w') as log:
            completed=subprocess.run(command+[str(binary),'run'],cwd=case,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT)
        text=(case/'run.log').read_text()
        if reject:
            if completed.returncode==0 or 'numerical/controller contract mismatch' not in text:
                raise AssertionError('cross-mode exact restart was not rejected by the stored identity')
            return case
        if completed.returncode!=0:raise AssertionError(f'{name} failed: {case / "run.log"}')
        if text.count('backend=cuda')!=args.ranks:raise AssertionError('C++ backend was not used')
        if args.memcheck and name=='mixed_tail':
            logs=list(case.glob('memcheck.*.log'))
            if len(logs)!=args.ranks or any('ERROR SUMMARY: 0 errors' not in p.read_text() or
                'LEAK SUMMARY: 0 bytes leaked' not in p.read_text() for p in logs):
                raise AssertionError('mixed restored state memory safety')
        return case

    def checkpoint(case,step=4):return case/f'outdat/new/checkpoints/step{step:012d}'
    report=dict(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),topology=topology,
        fixture=args.fixture,filter=args.filter,automatic=args.automatic,transport=args.transport,
        sync=args.sync,workspace=args.workspace,cases={})
    prefixes={}
    for mixed in (False,True):
        mode='mixed' if mixed else 'fp64'
        continuous=run(mode+'_continuous',mixed)
        quiet=run(mode+'_quiet',mixed,writing=False)
        prefix=run(mode+'_prefix',mixed,steps=2);prefixes[mode]=checkpoint(prefix,2)
        tail=run(mode+'_tail',mixed,restore=checkpoint(prefix,2))
        a,b=checkpoint(continuous),checkpoint(tail)
        count=exact_state(a/'state.h5',b/'state.h5')
        if (a/'control.bin').read_bytes()!=(b/'control.bin').read_bytes():raise AssertionError('control history differs')
        with (a/'control.bin').open('rb') as stream:
            stream.read(8);contract=np.fromfile(stream,np.int64,8)
        expected=24|(1 if args.workspace=='full' else 0)|(4 if mixed else 0)
        if int(contract[6])!=expected:raise AssertionError('wrong mixed/filter identity in control.bin')
        for rank in range(args.ranks):
            name=f'outdat/rhs.cpp_final.step00000004.rk01.rank{rank:08d}.bin'
            first=raw_field(continuous/name)[1]
            if first.tobytes()!=raw_field(quiet/name)[1].tobytes() or first.tobytes()!=raw_field(tail/name)[1].tobytes():
                raise AssertionError('output or restored run changed full q words')
        full=np.loadtxt(continuous/'flowstate.dat',skiprows=1,ndmin=2)
        resumed=np.loadtxt(tail/'flowstate.dat',skiprows=1,ndmin=2)
        if not np.array_equal(resumed,full[np.isin(full[:,0],resumed[:,0])]):raise AssertionError('restored scalars differ')
        if not np.array_equal(full,np.loadtxt(quiet/'flowstate.dat',skiprows=1,ndmin=2)) or \
            not np.array_equal(logged_steps(continuous),logged_steps(quiet)):
            raise AssertionError('output changed clocks/scalars')
        if not np.array_equal(logged_steps(tail,2,first_step=3),logged_steps(continuous)[-2:]):
            raise AssertionError('restored time/dt sequence differs')
        report['cases'][mode]=dict(exact_state_datasets=count,control_flag=int(contract[6]),
            exact_control_scalar_clock_and_full_q=True,output_isolation=True)
    run('reject_fp64_from_mixed',False,restore=prefixes['mixed'],reject=True)
    run('reject_mixed_from_fp64',True,restore=prefixes['fp64'],reject=True)
    report['cross_mode_exact_rejected']=True
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
