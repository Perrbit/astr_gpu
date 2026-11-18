"""N11: preserve actual RK/RNG history while correcting the statistics phase."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from run_cpp_tgv_gate import compare,convert_case,logged_steps,namelist
from run_cpp_insitu_gate import compare_hdf


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('before','after','template','mpiexec','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--family',choices=('keep','me6'),default='keep')
    parser.add_argument('--ranks',type=int,choices=(1,2),default=1)
    parser.add_argument('--axis',choices=tuple('xyz'),default='x')
    parser.add_argument('--legacy-random',action='store_true')
    parser.add_argument('--restart',action='store_true')
    parser.add_argument('--filter',action='store_true')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    if args.restart and not args.legacy_random:parser.error('this checkpoint fixture uses the existing legacy-random profile registration')
    topology=[1,1,1];topology['xyz'.index(args.axis)]=args.ranks
    env={k:v for k,v in os.environ.items() if not k.startswith('ASTR_')}
    env.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',ASTR_GPU_SYNC_MODE='explicit',
        ASTR_GPU_HALO_TRANSPORT='pinned',ASTR_FORCE_MPI_TOPOLOGY=','.join(map(str,topology)),
        ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',ASTR_VALIDATION_RHS_STEP='0',ASTR_VALIDATION_RHS_STEP_END='4')
    cases={};hashes={}
    for name,path,gpu in (('before_gpu',args.before,True),('after_gpu',args.after,True),('cpu',args.after,False)):
        case=out/name;case.mkdir();cases[name]=case
        shutil.copytree(args.template.resolve()/'datin',case/'datin');(case/'outdat/new').mkdir(parents=True)
        binary=case/'tested_astr';shutil.copy2(path.resolve(),binary)
        hashes[name]=hashlib.sha256(binary.read_bytes()).hexdigest()
        groups=convert_case((case/'datin/input.flatplate').read_text())
        groups['case_options']['use_gpu']=gpu
        groups['numerics'].update(conschm='643e',difschm='643e',recon_schem=3,lchardecomp=False,lfilter=args.filter,
            convection_form='keep' if args.family=='keep' else 'standard',filter_scheme='explicit10',
            viscous_form='me6_opti' if args.family=='me6' else 'standard')
        groups['boundary']['bctype']=[11,22,41,52,1,1]
        groups['boundary_options']=dict(nscbc_farfield_mode='nonreflecting',profile_inflow_mode='complete_state')
        groups['precision_options']=dict(gpu_precision_mode='fp64')
        if args.legacy_random:
            groups['boundary_options']['wall_blowing_mode']='legacy_random'
            (case/'datin/wallbs.dat').write_text('# wall forcing\n# amplitude beta xa xb xc\n1.d-4 2.d0 2.d0 5.d0 8.d0\n# time/span modes\n0 2\n')
        (case/'datin/input.case').write_text(namelist(groups))
        (case/'datin/input.output').write_text(namelist(dict(output=dict(directory='outdat/new',
            exact_restart=True,checkpoint_statistics=False,host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0),
            checkpoint=dict(enabled=args.legacy_random,mode='steps',interval_steps=2,keep=2,initial_frame=False),
            volume=dict(enabled=False),slices=dict(enabled=False))))
        command=[str(args.mpiexec.resolve()),'--bind-to','none','-n',str(args.ranks),str(binary),'run']
        with (case/'run.log').open('w') as log:
            subprocess.run(command,cwd=case,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
    counts={}
    for phase in ('pre_rhs','post_source','post_update'):
        paths=sorted((cases['before_gpu']/'outdat').glob(f'rhs.{phase}.*.bin'))
        if len(paths)!=12*args.ranks:raise AssertionError(('missing RK evidence',phase,len(paths)))
        for path in paths:
            if path.read_bytes()!=(cases['after_gpu']/'outdat'/path.name).read_bytes():
                raise AssertionError(('N11 changed actual RK words',path.name))
        counts[phase]=len(paths)
    if not np.array_equal(logged_steps(cases['before_gpu']),logged_steps(cases['after_gpu'])):
        raise AssertionError('N11 changed time/dt sequence')
    cpu=np.loadtxt(cases['cpu']/'flowstate.dat',skiprows=1,ndmin=2)
    before=np.loadtxt(cases['before_gpu']/'flowstate.dat',skiprows=1,ndmin=2)
    after=np.loadtxt(cases['after_gpu']/'flowstate.dat',skiprows=1,ndmin=2)
    scalar_error=compare(after,cpu)
    if args.legacy_random:
        for step in (2,4):
            control=Path(f'outdat/new/checkpoints/step{step:012d}/control.bin')
            a=(cases['before_gpu']/control).read_bytes();b=(cases['after_gpu']/control).read_bytes()
            if a[:8]!=b'ASTROC09':raise AssertionError('missing active RNG checkpoint record')
            if a!=b:raise AssertionError('N11 changed RNG or checkpoint control history')
    if args.restart:
        final=Path('outdat/new/checkpoints/step000000000004')
        for name,steps,restore in (('prefix',2,None),('tail',4,out/'prefix/outdat/new/checkpoints/step000000000002')):
            case=out/name;case.mkdir()
            shutil.copytree(cases['after_gpu']/'datin',case/'datin');(case/'outdat/new').mkdir(parents=True)
            (case/'datin/input.run').write_text(namelist(dict(run_control=dict(max_steps=steps),
                timestep=dict(mode='fixed',fixed_dt=1e-6),logging=dict(list_steps=1),repair=dict(crashfix=False),
                statistics=dict(enabled=False))))
            (case/'datin/input.output').write_text(namelist(dict(output=dict(directory='outdat/new',
                exact_restart=True,restore_directory=str(restore or ''),checkpoint_statistics=False,
                host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0),
                checkpoint=dict(enabled=True,mode='steps',interval_steps=2,keep=2,initial_frame=False),
                volume=dict(enabled=False),slices=dict(enabled=False))))
            command=[str(args.mpiexec.resolve()),'--bind-to','none','-n',str(args.ranks),str(args.after.resolve()),'run']
            with (case/'run.log').open('w') as log:
                subprocess.run(command,cwd=case,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        compare_hdf(cases['after_gpu']/final/'state.h5',out/'tail'/final/'state.h5',exact=True)
        for name in ('control.bin','archives.bin','insitu_control.bin'):
            if (cases['after_gpu']/final/name).read_bytes()!=(out/'tail'/final/name).read_bytes():
                raise AssertionError(('N11 continuation control or RNG words',name))
        tail=np.loadtxt(out/'tail/flowstate.dat',skiprows=1,ndmin=2)
        if not np.array_equal(tail,after[np.isin(after[:,0],tail[:,0])]):raise AssertionError('N11 resumed scalar sequence')
        for rank in range(args.ranks):
            path=Path(f'outdat/rhs.post_update.step00000003.rk03.rank{rank:08d}.bin')
            if (cases['after_gpu']/path).read_bytes()!=(out/'tail'/path).read_bytes():
                raise AssertionError('N11 resumed full-q words')
    report=dict(family=args.family,topology=topology,legacy_random=args.legacy_random,filtering=args.filter,binary_sha256=hashes,
        actual_RK_files_bytes_equal=counts,clock_sequence_unchanged=True,
        scalar_scaled_before_report_only=(np.max(abs(before-cpu),axis=0)/np.maximum(1.,np.max(abs(cpu),axis=0))).tolist(),
        scalar_scaled_after=scalar_error,active_rng_control_history_bytes_equal=args.legacy_random,
        exact_continuation_state_controls_scalars_full_q=args.restart,
        scope='Old CUDA Fortran statistics phase; unchanged actual RK/RNG history, not C++ joint boundary qualification')
    shutil.copy2(__file__,out/'runner.py');(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
