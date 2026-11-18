"""Existing low-order nonreflecting top, with real solver viscosity and restart."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from run_cpp_tgv_gate import ROOT,convert_case,namelist,compare,logged_steps,snapshot
from run_cpp_physical_gate import stages
from run_cpp_multi_tgv_gate import layouts,raw_field
from run_cpp_restart_gate import exact_state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','reference','template','mpiexec','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--freeze-only',action='store_true');parser.add_argument('--reference-results',type=Path)
    parser.add_argument('--ranks',type=int,choices=[1,2],default=1);parser.add_argument('--axis',choices=list('xyz'),default='y')
    parser.add_argument('--transport',choices=['pinned','device'],default='pinned')
    parser.add_argument('--sync',choices=['explicit','dependency'],default='explicit')
    parser.add_argument('--checkpoint',action='store_true');parser.add_argument('--memcheck',action='store_true')
    parser.add_argument('--analyze-reference',action='store_true',help='analyze already completed reference runs without advancement')
    parser.add_argument('--convection-form',choices=('standard','keep'),default='standard')
    parser.add_argument('--viscous-form',choices=('standard','me6_opti'),default='standard')
    args=parser.parse_args();out=args.output.resolve()
    if not args.analyze_reference:out.mkdir(parents=True,exist_ok=False)
    if not args.freeze_only:
        shutil.copy2(args.binary.resolve(),out/'tested_astr')
        args.binary=out/'tested_astr'
    topology=[1,1,1];topology['xyz'.index(args.axis)]=args.ranks
    env=os.environ|dict(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',ASTR_GPU_HALO_TRANSPORT=args.transport,
        ASTR_GPU_SYNC_MODE=args.sync,ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',ASTR_VALIDATION_RHS_STEP='0',
        ASTR_VALIDATION_RHS_STEP_END='4',ASTR_GPU_MP_POSITIVITY='off')
    if args.transport=='device':env.update(OMPI_MCA_pml='ucx',OMPI_MCA_opal_cuda_support='1')
    def run(name,binary,gpu,ranks,steps=4,restore=None,writing=False):
        case=out/name;(case/'datin').mkdir(parents=True);(case/'outdat/new').mkdir(parents=True)
        groups=convert_case((args.template/'input.tgv').read_text())
        groups['case_options'].update(use_gpu=gpu)
        groups['mesh'].update(ia=16,ja=16,ka=16,lihomo=True,ljhomo=False,lkhomo=True,lreadgrid=False)
        groups['physics'].update(diffterm=True)
        groups['numerics'].update(conschm='643e',difschm='643e',recon_schem=3,lchardecomp=False,lfilter=False,
            convection_form=args.convection_form,viscous_form=args.viscous_form,filter_scheme='explicit10')
        groups['boundary'].update(bctype=[1,1,41,52,1,1],twall=[1.]*6)
        groups['boundary_options']=dict(nscbc_farfield_mode='nonreflecting')
        groups['precision_options']=dict(gpu_precision_mode='fp64')
        (case/'datin/input.case').write_text(namelist(groups))
        (case/'datin/input.run').write_text(namelist(dict(run_control=dict(max_steps=steps),
            timestep=dict(mode='fixed',fixed_dt=1e-5),logging=dict(list_steps=1),repair=dict(crashfix=False),statistics=dict(enabled=False))))
        (case/'datin/input.output').write_text(namelist(dict(output=dict(directory='outdat/new',
            restore_directory=str(restore or ''),exact_restart=True,checkpoint_statistics=False,
            host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0),
            checkpoint=dict(enabled=writing,mode='steps',interval_steps=2,keep=2,initial_frame=False),
            volume=dict(enabled=False),slices=dict(enabled=False))))
        dims=[1,1,1];dims['xyz'.index(args.axis)]=ranks
        command=[str(args.mpiexec.resolve()),'--bind-to','none','-n',str(ranks)]
        if args.memcheck and name=='cpp':
            command+=['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer','--tool','memcheck',
                '--leak-check','full','--error-exitcode','99','--log-file',str(case/'memcheck.%p.log')]
        with (case/'run.log').open('w') as log:
            subprocess.run(command+[str(binary.resolve()),'run'],cwd=case,
                env=env|dict(ASTR_FORCE_MPI_TOPOLOGY=','.join(map(str,dims))),stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,check=True)
        return case
    if args.freeze_only:
        if args.analyze_reference:
            cpu=out/'cpu';gpu=out/'old_gpu'
        else:
            cpu=run('cpu',args.reference/'bin/astr_cpu',False,1)
            gpu=run('old_gpu',args.reference/'bin/astr_cuda_fortran',True,1)
        phase_errors={}
        for phase,primitive in [('pre_rhs',False),('pre_rhs_primitives',True),('post_source',False)]:
            paths=sorted((cpu/'outdat').glob('rhs.'+phase+'.*.bin'))
            if len(paths)!=12:raise AssertionError('missing reference stages')
            maximum=0.
            for path in paths:
                arrays=[]
                for source in (path,gpu/'outdat'/path.name):
                    if phase!='post_source':
                        header,value=raw_field(source,primitive=primitive);h=header[3]
                        value=value[tuple(slice(h,h+n+1) for n in header[:3])+(slice(None),)]
                    else:
                        with source.open('rb') as f:
                            header=np.fromfile(f,np.int32,4)
                            value=np.fromfile(f,np.float64).reshape(tuple(header[:3]+1)+(5,),order='F')
                    arrays.append(value)
                maximum=max(maximum,max(compare(*arrays)))
            phase_errors[phase]=maximum
        terminal=Path('outdat/rhs.post_update.step00000003.rk03.rank00000000.bin')
        report=dict(stages=phase_errors,terminal=compare(snapshot(gpu/terminal),snapshot(cpu/terminal)),
            convection_form=args.convection_form,viscous_form=args.viscous_form)
        report['binaries']={name:hashlib.sha256((args.reference/'bin'/name).read_bytes()).hexdigest()
            for name in ('astr_cpu','astr_cuda_fortran')}
        cpu_stats=np.loadtxt(cpu/'flowstate.dat',skiprows=1);gpu_stats=np.loadtxt(gpu/'flowstate.dat',skiprows=1)
        if not np.isfinite(cpu_stats).all() or not np.isfinite(gpu_stats).all():
            report.update(status='rejected_nonfinite_statistics',
                cpu_nonfinite_indices=np.argwhere(~np.isfinite(cpu_stats)).tolist(),
                gpu_nonfinite_indices=np.argwhere(~np.isfinite(gpu_stats)).tolist())
            (out/'rejected_reference.json').write_text(json.dumps(report,indent=2)+'\n')
            raise AssertionError('nonfinite original statistics; not a usable complete reference')
        report['scalars']=compare(gpu_stats,cpu_stats)
        (out/'reference.json').write_text(json.dumps(report,indent=2)+'\n');shutil.copyfile(__file__,out/'preparation.py')
        for p in out.rglob('*'):p.chmod(0o555 if p.is_dir() else 0o444)
        out.chmod(0o555);print(json.dumps(report,indent=2));return
    if args.reference_results is None:parser.error('--reference-results is required')
    current=run('cpp',args.binary,True,args.ranks,writing=args.checkpoint)
    if 'backend=cuda' not in (current/'run.log').read_text():raise AssertionError('native C++ backend was not used')
    report=dict(topology=topology,transport=args.transport,sync=args.sync,references={},
        convection_form=args.convection_form,viscous_form=args.viscous_form,
        binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest())
    for backend in ('cpu','old_gpu'):
        previous=args.reference_results/backend;maximum=0.
        whole=snapshot(previous/'outdat/rhs.post_update.step00000003.rk03.rank00000000.bin')
        for rank,layout in layouts(current).items():
            value=snapshot(current/f'outdat/rhs.cpp_final.step00000004.rk01.rank{rank:08d}.bin')
            subset=whole[tuple(slice(o,o+n+1) for o,n in zip(layout['origin'],layout['upper']))+(slice(None),)]
            maximum=max(maximum,max(compare(value,subset)))
        report['references'][backend]=dict(stages=stages(current,previous),terminal=maximum,
            scalars=max(compare(np.loadtxt(current/'flowstate.dat',skiprows=1),np.loadtxt(previous/'flowstate.dat',skiprows=1))))
    if args.memcheck:
        logs=list(current.glob('memcheck.*.log'))
        if len(logs)!=args.ranks or any('ERROR SUMMARY: 0 errors' not in p.read_text() or
            'LEAK SUMMARY: 0 bytes leaked' not in p.read_text() for p in logs):
            raise AssertionError('missing clean box memory check')
    if args.checkpoint:
        off=run('without-checkpoint',args.binary,True,args.ranks)
        for rank in layouts(current):
            path=f'outdat/rhs.cpp_final.step00000004.rk01.rank{rank:08d}.bin'
            if (current/path).read_bytes()!=(off/path).read_bytes():raise AssertionError('checkpoint changed q')
        if not np.array_equal(logged_steps(current),logged_steps(off)):raise AssertionError('checkpoint changed clock')
        prefix=run('prefix',args.binary,True,args.ranks,steps=2,writing=True)
        tail=run('tail',args.binary,True,args.ranks,restore=prefix/'outdat/new/checkpoints/step000000000002',writing=True)
        final=Path('outdat/new/checkpoints/step000000000004')
        state_count=exact_state(current/final/'state.h5',tail/final/'state.h5')
        for name in ('control.bin','archives.bin','insitu_control.bin'):
            if (current/final/name).read_bytes()!=(tail/final/name).read_bytes():raise AssertionError('restored '+name)
        full=np.loadtxt(current/'flowstate.dat',skiprows=1,ndmin=2);resumed=np.loadtxt(tail/'flowstate.dat',skiprows=1,ndmin=2)
        if not np.array_equal(resumed,full[np.isin(full[:,0],resumed[:,0])]):raise AssertionError('restored scalar words')
        if not np.array_equal(logged_steps(tail,2,3),logged_steps(current)[-2:]):raise AssertionError('restored dt/time sequence')
        for rank in layouts(current):
            name=f'outdat/rhs.cpp_final.step00000004.rk01.rank{rank:08d}.bin'
            if raw_field(current/name)[1].tobytes()!=raw_field(tail/name)[1].tobytes():raise AssertionError('restored full q words')
        report.update(checkpoint_isolation=True,exact_state_control_scalars=True,exact_state_datasets=state_count,
            exact_full_q_and_clock=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
