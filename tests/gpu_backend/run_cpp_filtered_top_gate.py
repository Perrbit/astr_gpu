"""Current C++ filtered bc52 stages against the repaired N7 real references."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from run_cpp_tgv_gate import convert_case,namelist,compare,logged_steps,snapshot
from run_cpp_physical_gate import stages
from run_cpp_multi_tgv_gate import layouts,raw_field
from run_cpp_restart_gate import exact_state
from run_filtered_top_reference import read_phase


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('binary','reference','mpiexec','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--ranks',type=int,choices=(1,2),default=1)
    parser.add_argument('--axis',choices=tuple('xyz'),default='x')
    parser.add_argument('--transport',choices=('pinned','device'),default='pinned')
    parser.add_argument('--sync',choices=('explicit','dependency'),default='explicit')
    parser.add_argument('--memcheck',action='store_true')
    parser.add_argument('--policy',choices=('compatibility','incoming_only','sbli_shock','nonreflecting'),default='compatibility')
    parser.add_argument('--joint-family',choices=('keep','me6','mp7ld_compact'))
    parser.add_argument('--compact',action='store_true')
    parser.add_argument('--filter',choices=('off','explicit10','compact10'),default='off')
    parser.add_argument('--workspace',choices=('scalar','full'),default='scalar')
    parser.add_argument('--checkpoint',action='store_true')
    parser.add_argument('--backend',choices=('cpp','cpu','cuda-fortran'),default='cpp')
    parser.add_argument('--source-prefix',type=Path,help='additional same-topology external step2 checkpoint, read nonexactly')
    args=parser.parse_args()
    if args.backend!='cpp' and args.ranks!=1:parser.error('original checkpoint registration is checked at NP1')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    binary=out/'tested_astr';shutil.copy2(args.binary.resolve(),binary);shutil.copy2(__file__,out/'runner.py')
    reference=args.reference.resolve();topology=[1,1,1];topology['xyz'.index(args.axis)]=args.ranks
    env={k:v for k,v in os.environ.items() if not k.startswith('ASTR_')}
    env.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',ASTR_GPU_HALO_TRANSPORT=args.transport,
        ASTR_GPU_SYNC_MODE=args.sync,ASTR_FORCE_MPI_TOPOLOGY=','.join(map(str,topology)),
        ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',ASTR_VALIDATION_RHS_STEP='0',ASTR_VALIDATION_RHS_STEP_END='4')
    env.update(ASTR_GPU_COMPACT_TRANSPORT='device-aware' if args.transport=='device' else 'pinned',
        ASTR_GPU_COMPACT_SYNC=args.sync,ASTR_GPU_FILTER_WORKSPACE=args.workspace)
    if args.transport=='device':env.update(OMPI_MCA_pml='ucx',OMPI_MCA_opal_cuda_support='1')
    if args.backend!='cpp':env.update(ASTR_GPU_HALO_TRANSPORT='pinned',ASTR_GPU_SYNC_MODE='explicit',
        OMPI_MCA_pml='ob1',OMPI_MCA_opal_cuda_support='0')

    def final_field(case,rank,step=4):
        label=f'cpp_final.step{step:08d}.rk01' if args.backend=='cpp' else f'post_update.step{step-1:08d}.rk03'
        return case/f'outdat/rhs.{label}.rank{rank:08d}.bin'

    def local_layouts(case):
        if args.backend=='cpp':return layouts(case)
        header,_=raw_field(final_field(case,0))
        return {0:dict(origin=(0,0,0),upper=header[:3],halo=(header[3],))}

    def run(name,observe,steps=4,restore=None,writing=False,exact=True,changed_target=False):
        case=out/name;case.mkdir()
        shutil.copytree(reference/'cpu_np1/datin',case/'datin',copy_function=shutil.copyfile)
        (case/'outdat/new').mkdir(parents=True)
        groups=convert_case((case/'datin/input.flatplate').read_text())
        groups['case_options']['use_gpu']=args.backend!='cpu'
        groups['numerics'].update(conschm='543e',difschm='643e',recon_schem=3,lchardecomp=False,lfilter=False,
            convection_form='standard',viscous_form='standard',filter_scheme='explicit10')
        if args.joint_family:
            if args.joint_family=='mp7ld_compact':
                groups['numerics'].update(conschm='743e',difschm='643c',recon_schem=5,lchardecomp=True)
            else:
                groups['numerics'].update(conschm='643e',convection_form='keep' if args.joint_family=='keep' else 'standard',
                    viscous_form='me6_opti' if args.joint_family=='me6' else 'standard')
        if args.compact:groups['numerics'].update(conschm='643c',difschm='643c')
        if args.filter!='off':groups['numerics'].update(lfilter=True,filter_scheme=args.filter)
        groups['boundary']['bctype']=[11,21,41,52,1,1]
        if args.joint_family:groups['boundary']['bctype'][1]=22
        groups['boundary_options']=dict(nscbc_farfield_mode=args.policy,profile_inflow_mode='complete_state')
        if args.policy=='sbli_shock':
            groups['boundary_options'].update(nscbc_farfield_shock_x=5.,nscbc_farfield_shock_angle_deg=36.)
        if changed_target:groups['boundary_options']['nscbc_farfield_rho']=1.01
        groups['precision_options']=dict(gpu_precision_mode='fp64')
        (case/'datin/input.case').write_text(namelist(groups))
        (case/'datin/input.run').write_text(namelist(dict(run_control=dict(max_steps=steps),
            timestep=dict(mode='fixed',fixed_dt=1e-6),logging=dict(list_steps=1),repair=dict(crashfix=False),
            statistics=dict(enabled=False))))
        (case/'datin/input.output').write_text(namelist(dict(output=dict(directory='outdat/new',
            restore_directory=str(restore or ''),exact_restart=exact,checkpoint_statistics=False,
            host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0),
            checkpoint=dict(enabled=writing,mode='steps',interval_steps=2,keep=2,initial_frame=False),
            volume=dict(enabled=False),slices=dict(enabled=False))))
        last=4 if args.backend=='cpp' else 3
        execution=env if observe else env|dict(ASTR_VALIDATION_RHS_STEP=str(last),ASTR_VALIDATION_RHS_STEP_END=str(last))
        command=[str(args.mpiexec.resolve()),'--bind-to','none','-n',str(args.ranks)]
        instrument=args.memcheck and name==('tail' if args.checkpoint else 'cpp')
        if instrument:
            command+=['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer','--tool','memcheck',
                '--leak-check','full','--error-exitcode','99','--log-file',str(case/'sanitizer.%p.log')]
        command += [str(binary),'run']
        with (case/'run.log').open('w') as log:
            completed=subprocess.run(command,cwd=case,env=execution,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        text=(case/'run.log').read_text()
        if changed_target:
            if completed.returncode==0 or 'boundary/driver configuration mismatch' not in text:
                raise AssertionError('changed active farfield target was not rejected')
            return case
        if completed.returncode:raise AssertionError(f'{name} failed: {case / "run.log"}')
        if args.backend=='cpp' and text.count('backend=cuda')!=args.ranks:raise AssertionError('C++ top main was not selected')
        return case

    current=run('cpp',True,writing=args.checkpoint)
    report=dict(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),topology=topology,
        transport=args.transport,sync=args.sync,policy=args.policy,backend=args.backend,
        scope='static-profile upper-y characteristic plate',joint_family=args.joint_family,
        compact=args.compact,
        filter=args.filter,workspace=args.workspace,
        checkpoint_tested=args.checkpoint,references={})
    for backend in ('cpu_np1','gpu_np1'):
        previous=reference/backend
        whole=snapshot(previous/'outdat/rhs.post_update.step00000003.rk03.rank00000000.bin')
        terminal=np.zeros(5)
        for rank,layout in local_layouts(current).items():
            value=snapshot(final_field(current,rank))
            expected=whole[tuple(slice(o,o+n+1) for o,n in zip(layout['origin'],layout['upper']))+(slice(None),)]
            terminal=np.maximum(terminal,compare(value,expected))
            if not np.all(value[...,0]>0) or not np.all(value[...,4]-.5*np.sum(value[...,1:4]**2,axis=-1)/value[...,0]>0):
                raise AssertionError('physical density/internal energy')
        errors={}
        if args.backend=='cpp':
            errors=stages(current,previous)
            if args.joint_family=='mp7ld_compact':
                paths=sorted((previous/'outdat').glob('rhs.boundary.*.bin'))
                if len(paths)!=12:raise AssertionError('missing staged boundary reference')
                maximum=0.
                for path in paths:
                    expected=read_phase(path,'boundary')
                    # CPU records boundary divergence before its later RHS sign change.
                    if backend=='cpu_np1':expected=-expected
                    for rank,layout in local_layouts(current).items():
                        actual=read_phase(current/'outdat'/path.name.replace('rank00000000',f'rank{rank:08d}'),'boundary')
                        body=tuple(slice(o,o+n+1) for o,n in zip(layout['origin'],layout['upper']))+(slice(None),)
                        maximum=max(maximum,max(compare(actual,expected[body])))
                errors['boundary_sign_normalized']=maximum
        else:
            for phase in ('pre_rhs','pre_rhs_primitives','post_source','post_update'):
                paths=list((previous/'outdat').glob(f'rhs.{phase}.*.bin'))
                if len(paths)!=12:raise AssertionError('missing original same-phase reference')
                errors[phase]=max(max(compare(read_phase(current/'outdat'/p.name,phase),read_phase(p,phase))) for p in paths)
        report['references'][backend]=dict(stages=errors,terminal=terminal.tolist(),
            statistics=compare(np.loadtxt(current/'flowstate.dat',skiprows=1),np.loadtxt(previous/'flowstate.dat',skiprows=1)))
        if not np.array_equal(logged_steps(current),logged_steps(previous)):raise AssertionError('top clock mismatch')
    quiet=run('quiet',False)
    for rank in local_layouts(current):
        if raw_field(final_field(current,rank))[1].tobytes()!=raw_field(final_field(quiet,rank))[1].tobytes():
            raise AssertionError('top output/observer changed full q')
    if not np.array_equal(logged_steps(current),logged_steps(quiet)) or \
        not np.array_equal(np.loadtxt(current/'flowstate.dat',skiprows=1),np.loadtxt(quiet/'flowstate.dat',skiprows=1)):
        raise AssertionError('top observer changed clock/scalars')
    report['observer_isolation']=True
    if args.checkpoint:
        end=Path('outdat/new/checkpoints/step000000000004')
        prefix=run('prefix',True,steps=2,writing=True)
        source=prefix/'outdat/new/checkpoints/step000000000002'
        tail=run('tail',True,restore=source,writing=True)
        count=exact_state(current/end/'state.h5',tail/end/'state.h5')
        for name in ('control.bin','archives.bin','insitu_control.bin'):
            if (current/end/name).read_bytes()!=(tail/end/name).read_bytes():raise AssertionError('restored '+name)
        for rank in local_layouts(current):
            if raw_field(final_field(current,rank))[1].tobytes()!=raw_field(final_field(tail,rank))[1].tobytes():
                raise AssertionError('restored complete q words')
        full=np.loadtxt(current/'flowstate.dat',skiprows=1,ndmin=2)
        resumed=np.loadtxt(tail/'flowstate.dat',skiprows=1,ndmin=2)
        if not np.array_equal(resumed,full[np.isin(full[:,0],resumed[:,0])]):raise AssertionError('restored scalar words')
        if not np.array_equal(logged_steps(tail,2,3),logged_steps(current)[-2:]):raise AssertionError('restored time/dt')
        nonexact=run('nonexact',True,restore=source,writing=True,exact=False)
        report['nonexact_terminal']=max(max(compare(snapshot(final_field(nonexact,r)),snapshot(final_field(current,r))))
            for r in local_layouts(current))
        if not np.array_equal(logged_steps(nonexact,2,3),logged_steps(current)[-2:]):raise AssertionError('nonexact fixed clock')
        report.update(exact_state_datasets=count,exact_controls_clock_scalars_and_full_q=True)
        if args.source_prefix:
            external=run('external_nonexact',True,restore=args.source_prefix.resolve(),writing=True,exact=False)
            report['external_nonexact_terminal']=max(max(compare(snapshot(final_field(external,r)),snapshot(final_field(current,r))))
                for r in local_layouts(current))
            if not np.array_equal(logged_steps(external,2,3),logged_steps(current)[-2:]):
                raise AssertionError('external nonexact fixed clock')
            report['external_source']=str(args.source_prefix.resolve())
        if args.policy in ('incoming_only','sbli_shock'):
            run('reject_target_change',False,restore=source,writing=True,changed_target=True)
            report['changed_target_rejected']=True
    if args.memcheck:
        tool_case=out/('tail' if args.checkpoint else 'cpp')
        tools=list(tool_case.glob('sanitizer.*.log'))
        if len(tools)!=args.ranks or any('ERROR SUMMARY: 0 errors' not in p.read_text() or
            'LEAK SUMMARY: 0 bytes leaked' not in p.read_text() for p in tools):raise AssertionError('top memory safety')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
