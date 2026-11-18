"""Ordinary C++ plate stages/statistics against frozen original solvers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from prepare_upwind_components import prepare_flat
from run_cpp_tgv_gate import compare,checkpoint,logged_steps,namelist,snapshot,cfl_log
from run_cpp_physical_gate import stages
from run_cpp_insitu_gate import compare_hdf
from run_cpp_multi_tgv_gate import layouts
from named_fixture import convert_case
from run_cpp_native_tgv import field as native_field,stage_report as native_stage_report
from run_native_precision_tgv_gate import cached_state_report


def native_primitives(path):
    with path.open('rb') as stream:
        upper=np.fromfile(stream,np.int32,4)
        values=np.fromfile(stream,np.float32)
    im,jm,km,halo=upper
    shape=(im+2*halo+1,jm+2*halo+1,km+2*halo+1,6)
    if values.size!=np.prod(shape):raise AssertionError('native primitive snapshot width/layout')
    result=values.reshape(shape,order='F')[halo:halo+im+1,halo:halo+jm+1,halo:halo+km+1]
    if not np.isfinite(result).all() or not (result[...,[0,4,5]]>0).all():
        raise AssertionError('invalid native primitive state')
    return result


def main():
    parser=argparse.ArgumentParser()
    for name in ('binary','reference','mpiexec','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--freeze-only',action='store_true')
    parser.add_argument('--advance-only',action='store_true',help='no formal output or restart; retain phase diagnostics')
    parser.add_argument('--precision',choices=['fp64','fp32'],default='fp64')
    parser.add_argument('--native-reference',type=Path,help='explicit original native32 CPU/CUDA Fortran executable')
    parser.add_argument('--analyze-only',action='store_true',help='read completed runs without rewriting them')
    parser.add_argument('--automatic',action='store_true')
    parser.add_argument('--quiet-compare',action='store_true',help='compare full q against final-only observation')
    parser.add_argument('--reference-results',type=Path)
    parser.add_argument('--variant',choices=['mp7_51','mp7_51_filter','mp7ld_explicit',
        'mp7ld_compact','mp7ld_compact_filter','mp7ld_compact_nscbc'],default='mp7_51')
    parser.add_argument('--ranks',type=int,choices=[1,2],default=1)
    parser.add_argument('--axis',choices=['x','y','z'],default='x')
    parser.add_argument('--transport',choices=['pinned','device'],default='pinned')
    parser.add_argument('--workspace',choices=['scalar','full'],default='scalar')
    parser.add_argument('--sync',choices=['explicit','dependency'],default='explicit')
    parser.add_argument('--memcheck',action='store_true')
    args=parser.parse_args();root=args.output.resolve()
    if args.precision=='fp32' and args.freeze_only and args.native_reference is None:
        parser.error('native FP32 freezing requires --native-reference, not a renamed FP64 reference')
    if args.native_reference is not None and (args.precision!='fp32' or not args.freeze_only):
        parser.error('--native-reference selects original native32 reference freezing only')
    if not args.analyze_only:
        root.mkdir(parents=True,exist_ok=False)
        shutil.copy2(__file__,root/'runner.py')
        if args.freeze_only and args.native_reference:
            shutil.copy2(args.native_reference,root/'original_astr')
        elif not args.freeze_only:
            shutil.copy2(args.binary,root/'tested_astr')
    compact=args.variant.startswith('mp7ld_compact');filtered=args.variant.endswith('_filter')
    def q_snapshot(path):
        return native_field(path)[2] if args.precision=='fp32' else snapshot(path)
    def compare_data(a,b):
        return compare(a.astype(np.float64),b.astype(np.float64),limit=None if args.precision=='fp32' else 2e-10)
    final=Path('outdat/new/checkpoints/step000000000004')
    env=os.environ|dict(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',
        ASTR_GPU_HALO_TRANSPORT=args.transport,ASTR_GPU_FILTER_WORKSPACE=args.workspace,ASTR_GPU_SYNC_MODE=args.sync,
        ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',ASTR_VALIDATION_RHS_STEP='0',ASTR_VALIDATION_RHS_STEP_END='4',
        ASTR_GPU_MP_POSITIVITY='off',ASTR_GPU_COMPACT_TRANSPORT='device-aware' if args.transport=='device' else 'pinned')
    env['ASTR_CFL_PROFILE_Y']='1'
    if args.transport=='device':env.update(OMPI_MCA_pml='ucx',OMPI_MCA_opal_cuda_support='1')
    def run(name,binary,gpu,ranks,steps=4,restore=None,writing=True,observe=True):
        case=root/name
        if args.analyze_only:return case
        prepare_flat(case,gpu)
        (case/'outdat/new').mkdir()
        groups=convert_case((case/'datin/input.flatplate').read_text())
        groups['numerics'].update(convection_form='standard',viscous_form='standard',
            difschm='643c' if compact else '643e',lfilter=filtered,filter_scheme='compact10' if compact else 'explicit10')
        if args.variant.startswith('mp7_51'):
            groups['numerics'].update(conschm='543e',recon_schem=3,lchardecomp=False)
            groups['boundary']['bctype'][3]=51
        if args.variant=='mp7ld_compact_nscbc':
            groups['boundary']['bctype']=[11,22,41,52,1,1]
            groups['boundary_options']=dict(nscbc_farfield_mode='incoming_only',profile_inflow_mode='complete_state')
        groups['precision_options']=dict(gpu_precision_mode='fp64')
        (case/'datin/input.case').write_text(namelist(groups))
        timestep=(dict(mode='auto',initial_dt=1e-6,max_dt=1e-5,cfl_target=.5,growth_factor=1.1)
            if args.automatic else dict(mode='fixed',fixed_dt=1e-6))
        (case/'datin/input.run').write_text(namelist(dict(run_control=dict(max_steps=steps),
            timestep=timestep,logging=dict(list_steps=1),repair=dict(crashfix=False),
            statistics=dict(enabled=False))))
        (case/'datin/input.output').write_text(namelist(dict(output=dict(directory='outdat/new',
            restore_directory=str(restore or ''),exact_restart=True,checkpoint_statistics=False,
            host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0),
            checkpoint=dict(enabled=writing,mode='steps',interval_steps=2,keep=2,initial_frame=False),
            volume=dict(enabled=False),slices=dict(enabled=False))))
        topology=[1,1,1];topology['xyz'.index(args.axis)]=ranks
        command=[str(args.mpiexec.resolve()),'--bind-to','none','-n',str(ranks)]
        if args.memcheck and name=='cpp':
            command+=['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer',
                '--tool','memcheck','--error-exitcode','99','--leak-check','full','--log-file',str(case/'sanitizer.%p.log')]
        command += [str(binary.resolve()),'run']
        execution=env if observe else env|dict(ASTR_VALIDATION_RHS_STEP=str(steps),ASTR_VALIDATION_RHS_STEP_END=str(steps))
        with (case/'run.log').open('w') as log:
            subprocess.run(command,cwd=case,env=execution|dict(ASTR_FORCE_MPI_TOPOLOGY=','.join(map(str,topology))),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        return case
    if args.freeze_only:
        cpu_binary=root/'original_astr' if args.native_reference else args.reference/'bin/astr_cpu'
        gpu_binary=root/'original_astr' if args.native_reference else args.reference/'bin/astr_cuda_fortran'
        cpu=run('cpu',cpu_binary,False,1,writing=not args.advance_only)
        gpu=run('old_gpu',gpu_binary,True,1,writing=not args.advance_only)
        last=Path('outdat/rhs.post_update.step00000003.rk03.rank00000000.bin')
        terminal=(compare_data(q_snapshot(gpu/last),q_snapshot(cpu/last)) if args.advance_only else
            compare_data(checkpoint(gpu/final/'state.h5'),checkpoint(cpu/final/'state.h5')))
        report=dict(variant=args.variant,precision=args.precision,automatic=args.automatic,
            advance_only=args.advance_only,terminal=terminal,
            scalars=compare_data(np.loadtxt(gpu/'flowstate.dat',skiprows=1),np.loadtxt(cpu/'flowstate.dat',skiprows=1)))
        report['binaries']={name:hashlib.sha256(path.read_bytes()).hexdigest()
            for name,path in [('astr_cpu',cpu_binary),('astr_cuda_fortran',gpu_binary)]}
        (root/'reference.json').write_text(json.dumps(report,indent=2)+'\n')
        for path in root.rglob('*'):path.chmod(0o555 if path.is_dir() else 0o444)
        root.chmod(0o555);print(json.dumps(report,indent=2));return
    if args.reference_results is None:parser.error('--reference-results is required after freezing')
    frozen=json.loads((args.reference_results/'reference.json').read_text())
    if (frozen['variant']!=args.variant or frozen.get('precision','fp64')!=args.precision or
            frozen.get('automatic',False)!=args.automatic):
        raise AssertionError('reference variant/numerical-kind/controller mismatch')
    current=run('cpp',root/'tested_astr',True,args.ranks,writing=not args.advance_only)
    if 'backend=cuda' not in (current/'run.log').read_text():raise AssertionError('native C++ main not selected')
    report=dict(variant=args.variant,precision=args.precision,automatic=args.automatic,ranks=args.ranks,
        axis=args.axis,transport=args.transport,sync=args.sync,references={})
    report['binary_sha256']=hashlib.sha256((root/'tested_astr').read_bytes()).hexdigest()
    clocks=logged_steps(current)
    if not np.isfinite(clocks).all() or not (clocks[:,3:]>0).all():
        raise AssertionError('invalid native plate complete-step control')
    report['clocks']=clocks.tolist()
    if args.automatic:
        rates=cfl_log(current)
        if not np.isfinite(rates).all() or np.max(rates)>float(np.nextafter(np.float32(.5),np.float32(np.inf))):
            raise AssertionError('native plate auto CFL target')
        report['cfl']=rates.tolist()
    if args.precision=='fp32':
        relation=np.zeros(5);thermal=0.
        phase_paths=sorted((current/'outdat').glob('rhs.pre_rhs.step*.bin'))
        if len(phase_paths)!=12*args.ranks:raise AssertionError('native plate RK primitive phases missing')
        for path in phase_paths:
            name=path.name.replace('rhs.pre_rhs.','rhs.pre_rhs_primitives.')
            q=native_field(path)[2];p=native_primitives(path.with_name(name))
            check=cached_state_report(q,p,'fp32',2.)
            relation=np.maximum(relation,[v['max_scaled'] for v in check['conservative']])
            thermal=max(thermal,check['temperature_max_abs'])
        report['same_phase_cached_relation']=dict(conservative_scaled_report_only=relation.tolist(),
            temperature_max_abs_report_only=thermal,density_words_equal=True)
    for backend in ('cpu','old_gpu'):
        previous=args.reference_results/backend
        if args.advance_only or args.precision=='fp32':
            expected=q_snapshot(previous/'outdat/rhs.post_update.step00000003.rk03.rank00000000.bin')
            terminal=[0.]*5
            for rank,layout in layouts(current).items():
                actual=q_snapshot(current/f'outdat/rhs.cpp_final.step00000004.rk01.rank{rank:08d}.bin')
                subset=expected[tuple(slice(o,o+n+1) for o,n in zip(layout['origin'],layout['upper']))+(slice(None),)]
                terminal=np.maximum(terminal,compare_data(actual,subset)).tolist()
        else:
            terminal=compare_data(checkpoint(current/final/'state.h5'),checkpoint(previous/final/'state.h5'))
        phase=(native_stage_report(current,previous,layouts(current)) if args.precision=='fp32' else stages(current,previous))
        if args.precision=='fp32':
            maximum=np.zeros(6)
            phase_paths=sorted((previous/'outdat').glob('rhs.pre_rhs_primitives.step*.bin'))
            if len(phase_paths)!=12:raise AssertionError('native reference RK primitive phases missing')
            for path in phase_paths:
                whole=native_primitives(path)
                for rank,layout in layouts(current).items():
                    actual=native_primitives(current/'outdat'/path.name.replace('rank00000000',f'rank{rank:08d}'))
                    subset=whole[tuple(slice(o,o+n+1) for o,n in zip(layout['origin'],layout['upper']))]
                    maximum=np.maximum(maximum,compare_data(actual,subset))
            phase['pre_rhs_primitives']=maximum.tolist()
        report['references'][backend]=dict(stages=phase,terminal=terminal,
            scalars=compare_data(np.loadtxt(current/'flowstate.dat',skiprows=1),np.loadtxt(previous/'flowstate.dat',skiprows=1)))
        if args.variant=='mp7ld_compact_nscbc':
            paths=sorted((previous/'outdat').glob('rhs.boundary.*.bin'))
            if len(paths)!=12:raise AssertionError('native joint boundary RK phases missing')
            maximum=np.zeros(5)
            dtype=np.float32 if args.precision=='fp32' else np.float64
            def boundary_rhs(path):
                with path.open('rb') as stream:
                    h=np.fromfile(stream,np.int32,4)
                    values=np.fromfile(stream,dtype).reshape(tuple(h[:3]+1)+(int(h[3]),),order='F')
                return values.astype(np.float64)
            for path in paths:
                whole=boundary_rhs(path)
                if backend=='cpu':whole=-whole
                for rank,layout in layouts(current).items():
                    actual=boundary_rhs(current/'outdat'/path.name.replace('rank00000000',f'rank{rank:08d}'))
                    region=tuple(slice(o,o+n+1) for o,n in zip(layout['origin'],layout['upper']))
                    maximum=np.maximum(maximum,compare_data(actual,whole[region]))
            report['references'][backend]['boundary_sign_normalized']=maximum.tolist()
    if args.quiet_compare:
        quiet=run('quiet',root/'tested_astr',True,args.ranks,writing=not args.advance_only,observe=False)
        for rank in layouts(current):
            name=f'outdat/rhs.cpp_final.step00000004.rk01.rank{rank:08d}.bin'
            if (current/name).read_bytes()!=(quiet/name).read_bytes():
                raise AssertionError('plate observer changed complete conservative halo words')
        if not np.array_equal(logged_steps(current),logged_steps(quiet)) or not np.array_equal(
                np.loadtxt(current/'flowstate.dat',skiprows=1),np.loadtxt(quiet/'flowstate.dat',skiprows=1)):
            raise AssertionError('plate observer changed clocks/scalars')
        report['full_q_and_clock_observer_isolation']=True
    if args.memcheck:
        logs=list(current.glob('sanitizer.*.log'))
        if len(logs)!=args.ranks or any('ERROR SUMMARY: 0 errors' not in path.read_text() for path in logs):
            raise AssertionError('missing clean per-rank memory evidence')
    if args.advance_only:
        report.update(formal_output_tested=False,exact_restart_tested=False)
        (root/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));return
    quiet=run('cpp-off',root/'tested_astr',True,args.ranks,writing=False)
    for rank in layouts(current):
        name=f'outdat/rhs.cpp_final.step00000004.rk01.rank{rank:08d}.bin'
        if args.precision=='fp32':
            if native_field(current/name)[1].tobytes()!=native_field(quiet/name)[1].tobytes():
                raise AssertionError('output changed native full q words')
        elif snapshot(current/name).tobytes()!=snapshot(quiet/name).tobytes():raise AssertionError('output changed q')
    if not np.array_equal(logged_steps(current),logged_steps(quiet)):raise AssertionError('output changed clock')
    prefix=run('cpp-prefix',root/'tested_astr',True,args.ranks,steps=2)
    tail=run('cpp-tail',root/'tested_astr',True,args.ranks,restore=prefix/'outdat/new/checkpoints/step000000000002')
    compare_hdf(current/final/'state.h5',tail/final/'state.h5',exact=True)
    if (current/final/'control.bin').read_bytes()!=(tail/final/'control.bin').read_bytes():raise AssertionError('restored controls differ')
    if (current/final/'insitu_control.bin').read_bytes()!=(tail/final/'insitu_control.bin').read_bytes():
        raise AssertionError('restored product controls differ')
    if not np.array_equal(logged_steps(tail,count=2,first_step=3),clocks[2:]):raise AssertionError('restored clocks differ')
    if args.precision=='fp32':
        for rank in layouts(current):
            name=f'outdat/rhs.cpp_final.step00000004.rk01.rank{rank:08d}.bin'
            if native_field(current/name)[1].tobytes()!=native_field(tail/name)[1].tobytes():
                raise AssertionError('restored native full q words differ')
    full=np.loadtxt(current/'flowstate.dat',skiprows=1,ndmin=2);resumed=np.loadtxt(tail/'flowstate.dat',skiprows=1,ndmin=2)
    if not np.array_equal(resumed,full[np.isin(full[:,0],resumed[:,0])]):raise AssertionError('restored scalars differ')
    report.update(output_state_and_clock_bits_equal=True,exact_state_and_control=True,exact_scalars=True)
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
