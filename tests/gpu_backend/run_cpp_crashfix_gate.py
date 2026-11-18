"""Ordinary crashfix stages and their existing checkpoint history contract."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import h5py
import numpy as np

from run_cpp_tgv_gate import prepare_case, namelist, checkpoint, compare, logged_steps
from run_cpp_multi_tgv_gate import raw_field
from run_cpp_multi_tgv_gate import layouts
from run_cpp_restart_gate import exact_state
from prepare_upwind_components import prepare_flat,convert_case


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','reference','template','mpiexec','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--ranks',type=int,choices=(1,2),default=1)
    parser.add_argument('--axis',choices=tuple('xyz'),default='x')
    parser.add_argument('--form',choices=('central','keep','mp7ld'),default='central')
    parser.add_argument('--filter',choices=('none','explicit10','compact10'),default='none')
    parser.add_argument('--transport',choices=('pinned','device'),default='pinned')
    parser.add_argument('--sync',choices=('explicit','dependency'),default='explicit')
    parser.add_argument('--workspace',choices=('scalar','full'),default='scalar')
    parser.add_argument('--memcheck',action='store_true')
    parser.add_argument('--seeded-probes',action='store_true',help='executables inject one invalid initial device node before steploop')
    parser.add_argument('--reference-cpu',action='store_true',help='use the actual CPU path when the original GPU main does not admit this combination')
    parser.add_argument('--fixture',choices=('tgv','plate'),default='tgv')
    parser.add_argument('--disable-restore',action='store_true',help='disable repair explicitly while preserving a nonzero restored history')
    parser.add_argument('--precision',choices=('fp64','fp32'),default='fp64')
    parser.add_argument('--automatic',action='store_true')
    parser.add_argument('--mixed-candidate',choices=('viscous_flux','derivative'))
    parser.add_argument('--freeze-only',action='store_true',help='capture the original trajectory before changing the new consumer')
    parser.add_argument('--advance-only',action='store_true',help='test fields/history without requiring formal output or restart support')
    parser.add_argument('--frozen-reference',type=Path,help='explicit old-GPU raw-stage reference captured by --freeze-only --advance-only')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    def difference(a,b):
        return compare(np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64),
            limit=None if args.precision=='fp32' else 2e-10)
    shutil.copyfile(__file__,out/'runner.py')
    binary=args.binary.resolve();reference=args.reference.resolve()
    shutil.copy2(binary,out/'tested_astr');shutil.copy2(reference,out/'reference_astr')
    topology=[1,1,1];topology['xyz'.index(args.axis)]=args.ranks
    env=os.environ|dict(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',
        ASTR_FORCE_MPI_TOPOLOGY=','.join(map(str,topology)),ASTR_GPU_HALO_TRANSPORT=args.transport,
        ASTR_GPU_SYNC_MODE=args.sync,ASTR_GPU_FILTER_WORKSPACE=args.workspace,
        ASTR_GPU_COMPACT_TRANSPORT='pinned',ASTR_GPU_COMPACT_SYNC=args.sync,
        ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',ASTR_VALIDATION_RHS_STEP='0',ASTR_VALIDATION_RHS_STEP_END='4')
    if args.transport=='device':env.update(OMPI_MCA_pml='ucx',OMPI_MCA_opal_cuda_support='1')
    if args.advance_only:env['ASTR_VALIDATION_CRASHFIX_HISTORY_PREFIX']='outdat/history'
    numerics=dict(conschm={'central':'643e','keep':'643c','mp7ld':'743e'}[args.form],difschm='643e',
        convection_form='keep' if args.form=='keep' else 'standard',recon_schem=5,lchardecomp=False,
        filter_scheme='explicit10' if args.filter=='none' else args.filter,lfilter=args.filter!='none')
    precision_options=(dict(gpu_precision_mode='mixed_workspace',gpu_mixed_candidate=args.mixed_candidate)
        if args.mixed_candidate else None)

    def run(name,program,gpu,steps=4,restore=None,writing=True,repair=True,exact=True):
        case=out/name
        dt=1e-4
        if args.fixture=='plate':
            if args.form!='mp7ld' or args.filter!='none':raise AssertionError('this existing MP7-LD plate fixture uses no global filter')
            dt=1e-6
            prepare_flat(case,gpu)
            groups=convert_case((case/'datin/input.flatplate').read_text())
            groups['numerics'].update(numerics|dict(lchardecomp=True,difschm='643c'))
            groups['precision_options']=precision_options or dict(gpu_precision_mode='fp64')
            (case/'datin/input.case').write_text(namelist(groups))
            (case/'outdat/new').mkdir()
        else:
            prepare_case(case,args.template.resolve(),gpu,False,False,dt=dt,steps=steps,numerics=numerics,
                precision_options=precision_options)
        timestep=(dict(mode='auto',initial_dt=dt,max_dt=2*dt,cfl_target=.5,growth_factor=1.1)
            if args.automatic else dict(mode='fixed',fixed_dt=dt))
        (case/'datin/input.run').write_text(namelist(dict(run_control=dict(max_steps=steps,reload_steps=1),
            timestep=timestep,logging=dict(list_steps=1),
            repair=dict(crashfix=repair),statistics=dict(enabled=False))))
        (case/'datin/input.output').write_text(namelist(dict(output=dict(directory='outdat/new',
            restore_directory=str(restore or ''),exact_restart=exact,checkpoint_statistics=False,
            host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0),
            checkpoint=dict(enabled=writing and not args.advance_only,mode='steps',interval_steps=2,keep=2,initial_frame=False),
            volume=dict(enabled=False),slices=dict(enabled=False))))
        if args.frozen_reference and name=='continuous':
            for key in ['input.case','input.run']:
                assert (case/'datin'/key).read_bytes()==(args.frozen_reference/'old_gpu/datin'/key).read_bytes()
        command=[str(args.mpiexec.resolve()),'--bind-to','none','-n',str(args.ranks)]
        instrument=args.memcheck and (name=='tail' or (args.advance_only and name=='continuous'))
        if instrument:
            command+=['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer',
                '--tool','memcheck','--leak-check','full','--error-exitcode','99',
                '--log-file',str(case/'sanitizer.%p.log')]
        command += [str(program),'run']
        run_env=env
        if program==reference:
            run_env=env|dict(ASTR_GPU_SYNC_MODE='explicit',ASTR_GPU_COMPACT_SYNC='explicit',
                ASTR_GPU_HALO_TRANSPORT='pinned',OMPI_MCA_pml='ob1',OMPI_MCA_opal_cuda_support='0')
        with (case/'run.log').open('w') as log:
            subprocess.run(command,cwd=case,env=run_env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        if program==binary and (case/'run.log').read_text().count('backend=cuda')!=args.ranks:
            raise AssertionError('C++ main state was not used')
        if args.mixed_candidate:
            marker=('ASTR_GPU_ACTIVE_MIXED_WORKSPACE=' if program==reference else 'ASTR_GPU_CPP mixed_workspace=')+args.mixed_candidate
            if marker not in (case/'run.log').read_text():raise AssertionError('requested mixed workspace is inactive')
        if instrument:
            logs=list(case.glob('sanitizer.*.log'))
            if len(logs)!=args.ranks or any('ERROR SUMMARY: 0 errors' not in p.read_text() or
                'LEAK SUMMARY: 0 bytes leaked' not in p.read_text() for p in logs):raise AssertionError('crashfix continuation tools')
        return case

    def batch(case,step=4):return case/f'outdat/new/checkpoints/step{step:012d}'
    if args.seeded_probes and args.reference_cpu:parser.error('the seeded entry currently covers GPU startup repair only')
    if args.frozen_reference:
        meta=json.loads((args.frozen_reference/'reference.json').read_text())
        assert meta['reference_only'] and not meta['checkpoint_tested']
        assert meta['reference_sha256']==hashlib.sha256(reference.read_bytes()).hexdigest()
        assert meta['topology']==topology and meta['mixed_candidate']==args.mixed_candidate and meta['precision']==args.precision
        previous=args.frozen_reference.resolve()/'old_gpu'
    else:previous=run('cpu' if args.reference_cpu else 'old_gpu',reference,not args.reference_cpu)
    if args.freeze_only:
        if args.advance_only:
            from run_cpp_crashfix_plate_gate import history
            states=[];marks=0
            for rank in range(args.ranks):
                header,field=raw_field(previous/f'outdat/rhs.post_update.step00000003.rk03.rank{rank:08d}.bin',precision=args.precision)
                h=header[3];states.append(field[tuple(slice(h,h+n+1) for n in header[:3])+(slice(None),)].reshape(-1,5))
                marks+=int(np.count_nonzero(history(previous,rank)[1]))
            state=np.concatenate(states,axis=0)
        else:
            state=checkpoint(batch(previous)/'state.h5')
            with h5py.File(batch(previous)/'state.h5') as handle:
                marks=int(np.count_nonzero(handle['crashfix_history'][...]))
        wide=state.astype(np.float64)
        assert np.isfinite(wide).all() and (wide[...,0]>0).all()
        assert (wide[...,4]-.5*np.sum(wide[...,1:4]**2,axis=-1)/wide[...,0]>0).all()
        assert not args.seeded_probes or marks>0
        (out/'reference.json').write_text(json.dumps(dict(reference_only=True,
            reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
            precision=args.precision,mixed_candidate=args.mixed_candidate,topology=topology,
            checkpoint_tested=not args.advance_only,nonzero_history_words=marks,clocks=logged_steps(previous).tolist()),indent=2)+'\n')
        for path in previous.rglob('*'):path.chmod(0o555 if path.is_dir() else 0o444)
        previous.chmod(0o555)
        print('Original reference frozen; no C++ advancement performed')
        return
    current=run('continuous',binary,True)
    report=dict(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
        topology=topology,form=args.form,filter=args.filter,transport=args.transport,
        sync=args.sync,workspace=args.workspace,fixture=args.fixture,seeded_probes=args.seeded_probes,
        reference_backend='cpu' if args.reference_cpu else 'cuda_fortran')
    report.update(precision=args.precision,automatic=args.automatic,mixed_candidate=args.mixed_candidate,
        numeric_difference_limit=None if args.precision=='fp32' else 2e-10)
    if args.frozen_reference:report['frozen_raw_reference']=str(args.frozen_reference.resolve())
    if args.advance_only:
        from run_cpp_crashfix_plate_gate import history
        from run_cpp_mp_positivity_gate import read_rhs
        dtype=np.float32 if args.precision=='fp32' else np.float64
        phases={name:np.zeros(5) for name in ['pre_rhs','post_update','conv','full']}
        marks=0
        for rank in range(args.ranks):
            for step in range(4):
                for stage in range(1,4):
                    suffix=f'.step{step:08d}.rk{stage:02d}.rank{rank:08d}.bin'
                    for phase in phases:
                        values=[]
                        for case in [previous,current]:
                            if phase in ['pre_rhs','post_update']:
                                header,field=raw_field(case/'outdat'/('rhs.'+phase+suffix),precision=args.precision)
                                h=header[3];field=field[tuple(slice(h,h+n+1) for n in header[:3])+(slice(None),)]
                                q=field.astype(np.float64)
                                assert np.isfinite(q).all() and (q[...,0]>0).all()
                                assert (q[...,4]-.5*np.sum(q[...,1:4]**2,axis=-1)/q[...,0]>0).all()
                            else:field=read_rhs(case/'outdat'/('rhs.'+phase+suffix),dtype)
                            values.append(field)
                        phases[phase]=np.maximum(phases[phase],difference(values[1],values[0]))
            old_h,old=history(previous,rank);new_h,new=history(current,rank)
            assert np.array_equal(old_h,new_h) and old.tobytes()==new.tobytes()
            marks+=int(np.count_nonzero(new))
        assert not args.seeded_probes or marks>0
        assert np.array_equal(logged_steps(current),logged_steps(previous))
        scalar=np.loadtxt(current/'flowstate.dat',skiprows=1,ndmin=2)
        old_scalar=np.loadtxt(previous/'flowstate.dat',skiprows=1,ndmin=2)
        scalar_scaled=difference(scalar,old_scalar)
        scalar_absolute=np.max(abs(scalar-old_scalar),axis=0)
        if args.mixed_candidate:
            assert scalar_absolute.max()<=(1e-8 if args.mixed_candidate=='derivative' else 1e-10)
        for case in [previous,current]:
            assert not list((case/'outdat/new/checkpoints').glob('step*'))
            assert (case/'run.log').read_text().count('CRASHFIX_HISTORY_OBSERVER_Q_WORDS_PASS')==args.ranks
        report.update(checkpoint_tested=False,phases_scaled={k:v.tolist() for k,v in phases.items()},
            nonzero_history_words=marks,critical_history_equal=True,clock_equal=True,observer_q_words_equal=True,
            scalar_scaled=scalar_scaled,scalar_max_abs=scalar_absolute.tolist(),memcheck=args.memcheck)
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
        return
    state=checkpoint(batch(current)/'state.h5')
    if state.dtype.itemsize!=(4 if args.precision=='fp32' else 8):raise AssertionError('crashfix checkpoint numerical width')
    if not np.all(state[...,0]>0) or not np.all(state[...,4]-.5*np.sum(state[...,1:4]**2,axis=-1)/state[...,0]>0):
        raise AssertionError('physical density/internal energy must remain positive')
    if args.frozen_reference:
        terminal=np.zeros(5)
        for rank,layout in layouts(current).items():
            header,field=raw_field(previous/f'outdat/rhs.post_update.step00000003.rk03.rank{rank:08d}.bin',precision=args.precision)
            h=header[3]
            spans=tuple(slice(o,min(o+n+1,state.shape[a])) for a,(o,n) in enumerate(zip(layout['origin'],layout['upper'])))
            body=tuple(slice(h,h+s.stop-s.start) for s in spans)
            terminal=np.maximum(terminal,difference(state[spans],field[body]))
        report['terminal']=terminal.tolist()
    else:report['terminal']=difference(state,checkpoint(batch(previous)/'state.h5'))
    report['statistics']=difference(np.loadtxt(current/'flowstate.dat',skiprows=1,ndmin=2),
        np.loadtxt(previous/'flowstate.dat',skiprows=1,ndmin=2))
    if args.mixed_candidate:
        scalar_absolute=np.max(abs(np.loadtxt(current/'flowstate.dat',skiprows=1,ndmin=2)-
            np.loadtxt(previous/'flowstate.dat',skiprows=1,ndmin=2)),axis=0)
        assert scalar_absolute.max()<=(1e-8 if args.mixed_candidate=='derivative' else 1e-10)
        report['scalar_max_abs']=scalar_absolute.tolist()
    # Each same-topology rank is compared to its own original solver rank.
    for rank in range(args.ranks):
        maximum=0.
        paths=list((previous/'outdat').glob(f'rhs.pre_rhs.*.rank{rank:08d}.bin'))
        if len(paths)!=12:raise AssertionError('twelve same-phase prepared states required per rank')
        for path in paths:
            h,a=raw_field(current/'outdat'/path.name,precision=args.precision);_,b=raw_field(path,precision=args.precision)
            selection=tuple(slice(h[3],h[3]+n+1) for n in h[:3])+(slice(None),)
            maximum=max(maximum,max(difference(a[selection],b[selection])))
        report[f'prepared_rank{rank}']=maximum
    quiet=run('quiet',binary,True,writing=False)
    prefix=run('prefix',binary,True,steps=2)
    tail=run('tail',binary,True,restore=batch(prefix,2))
    report['exact_datasets']=exact_state(batch(current)/'state.h5',batch(tail)/'state.h5')
    for name in ('control.bin','archives.bin','insitu_control.bin'):
        if (batch(current)/name).read_bytes()!=(batch(tail)/name).read_bytes():raise AssertionError('restored '+name)
    for rank in range(args.ranks):
        name=f'outdat/rhs.cpp_final.step00000004.rk01.rank{rank:08d}.bin'
        words=raw_field(current/name,precision=args.precision)[1].tobytes()
        if words!=raw_field(quiet/name,precision=args.precision)[1].tobytes() or words!=raw_field(tail/name,precision=args.precision)[1].tobytes():
            raise AssertionError('checkpoint output/restore changed complete q words')
    a=np.loadtxt(current/'flowstate.dat',skiprows=1,ndmin=2)
    b=np.loadtxt(tail/'flowstate.dat',skiprows=1,ndmin=2)
    if not np.array_equal(a,np.loadtxt(quiet/'flowstate.dat',skiprows=1,ndmin=2)) or \
        not np.array_equal(b,a[np.isin(a[:,0],b[:,0])]):raise AssertionError('scalar isolation/continuation')
    if not np.array_equal(logged_steps(current),logged_steps(quiet)) or \
        not np.array_equal(logged_steps(tail,2,3),logged_steps(current)[-2:]):raise AssertionError('clock isolation/continuation')
    with h5py.File(batch(current)/'state.h5') as f:
        report['nonzero_history_words']=int(np.count_nonzero(f['crashfix_history'][...]))
    if args.seeded_probes and report['nonzero_history_words']==0:raise AssertionError('repair did not persist any history')
    if args.disable_restore:
        if not args.seeded_probes:raise AssertionError('disable test requires actual nonzero repair history')
        disabled=run('disabled_continuous',binary,True,steps=6,restore=batch(prefix,2),repair=False,exact=False)
        disabled_prefix=run('disabled_prefix',binary,True,restore=batch(prefix,2),repair=False,exact=False)
        # This transition deliberately produces lean checkpoints. A later read
        # remains nonexact, rather than claiming an exact restore from role11.
        disabled_tail=run('disabled_tail',binary,True,steps=6,restore=batch(disabled_prefix),repair=False,exact=False)
        disabled_difference=difference(checkpoint(batch(disabled_tail,6)/'state.h5'),
            checkpoint(batch(disabled,6)/'state.h5'))
        with h5py.File(batch(disabled,6)/'state.h5') as f:
            if f['crashfix_layout'][1]!=0 or not np.any(f['crashfix_history'][...]):
                raise AssertionError('disabling repair lost effective history or stored enabled mode')
            history=f['crashfix_history'][...]
        with h5py.File(batch(disabled_tail,6)/'state.h5') as f:
            if not np.array_equal(history,f['crashfix_history'][...]):raise AssertionError('disabled restored history changed')
        with h5py.File(batch(prefix,2)/'state.h5') as f:
            if not np.array_equal(history,f['crashfix_history'][...]):raise AssertionError('disabled history changed')
        a=np.loadtxt(disabled/'flowstate.dat',skiprows=1,ndmin=2)
        b=np.loadtxt(disabled_tail/'flowstate.dat',skiprows=1,ndmin=2)
        disabled_statistics=difference(b,a[np.isin(a[:,0],b[:,0])])
        if not np.array_equal(logged_steps(disabled_tail,2,5),logged_steps(disabled,4,3)[-2:]):
            raise AssertionError('disabled timestep continuity')
        report['disabled_history']=dict(nonzero_history_preserved=True,physical_difference=disabled_difference,
            statistics_difference=disabled_statistics,exact_clock=True,
            transition='explicit nonexact from enabled; then same-mode nonexact lean continuation')
    report.update(output_isolation=True,exact_control_clock_scalars_full_q=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
