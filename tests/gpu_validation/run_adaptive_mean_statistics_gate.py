"""Actual complete-step flatplate statistics, adaptive clocks and exact continuation."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import h5py
import numpy as np

from run_adaptive_tgv_gate import compare_state,STEP

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/input'))
from convert_legacy_input import convert_case,namelist


def runtime_packet(step,lifecycle,air5=False,statistics_enabled=True,handover=False):
    initial_dt=1e-10 if air5 else 1e-6
    statistics='enabled=t,sample_clock="steps",sample_steps=2,write_clock="steps",write_steps=4'
    if air5:
        statistics='enabled=t,sample_clock="time",sample_steps=0,sample_time=2.5e-10,write_clock="time",write_time=4e-10'
    if lifecycle:
        if step==3:
            statistics=f'enabled=t,sample_clock="time",sample_time={3*initial_dt:.17g}'
        elif step in (4,5):
            statistics='enabled=f'
        elif step>=6:
            period=(3 if step<9 else 2)*initial_dt
            statistics=f'enabled=t,sample_clock="time",sample_time={period:.17g},write_clock="time",write_time={4*initial_dt:.17g}'
    if not statistics_enabled:
        statistics='enabled=f'
    if handover and step>=4:
        statistics='enabled=f'
    return (f'&run_control max_steps=10,reload_steps={1 if lifecycle or handover else 100} /\n'
        f'&timestep mode="auto",initial_dt={initial_dt:.17g} /\n&logging list_steps=0 /\n'
        f'&statistics {statistics} /\n')


def run(args,backend,ranks,label,restore=None,topology=None,statistics_enabled=True):
    case=args.output/label
    if args.analyze_only:
        return case,read_clocks(case,restore is not None)
    model_environment={}
    if args.case=='air5':
        from run_output_air5_restart_validation import prepare
        prepare(case,backend,10,mean_statistics=statistics_enabled)
        source=case/'datin/input.air5_c4'
        model_environment=dict(ASTR_AIR5_SOURCE_MODE='coupled',ASTR_AIR5_COMPENSATION='on',
            ASTR_AIR5_CONVECTION_LIMITER='symmetric_species',ASTR_AIR5_DIFFUSION_LIMITER='layered',
            ASTR_AIR5_TOP_MODE='prescribed',ASTR_AIR5_FILTER_MODE='conservative-flux')
    else:
        subprocess.run([sys.executable,str(ROOT/'tests/gpu_validation/prepare_s1_flatplate_case.py'),
            '--dst-case',str(case),'--use-gpu','t' if backend=='gpu' else 'f',
            '--im','16','--jm','16','--km','16','--conschm','543e','--lfilter','t',
            '--warp-x','0.08' if args.case=='curve' else '0',
            '--warp-y','0.04' if args.case=='curve' else '0',
            '--maxstep','9','--deltat','1e-6'],check=True,capture_output=True,text=True)
        source=case/'datin/input.flatplate'
    (case/'outdat/new').mkdir(parents=True)
    (case/'datin/input.case').write_text(namelist(convert_case(source.read_text(),model_environment=model_environment)))
    (case/'datin/input.run').write_text(runtime_packet(0,args.lifecycle,args.case=='air5',statistics_enabled,
        getattr(args,'handover',False)))
    (case/'datin/input.output').write_text(
        '&output\n directory="outdat/new",exact_restart=t,host_budget_bytes=-1,device_budget_bytes=-1,\n'
        f' restore_directory="{restore or ""}"\n/\n'
        '&checkpoint\n enabled=t,interval_steps=5,keep=2\n/\n'
        '&volume\n enabled=f\n/\n&slices\n enabled=f\n/\n')
    env=os.environ.copy()
    env.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_coll='^cuda,hcoll,ucc',
        OMPI_MCA_osc='pt2pt',OMPI_MCA_opal_cuda_support='0',ASTR_FORCE_MPI_TOPOLOGY=topology or f'{ranks},1,1',
        ASTR_GPU_SYNC_MODE='explicit',ASTR_INSITU_TIMING='1')
    if args.case=='air5':
        env.update(ASTR_AIR5_FILTER_VALIDATION='on',ASTR_AIR5_TOP_GPU_VALIDATION='on')
    if args.lifecycle:
        replay=case/'run_replay'
        replay.mkdir()
        for step in range(11):
            (replay/f'controller{step:012d}').write_text(runtime_packet(step,
                not getattr(args,'handover',False),args.case=='air5',statistics_enabled,
                getattr(args,'handover',False)))
        env.update(LD_PRELOAD=str(args.replay_library),
            ASTR_CONTROLLER_TEST_FILE=str((case/'datin/input.run').resolve()),
            ASTR_CONTROLLER_TEST_REPLAY=str(replay.resolve()),ASTR_CONTROLLER_TEST_START_STEP='5' if restore else '0')
    with (case/'run.log').open('w') as log:
        completed=subprocess.run([str(args.mpiexec),'-n',str(ranks),str(getattr(args,backend)),
            'run','datin/input.case'],cwd=case,env=env,stdout=log,stderr=subprocess.STDOUT)
    if completed.returncode:
        raise AssertionError(f'{label}: exit={completed.returncode}; inspect {case/"run.log"}')
    return case,read_clocks(case,restore is not None)


def read_clocks(case,restored):
    clocks=[(int(m[1]),*[float(m[i]) for i in range(2,6)]) for m in STEP.finditer((case/'run.log').read_text())]
    expected=list(range(6 if restored else 1,11))
    if [row[0] for row in clocks]!=expected:
        raise AssertionError(f'{case}: missing complete steps')
    return clocks


def exact_file(a,b):
    with h5py.File(a) as first,h5py.File(b) as second:
        names=[]
        first.visititems(lambda name,obj:names.append(name) if isinstance(obj,h5py.Dataset) else None)
        other=[]
        second.visititems(lambda name,obj:other.append(name) if isinstance(obj,h5py.Dataset) else None)
        if names!=other:
            raise AssertionError('statistics dataset identities differ')
        for name in names:
            x,y=first[name][:],second[name][:]
            if x.dtype!=y.dtype or x.shape!=y.shape or not np.array_equal(x,y):
                raise AssertionError(f'exact statistics differ: {name}')


def compare_plane(cpu_case,gpu_case,final):
    with h5py.File(cpu_case/'datin/grid.flatplate.h5') as grid:
        xyz=np.stack([grid[name][:] for name in ('x','y','z')])
    ds=np.sqrt(np.sum(np.diff(xyz,axis=1)**2,axis=0))
    errors={}
    with h5py.File(cpu_case/final/'statistics.h5') as cpu,h5py.File(gpu_case/final/'statistics.h5') as gpu:
        cpu_moments=[];gpu_moments=[]
        for gpu_slot,cpu_slot in enumerate([1,2,3,4,6,5,7,8,9,10,11,12],start=2):
            values=cpu[f'q{cpu_slot:04d}'][:]
            expected=np.sum(0.5*(values[:-1]+values[1:])*ds,axis=0)
            observed=np.sum(gpu[f'q{gpu_slot:04d}'][:],axis=0)
            if 'inherited' in gpu:
                observed+=np.sum(gpu['inherited'][f'q{gpu_slot:04d}'][:],axis=0)
            error=float(np.max(abs(observed-expected)))/max(1,float(np.max(abs(expected))))
            if error>2e-10:
                raise AssertionError(f'CPU/GPU time and z-integrated moment {gpu_slot}: {error}')
            errors[f'gpu_q{gpu_slot:04d}']=error
            cpu_moments.append(expected);gpu_moments.append(observed)
        cpu_duration=np.asarray(cpu['metadata'][8],dtype=np.int64).view(np.float64).item()
        gpu_duration=np.asarray(gpu['metadata'][13],dtype=np.int64).view(np.float64).item()
        if cpu_duration!=gpu_duration or cpu_duration<=0:
            raise AssertionError('CPU/GPU integration durations differ or lack coverage')
        measure=np.sum(ds,axis=0)*cpu_duration
        def derived(moments):
            density=moments[0]
            u,v,w=[moments[i]/density for i in (1,2,3)]
            return [density/measure,u,v,w,moments[4]/density,moments[5]/measure,
                *[moments[i]/density-product for i,product in zip(range(6,12),(u*u,v*v,w*w,u*v,u*w,v*w))]]
        for name,a,b in zip(('rho','u','v','w','T','p','uu','vv','ww','uv','uw','vw'),
                            derived(cpu_moments),derived(gpu_moments)):
            error=float(np.max(abs(a-b)))/max(1,float(np.max(abs(a))))
            if error>2e-10:
                raise AssertionError(f'CPU/GPU normalized mean/stress {name}: {error}')
            errors['normalized_'+name]=error
    return errors


def compare_air5_statistics(cpu_case,gpu_case,final):
    errors={}
    with h5py.File(cpu_case/final/'statistics.h5') as cpu,h5py.File(gpu_case/final/'statistics.h5') as gpu:
        duration=np.asarray(cpu['metadata'][8],dtype=np.int64).view(np.float64).item()
        other_duration=np.asarray(gpu['metadata'][8],dtype=np.int64).view(np.float64).item()
        if duration!=other_duration or duration<=0:
            raise AssertionError('AIR5 CPU/GPU coverage differs')
        for slot in range(1,45):
            a=cpu[f'q{slot:04d}'][:]/duration
            b=gpu[f'q{slot:04d}'][:]/duration
            error=float(np.max(abs(a-b)))/max(1,float(np.max(abs(a))))
            if error>2e-10:
                raise AssertionError(f'AIR5 normalized integral {slot}: {error}')
            errors[f'time_mean_q{slot:04d}']=error
        for slot in range(45,59):
            a=cpu[f'q{slot:04d}'][:];b=gpu[f'q{slot:04d}'][:]
            error=float(np.max(abs(a-b)))/max(1,float(np.max(abs(a))))
            if error>2e-10:
                raise AssertionError(f'AIR5 endpoint {slot}: {error}')
            errors[f'endpoint_q{slot:04d}']=error
    return errors


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cpu','gpu','mpiexec','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--ranks',nargs='+',type=int,default=[1,2])
    parser.add_argument('--analyze-only',action='store_true')
    parser.add_argument('--lifecycle',action='store_true')
    parser.add_argument('--handover',action='store_true',help='disable statistics exactly at the old sample/write event')
    parser.add_argument('--replay-library',type=Path)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--cross-topology',action='store_true')
    parser.add_argument('--case',choices=('flatplate','curve','air5'),default='flatplate')
    args=parser.parse_args()
    if args.handover:
        args.lifecycle=True
    if args.lifecycle and args.replay_library is None:
        parser.error('--lifecycle requires --replay-library')
    for name in ('cpu','gpu','mpiexec','output'):
        setattr(args,name,getattr(args,name).resolve())
    if args.lifecycle:
        args.replay_library=args.replay_library.resolve(strict=True)
    if not args.analyze_only:
        args.output.mkdir(parents=True,exist_ok=False)
    final=Path('outdat/new/checkpoints/step000000000010')
    report={}
    if args.handover:
        if args.case!='flatplate':
            parser.error('--handover uses the qualified nonreacting flatplate fixture')
        references={}
        for backend in ('cpu','gpu'):
            source,clocks=run(args,backend,2,backend)
            path=source/('outdat/meanflow.h5' if backend=='cpu' else 'outdat/statistics.step000000000004.h5')
            if not path.is_file():
                raise AssertionError(f'{backend}: old statistics sample/write at step 4 was lost before disable')
            with h5py.File(path) as data:
                if backend=='cpu':
                    count=int(data['nsamples'][0]);step=int(data['nstep'][0])
                    duration=float(data['mean_time_weight'][0])
                else:
                    metadata=data['metadata'][:]
                    count=int(metadata[7]);step=int(data['identity'][8])
                    duration=np.asarray(metadata[13],dtype=np.int64).view(np.float64).item()
                    next_dt=np.asarray(data['identity'][11],dtype=np.int64).view(np.float64).item()
                    if next_dt!=clocks[3][4]:
                        raise AssertionError('deferred statistics write lost the selected dt_next')
                if count!=3 or step!=4 or duration!=clocks[3][2]:
                    raise AssertionError(f'{backend}: old interval must contain endpoints 0,2,4 and end at t4')
            restored,tail=run(args,backend,2,backend+'_resumed',source/'outdat/new/checkpoints/step000000000005')
            if tail!=clocks[5:]:
                raise AssertionError('handover exact restart clocks differ')
            compare_state(source/final/'state.h5',restored/final/'state.h5',True)
            for name in ('control.bin','archives.bin','insitu_control.bin'):
                if (source/final/name).read_bytes()!=(restored/final/name).read_bytes():
                    raise AssertionError(f'handover exact continuation differs: {name}')
            references[backend]=source/final/'state.h5'
            report[backend]=dict(samples=count,write_step=step,time_weight=duration,exact_restart=True)
        field={}
        compare_state(references['cpu'],references['gpu'],False,field)
        report.update(cpu_gpu=field,status='passed')
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
        return
    for ranks in args.ranks:
        references={}
        for backend in ('cpu','gpu'):
            label=f'{backend}_np{ranks}'
            continuous,clocks=run(args,backend,ranks,label)
            restored,tail=run(args,backend,ranks,label+'_resumed',
                continuous/'outdat/new/checkpoints/step000000000005')
            if tail!=clocks[5:]:
                raise AssertionError('exact complete-step times differ')
            compare_state(continuous/final/'state.h5',restored/final/'state.h5',True)
            exact_file(continuous/final/'statistics.h5',restored/final/'statistics.h5')
            for name in ('control.bin','archives.bin','insitu_control.bin'):
                if (continuous/final/name).read_bytes()!=(restored/final/name).read_bytes():
                    raise AssertionError(f'exact schedule/controller differs: {name}')
            with h5py.File(continuous/final/'statistics.h5') as data:
                metadata=data['metadata'][:]
                compact=backend=='gpu' and args.case!='air5'
                count=int(metadata[7 if compact else 1])
                if count!=(3 if args.lifecycle else (7 if args.case=='air5' else 6)):
                    raise AssertionError(f'statistics sample sequence differs; count={count}')
                duration=np.asarray(metadata[13 if compact else 8],dtype=np.int64).view(np.float64).item()
                expected_duration=clocks[-1][2]-(clocks[5][2] if args.lifecycle else 0)
                if duration!=expected_duration:
                    raise AssertionError('window duration crossed a disabled interval or missed a sample')
            if backend=='gpu':
                for step in ((9,10) if args.lifecycle else ((4,7,9) if args.case=='air5' else (4,8))):
                    if not (continuous/f'outdat/statistics.step{step:012d}.h5').is_file():
                        raise AssertionError('independent device statistics write missing')
            else:
                with h5py.File(continuous/'outdat/meanflow.h5') as data:
                    if int(data['nstep'][0])!=(10 if args.lifecycle else (9 if args.case=='air5' else 8)):
                        raise AssertionError('independent CPU statistics write clock differs')
                if args.lifecycle:
                    with h5py.File(continuous/'bakup/meanflow.h5') as data:
                        if int(data['nstep'][0])!=9:
                            raise AssertionError('CPU time write must also cross its first target at step 9')
            if args.reference:
                compare_state(args.reference/label/final/'state.h5',continuous/final/'state.h5',True)
            if args.case=='air5' and not args.lifecycle:
                disabled,_=run(args,backend,ranks,label+'_statistics_off',statistics_enabled=False)
                compare_state(disabled/final/'state.h5',continuous/final/'state.h5',True)
                from run_output_air5_restart_validation import state_sanity
                state_sanity(continuous/final/'state.h5')
            timing=re.findall(r'ASTR_MEAN_TIMING[^\n]+',(continuous/'run.log').read_text())
            if len(timing)!=1:
                raise AssertionError('sample/write timing record missing')
            report[label]=dict(exact_state_and_statistics=True,samples=count,time_weight=duration,clocks=clocks,timing=timing[0])
            references[backend]=continuous
        field={}
        compare_state(references['cpu']/final/'state.h5',references['gpu']/final/'state.h5',False,field)
        compare_statistics=compare_air5_statistics if args.case=='air5' else compare_plane
        statistics_key='statistics' if args.case=='air5' else 'plane_moments'
        report[f'np{ranks}_cpu_gpu']={'field':field,
            statistics_key:compare_statistics(references['cpu'],references['gpu'],final)}
        if args.cross_topology and ranks==2:
            migrated,_=run(args,'gpu',2,'gpu_np2_zresumed',
                references['gpu']/'outdat/new/checkpoints/step000000000005',topology='1,1,2')
            field={}
            compare_state(references['gpu']/final/'state.h5',migrated/final/'state.h5',False,field)
            report['gpu_x_to_z_repartition']={'field':field,
                statistics_key:compare_statistics(references['cpu'],migrated,final)}
    report['status']='passed'
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
