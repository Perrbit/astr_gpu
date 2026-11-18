"""Actual MP3 ASTR advancement against frozen originals, without checkpoint changes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import h5py
import numpy as np

from run_cpp_tgv_gate import compare,logged_steps,cfl_log,snapshot,namelist
from run_cpp_multi_tgv_gate import layouts,raw_field
from prepare_characteristic_flux_reference import sensor


FIELD_ATOL=2e-6
STAT_ATOL=2e-6
FIELD_NAMES=('density','velocity_x','velocity_y','velocity_z','pressure','temperature')


def difference(actual,reference):
    if actual.shape!=reference.shape or not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise AssertionError('characteristic field shape/finiteness')
    absolute=np.max(abs(actual-reference),axis=tuple(range(actual.ndim-1)))
    scale=np.maximum(1.,np.max(abs(reference),axis=tuple(range(reference.ndim-1))))
    return dict(max_abs=absolute.tolist(),scaled=(absolute/scale).tolist())


def admissible(q):
    internal=q[...,4]-.5*np.sum(q[...,1:4]**2,axis=-1)/q[...,0]
    if not np.isfinite(q).all() or not (q[...,0]>0).all() or not (internal>0).all():
        raise AssertionError('MP3 physical rho/internal energy')


def primitives(q):
    rho=q[...,0];velocity=q[...,1:4]/rho[...,None]
    pressure=.4*(q[...,4]-.5*rho*np.sum(velocity**2,axis=-1))
    return np.concatenate((rho[...,None],velocity,pressure[...,None],(12.6*pressure/rho)[...,None]),axis=-1)


def read_phase(path,phase):
    if phase in ('pre_rhs','post_update','pre_rhs_primitives'):
        header,value=raw_field(path,primitive=phase=='pre_rhs_primitives')
        halo=header[3]
        return value[tuple(slice(halo,halo+n+1) for n in header[:3])+(slice(None),)]
    with path.open('rb') as stream:
        header=np.fromfile(stream,np.int32,4)
        return np.fromfile(stream,np.float64).reshape((*header[:3]+1,header[3]),order='F')


def phases(case,previous,mixed):
    report={};partitions=layouts(case)
    for phase in ('pre_rhs','pre_rhs_primitives','post_update','conv','full','post_source'):
        maximum_abs=np.zeros(6 if phase=='pre_rhs_primitives' else 5);maximum_scaled=maximum_abs.copy()
        files=sorted((previous/'outdat').glob('rhs.'+phase+'.*.bin'))
        if len(files)!=6:raise AssertionError('six frozen stage snapshots required for '+phase)
        for path in files:
            expected=read_phase(path,phase)
            for rank,layout in partitions.items():
                actual=read_phase(case/'outdat'/path.name.replace('rank00000000',f'rank{rank:08d}'),phase)
                subset=expected[tuple(slice(o,o+n+1) for o,n in zip(layout['origin'],layout['upper']))+(slice(None),)]
                result=difference(actual,subset)
                maximum_abs=np.maximum(maximum_abs,result['max_abs']);maximum_scaled=np.maximum(maximum_scaled,result['scaled'])
                if phase in ('pre_rhs','post_update'):admissible(actual)
                if phase=='pre_rhs_primitives' and not all((actual[...,c]>0).all() for c in (0,4,5)):
                    raise AssertionError('MP3 cached rho/p/T')
        if not mixed and max(maximum_scaled)>2e-10:raise AssertionError(('FP64 stage',phase,maximum_scaled))
        if mixed and phase in ('pre_rhs','pre_rhs_primitives','post_update') and max(maximum_abs)>FIELD_ATOL:
            raise AssertionError(('MP3 stage field',phase,maximum_abs))
        report[phase]=dict(max_abs=maximum_abs.tolist(),scaled=maximum_scaled.tolist(),
            gate='MP3 field absolute2e-6' if mixed and phase in ('pre_rhs','pre_rhs_primitives','post_update') else
                ('report only; no frozen MP3 trajectory RHS gate' if mixed else 'FP64 scaled2e-10'))
    return report


def sensors(case,previous,mixed,xphysical=False):
    rows=[]
    for path in sorted((previous/'outdat').glob('rhs.sensor.*.bin')):
        original,mask=sensor(path)
        original=original.reshape((17,17,17),order='F');mask=mask.reshape((17,17,17),order='F')
        maximum=0.
        for rank,layout in layouts(case).items():
            raw,marks=sensor(case/'outdat'/path.name.replace('rank00000000',f'rank{rank:08d}'))
            shape=tuple(n+1 for n in layout['upper']);raw=raw.reshape(shape,order='F');marks=marks.reshape(shape,order='F')
            subset=tuple(slice(o,o+n+1) for o,n in zip(layout['origin'],layout['upper']))
            if not np.isfinite(raw).all() or not np.array_equal(marks,mask[subset]):
                raise AssertionError('MP3 raw sensor nonfinite or integer masks differ')
            maximum=max(maximum,float(np.max(abs(raw-original[subset]))))
        if not mixed and not xphysical and maximum>2e-10:raise AssertionError('FP64 sensor comparison')
        rows.append(dict(file=path.name,raw_max_abs=maximum,mask_mismatches=0,
            raw_gate='report trajectory raw; exact masks and independent same-input operator gate' if xphysical else
                ('report mixed trajectory raw; exact masks' if mixed else 'FP64 absolute2e-10')))
    if len(rows)!=6:raise AssertionError('six sensor phases required')
    return rows


def archives(case,reference):
    with h5py.File(reference/'datin/grid.h5') as geometry:
        coordinates=np.stack([geometry[name][...] for name in 'xyz'],axis=-1)
    rows={};partitions=layouts(case)
    status=json.loads((case/'outdat/new/products.status.json').read_text())
    if len(status['products'])!=3 or any(item['state']!='active' or item['last_output_step']!=2 for item in status['products']):
        raise AssertionError('three active completed-step numerical slice products required')
    for step in (1,2):
        q=np.empty((17,17,17,5))
        for rank,layout in partitions.items():
            value=snapshot(case/f'outdat/rhs.post_update.step{step-1:08d}.rk03.rank{rank:08d}.bin')
            q[tuple(slice(o,o+n+1) for o,n in zip(layout['origin'],layout['upper']))+(slice(None),)]=value
        expected=primitives(q).transpose(2,1,0,3)
        base=case/'outdat/new/fields';frame=base/'segment00000000'/f'step{step:012d}'
        if not (frame/'COMPLETE').is_file():raise AssertionError('missing complete MP3 archive frame')
        maximum=0.
        with h5py.File(frame/'data.h5') as data,h5py.File(base/'resources/data.h5') as grid:
            identity=data['metadata'][...]
            if identity[1]!=step or identity[2:5].view(np.float64)[0]!=step*1e-5:
                raise AssertionError('MP3 archive complete-step phase')
            if not np.array_equal(grid['coordinates'][...],coordinates):raise AssertionError('MP3 volume geometry')
            for component,name in enumerate(FIELD_NAMES):
                if data[name].dtype!=np.dtype('float64'):raise AssertionError('MP3 authority archives must remain FP64')
                maximum=max(maximum,max(compare(data[name][...][...,None],expected[...,component,None])))
            vector=np.stack([data['velocity_'+axis][...] for axis in 'xyz'],axis=-1)
            if vector.tobytes()!=data['velocity'][...].tobytes():raise AssertionError('MP3 archived vector/component bits')
        rows[f'fields/step{step}']=dict(own_q_reconstructed_primitive_scaled=maximum,complete_step=step,time=step*1e-5)
        for axis,index in (('i',8),('j',0),('k',16)):
            segments=list((case/'outdat/new/slice_products').glob(f'plane-{axis}{index}/*/segment*'))
            if len(segments)!=1:raise AssertionError('one active numerical slice segment required')
            segment=segments[0]
            if sorted(path.name for path in segment.glob('step*.h5'))!=[f'step{n:012d}.h5' for n in (0,1,2)]:
                raise AssertionError('MP3 catalog initial/completed slice inventory')
            selected=[slice(None)]*3;selected[2-'ijk'.index(axis)]=index;selected=tuple(selected);maximum=0.
            with h5py.File(segment/f'step{step:012d}.h5') as data,h5py.File(segment/'geometry.h5') as grid:
                if data['step'][()]!=step or data['clock'][0]!=step*1e-5:raise AssertionError('MP3 numerical slice phase')
                if not np.array_equal(grid['coordinates'][...],coordinates[selected]):raise AssertionError('MP3 slice geometry/index')
                for component,name in enumerate(FIELD_NAMES):
                    if data[name].dtype!=np.dtype('float64'):raise AssertionError('MP3 numerical slices must remain FP64')
                    maximum=max(maximum,max(compare(data[name][...][...,None],expected[selected][...,component,None])))
                if 'velocity' in data or 'coordinates' in data:raise AssertionError('catalog slice repeated vector or geometry')
            rows[f'{axis}{index}/step{step}']=dict(own_q_reconstructed_primitive_scaled=maximum,complete_step=step,time=step*1e-5)
    rows['initial_catalog_scope']='inventory only; completed steps1/2 compared to authority, no initial-phase field claim'
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','reference','mpiexec','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--ranks',type=int,choices=(1,2),default=1)
    parser.add_argument('--axis',choices=tuple('xyz'),default='x')
    parser.add_argument('--transport',choices=('pinned','device'),default='pinned')
    parser.add_argument('--sync',choices=('explicit','dependency'),default='explicit')
    parser.add_argument('--sanitizer',choices=('memcheck','initcheck'))
    parser.add_argument('--archives',action='store_true')
    parser.add_argument('--xphysical-sensor-trajectory',action='store_true',
        help='approved bc50 trajectory raw report; keep exact masks and the separate same-input sensor gate')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    binary=args.binary.resolve();reference=args.reference.resolve()
    boundary='xphysical' if args.xphysical_sensor_trajectory else 'periodic'
    if args.xphysical_sensor_trajectory and json.loads((reference/'reference.json').read_text())['boundary']!='xphysical':
        raise AssertionError('approved trajectory distinction requires the explicit bc50 xphysical reference')
    shutil.copyfile(__file__,out/'runner.py');shutil.copyfile(binary,out/'tested_astr')
    topology=[1,1,1];topology['xyz'.index(args.axis)]=args.ranks
    env=os.environ|dict(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',ASTR_GPU_HALO_TRANSPORT=args.transport,
        ASTR_FORCE_MPI_TOPOLOGY=','.join(map(str,topology)),ASTR_GPU_SYNC_MODE=args.sync,
        ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',ASTR_VALIDATION_RHS_STEP='0',ASTR_VALIDATION_RHS_STEP_END='2')
    if args.transport=='device':env.update(OMPI_MCA_pml='ucx',OMPI_MCA_opal_cuda_support='1')
    report=dict(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),topology=topology,
        transport=args.transport,sync=args.sync,field_atol=FIELD_ATOL,statistics_atol=STAT_ATOL,boundary=boundary,cases={})
    def run(name,mixed,observed=True):
        case=out/name;case.mkdir()
        previous=reference/('old_gpu_mixed' if mixed else 'old_gpu_fp64')
        shutil.copytree(previous/'datin',case/'datin',copy_function=shutil.copyfile);(case/'datin').chmod(0o755)
        (case/'outdat/new').mkdir(parents=True)
        if args.archives:
            (case/'datin/input.output').write_text(namelist(dict(output=dict(directory='outdat/new',exact_restart=True,
                host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0),checkpoint=dict(enabled=False),
                volume=dict(enabled=observed,mode='steps',interval_steps=1,initial_frame=False,final_frame=False),
                slices=dict(enabled=False))))
            definitions=[]
            if observed:
                for axis,index in (('i',8),('j',0),('k',16)):
                    definitions.append(dict(id=f'plane-{axis}{index}',kind='slice',enable_request=1,source='instantaneous',
                        fields=list(FIELD_NAMES),sampling=dict(mode='grid_index',axis=axis,index=index,
                            index_ranges={other:[0,16] for other in 'ijk' if other!=axis}),
                        output=dict(channel='data',format='hdf5-xdmf'),schedule=dict(mode='steps',interval=1)))
            (case/'input.products.jsonl').write_text(''.join(json.dumps(item)+'\n' for item in
                [dict(catalog_schema='astr-insitu-products-v1'),*definitions]))
        command=[str(args.mpiexec.resolve()),'--bind-to','none','-n',str(args.ranks)]
        if args.sanitizer and name=='mixed':
            command+=['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer','--tool',args.sanitizer,
                '--error-exitcode','99','--log-file',str(case/'sanitizer.%p.log')]
            if args.sanitizer=='memcheck':command+=['--leak-check','full']
        with (case/'run.log').open('w') as log:
            subprocess.run(command+[str(binary),'run'],cwd=case,
                env=env|dict(ASTR_VALIDATION_RHS_STEP='0' if observed else '2'),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        text=(case/'run.log').read_text()
        if text.count('backend=cuda')!=args.ranks or ('ASTR_GPU_CPP mixed_workspace=characteristic_flux' in text)!=mixed:
            raise AssertionError('actual C++ MP3 selection')
        if list((case/'outdat/new/checkpoints').glob('step*')):raise AssertionError('MP3 gate must not write checkpoints')
        return case
    fields={};statistics={};cases={}
    for mode in ('fp64','mixed'):
        mixed=mode=='mixed';case=run(mode,mixed);cases[mode]=case
        previous=reference/('old_gpu_mixed' if mixed else 'old_gpu_fp64')
        expected=snapshot(previous/'outdat/rhs.post_update.step00000001.rk03.rank00000000.bin')
        actual=np.empty_like(expected)
        for rank,layout in layouts(case).items():
            values=snapshot(case/f'outdat/rhs.cpp_final.step00000002.rk01.rank{rank:08d}.bin');admissible(values)
            actual[tuple(slice(o,o+n+1) for o,n in zip(layout['origin'],layout['upper']))+(slice(None),)]=values
        state=difference(actual,expected);primitive=difference(primitives(actual),primitives(expected))
        scalars=np.loadtxt(case/'flowstate.dat',skiprows=1,ndmin=2)
        old_scalars=np.loadtxt(previous/'flowstate.dat',skiprows=1,ndmin=2);stat=difference(scalars,old_scalars)
        if not np.array_equal(scalars[:,:2],old_scalars[:,:2]):raise AssertionError('MP3 statistic sample identity')
        if mixed:
            if max(state['max_abs'])>FIELD_ATOL or max(primitive['max_abs'])>FIELD_ATOL or max(stat['max_abs'])>STAT_ATOL:
                raise AssertionError('MP3 existing field/statistics gates')
        else:
            compare(actual,expected);compare(primitives(actual),primitives(expected));compare(scalars,old_scalars)
            cpu=snapshot(reference/'cpu/outdat/rhs.post_update.step00000001.rk03.rank00000000.bin')
            report['cpu_fp64_terminal_scaled']=compare(actual,cpu)
        if not np.array_equal(logged_steps(case,2),logged_steps(previous,2)):raise AssertionError('MP3 complete-step clocks')
        cfl=cfl_log(case,2)
        if not np.isfinite(cfl).all():raise AssertionError('MP3 CFL finiteness')
        cfl_result=difference(cfl,cfl_log(previous,2))
        if not mixed:compare(cfl,cfl_log(previous,2))
        report['cases'][mode]=dict(state=state,primitive=primitive,statistics=stat,cfl=cfl_result,
            phases=phases(case,previous,mixed),sensors=sensors(case,previous,mixed,boundary=='xphysical'))
        if args.archives:report['cases'][mode]['archives']=archives(case,previous)
        fields[mode],statistics[mode]=actual,scalars
    initial='outdat/rhs.sensor.step00000000.rk01.rank'
    for rank in layouts(cases['mixed']):
        first,_=sensor(cases['fp64']/(initial+f'{rank:08d}.bin'))
        second,_=sensor(cases['mixed']/(initial+f'{rank:08d}.bin'))
        if first.tobytes()!=second.tobytes():raise AssertionError('same initial C++ state changed raw sensor bits')
    candidate_field=difference(fields['mixed'],fields['fp64'])
    candidate_primitive=difference(primitives(fields['mixed']),primitives(fields['fp64']))
    candidate_statistics=difference(statistics['mixed'],statistics['fp64'])
    if max(candidate_field['max_abs'])>FIELD_ATOL or max(candidate_primitive['max_abs'])>FIELD_ATOL or \
        max(candidate_statistics['max_abs'])>STAT_ATOL:raise AssertionError('MP3 candidate-vs-FP64 mature gates')
    quiet=run('mixed_final_observation_only',True,observed=False)
    for rank in layouts(cases['mixed']):
        name=f'outdat/rhs.cpp_final.step00000002.rk01.rank{rank:08d}.bin'
        if (cases['mixed']/name).read_bytes()!=(quiet/name).read_bytes():raise AssertionError('sensor/RHS observations changed q bytes')
    if not np.array_equal(np.loadtxt(quiet/'flowstate.dat',skiprows=1,ndmin=2),statistics['mixed']) or \
        not np.array_equal(logged_steps(quiet,2),logged_steps(cases['mixed'],2)):
        raise AssertionError('sensor/RHS observations changed statistics or clocks')
    if args.sanitizer:
        logs=list(cases['mixed'].glob('sanitizer.*.log'))
        if len(logs)!=args.ranks or any('ERROR SUMMARY: 0 errors' not in path.read_text() for path in logs):
            raise AssertionError('MP3 main sanitizer report')
        if args.sanitizer=='memcheck' and any('LEAK SUMMARY: 0 bytes leaked' not in path.read_text() for path in logs):
            raise AssertionError('MP3 main device leak')
    report.update(candidate_vs_fp64_field=candidate_field,candidate_vs_fp64_primitive=candidate_primitive,
        candidate_vs_fp64_statistics=candidate_statistics,initial_sensor_bits_equal=True,
        observation_q_statistics_clocks_exact=True,archives_tested=args.archives,
        checkpoint_tested=False,native_fp32_tested=False)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(topology=topology,transport=args.transport,sync=args.sync,
        mixed_old_q_max_abs=max(report['cases']['mixed']['state']['max_abs']),
        mixed_old_rhs_max_abs=max(report['cases']['mixed']['phases']['full']['max_abs']),
        candidate_fp64_q_max_abs=max(candidate_field['max_abs']),
        candidate_fp64_primitive_max_abs=max(candidate_primitive['max_abs']),
        candidate_fp64_statistics_max_abs=max(candidate_statistics['max_abs']),
        integer_masks_exact=True,observation_state_statistics_clocks_exact=True),indent=2))


if __name__=='__main__':main()
