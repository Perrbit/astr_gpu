"""Ordinary native-FP32 C++ TGV advancement; Python only prepares and compares."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import h5py

from run_cpp_tgv_gate import prepare_case,namelist,logged_steps,cfl_log,compare

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests/gpu_validation'))
from compare_q_validation_snapshots import read_q_snapshot


def field(path):
    record=read_q_snapshot(path,precision='fp32')
    im,jm,km,h,components=record.header
    full=record.values.reshape((im+2*h+1,jm+2*h+1,km+2*h+1,components),order='F')
    body=full[h:h+im+1,h:h+jm+1,h:h+km+1,:]
    if not (body[...,0]>0).all():raise AssertionError('nonpositive native density')
    internal=body[...,4].astype(np.float64)-.5*np.sum(body[...,1:4].astype(np.float64)**2,axis=-1)/body[...,0]
    if not (internal>0).all():raise AssertionError('nonpositive native internal energy')
    return record.header,full,body


def stage_report(current,reference,partition):
    report={}
    for label in ('pre_rhs','post_update'):
        paths=sorted((reference/'outdat').glob(f'rhs.{label}.*.bin'))
        if len(paths)!=12:raise AssertionError('native reference complete-step/RK identities')
        maximum=np.zeros(5)
        for path in paths:
            _,_,whole=field(path)
            for rank,layout in partition.items():
                _,_,actual=field(current/'outdat'/path.name.replace('rank00000000',f'rank{rank:08d}'))
                region=tuple(slice(o,o+n+1) for o,n in zip(layout['origin'],layout['upper']))
                maximum=np.maximum(maximum,compare(actual.astype(np.float64),whole[region],limit=None))
        report[label]=maximum.tolist()
    for label in ('conv','full'):
        paths=sorted((reference/'outdat').glob(f'rhs.{label}.*.bin'))
        if len(paths)!=12:raise AssertionError('native RHS stage identities')
        maximum=np.zeros(5)
        for path in paths:
            with path.open('rb') as stream:
                header=np.fromfile(stream,np.int32,4)
                whole=np.fromfile(stream,np.float32).reshape(tuple(header[:3]+1)+(int(header[3]),),order='F')
            for rank,layout in partition.items():
                with (current/'outdat'/path.name.replace('rank00000000',f'rank{rank:08d}')).open('rb') as stream:
                    h=np.fromfile(stream,np.int32,4)
                    actual=np.fromfile(stream,np.float32).reshape(tuple(h[:3]+1)+(int(h[3]),),order='F')
                region=tuple(slice(o,o+n+1) for o,n in zip(layout['origin'],layout['upper']))
                maximum=np.maximum(maximum,compare(actual.astype(np.float64),whole[region],limit=None))
        report[label]=maximum.tolist()
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','template','mpiexec','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--reference',type=Path,help='immutable original native32 four-step reference with matching derivative/filter')
    parser.add_argument('--ranks',type=int,choices=(1,2),default=1)
    parser.add_argument('--axis',choices=('x','y','z'),default='x')
    parser.add_argument('--automatic',action='store_true')
    parser.add_argument('--filter',choices=('off','explicit10','compact10'),default='off')
    parser.add_argument('--derivative',choices=('explicit','compact'),default='explicit')
    parser.add_argument('--convection',choices=('standard','keep'),default='standard')
    parser.add_argument('--workspace',choices=('scalar','full'),default='scalar')
    parser.add_argument('--sync',choices=('explicit','dependency'),default='explicit')
    parser.add_argument('--transport',choices=('pinned','device'),default='pinned')
    parser.add_argument('--quiet-compare',action='store_true')
    parser.add_argument('--memcheck',action='store_true')
    parser.add_argument('--restart',choices=('exact','nonexact'),help='same-build two-plus-two checkpoint continuation')
    parser.add_argument('--archives',action='store_true',help='completed-step volume and fixed-index slices, no rendering')
    parser.add_argument('--analyze-only',action='store_true',help='read completed runs without advancing or rewriting input')
    args=parser.parse_args()
    if args.reference and args.automatic:
        parser.error('the original reference selection here uses fixed dt')
    if args.automatic and args.restart=='nonexact':
        parser.error('this same-phase comparison uses fixed dt for nonexact controller reset')
    if args.reference:
        for backend in ('cpu','gpu'):
            source=args.reference/f'filter{int(args.filter!="off")}_{backend}'
            chosen=re.search(r"\bconvection_form\s*=\s*['\"]([^'\"]+)['\"]",(source/'datin/input.case').read_text())
            if chosen is None or chosen[1]!=args.convection:raise AssertionError('native convection reference mismatch')
    root=args.output.resolve()
    if not args.analyze_only:
        root.mkdir(parents=True,exist_ok=False)
        shutil.copy2(args.binary,root/'tested_astr');shutil.copy2(__file__,root/'runner.py')
    topology=[1,1,1];topology['xyz'.index(args.axis)]=args.ranks
    env={name:value for name,value in os.environ.items() if not name.startswith('ASTR_')}
    env.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',ASTR_GPU_HALO_TRANSPORT=args.transport,
        ASTR_GPU_SYNC_MODE=args.sync,ASTR_GPU_FILTER_WORKSPACE=args.workspace,
        ASTR_GPU_COMPACT_TRANSPORT='device-aware' if args.transport=='device' else 'pinned',
        ASTR_GPU_COMPACT_SYNC=args.sync,ASTR_FORCE_MPI_TOPOLOGY=','.join(map(str,topology)),ASTR_CFL_PROFILE_Y='1',
        ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',ASTR_VALIDATION_RHS_STEP='0',ASTR_VALIDATION_RHS_STEP_END='4')
    if args.transport=='device':env.update(OMPI_MCA_pml='ucx',OMPI_MCA_opal_cuda_support='1')
    scheme='643c' if args.derivative=='compact' else '643e'

    def run(name,observe=True,steps=4,restore=None,writing=True):
        case=root/name
        if args.analyze_only:return case
        prepare_case(case,args.template.resolve(),True,True,args.automatic,grid=16,steps=steps,dt=1e-4,max_dt=1e-3,
            numerics=dict(conschm=scheme,difschm=scheme,lfilter=args.filter!='off',
                filter_scheme='explicit10' if args.filter=='off' else args.filter,convection_form=args.convection,
                recon_schem=0,lchardecomp=False),
            physics=dict(diffterm=True))
        (case/'datin/input.output').write_text(namelist(dict(output=dict(directory='outdat/new',
            exact_restart=args.restart=='exact',restore_directory=str(restore or ''),
            product_poll_steps=1,
            host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0),
            checkpoint=dict(enabled=writing and bool(args.restart),mode='steps',interval_steps=2,keep=2),
            volume=dict(enabled=writing and args.archives,mode='steps',interval_steps=2,
                velocity_gradient=True,vorticity=True,qcriterion=True),slices=dict(enabled=False))))
        if args.archives:
            products=[dict(catalog_schema='astr-insitu-products-v1')]
            if writing:
                for axis in ('i','j','k'):
                    products.append(dict(id=axis+'8',kind='slice',enable_request=1,source='instantaneous',
                        fields=['density','velocity_x','velocity_y','velocity_z','pressure','temperature',
                            'Q_rs','velocity_divergence','density_gradient_magnitude'],
                        sampling=dict(mode='grid_index',axis=axis,index=8,
                            index_ranges={a:[0,16] for a in ('i','j','k') if a!=axis}),
                        output=dict(channel='data',format='hdf5-xdmf'),schedule=dict(mode='steps',interval=2)))
            (case/'input.products.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in products))
        command=[str(args.mpiexec.resolve()),'--bind-to','none','-n',str(args.ranks)]
        if args.memcheck and observe:
            command+=['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer',
                '--tool','memcheck','--error-exitcode','99','--leak-check','full',
                '--log-file',str(case/'sanitizer.%p.log')]
        command+=[str(root/'tested_astr'),'run']
        environment=env if observe else env|dict(ASTR_VALIDATION_RHS_STEP='4')
        with (case/'run.log').open('w') as log:
            subprocess.run(command,cwd=case,env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        text=(case/'run.log').read_text()
        if 'backend=cuda' not in text:raise AssertionError('ordinary C++ GPU selection')
        return case

    current=run('cpp')
    clocks=logged_steps(current);cfl=cfl_log(current)
    if not np.isfinite(clocks).all() or not np.isfinite(cfl).all() or not (clocks[:,3:]>0).all():
        raise AssertionError('native controller/CFL finite positive values')
    if args.automatic and np.max(cfl)>float(np.nextafter(np.float32(.5),np.float32(np.inf))):
        raise AssertionError('native automatic CFL target')
    from run_cpp_multi_tgv_gate import layouts
    partition=layouts(current)
    fields={};report=dict(ranks=args.ranks,topology=topology,automatic=args.automatic,
        derivative=args.derivative,convection=args.convection,filter=args.filter,workspace=args.workspace,sync=args.sync,transport=args.transport,
        clocks=clocks.tolist(),cfl=cfl.tolist(),references={})
    for rank in partition:
        header,all_values,body=field(current/f'outdat/rhs.cpp_final.step00000004.rk01.rank{rank:08d}.bin')
        fields[rank]=(all_values,body)
    if args.reference:
        for backend in ('cpu','gpu'):
            source=args.reference/f'filter{int(args.filter!="off")}_{backend}'
            _,_,whole=field(source/'outdat/rhs.post_update.step00000003.rk03.rank00000000.bin')
            errors=np.zeros(5)
            for rank,layout in partition.items():
                region=tuple(slice(o,o+n+1) for o,n in zip(layout['origin'],layout['upper']))
                errors=np.maximum(errors,compare(fields[rank][1].astype(np.float64),whole[region],limit=None))
            scalars=np.loadtxt(current/'flowstate.dat',skiprows=1,ndmin=2)
            original=np.loadtxt(source/'flowstate.dat',skiprows=1,ndmin=2)
            report['references'][backend]=dict(q_scaled=errors.tolist(),
                scalar_scaled=compare(scalars,original,limit=None),
                clocks_scaled=compare(clocks,logged_steps(source),limit=None),
                stages=stage_report(current,source,partition))
    if args.quiet_compare:
        quiet=run('quiet',False,writing=False)
        if not np.array_equal(logged_steps(quiet),clocks):raise AssertionError('native observation changed controller words')
        for rank in partition:
            _,whole,_=field(quiet/f'outdat/rhs.cpp_final.step00000004.rk01.rank{rank:08d}.bin')
            if whole.tobytes()!=fields[rank][0].tobytes():raise AssertionError('native observation changed full q words')
        report['observation_full_q_clock_words_equal']=True
    if args.restart:
        from run_cpp_restart_gate import datasets,exact_state
        prefix=run('prefix',steps=2)
        saved=prefix/'outdat/new/checkpoints/step000000000002'
        resumed=run('resumed',restore=saved)
        last='outdat/new/checkpoints/step000000000004'
        a,b=current/last,resumed/last
        state=datasets(a/'state.h5')
        if any(state[f'q{c:04d}'].dtype!=np.dtype('float32') for c in range(1,6)):
            raise AssertionError('native checkpoint numerical storage width')
        identity=state['identity']
        continuation={}
        for rank in partition:
            _,whole,body=field(resumed/f'outdat/rhs.cpp_final.step00000004.rk01.rank{rank:08d}.bin')
            continuation[str(rank)]=compare(body.astype(np.float64),fields[rank][1].astype(np.float64),limit=None)
            if args.restart=='exact' and whole.tobytes()!=fields[rank][0].tobytes():
                raise AssertionError('native exact continuation full q words')
        if not np.array_equal(logged_steps(resumed,count=2,first_step=3),clocks[2:]):
            raise AssertionError('native continuation complete-step control phase')
        scalar=np.loadtxt(current/'flowstate.dat',skiprows=1,ndmin=2)
        tail=np.loadtxt(resumed/'flowstate.dat',skiprows=1,ndmin=2)
        matching=scalar[np.isin(scalar[:,0],tail[:,0])]
        if args.restart=='exact':
            count=exact_state(a/'state.h5',b/'state.h5')
            for name in ('control.bin','insitu_control.bin'):
                if (a/name).read_bytes()!=(b/name).read_bytes():
                    raise AssertionError('native exact continuation control words '+name)
            archive_a=(a/'archives.bin').read_bytes();archive_b=(b/'archives.bin').read_bytes()
            if args.archives:
                # The final six int64 values identify each run's actual output
                # segment. A resumed segment records its parent, not the initial
                # segment's provenance; all preceding scheduling state is exact.
                if archive_a[:8]!=b'ASTROA02' or archive_a[:-48]!=archive_b[:-48]:
                    raise AssertionError('native exact archive schedule words')
                report['archive_segment_identity']={}
                for case,wire in ((current,archive_a),(resumed,archive_b)):
                    segment_identity=np.frombuffer(wire[-48:],dtype='<i8').reshape(3,2)
                    segment=case/f'outdat/new/fields/segment{segment_identity[0,0]:08d}/SEGMENT'
                    if segment_identity[1,0]!=segment.stat().st_size or not np.array_equal(segment_identity[:,1],[-1,0,0]):
                        raise AssertionError('native archive segment identity/disabled slices')
                    report['archive_segment_identity'][case.name]=segment_identity.tolist()
            elif archive_a!=archive_b:
                raise AssertionError('native exact inactive archive controls')
            report['archive_schedule_words_equal']=True
            if not np.array_equal(matching,tail):raise AssertionError('native exact scalar words')
            report['exact_datasets']=count
        else:
            if state['rank_extras'].size or 'q0006' in state or int(identity[12])!=11:
                raise AssertionError('native nonexact checkpoint contains exact caches or redundant rank data')
            if (current/'outdat/new/resources/geometry.h5').exists():
                raise AssertionError('native nonexact checkpoint retains exact geometry resource')
        report.update(restart=args.restart,checkpoint_real_bytes=4,
            continuation_q_scaled_report_only=continuation,continuation_scalar_scaled_report_only=compare(tail,matching,limit=None),
            checkpoint_identity=identity.tolist())
    if args.archives:
        archives=sorted((current/'outdat/new').rglob('*.h5'))
        report['archive_files']=[str(p.relative_to(current)) for p in archives if 'checkpoints' not in p.parts]
        if not report['archive_files']:raise AssertionError('native archive events missing')
        if len(list((current/'outdat/new/fields').glob('segment*/step*/data.h5')))!=2:
            raise AssertionError('native volume complete-step events')
        for axis in ('i','j','k'):
            paths=sorted((current/'outdat/new/slice_products'/f'{axis}8').rglob('step*.h5'))
            if [p.name for p in paths]!=[f'step{s:012d}.h5' for s in (0,2,4)]:
                raise AssertionError('native index slice events '+axis)
            if args.restart:
                tail=sorted((resumed/'outdat/new/slice_products'/f'{axis}8').rglob('step*.h5'))
                if [p.name for p in tail]!=['step000000000004.h5']:
                    raise AssertionError('native slice restore schedule '+axis)
                if args.restart=='exact':exact_state(paths[-1],tail[0])
        # The source-of-truth field is checked separately; inspect product arrays,
        # not integer layout metadata or undefined exterior cache padding.
        report['archive_float_dtypes']={}
        for path in archives:
            if 'checkpoints' in path.parts:continue
            for name,value in datasets(path).items():
                if value.dtype.kind=='f':
                    if not np.isfinite(value).all():raise AssertionError('nonfinite native archive '+name)
                    if name not in ('clock','reference_state') and value.dtype!=np.dtype('float32'):
                        raise AssertionError('non-native spatial archive storage '+name)
                    report['archive_float_dtypes'][str(path.relative_to(current))+':'+name]=str(value.dtype)
        volume=current/'outdat/new/fields/segment00000000/step000000000004/data.h5'
        with h5py.File(volume) as data:
            density=data['density'][...].transpose(2,1,0)
            for rank,layout in partition.items():
                region=tuple(slice(o,o+n+1) for o,n in zip(layout['origin'],layout['upper']))
                if density[region].tobytes()!=fields[rank][1][...,0].tobytes():
                    raise AssertionError('complete-step volume density differs from authoritative q')
            report['slice_volume_scaled_report_only']={}
            for axis in ('i','j','k'):
                final=next((current/'outdat/new/slice_products'/f'{axis}8').rglob('step000000000004.h5'))
                with h5py.File(final) as plane:
                    names=[name for name in plane if name in data and plane[name].ndim==2]
                    report['slice_volume_scaled_report_only'][axis]={name:compare(
                        plane[name][...].astype(np.float64)[...,None],
                        np.take(data[name][...],8,axis=2-'ijk'.index(axis)).astype(np.float64)[...,None],
                        limit=None)[0] for name in names}
        if args.restart=='exact':
            exact_state(volume,resumed/'outdat/new/fields/segment00000000/step000000000004/data.h5')
        report['volume_density_matches_authoritative_q_words']=True
    if args.memcheck:
        paths=list(current.glob('sanitizer.*.log'))
        if len(paths)!=args.ranks or any('ERROR SUMMARY: 0 errors' not in p.read_text() or
                'LEAK SUMMARY: 0 bytes leaked' not in p.read_text() for p in paths):raise AssertionError('native main Memcheck')
    report.update(binary_sha256=hashlib.sha256((root/'tested_astr').read_bytes()).hexdigest(),precision='fp32',
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        finite_positive_physical_state=True,scope='Ordinary native C++ periodic TGV four-step integration and explicitly selected lifecycle/products; not long-window physics or other optional capability qualification')
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
