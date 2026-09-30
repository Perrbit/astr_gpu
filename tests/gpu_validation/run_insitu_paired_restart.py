"""GPU TGV immutable paired-checkpoint restart in independent MPI processes."""
import argparse
import json
import hashlib
import os
import re
from pathlib import Path
import subprocess
import shutil
import sys

import numpy as np
from run_insitu_sample_validation import read_sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('gpu', 'mpiexec', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--render-backend-dir', type=Path)
    parser.add_argument('--legacy-pair-reference', type=Path,
                        help='Read immutable ASTRPS01 batches with the native GPU reader')
    parser.add_argument('--extracts',action='store_true')
    parser.add_argument('--device-budget', action='store_true', help='apply approved local device budget to statistics allocations')
    parser.add_argument('--host-budget', action='store_true', help='apply approved local node budget to explicit statistics arrays')
    parser.add_argument('--streamlines', action='store_true')
    parser.add_argument('--mean-streamlines', action='store_true')
    parser.add_argument('--continuous-no-field-io', type=Path,
                        help='continuous no-field-I/O gate; compare against an existing paired run directory')
    parser.add_argument('--no-oracle-io',action='store_true')
    parser.add_argument('--observe-resources',action='store_true')
    parser.add_argument('--native',action='store_true',help='Use formal namelist, including BUILD_TESTING=OFF')
    args = parser.parse_args()
    if args.legacy_pair_reference and not args.native:
        parser.error('--legacy-pair-reference requires --native')
    if args.native and args.continuous_no_field_io:
        parser.error('use run_insitu_native_render.py for native continuous no-field-I/O')
    if args.native and args.render_backend_dir and not args.mean_streamlines:
        parser.error('native TGV preset requires --extracts --streamlines --mean-streamlines')
    if args.no_oracle_io and not args.continuous_no_field_io:
        parser.error('--no-oracle-io requires --continuous-no-field-io reference')
    if args.observe_resources and not args.continuous_no_field_io:
        parser.error('--observe-resources requires continuous mode')
    if args.extracts and not args.render_backend_dir:
        parser.error('--extracts requires --render-backend-dir')
    if args.streamlines and not args.extracts:
        parser.error('--streamlines requires --extracts')
    if args.mean_streamlines and not args.streamlines:
        parser.error('--mean-streamlines requires --streamlines')
    products = ['q_surface','velocity_slice']
    if args.streamlines:
        products += ['instantaneous_streamlines','crossing_streamlines']
    if args.mean_streamlines:
        products += ['mean_reynolds_streamlines','mean_favre_streamlines']
    root = Path(__file__).resolve().parents[2]
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    env = {k:v for k,v in os.environ.items() if not k.startswith('ASTR_')}
    env.update(ASTR_GPU_SYNC_MODE='explicit', ASTR_GPU_HALO_TRANSPORT='pinned',
               ASTR_GPU_PRECISION_MODE='fp64', ASTR_INSITU_SAMPLE_PREFIX='outdat/sample',
               ASTR_INSITU_TEST_INITIAL='1', ASTR_INSITU_TEST_STATISTICS_WINDOW='0.0005 0.0035')
    report = {'status':'running', 'checks':[]}
    baseline_resources=None
    if args.device_budget:
        env['ASTR_INSITU_TEST_DEVICE_BUDGET']=f'{2*1024**3} {1024**3}'
    if args.host_budget:
        env['ASTR_INSITU_TEST_HOST_BUDGET']=str(4*1024**3)
    if args.continuous_no_field_io:
        env.update(ASTR_GPU_BENCHMARK_NO_FIELD_IO='1',ASTR_GPU_RK_TIMING='1')
    if args.no_oracle_io:
        env['ASTR_INSITU_TEST_ORACLE_IO']='0'

    def pair_runtime(runtime,case,batch_prefix='',restore=''):
        result=dict(runtime)
        if args.native:
            result={k:v for k,v in result.items() if not k.startswith('ASTR_INSITU_')}
            config=case/'insitu.nml'
            def quote(value):
                return "'"+str(value).replace("'","''")+"'"
            render='render=f,\n'
            if args.render_backend_dir:
                render=("render=t, schedule_mode='steps', step_interval=2, initial_frame=t, final_frame=f,\n"
                        f'implementation_path={quote(args.render_backend_dir.resolve())},\n'
                        f"pipeline_file={quote(root/'scripts/insitu/tgv_pipeline.py')},\n")
            config.write_text('&insitu_run enabled=t, statistics=t, '+render+
                              'statistics_window=0.0005,0.0035,\n'
                              "output_directory='outdat', host_budget_bytes=4294967296,\n"
                              'device_budget_bytes=2147483648, device_reserve_bytes=1073741824,\n'
                              f'batch_prefix={quote(batch_prefix)}, restore_batch={quote(restore)}\n/\n')
            result['ASTR_INSITU_CONFIG']=str(config)
        else:
            if batch_prefix:
                result['ASTR_INSITU_TEST_BATCH_PREFIX']=batch_prefix
            if restore:
                result['ASTR_INSITU_TEST_RESTORE_BATCH']=str(restore)
        return result

    def prepare(case, restart=False):
        subprocess.run([sys.executable,str(root/'tests/gpu_validation/prepare_tgv_case.py'),
                        '--src-case',str(root/'examples/Taylor_Green_Vortex'), '--dst-case',str(case),
                        '--use-gpu','t','--grid','32,32,32','--maxstep','4' if args.native else '3',
                        '--feqchkpt','1000' if args.continuous_no_field_io else '2',
                        '--deltat','1.d-3','--lfilter','t','--diffterm','t','--scheme','643e'],
                       check=True, timeout=30)
        if restart:
            path=case/'datin/input.tgv'
            lines=path.read_text().splitlines()
            found=False
            for i,line in enumerate(lines):
                if line.lstrip().startswith('# lrestar'):
                    for j in range(i+1,len(lines)):
                        if lines[j].strip() and not lines[j].lstrip().startswith('#'):
                            lines[j]='t'
                            found=True
                            break
                    break
            if not found:
                raise ValueError('Missing restart input marker')
            path.write_text('\n'.join(lines)+'\n')

    def run(case,ranks,runtime,expected_error=None):
        launch=[]
        if args.render_backend_dir and not args.native:
            (case/'outdat').mkdir(exist_ok=True)
            runtime=dict(runtime)
            for key in ('DISPLAY','PYTHONPATH','CATALYST_IMPLEMENTATION_PREFER_ENV'):
                runtime.pop(key,None)
            frames=[4] if 'ASTR_INSITU_TEST_RESTORE_BATCH' in runtime else [0,2,4]
            runtime.update(ASTR_INSITU_TEST_BACKEND=str(args.render_backend_dir.resolve()),
                           ASTR_INSITU_TEST_PIPELINE=str(root/'tests/gpu_validation/insitu_tgv_pipeline.py'),
                           ASTR_PROBE_OUTPUT=str(case/'outdat'),ASTR_INSITU_TEST_STEP_INTERVAL='2',
                           ASTR_INSITU_TEST_EXPECTED_STEPS=json.dumps(frames))
            if args.extracts:
                runtime['ASTR_INSITU_TEST_EXTRACTS']='1'
            if args.streamlines:
                runtime['ASTR_INSITU_TEST_STREAMLINES']='1'
            if args.mean_streamlines:
                runtime['ASTR_INSITU_TEST_MEAN_STREAMLINES']='1'
            launch=[sys.executable,str(root/'tests/gpu_validation/insitu_device_identity.py'),'--']
        if args.native and args.render_backend_dir:
            runtime=dict(runtime)
            for key in ('DISPLAY','PYTHONPATH','CATALYST_IMPLEMENTATION_PREFER_ENV','VTK_EGL_DEVICE_INDEX'):
                runtime.pop(key,None)
            frames=[4] if case.name.endswith('_resumed') else [0,2,4]
        with (case/'run.log').open('w') as log:
            command=[str(args.mpiexec.resolve()),'--mca','coll_hcoll_enable','0','-np',str(ranks),
                     *launch,str(args.gpu.resolve()),'run','datin/input.tgv']
            if args.observe_resources:
                from insitu_resource_monitor import run_monitored
                run_monitored(command,case,runtime,log,case/'resources.json',baseline_resources)
                completed=subprocess.CompletedProcess(command,0)
            else:
                completed=subprocess.run(command,cwd=case,env=runtime,
                                         stdout=log,stderr=subprocess.STDOUT,timeout=180)
        if expected_error:
            if completed.returncode==0 or expected_error not in (case/'run.log').read_text():
                raise ValueError('Malformed batch was not rejected by the expected guard')
            if list((case/'outdat').glob('sample.*.bin')):
                raise ValueError('Malformed batch advanced or sampled flow')
            return
        completed.check_returncode()
        if 'The job is done!' not in (case/'run.log').read_text():
            raise ValueError('Solver did not complete')
        if args.native:
            transfers=re.findall(r'ASTR_INSITU_GPU_STATS rank=(\d+) samples=(\d+) full_output_downloads=(\d+)',
                                 (case/'run.log').read_text())
            expected_samples=3 if case.name.endswith('_resumed') else 6
            assert sorted(int(rank) for rank,_,_ in transfers)==list(range(ranks)),transfers
            assert all(int(n)==expected_samples and int(d)==1 for _,n,d in transfers),transfers
            flows=re.findall(r'ASTR_INSITU_GPU_FLOW rank=(\d+) frame_downloads=(\d+)',(case/'run.log').read_text())
            expected_frames=(1 if case.name.endswith('_resumed') else 3) if args.render_backend_dir else 0
            assert len(flows)==ranks and all(int(n)==expected_frames for _,n in flows),flows
        if args.render_backend_dir:
            from PIL import Image
            for rank in range(ranks):
                lifecycle=json.loads((case/'outdat'/f'lifecycle_rank{rank}.json').read_text())
                expected_frames=[[step,step*1.e-3] for step in frames] if args.native else frames
                if lifecycle != {'frames':expected_frames,'finalized':True}:
                    raise ValueError('Renderer lifecycle/step sequence mismatch after restart')
            for product in products:
                paths=list((case/'outdat').glob(f'{product}.step*.jpeg'))
                product_frames=[step for step in frames if step>0] if product.startswith('mean_') else frames
                if len(paths)!=len(product_frames):
                    raise ValueError('Unexpected rendered frame count')
                for path in paths:
                    with Image.open(path) as image:
                        rgb=np.asarray(image.convert('RGB'))
                        if image.size!=(800,600) or np.count_nonzero(np.min(rgb,axis=2)<220)<1000:
                            raise ValueError('Blank or wrong-sized restart rendering')
                        if args.native:
                            assert path.with_suffix('.eps').is_file()
                        else:
                            image.convert('RGB').save(path.with_suffix('.eps'))

    try:
        for ranks in (1,2):
            runtime=dict(env,ASTR_FORCE_MPI_TOPOLOGY=f'{ranks},1,1')
            if args.observe_resources:
                from insitu_resource_monitor import run_monitored
                off=args.output/f'np{ranks}_off'
                prepare(off)
                off_runtime={k:v for k,v in runtime.items() if not k.startswith('ASTR_INSITU_')}
                with (off/'run.log').open('w') as log:
                    baseline_resources=run_monitored(
                        [str(args.mpiexec.resolve()),'--mca','coll_hcoll_enable','0','-np',str(ranks),
                         str(args.gpu.resolve()),'run','datin/input.tgv'],off,off_runtime,log,off/'resources.json')
            source=args.output/f'np{ranks}_continuous'
            resumed=args.output/f'np{ranks}_resumed'
            prepare(source)
            if args.continuous_no_field_io:
                run(source,ranks,runtime)
                out=source/'outdat'
                if list(out.glob('*.h5')) or list(out.glob('restart*')):
                    raise ValueError('Conventional field/checkpoint files written in no-field-I/O mode')
                reference=args.continuous_no_field_io.resolve()/f'np{ranks}_continuous/outdat'
                samples=list(out.glob('sample.step*.bin'))
                if len(samples)!=(0 if args.no_oracle_io else 5*ranks):
                    raise ValueError('Missing no-field-I/O validation snapshots')
                if args.no_oracle_io:
                    if list(out.glob('sample*.bin')):
                        raise ValueError('Oracle full-field files still exist')
                    geometry=list(out.rglob('*.vtp'))+list(out.glob('*.pvtp'))
                    if not geometry:
                        raise ValueError('No spatial extraction files')
                    for path in geometry:
                        if path.read_bytes()!=(reference/path.relative_to(out)).read_bytes():
                            raise ValueError(f'Geometry differs without oracle I/O: {path.name}')
                for snapshot in samples:
                    if snapshot.read_bytes()!=(reference/snapshot.name).read_bytes():
                        raise ValueError('No-field-I/O state differs from field-I/O reference')
                for snapshot in out.glob('sample.*statistics.step*.bin'):
                    if snapshot.read_bytes()!=(reference/snapshot.name).read_bytes():
                        raise ValueError('No-field-I/O statistics differ from reference')
                if args.render_backend_dir:
                    from PIL import Image
                    for picture in out.glob('*.jpeg'):
                        with Image.open(picture) as a,Image.open(reference/picture.name) as b:
                            np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
                report['checks'].append({'np':ranks,'conventional_field_io':'absent',
                                         'oracle_snapshots':'absent' if args.no_oracle_io else 'byte-identical',
                                         'spatial_files':'byte-identical' if args.no_oracle_io else 'produced',
                                         'statistics':'not serialized' if args.no_oracle_io else 'byte-identical',
                                         'rendered_pixels':'exact'})
                print(f'PASS NP={ranks}: continuous no-field-I/O rendering',flush=True)
                continue
            run(source,ranks,pair_runtime(runtime,source,batch_prefix='outdat/paired'))
            batch=source/'outdat/paired.step00000002'
            if not (batch/'manifest.bin').is_file():
                raise ValueError('No complete batch')
            original={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in batch.iterdir() if p.is_file()}
            prepare(resumed,restart=True)
            run(resumed,ranks,pair_runtime(runtime,resumed,batch_prefix='outdat/paired' if args.native else '',restore=batch))
            if args.native:
                import h5py
                if list((resumed/'outdat').glob('sample.step*.bin')):
                    raise ValueError('Native restart wrote test snapshots')
                for rank in range(ranks):
                    names=[f'sample.statistics.step00000005.rank{rank:08d}.bin',
                           f'paired.step00000004/statistics.rank{rank:08d}.bin']
                    for name in names:
                        if (source/'outdat'/name).read_bytes() != (resumed/'outdat'/name).read_bytes():
                            raise ValueError(f'Native restart differs: {name}')
                    # Publication generation is a wall-clock identifier, not numerical state.
                    dtype=np.dtype([('magic','S8'),('header','<i4',(29,)),('step','<i8'),
                                    ('generation','<i8'),('parameters','<f8',(12,)),('names','S32',(4,))])
                    qname=f'restart_q.rank{rank:08d}.bin'
                    with (source/'outdat'/qname).open('rb') as a, (resumed/'outdat'/qname).open('rb') as b:
                        ha=np.fromfile(a,dtype=dtype,count=1)
                        hb=np.fromfile(b,dtype=dtype,count=1)
                        assert len(ha)==len(hb)==1 and ha['magic'][0]==b'ASTRQ001'
                        for name in dtype.names:
                            if name!='generation':
                                np.testing.assert_array_equal(ha[name],hb[name])
                        qa=np.fromfile(a,dtype='<f8')
                        qb=np.fromfile(b,dtype='<f8')
                        assert qa.size>0 and np.isfinite(qa).all()
                        assert qa.tobytes()==qb.tobytes(), 'exact q payload differs'
                with h5py.File(source/'outdat/flowfield.h5') as a, h5py.File(resumed/'outdat/flowfield.h5') as b:
                    for name in ('ro','u1','u2','u3','p','t','nstep','time'):
                        if a[name][...].tobytes()!=b[name][...].tobytes():
                            raise ValueError(f'Native restart changed {name}')
                report['checks'].append({'np':ranks,'native_final_step':5,'paired_checkpoint_step':4,
                                         'exact_q':'byte-identical','host_device_statistics_state':'byte-identical',
                                         'statistics_output':'byte-identical','HDF_fields':'bitwise-identical'})
            else:
                if len(list((resumed/'outdat').glob('sample.step*.bin'))) != 2*ranks:
                    raise ValueError('Restart duplicated or omitted a completed sample')
                for step in (3,4):
                    for rank in range(ranks):
                        suffix=f'.step{step:08d}.rank{rank:08d}.bin'
                        a=source/'outdat'/('sample'+suffix)
                        b=resumed/'outdat'/('sample'+suffix)
                        ha,xa,fa=read_sample(a)
                        hb,xb,fb=read_sample(b)
                        np.testing.assert_array_equal(ha,hb)
                        np.testing.assert_array_equal(xa,xb)
                        np.testing.assert_array_equal(fa,fb)
                        for name in ('sample.statistics','sample.device_statistics'):
                            if (source/'outdat'/(name+suffix)).read_bytes() != (resumed/'outdat'/(name+suffix)).read_bytes():
                                raise ValueError(f'Nonidentical restart statistics: {name}{suffix}')
                report['checks'].append({'np':ranks,'complete_steps':[3,4],
                                         'fields':'exact','statistics_files':'byte-identical'})
            if args.render_backend_dir:
                from PIL import Image
                for product in products:
                    name=f'{product}.step00000004.jpeg'
                    with Image.open(source/'outdat'/name) as a, Image.open(resumed/'outdat'/name) as b:
                        np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
                report['checks'].append({'np':ranks,'render_steps_continuous':[0,2,4],
                                         'render_steps_resumed':[4],'step4_pixels':'exact'})
                if args.native:
                    geometry=list((resumed/'outdat').rglob('*.vtp'))+list((resumed/'outdat').glob('*.pvtp'))
                    assert geometry
                    for path in geometry:
                        assert path.read_bytes()==(source/'outdat'/path.relative_to(resumed/'outdat')).read_bytes(),path
                    report['checks'].append({'np':ranks,'restored_geometry':'byte-identical'})
            if args.legacy_pair_reference:
                legacy_batch=(args.legacy_pair_reference.resolve()/
                              f'np{ranks}_continuous/outdat/paired.step00000002')
                before={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in legacy_batch.iterdir() if p.is_file()}
                for rank in range(ranks):
                    with (legacy_batch/f'statistics.rank{rank:08d}.bin').open('rb') as f:
                        assert f.read(8)==b'ASTRPS01', 'reference must use old statistics format'
                legacy=args.output/f'np{ranks}_legacy_resumed'
                prepare(legacy,restart=True)
                run(legacy,ranks,pair_runtime(runtime,legacy,restore=legacy_batch))
                for rank in range(ranks):
                    name=f'sample.statistics.step00000005.rank{rank:08d}.bin'
                    assert (legacy/'outdat'/name).read_bytes()==(resumed/'outdat'/name).read_bytes(),name
                with h5py.File(legacy/'outdat/flowfield.h5') as a, h5py.File(resumed/'outdat/flowfield.h5') as b:
                    for name in ('ro','u1','u2','u3','p','t','nstep','time'):
                        assert a[name][...].tobytes()==b[name][...].tobytes(),name
                if args.render_backend_dir:
                    for path in (legacy/'outdat').glob('*.jpeg'):
                        with Image.open(path) as a, Image.open(resumed/'outdat'/path.name) as b:
                            np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
                    geometry=list((legacy/'outdat').rglob('*.vtp'))+list((legacy/'outdat').glob('*.pvtp'))
                    assert geometry
                    for path in geometry:
                        assert path.read_bytes()==(resumed/'outdat'/path.relative_to(legacy/'outdat')).read_bytes(),path
                after={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in legacy_batch.iterdir() if p.is_file()}
                assert before==after, 'legacy reference batch was modified'
                report['checks'].append({'np':ranks,'ASTRPS01_restart':'passed',
                                         'statistics':'byte-identical','HDF_fields':'bitwise-identical',
                                         'reference_batch':'unchanged'})
            if ranks==2:
                for fault in ('missing','corrupt'):
                    broken=args.output/f'batch_{fault}'
                    shutil.copytree(batch,broken)
                    victim=broken/'statistics.rank00000001.bin'
                    if fault=='missing':
                        victim.unlink()
                    else:
                        with victim.open('r+b') as stream:
                            stream.seek(-1,2)
                            value=stream.read(1)[0]
                            stream.seek(-1,2)
                            stream.write(bytes([value^1]))
                    case=args.output/f'np2_reject_{fault}'
                    prepare(case,restart=True)
                    run(case,ranks,pair_runtime(runtime,case,restore=broken),
                        expected_error='paired batch incomplete or changed')
                    report['checks'].append({'np':2,'rejected_before_sampling':fault})
                if args.native:
                    faults=['window','topology']
                    if args.render_backend_dir:
                        faults += ['render_disabled','schedule_interval']
                    else:
                        faults += ['render_enabled']
                    for fault in faults:
                        case=args.output/f'native_reject_{fault}'
                        prepare(case,restart=True)
                        rejected_runtime=pair_runtime(runtime,case,restore=batch)
                        rejected_ranks=ranks
                        expected='paired regional statistics record failed'
                        if fault=='window':
                            config=case/'insitu.nml'
                            config.write_text(config.read_text().replace('0.0005,0.0035','0.0006,0.0035'))
                        elif fault=='topology':
                            rejected_ranks=1
                            rejected_runtime['ASTR_FORCE_MPI_TOPOLOGY']='1,1,1'
                            expected='paired batch incomplete or changed'
                        else:
                            config=case/'insitu.nml'
                            content=config.read_text()
                            if fault=='render_disabled':
                                content=content.replace('render=t','render=f')
                                expected='paired rendering configuration mismatch'
                            elif fault=='render_enabled':
                                content=content.replace('render=f',
                                    "render=t, step_interval=2, implementation_path='unused', pipeline_file='unused'")
                                expected='paired rendering configuration mismatch'
                            else:
                                content=content.replace('step_interval=2','step_interval=3')
                                expected='paired schedule record failed'
                            config.write_text(content)
                        run(case,rejected_ranks,rejected_runtime,expected_error=expected)
                        report['checks'].append({'np':rejected_ranks,'rejected_before_sampling':fault})
            after={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in batch.iterdir() if p.is_file()}
            if after != original:
                raise ValueError('Restart changed its immutable source batch')
            print(f'PASS NP={ranks}: new-process GPU paired restart',flush=True)
        report['status']=('passed-bounded-continuous-no-field-io' if args.continuous_no_field_io
                          else 'passed-bounded-gpu-paired-restart')
    except BaseException as error:
        report.update(status='failed',error=str(error))
        raise
    finally:
        (args.output/'summary.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
