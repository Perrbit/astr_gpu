"""Matched 256^3 TGV device visualization on/off timing, without field I/O."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
MPI = Path('/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpiexec')
LIBRARY = Path('/home/dell/workspace/astr_dependencies/install/paraview-6.1.1-hpcx-gcc13/lib/catalyst')
PRODUCTS = ('q_surface', 'instantaneous_streamlines')


def prepare(case, steps, enabled, library, transport):
    subprocess.run([sys.executable, str(ROOT/'tests/gpu_validation/prepare_tgv_case.py'),
        '--src-case', str(ROOT/'examples/Taylor_Green_Vortex'), '--dst-case', str(case),
        '--use-gpu', 't', '--grid', '256,256,256', '--maxstep', str(steps-1),
        '--feqchkpt', '1000', '--feqlist', '10', '--deltat', '1.d-4',
        '--lfilter', 't', '--diffterm', 't', '--scheme', '643e'], check=True)
    (case/'outdat/render').mkdir(parents=True)
    (case/'datin/input.output').write_text(
        "&output\n directory='outdat/output', buffer_bytes=4096,\n"
        'host_budget_bytes=67108864, device_budget_bytes=67108864\n/\n'
        '&checkpoint\n enabled=f\n/\n&volume\n enabled=f\n/\n&slices\n enabled=f\n/\n')
    configuration = '&insitu_run enabled=f, statistics=f, render=f /\n'
    if enabled:
        configuration = f"""&insitu_run
 enabled=t, statistics=f, render=t, output_directory='outdat/render',
 derivative_backend='gpu', processing_backend='device', products='tgv256_demo',
 postprocess_transport='{transport}',
 schedule_mode='steps', step_interval=1, initial_frame=f, final_frame=f,
 host_budget_bytes=17179869184, device_budget_bytes=6442450944,
 device_reserve_bytes=2147483648,
 implementation_path='{library}', pipeline_file='{ROOT/'scripts/insitu/tgv_pipeline.py'}'
/
"""
    (case/'insitu.nml').write_text(configuration)


def environment(case, transport):
    env = {k:v for k,v in os.environ.items() if not k.startswith('ASTR_')}
    for key in ('DISPLAY', 'PYTHONPATH', 'CATALYST_IMPLEMENTATION_PREFER_ENV', 'VTK_EGL_DEVICE_INDEX'):
        env.pop(key, None)
    env.update(ASTR_INSITU_CONFIG=str(case/'insitu.nml'), ASTR_INSITU_TIMING='1',
        ASTR_COMPLETE_STEP_TIMING='1', ASTR_GPU_RK_TIMING='1', ASTR_GPU_RANK_RK_TIMING='1',
        ASTR_FORCE_MPI_TOPOLOGY='2,1,1', ASTR_GPU_SYNC_MODE='explicit',
        ASTR_GPU_HALO_TRANSPORT='pinned', ASTR_GPU_PRECISION_MODE='fp64',
        ASTR_GPU_FILTER_WORKSPACE='scalar', ASTR_GPU_BENCHMARK_NO_FIELD_IO='1')
    if transport == 'device-aware':
        env.update(OMPI_MCA_pml='ucx', OMPI_MCA_coll='^hcoll,ucc,cuda',
            OMPI_MCA_coll_hcoll_enable='0', OMPI_MCA_osc='pt2pt', UCX_MEMTYPE_CACHE='n',
            UCX_CUDA_COPY_ENABLE_FABRIC='no', UCX_CUDA_COPY_DMABUF='no',
            UCX_CUDA_IPC_ENABLE_MNNVL='no', UCX_TLS='self,sm,cuda_copy,cuda_ipc')
    else:
        env.update(OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,tcp', OMPI_MCA_osc='pt2pt',
            OMPI_MCA_opal_cuda_support='0', OMPI_MCA_coll_ucc_enable='0')
    return env


def timing(case, steps):
    stages = {}
    log = (case/'run.log').read_text()
    def add(stage, rank, value):
        rank, value = int(rank), float(value)
        if rank not in (0, 1) or value < 0 or not math.isfinite(value):
            raise ValueError('Invalid timing rank/value')
        stages.setdefault(stage, {0:[], 1:[]})[rank].append(value)
    for stage, step, rank, value in re.findall(
            r'ASTR_INSITU_STAGE_TIMING (\w+) (-?\d+) (\d+)\s+(\S+)', log):
        add(stage, rank, value)
    for line in log.splitlines():
        if 'ASTR_INSITU_PIPELINE_TIMING ' in line:
            item = json.loads(line.split('ASTR_INSITU_PIPELINE_TIMING ', 1)[1])
            for stage, value in item['seconds'].items():
                add('python_'+stage, item['rank'], value)
    for rank, step, prepare_seconds, integrate_seconds, total in re.findall(
            r'ASTR_GPU_RANK_RK_TIMING (\d+) (\d+)\s+(\S+)\s+(\S+)\s+(\S+)', log):
        add('pure_rk', rank, total)
    for stage in ('solver_initialization', 'completed_window', 'solver_total_after_mpi'):
        if stage not in stages or any(len(v) != 1 for v in stages[stage].values()):
            raise AssertionError('Missing whole-window timing: '+stage)
    for stage in ('advance_inclusive', 'pure_rk', 'insitu_sample_inclusive'):
        if stage not in stages or any(len(v) != steps for v in stages[stage].values()):
            raise AssertionError('Incomplete step count: '+stage)
    return {stage:dict(seconds=max(map(sum, values.values())),
        per_rank_seconds={str(rank):sum(v) for rank,v in values.items()},
        per_rank_events={str(rank):len(v) for rank,v in values.items()},
        first_event_seconds=max(v[0] if v else 0 for v in values.values()),
        remaining_events_seconds=max(sum(v[1:]) for v in values.values()))
        for stage,values in stages.items()}


def verify(case, steps, enabled):
    from PIL import Image
    import numpy as np
    out = case/'outdat'
    prohibited = [p for p in out.rglob('*') if p.is_file() and
        (p.suffix in ('.h5','.vtp','.pvtp') or p.name=='COMPLETE' or 'checkpoint' in p.parts)]
    if prohibited:
        raise AssertionError('Unexpected field/checkpoint/geometry output: '+str(prohibited[:3]))
    log = (case/'run.log').read_text()
    if any(message in log for message in ('MPI_ABORT', 'COMPUTATION CRASHED', 'ASTR device products failed')):
        raise AssertionError('Solver did not complete successfully')
    if not enabled:
        assert not list(out.rglob('*.jpeg')) and not list(out.rglob('*.eps'))
        assert 'ASTR_INSITU_DEVICE_FRAME ' not in log
        return {}
    frames = re.findall(r'ASTR_INSITU_DEVICE_FRAME rank=(\d+) step=(\d+) transport=(\S+)', log)
    assert sorted((int(rank),int(step)) for rank,step,_ in frames) == [
        (rank,step) for rank in range(2) for step in range(1,steps+1)]
    downloads = re.findall(r'ASTR_INSITU_GPU_FLOW rank=(\d+) frame_downloads=(\d+)',log)
    assert sorted(downloads)==[('0','0'),('1','0')]
    assert len(list(out.rglob('*.jpeg'))) == 2*steps
    assert len(list(out.rglob('*.eps'))) == 2*steps
    resources = []
    for rank in range(2):
        render = out/'render'
        lifecycle = json.loads((render/f'lifecycle_rank{rank}.json').read_text())
        assert lifecycle['finalized'] and [item[0] for item in lifecycle['frames']] == list(range(1,steps+1))
        with (render/f'resources.rank{rank:08d}.csv').open() as stream:
            identity = stream.readline().strip()
            rows = list(csv.DictReader(stream))
        assert sum(row['stage']=='frame_after' for row in rows)==steps
        resource = dict(rank=rank,identity=identity,
            host_increment_peak=max(int(row['host_increment_bytes']) for row in rows),
            device_increment_peak=max(int(row['device_increment_bytes']) for row in rows),
            device_free_min=min(int(row['device_free_bytes']) for row in rows))
        assert resource['host_increment_peak']<=16*1024**3
        assert resource['device_increment_peak']<=6*1024**3
        assert resource['device_free_min']>=2*1024**3
        resources.append(resource)
        for step in range(1,steps+1):
            record = json.loads((render/f'mesh_step{step:08d}_rank{rank}.json').read_text())
            assert record['processing_backend']=='device' and set(record['products'])==set(PRODUCTS)
            for product in record['products'].values():
                assert product['image']['status']=='published' and product['color_field']=='speed'
                assert product['color_range']==[0.,1.]
    for name in PRODUCTS:
        for step in range(1,steps+1):
            path=out/'render'/f'{name}.step{step:08d}.jpeg'
            assert path.with_suffix('.eps').is_file()
            with Image.open(path) as image:
                pixels = np.asarray(image.convert('RGB'))
                assert pixels.shape==(960,1280,3) and (pixels<240).any()
    return dict(resources=resources,images=2*steps,eps=2*steps)


def launch(args, name, steps, enabled):
    case = args.output/name
    prepare(case, steps, enabled, args.library, args.transport)
    print(f'START {name}: {steps} steps, visualization={enabled}', flush=True)
    started = perf_counter()
    with (case/'run.log').open('w') as stream:
        subprocess.run([str(args.mpiexec),'--mca','coll_hcoll_enable','0','-np','2',
            str(args.executable),'run','datin/input.tgv'],cwd=case,
            env=environment(case,args.transport),stdout=stream,stderr=subprocess.STDOUT,check=True)
    launch_seconds = perf_counter()-started
    result = dict(case=str(case),launch_seconds=launch_seconds,timing=timing(case,steps),
        checks=verify(case,steps,enabled))
    (case/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'PASS {name}: complete window {result["timing"]["completed_window"]["seconds"]:.6f} s',flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--executable',type=Path,default=ROOT/'build_insitu_gpu/bin/astr')
    parser.add_argument('--mpiexec',type=Path,default=MPI)
    parser.add_argument('--library',type=Path,default=LIBRARY)
    parser.add_argument('--transport',choices=('device-aware','pinned'),default='device-aware')
    parser.add_argument('--smoke-only',action='store_true')
    parser.add_argument('--skip-smoke',action='store_true',help='Use only after this exact executable passed the two-step check')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.executable = args.executable.resolve(strict=True)
    args.mpiexec = args.mpiexec.resolve(strict=True)
    args.library = args.library.resolve(strict=True)
    args.output.mkdir(parents=True,exist_ok=False)
    report = dict(status='running',grid=[256]*3,np=2,topology=[2,1,1],dt=1e-4,
        steps=100,products=list(PRODUCTS),q_threshold=0.,color='speed',image_resolution=[1280,960],
        statistics=False,checkpoint=False,volume=False,slices=False,vtk_geometry=False,
        postprocess_transport=args.transport,solver_transport='pinned',precision='fp64',filter_workspace='scalar',
        timing_definition='Max of rank-local elapsed whole windows; initialization excluded from completed_window, first-frame lazy setup included; nested stages are not summed',
        repetitions=1,provenance=dict(executable=str(args.executable),
            executable_sha256=hashlib.sha256(args.executable.read_bytes()).hexdigest(),
            pipeline_sha256=hashlib.sha256((ROOT/'scripts/insitu/tgv_pipeline.py').read_bytes()).hexdigest()))
    try:
        if not args.skip_smoke:
            report['smoke'] = launch(args,'smoke_on',2,True)
        if not args.smoke_only:
            report['on'] = launch(args,'visualization_on',100,True)
            report['off'] = launch(args,'visualization_off',100,False)
            for name in ('input.tgv','input.output','controller'):
                if (args.output/'visualization_on/datin'/name).read_bytes() != (args.output/'visualization_off/datin'/name).read_bytes():
                    raise AssertionError('Unmatched solver input: '+name)
            import numpy as np
            diagnostics = [np.loadtxt(args.output/f'visualization_{mode}/flowstate.dat',skiprows=1,ndmin=2)
                for mode in ('on','off')]
            if diagnostics[0].shape!=diagnostics[1].shape or not all(np.isfinite(v).all() for v in diagnostics):
                raise AssertionError('Statistics shape or finite-value check failed')
            np.testing.assert_array_equal(diagnostics[0][:,:2],diagnostics[1][:,:2])
            differences = np.max(abs(diagnostics[0][:,2:]-diagnostics[1][:,2:]),axis=0)
            if len(differences)!=3 or np.any(differences>2e-10):
                raise AssertionError('Visualization changed solver statistics: '+str(differences))
            report['statistics_comparison']=dict(samples=len(diagnostics[0]),
                maxabs=dict(zip(('kinetic_energy','enstrophy','dissipation'),differences.tolist())))
            on,off = (report[mode]['timing']['completed_window']['seconds'] for mode in ('on','off'))
            report['comparison'] = dict(on_seconds=on,off_seconds=off,extra_seconds=on-off,
                on_off_ratio=on/off,overhead_percent=(on/off-1)*100,extra_seconds_per_step=(on-off)/100)
        report['status']='passed-bounded-device-demonstration' if args.smoke_only else 'passed-matched-pair'
    except BaseException as error:
        report.update(status='failed',error=str(error))
        raise
    finally:
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
