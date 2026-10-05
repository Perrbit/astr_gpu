"""IS7 bounded matched observation, not a production performance benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
from PIL import Image

import h5py
import numpy as np

from test_insitu_curve_derivatives import arguments,ROOT,FINAL,GPU,CPU,MPI,LIBRARY
from test_output_insitu_restart import configuration,check_render
from run_output_restart_validation import run_case,compare_fields
from test_insitu_channel_statistics import read_statistics


def preset():
    return configuration(interval=2).replace('statistics_window=0.0005,0.0115',
        "statistics_window=0.003,0.004, derivative_backend='gpu'")


def read_timing(case,ranks):
    stages={}
    def add(stage,rank,value):
        if rank not in range(ranks) or not np.isfinite(value) or value<0:
            raise ValueError('Invalid timing duration/rank')
        stages.setdefault(stage,{r:[] for r in range(ranks)})[rank].append(value)
    log=(case/'run.log').read_text()
    for stage,step,rank,value in re.findall(
            r'ASTR_INSITU_STAGE_TIMING (\w+) (-?\d+) (\d+)\s+(\S+)',log):
        add(stage,int(rank),float(value))
    for line in log.splitlines():
        if 'ASTR_INSITU_PIPELINE_TIMING ' in line:
            data=json.loads(line.split('ASTR_INSITU_PIPELINE_TIMING ',1)[1])
            for stage,value in data['seconds'].items():
                add(stage,data['rank'],value)
    for rank,a,b in re.findall(r'ASTR_INSITU_TRANSFER_TIMING rank=(\d+) sync download_inclusive=\s*(\S+)\s+(\S+)',log):
        add('flow_sync',int(rank),float(a));add('flow_download_inclusive',int(rank),float(b))
    for rank,a,b in re.findall(r'ASTR_INSITU_GPU_DERIVATIVE_TIMING rank=(\d+) prepare_inclusive compute_download_inclusive=\s*(\S+)\s+(\S+)',log):
        add('derivative_prepare_inclusive',int(rank),float(a))
        add('derivative_compute_download_inclusive',int(rank),float(b))
    for rank,step,a,b in re.findall(r'ASTR_INSITU_BRIDGE_TIMING rank=(\d+) step=(\d+) copy=(\S+) execute_inclusive=(\S+)',log):
        add('bridge_copy',int(rank),float(a));add('catalyst_execute_inclusive',int(rank),float(b))
    # Max of each rank's accumulated duration, never sum ranks or sum nested phases.
    result={stage:dict(seconds=max(sum(values) for values in per_rank.values()),
        per_rank_seconds={str(r):sum(values) for r,values in per_rank.items()},
        per_rank_events={str(r):len(values) for r,values in per_rank.items()})
        for stage,per_rank in stages.items()}
    for stage in ('solver_initialization','solver_total_after_mpi','completed_window'):
        if stage not in result or set(result[stage]['per_rank_events'].values())!={1}:
            raise ValueError('Missing or duplicate whole-window timing: '+stage)
    if result.get('advance_inclusive',{}).get('per_rank_events')!={str(r):4 for r in range(ranks)}:
        raise ValueError('Timing window is not four complete steps')
    return result


def exact_authority(a,b,statistics=False,enable_changed=False):
    for name in ('state.h5',)+(('statistics.h5',) if statistics else ()):
        compare_fields(a/FINAL/name,b/FINAL/name)
    for name in ('control.bin',)+(('insitu_control.bin',) if statistics else ()):
        left,right=(a/FINAL/name).read_bytes(),(b/FINAL/name).read_bytes()
        if enable_changed and name=='control.bin':
            # ASTROC04 contract(12) intentionally records the new statistics provider.
            ca=np.frombuffer(left[8:120],dtype='<i8').copy()
            cb=np.frombuffer(right[8:120],dtype='<i8').copy()
            if ca[11]!=0 or cb[11]!=1:
                raise AssertionError('Unexpected statistics provider contract')
            ca[11]=cb[11]
            left=left[:8]+ca.tobytes()+left[120:]
        if left!=right:
            raise AssertionError('Timing changed authority: '+name)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume-checks',action='store_true',help='Reuse verified immutable matched pairs, then rerun affected correctness checks')
    args=parser.parse_args()
    args.output=args.output.resolve()
    if not args.resume_checks:args.output.mkdir(parents=True,exist_ok=False)
    report=dict(scope='local 32^3 TGV, four complete steps dt=1e-3, one matched pair per NP; no production extrapolation',
        status='running',pairs=[],checks=[],budget=dict(host_node=4*1024**3,
        device_gpu=2*1024**3,device_free=1024**3,directory=256*1024**2),
        aggregation='maximum of per-rank accumulated local durations; not rank sums or sum of phase maxima',
        nesting='solver_total includes initialization, completed_window and finalize; sample includes statistics, frame transfer and bridge; bridge includes lazy Catalyst initialization and execute; Python execute includes all Python phases; derivative_compute_download combines GPU work, explicit sync, packing and D2H; screenshot_capture can re-render; writer can include lazy pipeline work',
        initialization='solver_initialization begins after MPI init and timing setup; pre-MPI CUDA bind/MPI init excluded, mpiexec launch wall includes them; first-frame Catalyst and view setup are inside completed_window',
        provenance={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (GPU,CPU,ROOT/'scripts/insitu/tgv_pipeline.py',ROOT/'src/benchmark_runtime.F90')})
    if args.resume_checks:
        previous=json.loads((args.output/'report.json').read_text())
        if previous['provenance']!=report['provenance'] or [p['np'] for p in previous['pairs']]!=[1,2]:
            raise ValueError('Cannot reuse unmatched executable/pipeline evidence')
        for ranks in (1,2):
            off=args.output/f'gpu_np{ranks}_timing_off';on=args.output/f'gpu_np{ranks}_timing_on'
            exact_authority(off,on,enable_changed=True)
            check_render(on,ranks,[2,4])
            for mode,case in (('off',off),('on',on)):
                if read_timing(case,ranks)!=previous['pairs'][ranks-1]['timing'][mode]:
                    raise ValueError('Timing evidence changed')
        prefix='failed-check' if previous['status']=='failed' else 'before-refresh'
        receipt=args.output/f'report.{prefix}-{len(list(args.output.glob("report."+prefix+"-*.json"))):02d}.json'
        shutil.copyfile(args.output/'report.json',receipt)
        report['pairs']=previous['pairs']
        report['reuse']='Matched pairs unchanged; fixed relative restore path in validation driver; no performance rerun'
    try:
        for ranks in (() if args.resume_checks else (1,2)):
            local=arguments(args.output,'gpu','x')
            config=preset()
            off,size_off=run_case(local,ROOT,'gpu',ranks,'timing_off',4,grid='32,32,32',
                checkpoint_interval=1,insitu_config=config.replace('enabled=t','enabled=f',1),
                insitu_timing=True,monitor_resources=True)
            on,size_on=run_case(local,ROOT,'gpu',ranks,'timing_on',4,grid='32,32,32',
                checkpoint_interval=1,insitu_config=config,insitu_timing=True,
                monitor_resources=True,resource_baseline=json.loads((off/'resources.sampled.json').read_text()))
            # Inputs and output schedules must match except the new enable flag.
            for name in ('input.tgv','input.output','controller'):
                if (off/'datin'/name).read_bytes()!=(on/'datin'/name).read_bytes():
                    raise AssertionError('Unmatched physics/output input: '+name)
            exact_authority(off,on,enable_changed=True)
            check_render(on,ranks,[2,4])
            for step,expected in ((2,4),(4,6)):
                for rank in range(ranks):
                    products=json.loads((on/f'outdat/render/mesh_step{step:08d}_rank{rank}.json').read_text())['products']
                    if len(products)!=expected:
                        raise AssertionError('Missing agreed products')
                    if any(p['image']['status']!='published' for p in products.values()):
                        raise AssertionError('Missing agreed image')
            times={mode:read_timing(case,ranks) for mode,case in (('off',off),('on',on))}
            required=('statistics_inclusive','flow_sync','flow_download_inclusive',
                'derivative_prepare_inclusive','derivative_compute_download_inclusive','bridge_copy',
                'catalyst_initialization_inclusive','extraction','render_inclusive',
                'geometry_write_inclusive','screenshot_capture_inclusive','image_encoding','image_publication')
            required+=('diagnostic_compute_pack_sync','diagnostic_d2h')
            if not set(required)<=times['on'].keys():
                raise AssertionError('Missing phase attribution: '+str(set(required)-times['on'].keys()))
            report['pairs'].append(dict(np=ranks,topology=[ranks,1,1],timing=times,
                launch_seconds={mode:json.loads((case/'timing.launch.json').read_text())['seconds']
                    for mode,case in (('off',off),('on',on))},
                directory_bytes=dict(off=size_off,on=size_on),resources=json.loads((on/'resources.sampled.json').read_text()),
                complete_window_extra_seconds=times['on']['completed_window']['seconds']-times['off']['completed_window']['seconds']))
            (args.output/'report.json').write_text(json.dumps(report,indent=2))
        local=arguments(args.output,'gpu','x')
        on=args.output/'gpu_np2_timing_on'
        def check_case(local,backend,name,config,**kwargs):
            case=args.output/f'{backend}_np2_{name}'
            if args.resume_checks and case.is_dir():
                if not (case/FINAL/'COMPLETE').is_file() or (case/'datin/input.insitu').read_text()!=config:
                    raise ValueError('Existing correctness artifact is incomplete or changed: '+str(case))
                if 'MPI_ABORT' in (case/'run.log').read_text():
                    raise ValueError('Cannot reuse failed solver artifact')
                if kwargs.get('memcheck'):
                    counts=[int(v) for p in case.glob('memcheck.*.log')
                        for v in re.findall(r'ERROR SUMMARY: (\d+) errors',p.read_text())]
                    if len(counts)<2 or any(counts):raise ValueError('Missing clean immutable memcheck evidence')
                return case,0
            return run_case(local,ROOT,backend,2,name,4,grid='32,32,32',checkpoint_interval=1,insitu_config=config,**kwargs)
        untimed,_=check_case(local,'gpu','observer_check',preset())
        exact_authority(on,untimed,statistics=True)
        check_render(untimed,2,[2,4])
        resumed,_=check_case(local,'gpu','restart_check',preset(),
            insitu_timing=True,restore=on/'outdat/new/checkpoints/step000000000003')
        exact_authority(on,resumed,statistics=True)
        check_render(resumed,2,[4])
        for rank in range(2):
            name=f'outdat/render/sample.statistics.step00000004.rank{rank:08d}.bin'
            if not (on/name).read_bytes()==(untimed/name).read_bytes()==(resumed/name).read_bytes():
                raise AssertionError('Observer changed exported statistics')
        for path in (resumed/'outdat/render').rglob('*step00000004*.vtp'):
            if path.read_bytes()!=(on/'outdat/render'/path.relative_to(resumed/'outdat/render')).read_bytes():
                raise AssertionError('Restart changed extracted geometry')
        for path in (on/'outdat/render').glob('*.jpeg'):
            for other in (untimed,resumed) if '00000004' in path.name else (untimed,):
                with Image.open(path) as a,Image.open(other/'outdat/render'/path.name) as b:
                    np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
        report['checks']+=['matched off/on field exact NP1/2; control differs only in defined new statistics-provider flag',
            'GPU timed/untimed authority and statistics exact NP2',
            'GPU step3+1 paired restart authority, statistics and geometry exact NP2',
            'all ten agreed product frames, JPEG/EPS/VTK and clock/UUID verified NP1/2',
            'matched external 20ms and native phase budget observers NP1/2']
        cpu=arguments(args.output,'cpu','x')
        config=preset().replace('render=.true.','render=.false.').replace("derivative_backend='gpu'","derivative_backend='cpu'")
        timed,_=check_case(cpu,'cpu','timing_statistics',config,insitu_timing=True)
        plain,_=check_case(cpu,'cpu','untimed_statistics',config)
        exact_authority(timed,plain,statistics=True)
        cpu_restart,_=check_case(cpu,'cpu','restart_statistics',config,
            insitu_timing=True,restore=timed/'outdat/new/checkpoints/step000000000003')
        exact_authority(timed,cpu_restart,statistics=True)
        for name in ('state.h5','statistics.h5'):
            with h5py.File(timed/FINAL/name) as a,h5py.File(on/FINAL/name) as b:
                for key in a:
                    if key.startswith('q'):
                        selection=(slice(0,32),)*3 if name=='statistics.h5' else (...,)
                        # GPU statistics omit periodic duplicate nodes; CPU keeps them.
                        np.testing.assert_allclose(a[key][selection],b[key][selection],rtol=0,atol=2e-10)
        for rank in range(2):
            name=f'outdat/render/sample.statistics.step00000004.rank{rank:08d}.bin'
            ch,cm,cf=read_statistics(timed/name);gh,gm,gf=read_statistics(on/name)
            np.testing.assert_array_equal(ch,gh)
            np.testing.assert_allclose(cm,gm,rtol=0,atol=2e-10)
            np.testing.assert_allclose(cf,gf,rtol=0,atol=2e-10)
        report['checks'].append('CPU timed/untimed and step3+1 restart exact; CPU/GPU fields and statistics <=2e-10 NP2')
        check_case(local,'gpu','memcheck',preset(),insitu_timing=True,memcheck=True)
        report['checks'].append('GPU all-product timed NP2 memcheck zero errors')
        report['status']='passed-bounded-IS7-observation-not-production-certification'
    except BaseException as error:
        report.update(status='failed',error=str(error))
        raise
    finally:
        (args.output/'report.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
