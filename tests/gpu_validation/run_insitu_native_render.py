"""Formal TGV render with no test environment or conventional field files."""
import argparse
import csv
import json
import os
import re
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image


def check_native_resources(out,ranks,frames=3):
    records=[]
    expected=['baseline','initialized']+['frame_before','frame_after']*frames+[
        'finalize_before','finalize_after','statistics_exported','session_released']
    for rank in range(ranks):
        with (out/f'resources.rank{rank:08d}.csv').open() as stream:
            identity=stream.readline().strip()
            rows=list(csv.DictReader(stream))
        assert identity.startswith('# device=GPU-'),identity
        assert [row['stage'] for row in rows]==expected,rows
        for row in rows:
            assert int(row['host_increment_bytes'])<=4*1024**3,row
            assert int(row['device_increment_bytes'])<=2*1024**3,row
            assert int(row['device_free_bytes'])>=1024**3,row
        assert int(rows[0]['host_increment_bytes'])==0
        assert int(rows[0]['device_increment_bytes'])==0
        records.append(dict(rank=rank,identity=identity,
            host_increment_max=max(int(row['host_increment_bytes']) for row in rows),
            device_increment_max=max(int(row['device_increment_bytes']) for row in rows)))
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('gpu','mpiexec','backend','reference','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--time-schedule',action='store_true')
    p.add_argument('--observe-resources',action='store_true')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[2]
    report={'status':'running','checks':[]}
    try:
        for ranks in (1,2):
            case=(args.output/f'np{ranks}_continuous').resolve()
            reference=args.reference/f'np{ranks}_continuous'
            case.mkdir()
            shutil.copytree(reference/'datin',case/'datin')
            config=case/'insitu.nml'
            schedule="schedule_mode='time', time_interval=0.002" if args.time_schedule else \
                     "schedule_mode='steps', step_interval=2"
            config.write_text('&insitu_run enabled=t, statistics=t, render=t,\n'
                              "output_directory='outdat', statistics_window=0.0005,0.0035,\n"
                              f"{schedule}, initial_frame=t, final_frame=t,\n"
                              'host_budget_bytes=4294967296, device_budget_bytes=2147483648,\n'
                              'device_reserve_bytes=1073741824,\n'
                              f"implementation_path='{args.backend.resolve()}',\n"
                              f"pipeline_file='{root/'scripts/insitu/tgv_pipeline.py'}'\n/\n")
            env={k:v for k,v in os.environ.items() if not k.startswith('ASTR_')}
            for key in ('DISPLAY','PYTHONPATH','CATALYST_IMPLEMENTATION_PREFER_ENV','VTK_EGL_DEVICE_INDEX'):
                env.pop(key,None)
            env.update(ASTR_INSITU_CONFIG=str(config),ASTR_GPU_SYNC_MODE='explicit',
                       ASTR_GPU_HALO_TRANSPORT='pinned',ASTR_GPU_PRECISION_MODE='fp64',
                       ASTR_FORCE_MPI_TOPOLOGY=f'{ranks},1,1',ASTR_GPU_BENCHMARK_NO_FIELD_IO='1',
                       ASTR_GPU_RK_TIMING='1')
            command=[str(args.mpiexec.resolve()),'--mca','coll_hcoll_enable','0',
                     '-np',str(ranks),str(args.gpu.resolve()),'run','datin/input.tgv']
            baseline=None
            if args.observe_resources:
                from insitu_resource_monitor import run_monitored
                off=(args.output/f'np{ranks}_off').resolve()
                off.mkdir()
                shutil.copytree(case/'datin',off/'datin')
                off_env=dict(env)
                off_env.pop('ASTR_INSITU_CONFIG')
                with (off/'run.log').open('w') as log:
                    baseline=run_monitored(command,off,off_env,log,off/'resources.json')
            with (case/'run.log').open('w') as log:
                if args.observe_resources:
                    resources=run_monitored(command,case,env,log,case/'resources.json',baseline)
                    report['checks'].append(dict(np=ranks,sampled_resources=resources))
                else:
                    subprocess.run(command,cwd=case,env=env,stdout=log,
                                   stderr=subprocess.STDOUT,check=True,timeout=240)
            out=case/'outdat'
            transfers=re.findall(r'ASTR_INSITU_GPU_STATS rank=(\d+) samples=(\d+) full_output_downloads=(\d+)',
                                 (case/'run.log').read_text())
            assert sorted(int(rank) for rank,_,_ in transfers)==list(range(ranks)),transfers
            assert all(int(samples)==5 and int(downloads)==1 for _,samples,downloads in transfers),transfers
            flows=re.findall(r'ASTR_INSITU_GPU_FLOW rank=(\d+) frame_downloads=(\d+)',(case/'run.log').read_text())
            assert len(flows)==ranks and all(int(n)==3 for _,n in flows),flows
            report['checks'].append(dict(np=ranks,device_statistics_full_output_downloads=1,
                                         flow_downloads_per_rank=3,
                                         samples_per_rank=5))
            report['checks'].append(dict(np=ranks,native_resources=check_native_resources(out,ranks)))
            assert not list(out.glob('*.h5')) and not list(out.glob('restart*'))
            assert not list(out.glob('sample.step*.bin')) and not list(out.glob('sample.schedule*.txt'))
            for rank in range(ranks):
                state=json.loads((out/f'lifecycle_rank{rank}.json').read_text())
                assert state=={'frames':[[0,0.0],[2,0.002],[4,0.004]],'finalized':True},state
            images=list(out.glob('*.jpeg'))
            assert len(images)==16,len(images)
            for picture in images:
                assert picture.with_suffix('.eps').is_file()
                with Image.open(picture) as a,Image.open(reference/'outdat'/picture.name) as b:
                    np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
            geometry=list(out.rglob('*.vtp'))+list(out.glob('*.pvtp'))
            assert geometry
            for path in geometry:
                assert path.read_bytes()==(reference/'outdat'/path.relative_to(out)).read_bytes(),path
            report['checks'].append(dict(np=ranks,images='pixel-identical',geometry='byte-identical',
                                         test_snapshots='absent',conventional_field_io='absent'))
        report['status']='passed-native-render-not-full-resource-admission'
    except BaseException as error:
        report.update(status='failed',error=str(error))
        raise
    finally:
        (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
