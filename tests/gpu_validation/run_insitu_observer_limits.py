"""Exercise native observed resource refusal, including two ranks on one GPU."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('gpu','mpiexec','reference','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    MiB=1024**2
    cases=[('host',1,'0',128*MiB,2048*MiB,1024*MiB,'node host increment'),
           ('device',1,'0',4096*MiB,64*MiB,1024*MiB,'physical GPU increment'),
           ('shared_device',2,'0',4096*MiB,160*MiB,1024*MiB,'physical GPU increment'),
           ('reserve',1,'0',4096*MiB,2048*MiB,1024**4,'free memory below reserve')]
    report={'status':'running','checks':[]}
    try:
        for name,ranks,visible,host,device,reserve,reason in cases:
            case=(args.output/name).resolve()
            source=args.reference/f'np{ranks}_continuous'
            case.mkdir()
            shutil.copytree(source/'datin',case/'datin')
            config=(source/'insitu.nml').read_text().replace('host_budget_bytes=4294967296',f'host_budget_bytes={host}')
            config=config.replace('device_budget_bytes=2147483648',f'device_budget_bytes={device}')
            config=config.replace('device_reserve_bytes=1073741824',f'device_reserve_bytes={reserve}')
            (case/'insitu.nml').write_text(config)
            env={k:v for k,v in os.environ.items() if not k.startswith('ASTR_')}
            for key in ('DISPLAY','PYTHONPATH','CATALYST_IMPLEMENTATION_PREFER_ENV','VTK_EGL_DEVICE_INDEX'):
                env.pop(key,None)
            env.update(CUDA_VISIBLE_DEVICES=visible,ASTR_INSITU_CONFIG=str(case/'insitu.nml'),
                       ASTR_GPU_SYNC_MODE='explicit',ASTR_GPU_HALO_TRANSPORT='pinned',
                       ASTR_GPU_PRECISION_MODE='fp64',ASTR_FORCE_MPI_TOPOLOGY=f'{ranks},1,1',
                       ASTR_GPU_BENCHMARK_NO_FIELD_IO='1',ASTR_GPU_RK_TIMING='1')
            command=[str(args.mpiexec.resolve()),'--mca','coll_hcoll_enable','0',
                     '-np',str(ranks),str(args.gpu.resolve()),'run','datin/input.tgv']
            with (case/'run.log').open('w') as log:
                process=subprocess.Popen(command,cwd=case,env=env,stdout=log,
                                         stderr=subprocess.STDOUT,start_new_session=True)
                try:
                    code=process.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL)
                    process.wait()
                    raise RuntimeError('Native resource refusal did not terminate MPI')
            text=(case/'run.log').read_text()
            assert code!=0 and 'ASTR INSITU RESOURCE OBSERVER ERROR:' in text and reason in text,text[-3000:]
            assert 'The job is done!' not in text
            observations=[]
            for path in sorted((case/'outdat').glob('resources.rank*.csv')):
                with path.open() as stream:
                    identity=stream.readline().strip()
                    rows=list(csv.DictReader(stream))
                observations.append(dict(rank_file=path.name,identity=identity,last=rows[-1]))
            assert len(observations)==ranks
            report['checks'].append(dict(case=name,np=ranks,reason=reason,observations=observations))
            print('PASS:',name,flush=True)
        report['status']='passed-native-observed-resource-refusal'
    except BaseException as error:
        report.update(status='failed',error=str(error))
        raise
    finally:
        (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
