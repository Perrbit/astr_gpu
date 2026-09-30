"""Reject bridge vector allocation on one rank without exhausting real memory."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('gpu','mpiexec','interposer','reference','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    report={'status':'running','checks':[]}
    try:
        for ranks in (1,2):
            case=(args.output/f'np{ranks}').resolve()
            source=args.reference/f'np{ranks}_continuous'
            case.mkdir()
            shutil.copytree(source/'datin',case/'datin')
            shutil.copyfile(source/'insitu.nml',case/'insitu.nml')
            env={k:v for k,v in os.environ.items() if not k.startswith('ASTR_')}
            for key in ('DISPLAY','PYTHONPATH','CATALYST_IMPLEMENTATION_PREFER_ENV','VTK_EGL_DEVICE_INDEX'):
                env.pop(key,None)
            size=(32//ranks+1)*33*33*3*8
            env.update(ASTR_INSITU_CONFIG=str(case/'insitu.nml'),
                       ASTR_GPU_SYNC_MODE='explicit',ASTR_GPU_HALO_TRANSPORT='pinned',
                       ASTR_GPU_PRECISION_MODE='fp64',ASTR_FORCE_MPI_TOPOLOGY=f'{ranks},1,1',
                       ASTR_GPU_BENCHMARK_NO_FIELD_IO='1',ASTR_GPU_RK_TIMING='1',
                       ASTR_TEST_NEW_BYTES=str(size),
                       ASTR_TEST_NEW_RANK=str(ranks-1))
            command=[str(args.mpiexec.resolve()),'--mca','coll_hcoll_enable','0','-np',str(ranks),
                     'env','LD_PRELOAD='+str(args.interposer.resolve()),str(args.gpu.resolve()),
                     'run','datin/input.tgv']
            with (case/'run.log').open('w') as log:
                process=subprocess.Popen(command,cwd=case,env=env,stdout=log,
                                         stderr=subprocess.STDOUT,start_new_session=True)
                try:
                    code=process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL)
                    process.wait()
                    raise RuntimeError('MPI allocation failure did not terminate')
            text=(case/'run.log').read_text()
            if code==0 or f'rank={ranks-1} bytes={size}' not in text or \
                    'ASTR INSITU BRIDGE ERROR: execute: std::bad_alloc' not in text:
                raise RuntimeError('Failure did not reach the expected bridge exception handler')
            if list((case/'outdat').glob('*.jpeg')) or list((case/'outdat').glob('*.pvtp')):
                raise RuntimeError('Allocation failure emitted partial frame products')
            report['checks'].append(dict(np=ranks,failed_rank=ranks-1,bytes=size,
                                         result='reported-and-terminated-before-frame'))
        report['status']='passed-injected-bridge-allocation-failure'
    except BaseException as error:
        report.update(status='failed',error=str(error))
        raise
    finally:
        (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
