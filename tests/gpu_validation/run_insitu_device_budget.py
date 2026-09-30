"""Exercise real GPU statistics allocation admission, including shared GPUs."""
import argparse
import json
import os
import re
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('gpu', 'mpiexec', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    env = {k:v for k,v in os.environ.items() if not k.startswith('ASTR_')}
    env.update(ASTR_GPU_SYNC_MODE='explicit', ASTR_GPU_HALO_TRANSPORT='pinned',
               ASTR_GPU_PRECISION_MODE='fp64', ASTR_INSITU_SAMPLE_PREFIX='outdat/sample',
               ASTR_INSITU_TEST_INITIAL='1', ASTR_INSITU_TEST_STATISTICS_WINDOW='0 0.001')
    cases = [('np1_pass',1,'0',2*1024**3,1024**3,True),
             ('np2_split_pass',2,'0,1',12*1024**2,1024**3,True),
             ('np2_shared_reject',2,'0',12*1024**2,1024**3,False),
             ('np1_headroom_reject',1,'0',2*1024**3,1024**4,False),
             ('np2_host_reject',2,'0,1',2*1024**3,1024**3,False)]
    report = {'status':'running','checks':[]}
    try:
        for name, ranks, visible, limit, headroom, success in cases:
            case = args.output/name
            subprocess.run([sys.executable,str(root/'tests/gpu_validation/prepare_tgv_case.py'),
                            '--src-case',str(root/'examples/Taylor_Green_Vortex'),
                            '--dst-case',str(case),'--use-gpu','t','--grid','32,32,32',
                            '--maxstep','0','--feqchkpt','2','--deltat','1.d-3',
                            '--lfilter','t','--diffterm','t','--scheme','643e'],check=True,timeout=30,
                           stdout=subprocess.DEVNULL)
            runtime=dict(env,CUDA_VISIBLE_DEVICES=visible,ASTR_FORCE_MPI_TOPOLOGY=f'{ranks},1,1',
                         ASTR_INSITU_TEST_DEVICE_BUDGET=f'{limit} {headroom}',
                         ASTR_INSITU_TEST_HOST_BUDGET=str(32*1024**2 if name=='np2_host_reject' else 4*1024**3))
            with (case/'run.log').open('w') as log:
                result=subprocess.run([str(args.mpiexec.resolve()),'--mca','coll_hcoll_enable','0',
                                       '-np',str(ranks),str(args.gpu.resolve()),'run','datin/input.tgv'],
                                      cwd=case,env=runtime,stdout=log,stderr=subprocess.STDOUT,timeout=60)
            output=(case/'run.log').read_text()
            if success:
                if result.returncode or 'The job is done!' not in output:
                    raise ValueError(f'{name}: expected completion')
                if not list((case/'outdat').glob('sample.device_statistics.step*.bin')):
                    raise ValueError(f'{name}: no device statistics')
            else:
                message='statistics device allocation exceeds budget or free-memory headroom'
                if name=='np2_host_reject':
                    message='statistics host allocation exceeds node budget'
                    demands=re.findall(r'INSITU HOST BUDGET local/node/limit:\s+(\d+)\s+(\d+)\s+(\d+)',output)
                    if len(demands)!=2 or not all(int(a)<int(c)<int(b) for a,b,c in demands):
                        raise ValueError('Host refusal must be caused by node aggregate, not per-rank demand')
                if not result.returncode or message not in output:
                    raise ValueError(f'{name}: expected allocation refusal')
                if list((case/'outdat').glob('sample.device_statistics.step*.bin')):
                    raise ValueError(f'{name}: statistics produced after refused admission')
            report['checks'].append(dict(case=name,passed=True,allocation_admitted=success))
            print('PASS:',name,flush=True)
        report['status']='passed-statistics-controlled-allocation-only'
    except BaseException as error:
        report.update(status='failed',error=str(error))
        raise
    finally:
        (args.output/'summary.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
