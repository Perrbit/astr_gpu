"""Compare current-context native mapping with independent CUDA/EGL enumeration."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe',type=Path,required=True)
    parser.add_argument('--mpiexec',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    report={'status':'running','checks':[]}
    try:
        for mask,ranks in ((None,1),(None,2),('1,0',2),('1',2)):
            env=dict(os.environ)
            if mask is not None:
                env['CUDA_VISIBLE_DEVICES']=mask
            expected=json.loads(subprocess.check_output(
                [sys.executable,str(Path(__file__).with_name('insitu_device_identity.py'))],
                env=env,text=True,timeout=30))
            command=[str(args.mpiexec.resolve()),'--mca','coll_hcoll_enable','0',
                     '-np',str(ranks),str(args.probe.resolve())]
            result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=30)
            label=f'np{ranks}_mask{mask or "inherited"}'
            (args.output/(label+'.log')).write_text(result.stdout+result.stderr)
            result.check_returncode()
            actual=re.findall(r'MAPPING rank=(\d+) egl=(\d+) uuid=([0-9a-f-]{36})',result.stdout)
            assert len(actual)==ranks,result.stdout
            assert {int(row[0]) for row in actual}==set(range(ranks))
            for rank,index,uuid in actual:
                want=expected[int(rank)%len(expected)]
                assert int(index)==want['egl_index'] and uuid==want['uuid'],(actual,expected)
            report['checks'].append(dict(mask=mask,np=ranks,mapping=actual))
        result=subprocess.run([str(args.mpiexec.resolve()),'--mca','coll_hcoll_enable','0',
                               '-np','1',str(args.probe.resolve()),'no-context'],
                              capture_output=True,text=True,timeout=30)
        (args.output/'no_context.log').write_text(result.stdout+result.stderr)
        result.check_returncode()
        assert 'PASS no-context rejection rank=0' in result.stdout
        report['checks'].append({'no_context':'rejected'})
        report['status']='passed-current-context-mapping-not-render-context-validation'
    except BaseException as error:
        report.update(status='failed',error=str(error))
        raise
    finally:
        (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
