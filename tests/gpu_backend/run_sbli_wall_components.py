"""Small analytic MPI/ownership/memory checks for the shared SBLI wall observer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','mpiexec','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    binary=out/'frozen_component';shutil.copy2(a.binary,binary);binary.chmod(0o555)
    report=dict(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),cases=[])
    for ranks,mode,sanitizer in [(1,'pinned',''),(2,'pinned',''),(2,'device-aware',''),
        (2,'pinned','memcheck'),(1,'pinned','initcheck')]:
        name=f'np{ranks}_{mode}'+('_'+sanitizer if sanitizer else '')
        env=os.environ|dict(OMPI_MCA_pml='ucx' if mode=='device-aware' else 'ob1',
            OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',OMPI_MCA_coll='^cuda,hcoll,ucc',
            OMPI_MCA_opal_cuda_support='1' if mode=='device-aware' else '0')
        command=[str(a.mpiexec.resolve()),'-n',str(ranks)]
        if sanitizer:command+=['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer',
            '--tool',sanitizer,'--error-exitcode','99','--log-file',str(out/(name+'.%p.sanitizer'))]
        command +=[str(binary),mode]
        with (out/(name+'.log')).open('x') as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        text=(out/(name+'.log')).read_text()
        if 'CPP_SBLI_WALL_COMPONENT_PASS' not in text:raise AssertionError('missing analytic result')
        if sanitizer:
            logs=list(out.glob(name+'.*.sanitizer'))
            if len(logs)!=ranks or any('ERROR SUMMARY: 0 errors' not in p.read_text() for p in logs):
                raise AssertionError('missing clean component memory evidence')
        report['cases'].append(dict(ranks=ranks,transport=mode,sanitizer=sanitizer,maximum_scaled=float(text.split()[-1])))
        print(name,text.split()[-1],flush=True)
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
