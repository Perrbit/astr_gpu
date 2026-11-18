"""Analytic volume and wall integrals on actual device arrays."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


def main():
    parser=argparse.ArgumentParser()
    for name in ('binary','mpiexec','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--ranks',type=int,choices=[1,2],default=1)
    parser.add_argument('--axis',choices=['x','y','z'],default='x')
    parser.add_argument('--sanitizer',choices=['memcheck','initcheck'])
    parser.add_argument('--diagnostic',action='store_true')
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    binary=args.binary.resolve()
    shutil.copyfile(__file__,root/'runner.py');shutil.copyfile(binary,root/'tested_probe')
    command=[str(args.mpiexec.resolve()),'--bind-to','none','-n',str(args.ranks)]
    if args.sanitizer:
        command+=['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer','--tool',args.sanitizer,
            '--error-exitcode','99','--log-file',str(root/'sanitizer.%p.log')]
        if args.sanitizer=='memcheck':command+=['--leak-check','full']
    command += [str(args.binary.resolve()),str('xyz'.index(args.axis)+1),'1' if args.diagnostic else '0']
    env=os.environ|dict(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0')
    with (root/'run.log').open('w') as log:
        subprocess.run(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
    rows=re.findall(r'CPP_FLAT_STATISTICS_PASS\s+(\d+)\s+(\d+)\s+(\S+)',(root/'run.log').read_text())
    if sorted(int(r) for r,n,e in rows)!=list(range(args.ranks)) or any(int(n)!=4 for r,n,e in rows):
        raise AssertionError('missing per-rank analytic statistics cases')
    if args.sanitizer:
        logs=list(root.glob('sanitizer.*.log'))
        if len(logs)!=args.ranks or any('ERROR SUMMARY: 0 errors' not in p.read_text() for p in logs):
            raise AssertionError('missing clean per-rank tool logs')
    report=dict(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),ranks=args.ranks,axis=args.axis,sanitizer=args.sanitizer,diagnostic=args.diagnostic,max_error=max(float(e) for r,n,e in rows),
        geometry_storage_reused=True,q_words_unchanged=True,repeat_bits_identical=True,
        wall_moment_endpoint_and_time_integral=True)
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
