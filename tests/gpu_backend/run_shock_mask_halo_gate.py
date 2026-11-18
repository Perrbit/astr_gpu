"""Check byte face halos, local periodic wrap and physical ends on actual GPUs."""
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
    parser.add_argument('--transport',choices=['pinned','device'],default='pinned')
    parser.add_argument('--sanitizer',choices=['memcheck','initcheck'])
    parser.add_argument('--precision',choices=['fp64','fp32'],default='fp64')
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    shutil.copy2(args.binary,root/'tested_probe');shutil.copy2(__file__,root/'runner.py')
    env=os.environ|dict(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0')
    if args.transport=='device':env.update(OMPI_MCA_pml='ucx',OMPI_MCA_opal_cuda_support='1')
    command=[str(args.mpiexec.resolve()),'--bind-to','none','-n',str(args.ranks)]
    if args.sanitizer:
        command+=['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer',
            '--tool',args.sanitizer,'--error-exitcode','99','--log-file',str(root/'sanitizer.%p.log')]
        if args.sanitizer=='memcheck':command+=['--leak-check','full']
    command += [str(root/'tested_probe'),str('xyz'.index(args.axis)+1),args.transport]
    with (root/'run.log').open('w') as log:
        subprocess.run(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
    rows=re.findall(r'CPP_MASK_HALO_PASS\s+(\d+)\s+(\d+)\s+(\d+)',(root/'run.log').read_text())
    if sorted(int(r) for r,n,c in rows)!=list(range(args.ranks)) or any(int(n)!=8 for r,n,c in rows):
        raise AssertionError('missing per-rank mask cases')
    widths=re.findall(r'CPP_MASK_ELEMENT_BYTES\s+(\d+)\s+(\d+)',(root/'run.log').read_text())
    width=4 if args.precision=='fp32' else 8
    if sorted((int(r),int(w)) for r,w in widths)!=[(r,width) for r in range(args.ranks)]:
        raise AssertionError('native raw-sensor precision differs')
    if args.sanitizer:
        logs=list(root.glob('sanitizer.*.log'))
        if len(logs)!=args.ranks or any('ERROR SUMMARY: 0 errors' not in p.read_text() or
                args.sanitizer=='memcheck' and 'LEAK SUMMARY: 0 bytes leaked' not in p.read_text() for p in logs):
            raise AssertionError('missing clean per-rank tool logs')
    report=dict(ranks=args.ranks,axis=args.axis,transport=args.transport,sanitizer=args.sanitizer,
        cases_per_rank=8,physical_masks_exact=True,all_six_faces_exact=True,changed_mask_refresh=True,
        interface_flags_exact=True,changed_interface_flags=sum(int(c) for r,n,c in rows),precision=args.precision,
        binary_sha256=hashlib.sha256((root/'tested_probe').read_bytes()).hexdigest())
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
