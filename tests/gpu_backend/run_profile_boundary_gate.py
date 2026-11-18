"""Static inlet, pressure outlet and top-face commands with unchanged nonowner nodes."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess


def main():
    parser=argparse.ArgumentParser()
    for name in ('binary','reference','mpiexec','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--ranks',type=int,choices=[1,2],default=1)
    parser.add_argument('--axis',choices=['x','y','z'],default='x')
    parser.add_argument('--sanitizer',choices=['memcheck','initcheck'])
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    command=[str(args.mpiexec.resolve()),'--bind-to','none','-n',str(args.ranks)]
    if args.sanitizer:
        command+=['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer',
            '--tool',args.sanitizer,'--error-exitcode','99','--log-file',str(root/'sanitizer.%p.log')]
        if args.sanitizer=='memcheck':command+=['--leak-check','full']
    command += [str(args.binary.resolve()),str(args.reference.resolve()),str('xyz'.index(args.axis)+1)]
    env=os.environ|dict(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0')
    with (root/'run.log').open('w') as log:
        subprocess.run(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
    rows={}
    for tag in ('BOUNDARY','MACH'):
        found=re.findall(r'CPP_PROFILE_'+tag+r'_PASS\s+(\d+)\s+(\d+)\s+(\S+)',(root/'run.log').read_text())
        if sorted(int(r[0]) for r in found)!=list(range(args.ranks)):raise AssertionError('missing rank results')
        rows[tag.lower()]=[]
        for rank,count,error in found:
            rank=int(rank);inlet=not(args.ranks==2 and args.axis=='x' and rank==1)
            outlet=not(args.ranks==2 and args.axis=='x' and rank==0)
            top=not(args.ranks==2 and args.axis=='y' and rank==0)
            expected=12*(int(inlet)+int(top)+int(outlet)) if tag=='BOUNDARY' else 6*(int(inlet)+int(outlet))
            if int(count)!=expected:raise AssertionError('missing owned-face cases')
            rows[tag.lower()].append(dict(rank=rank,cases=int(count),max_error=float(error)))
    if args.sanitizer:
        logs=list(root.glob('sanitizer.*.log'))
        if len(logs)!=args.ranks or any('ERROR SUMMARY: 0 errors' not in p.read_text() for p in logs):
            raise AssertionError('missing clean per-rank tool results')
    report=dict(ranks=args.ranks,axis=args.axis,sanitizer=args.sanitizer,rows=rows,
        nonselected_words_identical=True,rhs_face_only_reset=True,plane_only_target=True)
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
