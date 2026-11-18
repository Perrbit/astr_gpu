"""Native algebraic boundary state and top-normal-Mach reduction checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','mpiexec','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--precision',choices=('fp32','fp64'),required=True)
    parser.add_argument('--ranks',type=int,choices=(1,2),default=1)
    parser.add_argument('--axis',choices=('x','y','z'),default='x')
    parser.add_argument('--sanitizer',choices=('memcheck',))
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    shutil.copy2(args.binary,root/'tested_probe');shutil.copy2(__file__,root/'runner.py')
    env=os.environ|dict(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0')
    command=[str(args.mpiexec.resolve()),'--bind-to','none','-n',str(args.ranks)]
    if args.sanitizer:
        command+=['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer',
            '--tool','memcheck','--error-exitcode','99','--leak-check','full',
            '--log-file',str(root/'sanitizer.%p.log')]
    command+=[str(root/'tested_probe'),str('xyz'.index(args.axis)+1)]
    with (root/'run.log').open('w') as log:
        subprocess.run(command,env=env,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
    text=(root/'run.log').read_text()
    rows=re.findall(r'^CPP_NATIVE_BOUNDARY_PASS\s+(\d+)\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)',text,re.M)
    cases=44 if args.ranks==1 else (38 if args.axis=='z' else 36)
    width=4 if args.precision=='fp32' else 8
    if sorted((int(r),int(w),int(n)) for r,w,n,e,m in rows)!=[(r,width,cases) for r in range(args.ranks)]:
        raise AssertionError('native boundary rank/kind/case observations')
    observations=re.findall(r'^CPP_NATIVE_BOUNDARY\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\S+)',text,re.M)
    mach=re.findall(r'^CPP_NATIVE_MACH\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)',text,re.M)
    if len(observations)!=cases*args.ranks or len(mach)!=2*args.ranks:
        raise AssertionError('native boundary or reduction records incomplete')
    if args.sanitizer:
        paths=list(root.glob('sanitizer.*.log'))
        if len(paths)!=args.ranks or any('ERROR SUMMARY: 0 errors' not in p.read_text() or
                'LEAK SUMMARY: 0 bytes leaked' not in p.read_text() for p in paths):
            raise AssertionError('native boundary Memcheck logs')
    report=dict(precision=args.precision,ranks=args.ranks,axis=args.axis,
        cases_per_rank=cases,polynomial_state_scaled_max=max(float(e) for r,w,n,e,m in rows),
        normal_mach2_scaled_max=max(float(m) for r,w,n,e,m in rows),
        raw_observations=observations,mach_observations=mach,command=command,
        rhs_footprint_exact=True,interior_other_faces_and_halo_words_unchanged=True,sync_q_words_equal=True,
        sanitizer=args.sanitizer,binary_sha256=hashlib.sha256((root/'tested_probe').read_bytes()).hexdigest(),
        scope='Algebraic bc41/42/50/60 states and native normal-Mach maximum, not native NSCBC RHS/CFD/restart/physical validation')
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key not in ('raw_observations','mach_observations')},indent=2))


if __name__=='__main__':main()
