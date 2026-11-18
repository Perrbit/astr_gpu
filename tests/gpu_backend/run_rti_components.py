"""Freeze original bc31/gravity functions, then compare the resident C++ commands."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np

from run_nscbc_outlet_components import compile_reference,ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-build',type=Path);parser.add_argument('--gpu-build',type=Path)
    parser.add_argument('--reference',type=Path);parser.add_argument('--binary',type=Path)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--mpiexec',type=Path,required=True)
    parser.add_argument('--ranks',type=int,choices=[1,2],default=1);parser.add_argument('--axis',choices=list('xyz'),default='y')
    parser.add_argument('--precision',choices=['fp64','fp32'],default='fp64')
    parser.add_argument('--sanitizer',choices=['memcheck','initcheck'])
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    env=os.environ|dict(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0')
    launch=[str(args.mpiexec.resolve()),'--bind-to','none','-n',str(args.ranks)];axis=str('xyz'.index(args.axis)+1)
    source=Path(__file__).with_name('rti_probe.F90');report=dict(ranks=args.ranks,axis=args.axis,precision=args.precision)
    dtype=np.float32 if args.precision=='fp32' else np.float64
    if args.reference is None:
        if args.cpu_build is None or args.gpu_build is None:parser.error('freeze requires CPU/old-GPU builds')
        for name,build,gpu in [('cpu',args.cpu_build,False),('gpu',args.gpu_build,True)]:
            binary,info=compile_reference(build,out/name,gpu,source);report[name]=info
            with (out/f'{name}.log').open('w') as log:
                subprocess.run(launch+[str(binary),str(out),axis],cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        report['pairs']=[]
        for rank in range(args.ranks):
            with (out/f'cpu.rank{rank:08d}.bin').open('rb') as a,(out/f'gpu.rank{rank:08d}.bin').open('rb') as b:
                h=np.fromfile(a,np.int32,6);other=np.fromfile(b,np.int32,6)
                if not np.array_equal(h,other):raise AssertionError('RTI reference layout')
                physical=int(np.prod(h[:3]+1));halo_nodes=int(np.prod(h[:3]+11));maximum=0.
                for case in range(6):
                    for nodes,components in ((halo_nodes,5),(halo_nodes,6),(physical,5)):
                        av=np.fromfile(a,dtype,nodes*components).reshape((nodes,components),order='F')
                        bv=np.fromfile(b,dtype,nodes*components).reshape((nodes,components),order='F')
                        if not np.isfinite(av).all() or not np.isfinite(bv).all():raise AssertionError('nonfinite RTI reference')
                        error=float(np.max(np.max(abs(av-bv),axis=0)/np.maximum(1.,np.max(abs(av),axis=0))))
                        if args.precision=='fp64' and error>2e-10:raise AssertionError(f'RTI reference mismatch {rank,case,error}')
                        maximum=max(maximum,error)
                if a.read(1) or b.read(1):raise AssertionError('RTI reference trailing data')
            report['pairs'].append(dict(rank=rank,max_error=maximum))
        shutil.copyfile(source,out/'probe_source.F90')
        report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [ROOT/'src/bc.F90',ROOT/'src/solver.F90',ROOT/'src_gpu/boundary_gpu.cuf',ROOT/'src_gpu/solver_gpu.cuf',source]}
    else:
        if args.binary is None:parser.error('comparison requires --binary')
        command=launch
        if args.sanitizer:
            command+=['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer','--tool',args.sanitizer,
                '--error-exitcode','99','--log-file',str(out/'sanitizer.%p.log')]
            if args.sanitizer=='memcheck':command+=['--leak-check','full']
        with (out/'run.log').open('w') as log:
            subprocess.run(command+[str(args.binary.resolve()),str(args.reference.resolve()),axis],cwd=out,env=env,
                stdout=log,stderr=subprocess.STDOUT,check=True)
        rows=re.findall(r'CPP_RTI_PASS\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)',(out/'run.log').read_text())
        if sorted(int(row[0]) for row in rows)!=list(range(args.ranks)) or any(int(row[1])!=24 for row in rows):
            raise AssertionError('missing RTI rank/cases')
        report['rows']=[dict(rank=int(r),cases=int(c),max_error=float(e),analytic_error=float(a)) for r,c,e,a in rows]
        if args.sanitizer:
            logs=list(out.glob('sanitizer.*.log'))
            if len(logs)!=args.ranks or any('ERROR SUMMARY: 0 errors' not in p.read_text() for p in logs):
                raise AssertionError('missing clean RTI sanitizer records')
        report['binary_sha256']=hashlib.sha256(args.binary.read_bytes()).hexdigest()
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('cpu','gpu','source_sha256')},indent=2))
    if args.reference is None:
        for p in out.rglob('*'):p.chmod(0o555 if p.is_dir() or p.name=='probe' else 0o444)
        out.chmod(0o555)


if __name__=='__main__':main()
