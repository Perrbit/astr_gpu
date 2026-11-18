"""Repaired GPU bc12 reference plus independent incoming acoustic/entropy modes."""
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
    parser.add_argument('--gpu-build',type=Path);parser.add_argument('--reference',type=Path)
    parser.add_argument('--binary',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mpiexec',type=Path,required=True);parser.add_argument('--ranks',type=int,choices=[1,2],default=1)
    parser.add_argument('--axis',choices=list('xyz'),default='x')
    parser.add_argument('--sanitizer',choices=['memcheck','initcheck'])
    parser.add_argument('--precision',choices=('fp64','fp32'),default='fp64')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    env=os.environ|dict(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0')
    launch=[str(args.mpiexec.resolve()),'--bind-to','none','-n',str(args.ranks)]
    axis=str('xyz'.index(args.axis)+1);report=dict(ranks=args.ranks,axis=args.axis,precision=args.precision)
    dtype=np.float32 if args.precision=='fp32' else np.float64
    source=Path(__file__).with_name('nscbc_outlet_probe.F90')
    if args.reference is None:
        if args.gpu_build is None:parser.error('freeze requires --gpu-build')
        binary,info=compile_reference(args.gpu_build,out/'gpu',True,source,['-DTEST_INLET'])
        report['gpu']=info
        with (out/'gpu.log').open('w') as log:
            subprocess.run(launch+[str(binary),str(out),axis],cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        report['analytic_errors']=[]
        speeds=[.2,2.,0.,-.2,-2.,1.,1.-1e-9,1.+1e-9,-1.,-1.-1e-9,-1.+1e-9,.2]
        if args.precision=='fp32':
            d=128*np.finfo(np.float32).eps
            speeds[6:8]=[1-d,1+d];speeds[9:11]=[-1-d,-1+d]
        speeds=np.asarray(speeds,dtype).astype(np.float64)
        for rank in range(args.ranks):
            with (out/f'gpu.rank{rank:08d}.bin').open('rb') as f:
                header=np.fromfile(f,np.int32,7)
                values=np.fromfile(f,dtype).astype(np.float64).reshape(tuple(header[:3]+1)+(5,24),order='F')
            if not np.isfinite(values).all():raise AssertionError('nonfinite inlet reference')
            owner=header[3]==0;maximum=0.
            for trial,u in enumerate(speeds):
                v=float(dtype(10. if trial==11 else .1));w=float(dtype(-.05));gm=float(dtype(1.4))
                p=float(dtype(dtype(1)/dtype(gm)));pfar=float(dtype(dtype(p)-dtype(.01)))
                ui,vi,wi=map(float,np.asarray([.4,.03,.02],dtype))
                energy=p/(gm-1)+.5*(u*u+v*v+w*w);enthalpy=energy+p
                delta=np.array([1.,u,v,w,energy])-np.array([1.,ui,vi,wi,
                    pfar/(gm-1)+.5*(ui**2+vi**2+wi**2)])
                dp=(gm-1)*(delta[4]-u*delta[1]-v*delta[2]-w*delta[3]+.5*(u*u+v*v+w*w)*delta[0])
                du=delta[1]-u*delta[0]
                plus=.5*(dp+du)*np.array([1.,u+1,v,w,enthalpy+u])
                minus=.5*(dp-du)*np.array([1.,u-1,v,w,enthalpy-u])
                tolerance=64*np.finfo(dtype).eps*max(1.,abs(u))
                incoming=(delta-plus-minus if u>tolerance else np.zeros(5))
                if u+1>tolerance:incoming+=plus
                if u-1>tolerance:incoming+=minus
                expected=np.zeros_like(values[...,trial])
                if owner:expected[0,:,:,:]=-(.25/16.)*incoming
                actual=values[...,trial]-values[-1,0,0,:,trial]
                maximum=max(maximum,float(np.max(abs(actual-expected))))
            if args.precision=='fp64' and maximum>2e-10:raise AssertionError(f'inlet analytic projection: {maximum}')
            report['analytic_errors'].append(dict(rank=rank,max_abs=maximum))
        shutil.copyfile(source,out/'probe_source.F90')
        report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [ROOT/'src_gpu/boundary_gpu.cuf',source]}
    else:
        if args.binary is None:parser.error('comparison requires --binary')
        command=launch
        if args.sanitizer:
            command+=['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer',
                '--tool',args.sanitizer,'--error-exitcode','99','--log-file',str(out/'sanitizer.%p.log')]
            if args.sanitizer=='memcheck':command+=['--leak-check','full']
        command += [str(args.binary.resolve()),str(args.reference.resolve()),axis]
        with (out/'run.log').open('w') as log:
            subprocess.run(command,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        rows=re.findall(r'CPP_NSCBC_INLET_PASS\s+(\d+)\s+(\d+)\s+(\S+)',(out/'run.log').read_text())
        if sorted(int(r[0]) for r in rows)!=list(range(args.ranks)) or any(int(r[1])!=48 for r in rows):
            raise AssertionError('missing inlet rank/cases')
        report['rows']=[dict(rank=int(r),cases=int(n),max_error=float(e)) for r,n,e in rows]
        text=(out/'run.log').read_text()
        differences=re.findall(r'CPP_NSCBC_NATIVE_DIFFERENCE\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\S+)',text)
        if len(differences)!=240*args.ranks:raise AssertionError('inlet per-variable phase records')
        report['per_variable_scaled']=[max(float(e) for r,d,t,s,c,e in differences if int(c)==component) for component in range(1,6)]
        widths=re.findall(r'NSCBC_PROBE_REAL_BYTES\s+(\d+)\s+(\d+)',text)
        if sorted((int(r),int(w)) for r,w in widths)!=[(r,4 if args.precision=='fp32' else 8) for r in range(args.ranks)]:
            raise AssertionError('inlet actual numerical width')
        if args.sanitizer:
            logs=list(out.glob('sanitizer.*.log'))
            if len(logs)!=args.ranks or any('ERROR SUMMARY: 0 errors' not in p.read_text() for p in logs):
                raise AssertionError('missing clean per-rank tool result')
        report['binary_sha256']=hashlib.sha256(args.binary.read_bytes()).hexdigest()
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='gpu'},indent=2))
    if args.reference is None:
        for p in out.rglob('*'):p.chmod(0o555 if p.is_dir() or p.name=='probe' else 0o444)
        out.chmod(0o555)


if __name__=='__main__':main()
