"""Freeze actual CPU/old CUDA outlet RHS, then compare native C++ components."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[2]


def compile_reference(build,output,gpu,source,definitions=()):
    build=build.resolve();output.mkdir(parents=True,exist_ok=False)
    link=shlex.split((build/'src/CMakeFiles/astr.dir/link.txt').read_text())
    includes,=[line.split('=',1)[1] for line in (build/'src/CMakeFiles/astr.dir/flags.make').read_text().splitlines()
               if line.startswith('Fortran_INCLUDES =')]
    obj=output/'probe.o';binary=output/'probe'
    command=[link[0],'-Mpreprocess','-Mextend','-O2','-I'+str(build/'src'),*shlex.split(includes),
        '-DREFERENCE_CUDA' if gpu else '-DREFERENCE_CPU',*definitions,'-c',str(source.resolve()),'-o',str(obj)]
    if gpu:command+=['-cuda','-gpu=lineinfo']
    subprocess.run(command,cwd=output,check=True)
    link[link.index('CMakeFiles/astr.dir/astr.F90.o')]=str(obj);link[link.index('-o')+1]=str(binary)
    link=[value for value in link if not value.startswith('-Wl,--dependency-file=')]
    subprocess.run(link,cwd=build/'src',check=True)
    return binary,dict(compile=command,link=link,sha256=hashlib.sha256(binary.read_bytes()).hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-build',type=Path);parser.add_argument('--gpu-build',type=Path)
    parser.add_argument('--reference',type=Path);parser.add_argument('--binary',type=Path)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--mpiexec',type=Path,required=True)
    parser.add_argument('--ranks',type=int,choices=[1,2],default=1)
    parser.add_argument('--axis',choices=list('xyz'),default='x')
    parser.add_argument('--sanitizer',choices=['memcheck','initcheck'])
    parser.add_argument('--precision',choices=('fp64','fp32'),default='fp64')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    env=os.environ|dict(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0')
    launch=[str(args.mpiexec.resolve()),'--bind-to','none','-n',str(args.ranks)]
    axis=str('xyz'.index(args.axis)+1);report=dict(ranks=args.ranks,axis=args.axis,precision=args.precision)
    if args.reference is None:
        if args.cpu_build is None or args.gpu_build is None:parser.error('freeze requires CPU and old GPU build')
        for name,build,gpu in [('cpu',args.cpu_build,False),('gpu',args.gpu_build,True)]:
            binary,info=compile_reference(build,out/name,gpu,Path(__file__).with_name('nscbc_outlet_probe.F90'));report[name]=info
            with (out/f'{name}.log').open('w') as log:
                subprocess.run(launch+[str(binary),str(out),axis],cwd=out,env=env,
                    stdout=log,stderr=subprocess.STDOUT,check=True)
        import numpy as np
        dtype=np.float32 if args.precision=='fp32' else np.float64
        report['pair_errors']=[];report['analytic_errors']=[]
        for rank in range(args.ranks):
            with (out/f'cpu.rank{rank:08d}.bin').open('rb') as a,(out/f'gpu.rank{rank:08d}.bin').open('rb') as b:
                ha=np.fromfile(a,np.int32,7);hb=np.fromfile(b,np.int32,7)
                if not np.array_equal(ha,hb):raise AssertionError('reference header')
                av=np.fromfile(a,dtype).astype(np.float64);bv=np.fromfile(b,dtype).astype(np.float64)
            error=float(np.max(abs(av-bv))/max(1.,float(np.max(abs(av)))))
            if not np.isfinite(av).all() or not np.isfinite(bv).all() or (args.precision=='fp64' and error>2e-10):
                raise AssertionError(f'CPU/old GPU outlet differs: {error}')
            report['pair_errors'].append(error)
            shape=tuple(ha[:3]+1)+(5,20)
            owner=ha[0]+ha[3]==16
            # Uniform Cartesian states have an independent acoustic/entropy
            # projection, without constructing either implementation's matrix.
            for label,values in [('cpu',av),('gpu',bv)]:
                fields=values.reshape(shape,order='F');analytic_max=0.
                speeds=[.2,2.,0.,-.2,-2.,-1.,-1.-1e-9,-1.+1e-9,-1e-9,.2]
                if args.precision=='fp32':
                    d=128*np.finfo(np.float32).eps
                    speeds[6:9]=[-1-d,-1+d,-d]
                speeds=np.asarray(speeds,dtype=dtype).astype(np.float64)
                for trial,u in enumerate(speeds):
                    v=float(dtype(10. if trial==9 else .1));w=float(dtype(-.05))
                    gm=float(dtype(1.4));gm1=float(dtype(dtype(gm)-dtype(1)))
                    p=float(dtype(dtype(1)/dtype(gm)));pfar=float(dtype(dtype(p)-dtype(.01)))
                    pout=float(dtype(.8 if trial==3 else 1.));energy=p/gm1+.5*(u*u+v*v+w*w)
                    h=energy+p;rhs=np.zeros(5)
                    if u<1.:
                        rate=3.75*max(0.,1-u*u)/16.
                        rhs=-rate*(p-pfar*pout)*np.array([1.,u-1,v,w,h-u])
                    if u<0.:
                        ui,vi,wi=map(float,np.asarray([.4,.03,.02],dtype))
                        delta=np.array([1.,u,v,w,energy])-np.array([pout,ui*pout,vi*pout,wi*pout,
                            pfar*pout/gm1+.5*pout*(ui**2+vi**2+wi**2)])
                        dp=gm1*(delta[4]-u*delta[1]-v*delta[2]-w*delta[3]+.5*(u*u+v*v+w*w)*delta[0])
                        du=delta[1]-u*delta[0]
                        plus=.5*(dp+du)*np.array([1.,u+1,v,w,h+u])
                        rhs-=.25*(-u)/16.*(delta-plus)+.25*max(-u-1.,0.)/16.*plus
                    actual=fields[...,trial].copy()
                    actual-=fields[0,0,0,:,trial]
                    expected=np.zeros_like(actual)
                    if owner:expected[-1,:,:,:]=rhs
                    analytic_max=max(analytic_max,float(np.max(abs(actual-expected))))
                if args.precision=='fp64' and analytic_max>2e-10:raise AssertionError(f'{label} independent uniform reference: {analytic_max}')
                report['analytic_errors'].append(dict(rank=rank,backend=label,max_abs=analytic_max))
        shutil.copyfile(Path(__file__).with_name('nscbc_outlet_probe.F90'),out/'probe_source.F90')
        report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [ROOT/'src/bc.F90',ROOT/'src_gpu/boundary_gpu.cuf',Path(__file__).with_name('nscbc_outlet_probe.F90')]}
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
        rows=re.findall(r'CPP_NSCBC_OUTLET_PASS\s+(\d+)\s+(\d+)\s+(\S+)',(out/'run.log').read_text())
        if sorted(int(row[0]) for row in rows)!=list(range(args.ranks)):raise AssertionError('missing outlet rank')
        if any(int(row[1])!=40 for row in rows):raise AssertionError('missing outlet cases')
        report['rows']=[dict(rank=int(r),cases=int(n),max_error=float(e)) for r,n,e in rows]
        text=(out/'run.log').read_text()
        differences=re.findall(r'CPP_NSCBC_NATIVE_DIFFERENCE\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\S+)',text)
        if len(differences)!=400*args.ranks:raise AssertionError('outlet per-variable phase records')
        report['per_variable_scaled']=[max(float(e) for r,d,t,s,c,e in differences if int(c)==component) for component in range(1,6)]
        widths=re.findall(r'NSCBC_PROBE_REAL_BYTES\s+(\d+)\s+(\d+)',text)
        if sorted((int(r),int(w)) for r,w in widths)!=[(r,4 if args.precision=='fp32' else 8) for r in range(args.ranks)]:
            raise AssertionError('outlet actual numerical width')
        if args.sanitizer:
            logs=list(out.glob('sanitizer.*.log'))
            if len(logs)!=args.ranks or any('ERROR SUMMARY: 0 errors' not in p.read_text() for p in logs):
                raise AssertionError('missing clean per-rank tool result')
        report['command']=command
        report['binary_sha256']=hashlib.sha256(args.binary.read_bytes()).hexdigest()
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('cpu','gpu','command')},indent=2))
    if args.reference is None:
        for p in out.rglob('*'):p.chmod(0o555 if p.is_dir() or p.name=='probe' else 0o444)
        out.chmod(0o555)


if __name__=='__main__':main()
