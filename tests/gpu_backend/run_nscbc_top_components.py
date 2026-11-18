"""Actual CPU/old-GPU upper-y NSCBC with analytic uniform characteristic checks."""
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


def analytic_uniform(plane,target,selection,trial,header,mach2,precision='fp64'):
    upper=header[:3];origin=header[3:6];expected=np.zeros(tuple(upper+1)+(5,))
    if origin[1]+upper[1]!=16:return expected
    dtype=np.float32 if precision=='fp32' else np.float64
    gm,ma,rho,t=map(float,np.asarray([1.4,3.,1.03,1.1],dtype))
    css=float(dtype(np.sqrt(dtype(t))/dtype(ma)));p=float(dtype(dtype(rho)*dtype(t)/dtype(dtype(gm)*dtype(ma)*dtype(ma))))
    speeds=[.2,-.2,2.,-2.,0.,1e-20,-1e-20,1.,-1.]
    u=float(dtype(10. if trial==0 else .7));v=float(dtype(dtype(css)*dtype(speeds[trial])));w=float(dtype(.03))
    q=np.array([rho,rho*u,rho*v,rho*w,p/(gm-1)+.5*rho*(u*u+v*v+w*w)])
    enthalpy=css*css/(gm-1)+.5*(u*u+v*v+w*w)
    tolerance=0. if selection==2 else 64*np.finfo(dtype).eps*max(css,abs(v))
    minus_vector=np.array([1.,u,v-css,w,enthalpy-v*css])
    plus_vector=np.array([1.,u,v+css,w,enthalpy+v*css])
    for k in range(upper[2]+1):
        for i in range(upper[0]+1):
            if selection==3:continue
            chosen=plane[i,k] if selection==2 else target
            delta=q-chosen
            dp=(gm-1)*(delta[4]-u*delta[1]-v*delta[2]-w*delta[3]+.5*(u*u+v*v+w*w)*delta[0])
            dv=delta[2]-v*delta[0]
            plus=.5*(dp+css*dv)/(css*css)*plus_vector
            minus=.5*(dp-css*dv)/(css*css)*minus_vector
            incoming=delta-plus-minus if v < -tolerance else np.zeros(5)
            if v+css < -tolerance:incoming+=plus
            if v-css < -tolerance:incoming+=minus
            if selection==0 and v>=-tolerance:
                rate=.25*max(0.,1-(v/css)**2)*css/16.
                expected[i,-1,k]=-rate*(p-1/(gm*ma*ma))/(css*css)*minus_vector if v-css < -tolerance else 0.
            else:
                rate=.25*css/16.
                if v>=-tolerance:rate=max(0.,.25*(1-mach2)*css/16.)
                expected[i,-1,k]=-rate*incoming
    return expected


def analytic_viscous(trial,header,precision='fp64'):
    dtype=np.float32 if precision=='fp32' else np.float64
    upper=header[:3];origin=header[3:6];gm=float(dtype(1.4));css=float(dtype(np.sqrt(dtype(1.1))/dtype(3)))
    speeds=[.2,-.2,2.,-2.,0.,1e-20,-1e-20,1.,-1.]
    u=float(dtype(10. if trial==0 else .7));v=float(dtype(dtype(css)*dtype(speeds[trial])));w=float(dtype(.03))
    enthalpy=css*css/(gm-1)+.5*(u*u+v*v+w*w)
    ii,jj,kk=np.indices(tuple(upper+1));strength=.001*(1+.01*(ii+origin[0])+.02*(jj+origin[1])+.03*(kk+origin[2]))
    if trial==4:strength[:]=0
    source=strength[...,None]*np.array([0.,2.,-3.,4.,-5.])
    expected=source.copy()
    if origin[1]+upper[1]!=16:return expected
    s=source[:,-1,:,:];dp=(gm-1)*(s[...,4]-u*s[...,1]-v*s[...,2]-w*s[...,3])
    plus=.5*(dp+css*s[...,2])[...,None]/(css*css)*np.array([1.,u,v+css,w,enthalpy+v*css])
    minus=.5*(dp-css*s[...,2])[...,None]/(css*css)*np.array([1.,u,v-css,w,enthalpy-v*css])
    tolerance=64*np.finfo(dtype).eps*max(css,abs(v))
    incoming=s-plus-minus if v < -tolerance else np.zeros_like(s)
    if v+css < -tolerance:incoming+=plus
    if v-css < -tolerance:incoming+=minus
    expected[:,-1,:,:]-=incoming
    return expected


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-build',type=Path);parser.add_argument('--gpu-build',type=Path)
    parser.add_argument('--reference',type=Path);parser.add_argument('--binary',type=Path)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--mpiexec',type=Path,required=True)
    parser.add_argument('--ranks',type=int,choices=[1,2],default=1);parser.add_argument('--axis',choices=list('xyz'),default='x')
    parser.add_argument('--sanitizer',choices=['memcheck','initcheck'])
    parser.add_argument('--reduction-only',action='store_true')
    parser.add_argument('--plane-intervals',type=int,default=16)
    parser.add_argument('--viscous-source',action='store_true')
    parser.add_argument('--precision',choices=('fp64','fp32'),default='fp64')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    env=os.environ|dict(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0')
    launch=[str(args.mpiexec.resolve()),'--bind-to','none','-n',str(args.ranks)]
    axis=str('xyz'.index(args.axis)+1);report=dict(ranks=args.ranks,axis=args.axis,plane_intervals=args.plane_intervals,
        viscous_source=args.viscous_source,precision=args.precision)
    dtype=np.float32 if args.precision=='fp32' else np.float64
    source=Path(__file__).with_name('nscbc_top_probe.F90')
    if args.reference is None and not args.reduction_only:
        if args.cpu_build is None or args.gpu_build is None:parser.error('freeze requires CPU/old-GPU builds')
        for name,build,gpu in [('cpu',args.cpu_build,False),('gpu',args.gpu_build,True)]:
            binary,info=compile_reference(build,out/name,gpu,source,['-DTEST_TOP_VISCOUS'] if args.viscous_source else ());report[name]=info
            with (out/f'{name}.log').open('w') as log:
                subprocess.run(launch+[str(binary),str(out),axis,str(args.plane_intervals)],cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        report['comparisons']=[]
        for rank in range(args.ranks):
            with (out/f'cpu.rank{rank:08d}.bin').open('rb') as a,(out/f'gpu.rank{rank:08d}.bin').open('rb') as b:
                header=np.fromfile(a,np.int32,8);other=np.fromfile(b,np.int32,8)
                if not np.array_equal(header,other):raise AssertionError('top reference header')
                shape=tuple(header[:3]+1)+(5,);count=int(np.prod(shape));pshape=(header[0]+1,header[2]+1,5)
                max_pair=0.;max_analytic=0.
                for fixture in range(2):
                    for selection in range(4):
                        if args.viscous_source and selection!=3:continue
                        for trial in range(12):
                            ta=np.fromfile(a,dtype,5);tb=np.fromfile(b,dtype,5)
                            pa=np.fromfile(a,dtype,int(np.prod(pshape))).reshape(pshape,order='F')
                            pb=np.fromfile(b,dtype,int(np.prod(pshape))).reshape(pshape,order='F')
                            ma=np.fromfile(a,dtype,1)[0];mb=np.fromfile(b,dtype,1)[0]
                            av=np.fromfile(a,dtype,count).astype(np.float64).reshape(shape,order='F')
                            bv=np.fromfile(b,dtype,count).astype(np.float64).reshape(shape,order='F')
                            if not np.array_equal(ta,tb) or not np.array_equal(pa,pb) or ma!=mb:
                                raise AssertionError('top reference inputs differ')
                            errors=[float(np.max(abs(av[...,c]-bv[...,c]))/max(1.,float(np.max(abs(av[...,c]))))) for c in range(5)]
                            if not np.isfinite(av).all() or not np.isfinite(bv).all() or (args.precision=='fp64' and max(errors)>2e-10):
                                raise AssertionError(f'top CPU/GPU reference: rank={rank} fixture={fixture} selection={selection} trial={trial} {errors}')
                            max_pair=max(max_pair,max(errors))
                            if trial<9:
                                predicted=(analytic_viscous(trial,header,args.precision) if args.viscous_source else
                                    analytic_uniform(pa,ta,selection,trial,header,ma,args.precision))+np.asarray([
                                        dtype(dtype(.125)+dtype(.001)*c) for c in range(1,6)],np.float64)
                                err=max(float(np.max(abs(av-predicted))),float(np.max(abs(bv-predicted))))
                                if args.precision=='fp64' and err>2e-10:raise AssertionError(f'top independent uniform characteristic: {rank,fixture,selection,trial,err}')
                                max_analytic=max(max_analytic,err)
                if a.read(1) or b.read(1):raise AssertionError('top reference trailing data')
            report['comparisons'].append(dict(rank=rank,pair_error=max_pair,analytic_error=max_analytic))
        shutil.copyfile(source,out/'probe_source.F90')
        report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [ROOT/'src/bc.F90',ROOT/'src_gpu/boundary_gpu.cuf',source]}
    else:
        if args.binary is None:parser.error('comparison requires --binary')
        command=launch
        if args.sanitizer:
            command+=['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer',
                '--tool',args.sanitizer,'--error-exitcode','99','--log-file',str(out/'sanitizer.%p.log')]
            if args.sanitizer=='memcheck':command+=['--leak-check','full']
        root=out if args.reduction_only else args.reference.resolve()
        command+=[str(args.binary.resolve()),str(root),axis,str(args.plane_intervals)]
        if args.reduction_only:command+=['mach-only']
        with (out/'run.log').open('w') as log:
            subprocess.run(command,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        marker=('CPP_NSCBC_TOP_MACH_PASS' if args.reduction_only else
            'CPP_NSCBC_TOP_VISCOUS_PASS' if args.viscous_source else 'CPP_NSCBC_TOP_PASS')
        rows=re.findall(marker+r'\s+(\d+)\s+(\d+)\s+(\S+)',(out/'run.log').read_text())
        expected=2 if args.reduction_only else 96 if args.viscous_source else 384
        if sorted(int(r[0]) for r in rows)!=list(range(args.ranks)) or any(int(r[1])!=expected for r in rows):
            raise AssertionError('missing top rank/cases')
        report['rows']=[dict(rank=int(r),cases=int(n),max_error=float(e)) for r,n,e in rows]
        text=(out/'run.log').read_text()
        widths=re.findall(r'NSCBC_TOP_PROBE_REAL_BYTES\s+(\d+)\s+(\d+)',text)
        if sorted((int(r),int(w)) for r,w in widths)!=[(r,4 if args.precision=='fp32' else 8) for r in range(args.ranks)]:
            raise AssertionError('top actual numerical width')
        if not args.reduction_only:
            values=re.findall(r'CPP_NSCBC_TOP_NATIVE_DIFFERENCE\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\S+)',text)
            if len(values)!=expected*5*args.ranks:raise AssertionError('top per-variable observations incomplete')
            report['per_variable_scaled']=[max(float(e) for r,d,f,s,t,b,c,e in values if int(c)==component) for component in range(1,6)]
            report['per_reference_scaled']={label:[max(float(e) for r,d,f,s,t,b,c,e in values
                if int(c)==component and int(b)==reference) for component in range(1,6)]
                for reference,label in ((1,'cpu'),(2,'old_gpu'))}
            sync_rows=re.findall(r'NSCBC_TOP_SYNC_WORDS_PASS\s+(\d+)\s+(\d+)',text)
            if sorted((int(r),int(n)) for r,n in sync_rows)!=[(r,expected) for r in range(args.ranks)]:
                raise AssertionError('top repeat/synchronization word checks missing')
            report['repeat_and_sync_words_equal']=True
        if args.sanitizer:
            logs=list(out.glob('sanitizer.*.log'))
            if len(logs)!=args.ranks or any('ERROR SUMMARY: 0 errors' not in p.read_text() for p in logs):
                raise AssertionError('missing clean per-rank top result')
        report['binary_sha256']=hashlib.sha256(args.binary.read_bytes()).hexdigest()
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('cpu','gpu')},indent=2))
    if args.reference is None and not args.reduction_only:
        for p in out.rglob('*'):p.chmod(0o555 if p.is_dir() or p.name=='probe' else 0o444)
        out.chmod(0o555)


if __name__=='__main__':main()
