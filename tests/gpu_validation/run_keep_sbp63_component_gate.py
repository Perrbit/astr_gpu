"""Root-built SBP6/3 CPU/GPU components versus independent exact matrices and wall fluxes."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from study_keep_sbp63_family import derive
from study_keep_sbp63_sat_wall import matrices,error_minimizer


def scaled(a,b):
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AssertionError('Component layout or finiteness differs')
    return float(np.max(abs(a-b)/np.maximum(1,np.maximum(abs(a),abs(b)))))


def oracle(path,n):
    data=np.loadtxt(path/f'component_n{n}.txt')
    if data.shape!=(n+1,40):
        raise AssertionError('Native component column layout differs')
    norms,left,centered,theta=derive();chosen,_=error_minimizer(norms,left,theta)
    h,d,b,r,q=matrices(n,norms,left,centered)
    h=np.diag(np.array(h,dtype=float));d=np.array(d.subs(theta,chosen),dtype=float)
    rows=np.loadtxt(path/f'matrix_n{n}.txt');native=np.zeros_like(d)
    native[rows[:,0].astype(int),rows[:,1].astype(int)]=rows[:,2]
    x=np.arange(n+1)/n;rho=1+x*x/20;t=(1+x/20)**2
    v=np.column_stack([.1+x/20,.025-x/30,x*x/50]);p=rho*t/(1.4*.09)
    inputs=np.column_stack([np.arange(n+1),x,h,rho,t,p,v])
    mass=.25*(rho[:,None]+rho[None,:])*(v[:,None,1]+v[None,:,1])
    average=.5*(v[:,None,:]+v[None,:,:]);flux=np.zeros((n+1,n+1,5))
    flux[...,0]=mass;flux[...,1:4]=mass[...,None]*average
    flux[...,2]+=.5*(p[:,None]+p[None,:])
    kinetic=.5*np.einsum('ic,jc->ij',v,v)
    internal=.5*((p/rho)[:,None]+(p/rho)[None,:])/.4
    flux[...,4]=mass*(kinetic+internal)+.5*(v[:,None,1]*p[None,:]+v[None,:,1]*p[:,None])
    convection=-2*np.einsum('ij,ijc->ic',d,flux)
    for i,s in ((0,-1.),(n,1.)):
        wall=np.array([0,0,p[i],0,0])
        convection[i]-=s*(wall-flux[i,i])/h[i]
    gu=np.zeros((n+1,3,3));gu[:,:,1]=d@v
    gt=np.zeros((n+1,3));gt[:,1]=d@t
    for i,s in ((0,-1.),(n,1.)):
        gu[i,:,1]-=s*v[i]/h[i];gt[i,1]+=s*(1-t[i])/h[i]
    sutherland=110.3/273.15
    mu=t*np.sqrt(t)*(1+sutherland)/(1000*(t+sutherland));kappa=mu/.72/.036
    stress=mu[:,None,None]*(gu+np.swapaxes(gu,1,2)-(2/3)*np.trace(gu,axis1=1,axis2=2)[:,None,None]*np.eye(3))
    energy=kappa[:,None]*gt+np.einsum('ic,icd->id',v,stress)
    bulk=np.column_stack([stress[:,:,1],energy[:,1]])
    face=np.zeros((n+1,5))
    for i,s in ((0,-1.),(n,1.)):
        normal=np.array([0,s,0]);delta=v[i]
        face[i,1:4]=stress[i]@normal-mu[i]/h[i]*(delta+normal*np.dot(delta,normal)/3)
        face[i,4]=kappa[i]*np.dot(gt[i],normal)-kappa[i]/h[i]*(t[i]-1)
    viscous=np.zeros((n+1,5));viscous[:,1:]=d@bulk
    for i,s in ((0,-1.),(n,1.)):
        viscous[i,1:]+=s*(s*face[i,1:]-bulk[i])/h[i]
    gradients=np.column_stack([gu.reshape(n+1,9,order='F'),gt])
    expected=np.column_stack([inputs,convection,viscous,gradients,bulk,face])
    errors=dict(matrix=scaled(native,d),inputs=scaled(data[:,:9],inputs),
        convection=scaled(data[:,9:14],convection),viscosity=scaled(data[:,14:19],viscous),
        gradients=scaled(data[:,19:31],gradients),bulk_flux=scaled(data[:,31:35],bulk),
        wall_flux=scaled(data[:,35:40],face))
    if max(errors.values())>2e-10:
        raise AssertionError(f'Native component differs from exact-matrix reference: n={n}: {errors}')
    return errors


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cpu','gpu','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--memcheck',action='store_true')
    args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();env['OMP_NUM_THREADS']='1'
    report=dict(status='running',cases={})
    for backend in ('cpu','gpu'):
        case=out/backend;case.mkdir()
        command=[str(getattr(args,backend).resolve()),str(case)]
        with (case/'run.log').open('w') as log:
            subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        text=(case/'run.log').read_text()
        if 'SBP63_COMPONENT_PASS' not in text:
            raise AssertionError('Native derivative and wall-balance checks are incomplete')
        report['cases'][backend]={n:oracle(case,n) for n in (16,32,64)}
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    report['cpu_gpu']={n:scaled(np.loadtxt(out/f'cpu/component_n{n}.txt'),np.loadtxt(out/f'gpu/component_n{n}.txt')) for n in (16,32,64)}
    if max(report['cpu_gpu'].values())>2e-10:
        raise AssertionError('Native CPU/GPU tables differ beyond the approved gate')
    if args.memcheck:
        case=out/'memcheck';case.mkdir()
        command=['compute-sanitizer','--tool','memcheck','--target-processes','all','--error-exitcode','99',
                 '--log-file',str(case/'memcheck.log'),str(args.gpu.resolve()),str(case)]
        with (case/'run.log').open('w') as log:
            subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        text=(case/'memcheck.log').read_text()
        if 'ERROR SUMMARY: 0 errors' not in text:
            raise AssertionError('GPU component memcheck did not report zero errors')
        for n in (16,32,64):
            if not np.array_equal(np.loadtxt(case/f'component_n{n}.txt'),np.loadtxt(out/f'gpu/component_n{n}.txt')):
                raise AssertionError('Memcheck changed a native component value')
        report['memcheck']='0 errors; instrumented/native component tables bit-identical'
    report['status']='passed_native_component_and_independent_reference'
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
