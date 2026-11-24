"""ASTR main-program W1: stretched laminar RHS with independent normal rows and restart."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import h5py
import numpy as np

from run_keep_f0_gate import checkpoint, compare, logged_steps, prepare_wall, stage
from check_keep_w1_scalar import audit, d2_matrix, operators
from check_keep_w1_shear_heat import normal_operators


def profile(y):
    r=y/np.pi
    u=.06*np.sin(np.pi*r)
    w=.021*np.sin(2*np.pi*r)
    t=1+.1*r*(1-r)+.03*np.sin(2*np.pi*r)
    uy=.06*np.cos(np.pi*r)
    wy=.042*np.cos(2*np.pi*r)
    ty=(.1*(1-2*r)+.06*np.pi*np.cos(2*np.pi*r))/np.pi
    uyy=-u
    wyy=-4*w
    tyy=-.2/np.pi**2-.12*np.sin(2*np.pi*r)
    return u,w,t,uy,wy,ty,uyy,wyy,tyy


def prepare(case,n,backend,steps,mode,restore=None,mixed=False):
    side=12
    prepare_wall(case,'41y',backend,'643e','643e','off',steps=steps,restore=restore)
    inp=case/'datin/input.case'
    inp.write_text(inp.read_text().replace('  ia = 16\n',f'  ia = {side}\n').replace('  ka = 16\n',f'  ka = {side}\n')
                   .replace('  ja = 16\n',f'  ja = {n}\n')+f"\n&boundary_options\n high_order_wall_mode = '{mode}'\n/\n")
    if restore:
        return
    eta=np.linspace(0.,1.,n+1)
    axes=(np.linspace(0.,2*np.pi,side+1),np.pi*(eta+.35*eta*(1-eta)),np.linspace(0.,2*np.pi,side+1))
    x,y,z=np.meshgrid(*axes,indexing='ij')
    u,w,t,*_=profile(y)
    v=np.zeros_like(u)
    if mixed:
        u+=.004*np.cos(x)*np.cos(z)*np.cos(y)**2
        v+=.003*np.sin(x)*np.cos(z)*np.cos(y)**2
        w+=.002*np.cos(x)*np.sin(z)*np.cos(y)**2
        t+=.004*np.cos(x+z)*np.cos(y)
    with h5py.File(case/'datin/grid.simple.h5','w') as f:
        for key,grid in zip('xyz',(x,y,z)):
            f[key]=grid.transpose(2,1,0)
    with h5py.File(case/'datin/flowini3d.h5','w') as f:
        for key,data in dict(ro=np.ones_like(u),u1=u,u2=v,u3=w,t=t).items():
            f[key]=data.transpose(2,1,0)


def viscous(case,rk=1):
    with h5py.File(case/'outdat/geometry.h5') as f:
        volume=f['jacob'][...].transpose(2,1,0)
    return (stage(case,'full',rk,0)-stage(case,'conv',rk,0))/volume[...,None]


def verify_laminar(case,n,left,inside,shift):
    with h5py.File(case/'datin/grid.simple.h5') as f:
        y=f['y'][0,:,0]
    with h5py.File(case/'outdat/geometry.h5') as f:
        jac_y=1/f['dxi22'][0,:,0]
    u,w,t,uy,wy,ty,uyy,wyy,tyy=profile(y)
    field=stage(case,'pre_rhs_primitives',1,0)[0,:,0,:]
    compare(field[:,[1,2,3,5]],np.stack([u,np.zeros_like(u),w,t],axis=-1))
    u,w,t=(field[:,i] for i in (1,3,5))
    s=110.3/273.15
    mu=t*np.sqrt(t)*(1+s)/(t+s)/3000
    kappa=mu/(.72*.4*.3**2)
    buy=mu*(1.5/t-1/(t+s))*ty
    bky=buy/(.72*.4*.3**2)
    h,_,e,sn,gamma=normal_operators(n+1,left,inside,shift)

    def solve_term(v,c,target,raw):
        b=c/jac_y
        q=b[[0,-1],None]*sn
        a=d2_matrix(b,left,inside)
        jump=v[[0,-1]]-target
        p=2*(raw*b[[0,-1]]/h[[0,-1]]+gamma*b[[0,-1]]**2/[b[:6].min(),b[-6:].min()])
        val=(a@v+(q.T@jump-e@(p*jump))/h)/jac_y
        work=(a@(v*v/2)-e@(q@(v*v/2))/h)/jac_y
        return val,work,q@v-p*jump

    rx,wx,taux=solve_term(u,mu,0.,2.)
    rz,wz,tauz=solve_term(w,mu,0.,2.)
    rt,_,flux=solve_term(t,kappa,1.,1.)
    expected=np.stack([np.zeros_like(rx),rx,np.zeros_like(rx),rz,rt+wx+wz],axis=-1)
    actual=viscous(case)[0,:,0,:]
    mismatch=compare(actual,expected)
    balance=abs(np.sum(h*jac_y*actual[:,4])-np.sum(flux))
    if balance>2.e-10:
        raise AssertionError(f'W1 physical energy balance: {balance}')
    continuum_x=mu*uyy+buy*uy
    continuum_z=mu*wyy+buy*wy
    continuum_e=u*continuum_x+w*continuum_z+mu*(uy*uy+wy*wy)+kappa*tyy+bky*ty
    error=np.abs(actual[:,[1,3,4]]-np.stack((continuum_x,continuum_z,continuum_e),axis=-1))
    side=np.array([-1,1])
    return dict(reference=mismatch, net_energy_error=balance,
                near_wall=error[:9].max(axis=0).tolist(), interior=error[9:-9].max(axis=0).tolist(),
                wall_shear=max(np.max(abs(taux-side*(mu*uy)[[0,-1]])),np.max(abs(tauz-side*(mu*wy)[[0,-1]]))),
                wall_heat=np.max(abs(flux-side*(kappa*ty)[[0,-1]])))


def mixed_balance(case,n,rk,left,inside,shift):
    h,_,_,_,gamma=normal_operators(n+1,left,inside,shift)
    with h5py.File(case/'outdat/geometry.h5') as f:
        jac=f['jacob'][...].transpose(2,1,0)[:-1,:,:-1]
        inverse_y=f['dxi22'][...].transpose(2,1,0)[:-1,:,:-1]
    phys=stage(case,'pre_rhs_primitives',rk,0)[:-1,:,:-1]
    dv=viscous(case,rk)[:-1,:,:-1]
    weight=jac*h[None,:,None]
    kinetic=np.sum(weight[...,None]*phys[...,1:4]*dv[...,1:4])
    theta=phys[...,5]
    s0=110.3/273.15
    b=theta*np.sqrt(theta)*(1+s0)/(theta+s0)*inverse_y/(3000*.72*.4*.3**2)
    wall_area=(np.pi/6)**2
    sn=-np.array([-25,48,-36,16,-3])/12
    net_heat=0.
    for field,prop in ((theta,b),(theta[:,::-1],b[:,::-1])):
        bw=prop[:,0,:]
        p=2*(bw/h[0]+gamma*bw**2/prop[:,:6,:].min(axis=1))
        net_heat+=wall_area*np.sum(bw*np.einsum('j,ijk->ik',sn,field[:,:5,:])-p*(field[:,0,:]-1))
    error=abs(np.sum(weight*dv[...,4])-net_heat)
    if not np.isfinite(kinetic) or kinetic>0 or error>2.e-10:
        raise AssertionError(f'W1 mixed energy/kinetic balance: {kinetic}, {error}')
    return dict(kinetic_rate=kinetic, net_energy_error=error)


def run(binary,mpi,root,coefficients):
    root.mkdir(parents=True,exist_ok=False)
    left,inside=operators(coefficients)
    shift=audit(coefficients)['separate_viscous_d1_shift']
    env=os.environ | dict(ASTR_FORCE_MPI_TOPOLOGY='1,1,1',ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',
                          ASTR_VALIDATION_RHS_STEP='0',ASTR_GEOMETRY_DUMP='outdat/geometry.h5',
                          ASTR_GPU_SYNC_MODE='explicit',ASTR_GPU_HALO_TRANSPORT='pinned',OMPI_MCA_opal_cuda_support='0')
    report={}

    def execute(name,n,backend,steps,restore=None,mixed=False,mode='keep_sbp63_pair',mismatch=False):
        case=root/name
        prepare(case,n,backend,steps,mode,restore,mixed)
        with (case/'run.log').open('w') as file:
            rc=subprocess.run([str(mpi),'-n','1',str(binary),'run'],cwd=case,env=env,
                              stdin=subprocess.DEVNULL,stdout=file,stderr=subprocess.STDOUT)
        logs=(case/'run.log').read_text()
        if mismatch:
            if rc.returncode==0 or 'compact operator identity mismatch' not in logs:
                raise AssertionError('old SBP mode accepted a paired W1 exact checkpoint')
        else:
            rc.check_returncode()
            if 's63pair-v1' not in logs:
                raise AssertionError('W1 paired viscosity was not selected')
            logged_steps(case,count=steps-2 if restore else steps,first_step=3 if restore else 1)
        return case

    for n in (24,48,96,192):
        cpu=execute(f'laminar-n{n}-cpu',n,'cpu',1)
        gpu=execute(f'laminar-n{n}-gpu',n,'gpu',1)
        report[n]=dict(cpu=verify_laminar(cpu,n,left,inside,shift),gpu=verify_laminar(gpu,n,left,inside,shift),
                       same_state=compare(viscous(gpu),viscous(cpu)))
        (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(f'laminar n{n}: W1 viscous stages match independent rows and CPU/GPU',flush=True)
    final='outdat/new/checkpoints/step000000000004/state.h5'
    first='outdat/new/checkpoints/step000000000002'
    ca={backend:execute(f'mixed-{backend}',24,backend,4,mixed=True) for backend in ('cpu','gpu')}
    report['mixed']=dict(stages=[compare(viscous(ca['gpu'],rk),viscous(ca['cpu'],rk)) for rk in (1,2,3)],
                         terminal=compare(checkpoint(ca['gpu']/final),checkpoint(ca['cpu']/final)))
    report['mixed']['balances']={backend:[mixed_balance(case,24,r,left,inside,shift) for r in (1,2,3)]
                                  for backend,case in ca.items()}
    for backend,case in ca.items():
        restored=execute(f'restore-{backend}',24,backend,4,restore=case/first,mixed=True)
        np.testing.assert_array_equal(checkpoint(restored/final).view(np.uint64),checkpoint(case/final).view(np.uint64))
    execute('s63sat-rejects-s63pair',24,'cpu',4,restore=ca['cpu']/first,mode='keep_sbp63',mismatch=True)
    report['mixed']['exact_checkpoint']='2+2 bitwise; original SAT identity rejected'
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('W1 mixed three-stage RHS, complete steps and restart passed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for flag in ('binary','mpiexec','output','coefficients'):
        p.add_argument('--'+flag,type=Path,required=True)
    a=p.parse_args()
    run(a.binary.resolve(),a.mpiexec.resolve(),a.output.resolve(),a.coefficients.resolve())
