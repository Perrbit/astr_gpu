"""Manufactured frozen viscous RHS for the existing bc41 ME6 nodal wall band."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import h5py
import numpy as np

from run_cpp_multi_tgv_gate import physical, raw_field
from run_keep_f0_gate import compare, prepare_wall, stage


def exact(n):
    r = np.linspace(0.,1.,n+1)
    s, c = np.sin(2*np.pi*r), np.cos(2*np.pi*r)
    p, q = r*(1-r), 1-2*r
    u = .2*(p+.3*s*p)
    uy = .2*(q*(1+.3*s)+.6*np.pi*p*c)/np.pi
    uyy = .2*(-2*(1+.3*s)+1.2*np.pi*q*c-.3*(2*np.pi)**2*p*s)/np.pi**2
    t = 1+.1*p+.03*s
    ty = (.1*q+.06*np.pi*c)/np.pi
    tyy = (-.2-.03*(2*np.pi)**2*s)/np.pi**2
    s0 = 110.3/273.15
    mu = t*np.sqrt(t)*(1+s0)/(t+s0)/3000
    muy = mu*(1.5/t-1/(t+s0))*ty
    conductivity_factor = 1/(.72*.4*.3**2)
    momentum = mu*uyy+muy*uy
    work_heat = u*momentum+mu*uy**2+conductivity_factor*(mu*tyy+muy*ty)
    return r, u, t, momentum, work_heat, (mu*uy)[[0,-1]], (conductivity_factor*mu*ty)[[0,-1]]


def prepare(case, n, backend):
    nx = 12
    prepare_wall(case,'41y',backend,'643e','643e','off',viscosity='me6_opti',steps=1)
    p = case/'datin/input.case'
    p.write_text(p.read_text().replace('  ia = 16\n', f'  ia = {nx}\n').replace('  ka = 16\n', f'  ka = {nx}\n')
                 .replace('  ja = 16\n', f'  ja = {n}\n'))
    r,u,t,*_ = exact(n)
    x,y,z = np.meshgrid(np.linspace(0.,2*np.pi,nx+1),np.pi*r,np.linspace(0.,2*np.pi,nx+1),indexing='ij')
    velocity = np.broadcast_to(u[None,:,None], x.shape)
    temperature = np.broadcast_to(t[None,:,None], x.shape)
    with h5py.File(case/'datin/grid.simple.h5','w') as grid:
        for name,v in zip('xyz',(x,y,z)):
            grid[name] = v.transpose(2,1,0)
    with h5py.File(case/'datin/flowini3d.h5','w') as state:
        for name,v in dict(ro=np.ones_like(x),u1=velocity,u2=np.zeros_like(x),u3=np.zeros_like(x),t=temperature).items():
            state[name] = v.transpose(2,1,0)


def wall_fluxes(cpu, gpu):
    path = cpu/'outdat/rhs.gradients.step00000000.rk01.rank00000000.bin'
    with path.open('rb') as f:
        i,j,k,*_ = np.fromfile(f,np.int32,5)
        shape = (i+1,j+1,k+1)
        volume = int(np.prod(shape))
        grad = np.fromfile(f,np.float64,9*volume).reshape((*shape,3,3),order='F')
        thermal = np.fromfile(f,np.float64,3*volume).reshape((*shape,3),order='F')
    mu_wall = 1/3000
    host_shear = mu_wall*(grad[0,[0,-1],0,0,1]+grad[0,[0,-1],0,1,0])
    host_heat = mu_wall*thermal[0,[0,-1],0,1]/(.72*.4*.3**2)
    device = {}
    for path in sorted((gpu/'outdat').glob('rhs.cpp_halo.step00000000.rk01.*.rank00000000.bin')):
        with path.open('rb') as f:
            header = np.fromfile(f,np.int32,9)
        if header[5] in (11,12) and header[6] == 1:
            header,values = raw_field(path,packet=True)
            body = physical(values,dict(upper=header[:3],halo=(header[3],)))
            device[header[5]] = body[0,[0,-1],0,1]
    return (host_shear,host_heat),(device[11],device[12])


def run(exe, mpiexec, root):
    root.mkdir(parents=True,exist_ok=False)
    env = os.environ | dict(ASTR_FORCE_MPI_TOPOLOGY='1,1,1',ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',
                            ASTR_VALIDATION_RHS_STEP='0',ASTR_VALIDATION_GRADIENTS='1',
                            ASTR_GEOMETRY_DUMP='outdat/geometry.h5',ASTR_GPU_SYNC_MODE='explicit',
                            ASTR_GPU_HALO_TRANSPORT='pinned',OMPI_MCA_opal_cuda_support='0')
    history = {}
    for n in (16,32,64,128):
        cases = []
        outputs = []
        _,u,t,momentum,energy,shear,heat = exact(n)
        for backend in ('cpu','gpu'):
            case = root/f'{backend}-n{n}'
            prepare(case,n,backend)
            with (case/'run.log').open('w') as f:
                subprocess.run([str(mpiexec),'-n','1',str(exe),'run'],cwd=case,env=env,
                               stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,check=True)
            prim = stage(case,'pre_rhs_primitives',1,0)
            compare(prim[0,:,0,[1,5]].T,np.stack((u,t),axis=-1))
            with h5py.File(case/'outdat/geometry.h5') as f:
                jac = f['jacob'][...].transpose(2,1,0)
            visc = (stage(case,'full',1,0)-stage(case,'conv',1,0))/jac[...,None]
            outputs.append(visc)
            cases.append(case)
        parity = compare(*outputs)
        wall = wall_fluxes(*cases)
        compare(np.stack(wall[0],axis=-1),np.stack(wall[1],axis=-1))
        report = {}
        for backend,field,flux in zip(('cpu','gpu'),outputs,wall):
            error = np.abs(field[0,:,0,[1,4]].T - np.stack((momentum,energy),axis=-1))
            idx = np.minimum(np.arange(n+1),n-np.arange(n+1))
            report[backend] = dict(wall_band=np.max(error[(idx>=1)&(idx<=3)],axis=0).tolist(),
                                   join=np.max(error[(idx>=4)&(idx<=7)],axis=0).tolist(),
                                   inner=np.max(error[idx>=8],axis=0).tolist(),
                                   wall_flux=np.max(abs(np.stack(flux)-np.stack([shear,heat])),axis=1).tolist())
        history[n] = dict(parity=parity,**report)
        (root/'report.json').write_text(json.dumps(history,indent=2)+'\n')
        print(n,report['cpu'],flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--mpiexec',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    run(args.binary.resolve(),args.mpiexec.resolve(),args.output.resolve())
