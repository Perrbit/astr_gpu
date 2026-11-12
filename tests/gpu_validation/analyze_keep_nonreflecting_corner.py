"""Independent frozen-flux check for the controlled KEEP nonreflecting fixture."""
import json
from pathlib import Path

import h5py
import numpy as np

from compare_q_validation_snapshots import read_q_snapshot
from review_keep_nonreflecting_corner import acoustic_projector


def check_corner(case,backend,output):
    suffix='.step00000000.rk01.rank00000000.bin'
    snap=read_q_snapshot(case/'outdat'/('rhs.pre_rhs'+suffix))
    im,jm,km,hm,nq=snap.header
    shape=(im+2*hm+1,jm+2*hm+1,km+2*hm+1)
    q=snap.values.reshape((*shape,nq),order='F')
    with (case/'outdat'/('rhs.pre_rhs_primitives'+suffix)).open('rb') as stream:
        np.fromfile(stream,np.int32,4)
        primitive=np.fromfile(stream,np.float64).reshape((*shape,6),order='F')
    with (case/'outdat'/('rhs.boundary'+suffix)).open('rb') as stream:
        np.fromfile(stream,np.int32,4)
        rhs=np.fromfile(stream,np.float64).reshape((im+1,jm+1,km+1,5),order='F')
    if backend=='cpu':
        rhs=-rhs
    with h5py.File(case/'outdat/new/resources/geometry.h5') as geometry:
        jac=geometry['q0004'][:].transpose(2,1,0)
        metric=np.empty((*jac.shape,3,3))
        for axis in range(3):
            for component in range(3):
                metric[...,axis,component]=geometry[f'q{5+axis+3*component:04d}'][:].transpose(2,1,0)
        lx=float(np.ptp(geometry['q0001'][:]))
    gamma=1.4;mach=.3;pinf=1/(gamma*mach**2)

    def node(i,j,k):
        return q[i+hm,j+hm,k+hm,:],primitive[i+hm,j+hm,k+hm,:]

    def mapped_flux(i,j,k,axis):
        state,cache=node(i,j,k)
        g=metric[i,j,k%km,axis,:]
        normal=np.dot(g,cache[1:4]);pressure=cache[4]
        flux=state*normal
        flux[1:4]+=pressure*g;flux[4]+=pressure*normal
        return jac[i,j,k%km]*flux

    records=[];identity=np.eye(5)
    for k in range(km+1):
        state,cache=node(im,jm,k)
        rho,u,v,w,p,temp=cache
        c=np.sqrt(temp)/mach
        if not 0<u<c:
            raise AssertionError('Corner oracle requires the approved positive subsonic x outflow')
        px=acoustic_projector(state,0,-1,gamma)
        dx=1.5*mapped_flux(im,jm,k,0)-2*mapped_flux(im-1,jm,k,0)+.5*mapped_flux(im-2,jm,k,0)
        rest=np.zeros(5)
        for component in range(3):
            cartesian=state*cache[1+component]
            cartesian[1+component]+=p;cartesian[4]+=p*cache[1+component]
            area=[jac[im-i,jm,k%km]*metric[im-i,jm,k%km,0,component] for i in range(3)]
            rest+=cartesian*(1.5*area[0]-2*area[1]+.5*area[2])
        minus=np.array([1.,u-c,v,w,(state[4]+p)/rho-c*u])
        rate=3.75*max(0.,1-(u/c)**2)*c/lx
        dx=(identity-px)@(dx-rest)+rest+jac[im,jm,k%km]*rate*(p-pinf)/c**2*minus
        dy=1.5*mapped_flux(im,jm,k,1)-2*mapped_flux(im,jm-1,k,1)+.5*mapped_flux(im,jm-2,k,1)
        dz=.5*(mapped_flux(im,jm,k+1,2)-mapped_flux(im,jm,k-1,2))
        raw=-(dx+dy+dz)
        tolerance=64*np.finfo(float).eps*max(c,abs(v))
        py=(identity-acoustic_projector(state,1,1,gamma)) if v<-tolerance else acoustic_projector(state,1,-1,gamma)
        observed=rhs[im,jm,k,:]
        expected=(identity-py)@raw
        records.append({'k':k,'reference_difference':float(np.max(abs(observed-expected))),
                        'incoming_residual':float(np.max(abs(py@observed))),
                        'outgoing_change':float(np.max(abs((identity-py)@(observed-raw))))})
    result={'backend':backend,'scope':'first-stage convective boundary RHS only; no viscous nonreflection claim',
            'gamma':gamma,'mach':mach,'inlet_top_max_abs':float(np.max(abs(rhs[0,jm,:,:]))),
            'reference_difference':max(r['reference_difference'] for r in records),
            'incoming_residual':max(r['incoming_residual'] for r in records),
            'outgoing_change':max(r['outgoing_change'] for r in records),'records':records}
    (output/f'{backend}_corner_diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
    if max(result[key] for key in ('reference_difference','incoming_residual','outgoing_change','inlet_top_max_abs'))>2e-10:
        raise AssertionError(f'KEEP nonreflecting corner contract failed: {result}')
    return result
