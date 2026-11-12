"""Attribute frozen NSCBC relaxation jumps to sign tests at grazing flow."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from compare_q_validation_snapshots import read_q_snapshot
from review_keep_nonreflecting_corner import acoustic_projector


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--policy',choices=['compatibility','incoming_only'],default='compatibility')
    args=parser.parse_args()
    suffix='.step00000000.rk01.rank00000000.bin'
    states=[]
    for backend in ('cpu','gpu'):
        case=args.directory/(backend+'_np1_keep_open_x')
        snap=read_q_snapshot(case/'outdat'/('rhs.pre_rhs'+suffix))
        im,jm,km,hm,nq=snap.header
        shape=(im+2*hm+1,jm+2*hm+1,km+2*hm+1)
        q=snap.values.reshape((*shape,nq),order='F')
        with (case/'outdat'/('rhs.pre_rhs_primitives'+suffix)).open('rb') as stream:
            np.fromfile(stream,np.int32,4)
            primitive=np.fromfile(stream,np.float64).reshape((*shape,6),order='F')
        with (case/'outdat'/('rhs.boundary'+suffix)).open('rb') as stream:
            np.fromfile(stream,np.int32,4)
            rhs=np.fromfile(stream,np.float64).reshape((im+1,jm+1,km+1,nq),order='F')
        with h5py.File(case/'outdat/new/resources/geometry.h5') as geometry:
            jac=geometry['q0004'][:].transpose(2,1,0)
            ly=float(np.ptp(geometry['q0002'][:]))
        states.append((q,primitive,rhs,jac,ly))
    gamma=1.4;mach=.3
    pinf=1/(gamma*mach**2)
    target=np.array([1.,1.,0.,0.,pinf/(gamma-1)+.5])
    records=[]
    for k in range(km+1):
        q,primitive,rhs,jac,ly=states[0]
        q0=q[im+hm,jm+hm,k+hm,:]
        ro,u,v,w,p,t=primitive[im+hm,jm+hm,k+hm,:]
        gpu_v=float(states[1][1][im+hm,jm+hm,k+hm,2])
        if (v<0)==(gpu_v<0):
            continue
        css=np.sqrt(t)/mach
        delta=q0-target
        velocity=np.array([u,v,w]);kinetic=.5*np.dot(velocity,velocity)
        dp=(gamma-1)*(delta[4]-np.dot(velocity,delta[1:4])+kinetic*delta[0])
        dm=delta[2]-v*delta[0]
        amplitude=(dp-css*dm)/(2*css**2)
        minus=np.array([1.,u,v-css,w,css**2/(gamma-1)+kinetic-css*v])
        rate=.25*css/ly
        jump=jac[im,jm,k]*rate*(amplitude-(p-pinf)/css**2)*minus
        if v>=0:
            jump=-jump
        observed=rhs[im,jm,k,:]+states[1][2][im,jm,k,:]
        # One-sided limits at the same frozen thermodynamic/tangential state.
        # This is algebra for the existing policies, not another CFD run.
        grazing=q0.copy();grazing[2]=0.
        minus_projector=acoustic_projector(grazing,1,-1,gamma)
        plus_projector=acoustic_projector(grazing,1,1,gamma)
        convective_projector=np.eye(5)-minus_projector-plus_projector
        top=primitive[hm:hm+im+1,hm+jm,hm:hm+km+1,:]
        mach2=float(np.max(top[...,2]**2/(top[...,5]/mach**2)))
        sigma_out=rate*max(0.,1-mach2)
        grazing_delta=grazing-target
        incoming_minus=jac[im,jm,k]*rate*((convective_projector+minus_projector)@grazing_delta)
        incoming_plus=jac[im,jm,k]*sigma_out*(minus_projector@grazing_delta)
        pressure_minus=jac[im,jm,k]*rate*(minus_projector@grazing_delta)
        pressure_plus=jac[im,jm,k]*rate*(p-pinf)/css**2*minus
        if args.policy=='incoming_only':
            jump=incoming_minus-incoming_plus
            if v>=0:
                jump=-jump
        records.append({'k':k,'cpu_v':float(v),'gpu_v':gpu_v,
                        'wave_tolerance':float(64*np.finfo(float).eps*css),
                        'observed':observed.tolist(),'predicted_branch_jump':jump.tolist(),
                        'residual_max_abs':float(np.max(abs(observed-jump))),
                        'one_sided_limits':{
                            'meaning':'positive mapped relaxation contribution only, not a measured incoming_only trajectory',
                            'top_normal_mach2_max':mach2,
                            'compatibility_jump_max_abs':float(np.max(abs(pressure_minus-pressure_plus))),
                            'incoming_only_jump':(incoming_minus-incoming_plus).tolist(),
                            'incoming_only_jump_max_abs':float(np.max(abs(incoming_minus-incoming_plus))),
                            'incoming_only_convective_jump_max_abs':float(np.max(abs(
                                jac[im,jm,k]*rate*(convective_projector@grazing_delta))))}})
    report={'scope':'frozen relaxation branch; no CFD advancement','policy':args.policy,
            'model':'pre-v2 strict normal-speed sign test' if args.policy=='compatibility' else 'strict incoming-wave sign test',
            'gamma':gamma,'mach':mach,'target':target.tolist(),'sign_changed_nodes':records,
            'explanation':('normal_speed<0 selects conservative-state relaxation; else selects pressure-only relaxation'
                if args.policy=='compatibility' else 'negative normal speed activates three convective incoming waves at finite relaxation rate')}
    output=args.output if args.output is not None else args.directory/'grazing_attribution.json'
    with output.open('x') as stream:
        stream.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
