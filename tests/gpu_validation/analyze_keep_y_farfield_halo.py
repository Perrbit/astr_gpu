"""Explain a frozen pressure-outlet/top-NSCBC periodic-seam discrepancy."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from compare_q_validation_snapshots import read_q_snapshot
from run_keep_y_farfield_gate import phase,incoming_projector


def state(case,name):
    snap=read_q_snapshot(case/f'outdat/rhs.{name}.step00000000.rk01.rank00000000.bin')
    im,jm,km,hm,nq=snap.header
    values=snap.values.reshape((im+2*hm+1,jm+2*hm+1,km+2*hm+1,nq),order='F')
    return values[hm:hm+im+1,hm:hm+jm+1,hm:hm+km+1]


def analyze(cpu,gpu):
    old=state(cpu,'initialized');fresh=state(cpu,'pre_rhs');n=old.shape[2]-1
    observed=-phase(cpu,'boundary')-phase(gpu,'boundary')
    with h5py.File(cpu/'outdat/new/resources/geometry.h5') as geometry:
        jac=geometry['q0004'][:].transpose(2,1,0)
        metric=np.empty(jac.shape+(3,3))
        for axis in range(3):
            for c in range(3):
                metric[...,axis,c]=geometry[f'q{5+axis+3*c:04d}'][:].transpose(2,1,0)
    def flux(q,k):
        rho=q[0];velocity=q[1:4]/rho
        pressure=.4*(q[4]-.5*np.dot(q[1:4],velocity))
        direction=metric[-1,-1,k,2];un=np.dot(direction,velocity)
        value=q*un;value[1:4]+=pressure*direction;value[4]+=pressure*un
        return jac[-1,-1,k]*value
    records=[]
    for k,wrapped,sign in ((0,n-1,1.),(n,1,-1.)):
        normal=metric[-1,-1,k,1];normal=normal/np.linalg.norm(normal)
        incoming,_=incoming_projector(fresh[-1,-1,k],normal)
        predicted=sign*.5*(np.eye(5)-incoming)@(flux(old[-1,-1,wrapped],wrapped)-flux(fresh[-1,-1,wrapped],wrapped))
        actual=observed[-1,-1,k]
        records.append(dict(k=k,predicted=predicted.tolist(),observed=actual.tolist(),
                            residual_max_abs=float(np.max(abs(predicted-actual)))))
    result=dict(scope='read-only first-stage characteristic projection of stale exit-plane z halo',
        observed_max_abs=float(np.max(abs(observed))),
        nonseam_max_abs=float(np.max(abs(observed[:,:,1:-1]))),
        predicted_seam_residual_max_abs=max(r['residual_max_abs'] for r in records),records=records)
    if result['predicted_seam_residual_max_abs']>2e-10:
        raise AssertionError(f'Stale-halo hypothesis does not explain the frozen data: {result}')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu',type=Path,required=True)
    parser.add_argument('--gpu',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=analyze(args.cpu,args.gpu)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
