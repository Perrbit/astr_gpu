"""Attribute the frozen x+/y+ corner CPU/GPU RHS difference to mapped z flux."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from compare_q_validation_snapshots import read_q_snapshot


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    cpu=args.directory/'cpu_np1_keep_open_x'
    gpu=args.directory/'gpu_np1_keep_open_x'
    suffix='.step00000000.rk01.rank00000000.bin'
    snapshot=read_q_snapshot(cpu/'outdat'/('rhs.pre_rhs'+suffix))
    im,jm,km,hm,nq=snapshot.header
    full_shape=(im+2*hm+1,jm+2*hm+1,km+2*hm+1)
    physical=(slice(hm,hm+im+1),slice(hm,hm+jm+1),slice(hm,hm+km+1))
    q=snapshot.values.reshape((*full_shape,nq),order='F')[physical]
    with (cpu/'outdat'/('rhs.pre_rhs_primitives'+suffix)).open('rb') as stream:
        np.fromfile(stream,np.int32,4)
        primitives=np.fromfile(stream,np.float64).reshape((*full_shape,6),order='F')[physical]
    with h5py.File(cpu/'outdat/new/resources/geometry.h5') as state:
        jac=state['q0004'][:].transpose(2,1,0)
        metric=np.stack([state[f'q{4+3+3*c:04d}'][:].transpose(2,1,0) for c in range(3)],axis=-1)
    normal=np.sum(metric*primitives[...,1:4],axis=-1)
    pressure=primitives[...,4]
    flux=q*normal[...,None]
    flux[...,1:4]+=pressure[...,None]*metric
    flux[...,4]+=pressure*normal
    flux*=jac[...,None]
    dz=0.5*(flux[:,:,2:,:]-flux[:,:,:-2,:])
    rhs=[]
    for case in (cpu,gpu):
        with (case/'outdat'/('rhs.boundary'+suffix)).open('rb') as stream:
            np.fromfile(stream,np.int32,4)
            rhs.append(np.fromfile(stream,np.float64).reshape((im+1,jm+1,km+1,nq),order='F'))
    difference=rhs[0]+rhs[1]  # CPU positive divergence, GPU negative divergence.
    corner=difference[im,jm,1:km,:]
    remaining=corner+dz[im,jm,:,:]
    other=difference.copy();other[im,jm,:,:]=0
    record={'phase':'post-boundary/pre-convection','corner':'x+ / y+, periodic-z interior nodes',
            'raw_corner_max_abs':float(np.max(abs(corner))),
            'negative_z_flux_difference_max_abs':float(np.max(abs(dz[im,jm,:,:]))),
            'residual_after_negative_z_flux_max_abs':float(np.max(abs(remaining))),
            'other_nodes_max_abs':float(np.max(abs(other))),
            'component_raw_max_abs':np.max(abs(corner),axis=0).tolist(),
            'component_residual_max_abs':np.max(abs(remaining),axis=0).tolist(),
            'interpretation':'CPU omits z transport on x+/y+ line; GPU assigns it to x+ boundary'}
    (args.directory/'corner_attribution.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    main()
