#!/usr/bin/env python3
"""Compare saved HB6 milestones directly, with explicit backend/refinement meaning."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from test_high_order_wall_runtime import defined_extras
from hb6_channel_reference import initial_fields


def kinetic_deviation(rho,velocity,base_velocity,grid):
    density=.5*rho*np.sum((velocity-base_velocity)**2,axis=-1)
    return float(np.trapezoid(density[:-1,:,:-1],dx=2/grid[1],axis=1).sum()*
                 (2*np.pi/grid[0])*(np.pi/grid[2]))


def response(left,right):
    base=json.loads(left.read_text()); perturbed=json.loads(right.read_text())
    assert [r['target_time'] for r in base]==[r['target_time'] for r in perturbed]==[1,5,10]
    p,q=base[0]['configuration'],perturbed[0]['configuration']
    assert p['wall']==q['wall']==41 and p['perturbation']==0 and q['perturbation']!=0
    assert p['grid']==q['grid'] and p['dt']==q['dt']
    grid=tuple(p['grid'])
    _,initial_base=initial_fields(grid)
    _,initial_perturbed=initial_fields(grid,perturbation=q['perturbation'])
    v0=np.stack([initial_base[f'u{m}'] for m in range(1,4)],axis=-1)
    v1=np.stack([initial_perturbed[f'u{m}'] for m in range(1,4)],axis=-1)
    energy0=kinetic_deviation(initial_perturbed['ro'],v1,v0,grid)
    assert energy0>0
    records=[]
    for a,b in zip(base,perturbed,strict=True):
        assert a['configuration']['steps']==b['configuration']['steps']
        velocities=[]
        for row in (a,b):
            with h5py.File(Path(row['checkpoint'])/'state.h5') as f:
                rho=f['q0001'][:].transpose(2,1,0)
                momentum=np.stack([f[f'q{m:04d}'][:].transpose(2,1,0) for m in range(2,5)],axis=-1)
            velocities.append(momentum/rho[...,None])
        energy=kinetic_deviation(rho,velocities[1],velocities[0],grid)
        assert np.isfinite(energy)
        records.append(dict(target_time=a['target_time'],kinetic_deviation=energy,
                            amplification=energy/energy0,
                            maximum_velocity_difference=float(np.max(abs(velocities[1]-velocities[0])))))
    return dict(kind='perturbation_response',grid=grid,initial_energy=energy0,records=records,
                definition='0.5 integral rho_perturbed |u_perturbed-u_unperturbed_ASTR|^2 dV',
                note='Velocity deviation norm, not total energy difference or a monotone-decay gate; physical trapezoidal quadrature')


def compare(left, right, kind):
    a=json.loads(left.read_text()); b=json.loads(right.read_text())
    assert [row['target_time'] for row in a]==[1,5,10]
    assert [row['target_time'] for row in b]==[1,5,10]
    tolerance=2e-10 if kind=='backend' else 1e-3
    records=[]
    for x,y in zip(a,b,strict=True):
        p,q=x['configuration'],y['configuration']
        assert p['wall']==q['wall'] and p['perturbation']==q['perturbation']
        gx=np.array(p['grid']); gy=np.array(q['grid'])
        if kind=='backend':
            np.testing.assert_array_equal(gx,gy)
            assert p['dt']==q['dt'] and p['steps']==q['steps']
            assert p['topology']==q['topology']
            stride=1
        else:
            np.testing.assert_array_equal(2*gx,gy)
            stride=2
        errors={}; raw={}
        with h5py.File(Path(x['checkpoint'])/'state.h5') as fa,h5py.File(Path(y['checkpoint'])/'state.h5') as fb:
            for m in range(1,12):
                name=f'q{m:04d}'
                va=fa[name][:]; vb=fb[name][::stride,::stride,::stride]
                errors[name]=float(np.max(abs(va-vb)/(1+abs(va))))
                raw[name]=float(np.max(abs(va-vb)))
            if kind=='backend':
                for rank,(va,vb) in enumerate(zip(defined_extras(fa),defined_extras(fb),strict=True)):
                    errors[f'physical_extras_rank{rank}']=float(np.max(abs(va-vb)/(1+abs(va))))
        row=dict(target_time=x['target_time'],scaled_errors=errors,absolute_errors=raw,
                 maximum_scaled=max(errors.values()))
        records.append(row)
        assert row['maximum_scaled']<=tolerance,row
    return dict(kind=kind,tolerance=tolerance,records=records,maximum_scaled=max(r['maximum_scaled'] for r in records))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left',type=Path)
    parser.add_argument('right',type=Path)
    parser.add_argument('--kind',choices=('backend','refinement','response'),default='backend')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=response(args.left,args.right) if args.kind=='response' else compare(args.left,args.right,args.kind)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
