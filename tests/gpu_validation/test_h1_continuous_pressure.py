"""Native Euler RHS near-boundary convergence with nonzero Ducros activation."""
import json
import os
import h5py
import numpy as np
import pytest

from test_grcbc_oblique_shock import EXE, launch
from run_native_precision_tgv_gate import rhs_state

pytestmark=pytest.mark.skipif(not EXE,reason='Set ASTR_SBLI_GRCBC_EXE to a root-CMake executable')
VARIANT=os.environ.get('ASTR_H1_REPAIR_VARIANT','continuous')


def manufactured_inputs(root,n,axis):
    z,y,x=np.meshgrid(np.linspace(0,.125,9),np.linspace(0,1,n+1),np.linspace(0,1,n+1),indexing='ij')
    coordinate=x if axis==0 else y
    rho=np.ones_like(x)
    velocity=np.zeros(x.shape+(3,));velocity[...,0]=1
    slope=-.1
    velocity[...,axis]+=slope*coordinate
    pressure=(1+.01*np.exp(4*(coordinate-1)))/5.6
    dp=.04*np.exp(4*(coordinate-1))/5.6
    energy=pressure/.4+.5*np.sum(velocity**2,axis=-1)
    q=np.concatenate([rho[...,None],velocity,energy[...,None]],axis=-1)
    reference=np.zeros_like(q)
    reference[...,0]=-slope
    reference[...,1:4]=-slope*velocity
    reference[...,axis+1]-=slope*velocity[...,axis]+dp
    reference[...,4]=-((3.5*dp+velocity[...,axis]*slope)*velocity[...,axis]+(energy+pressure)*slope)
    root.mkdir(parents=True)
    source=root/'field.h5'
    with h5py.File(source,'w') as file:
        for m in range(5):file[f'q{m+1:04d}']=q[...,m]
    profile=root/'inlet.prof'
    lines=['Manufactured density=provided pressure=provided','rho u v temperature pressure',
           '1 1 1 1','y rho u v temperature pressure']
    for j in range(n+1):
        u=velocity[0,j,0];p=pressure[0,j,0]
        lines.append(f'1.0000000000000000e+00 {u[0]:.16e} {u[1]:.16e} {5.6*p:.16e} {p:.16e}')
    profile.write_text('\n'.join(lines)+'\n')
    return source,profile,reference.transpose(2,1,0,3)


@pytest.mark.parametrize('axis',[0,1],ids=['x','y'])
def test_nonconstant_pressure_rhs_convergence(tmp_path,axis):
    errors=[];reports=[]
    for n in [32,64,128]:
        source,profile,reference=manufactured_inputs(tmp_path/f'inputs{n}',n,axis)
        cases=[];strengths=[]
        for backend in ['cpu','gpu']:
            case=launch(tmp_path/f'{backend}{n}',backend,n=n,steps=1,entry_x=10,
                field_source=source,inlet_profile_source=profile,diagnostics=True,h1_variant=VARIANT)
            suffix='step00000000.rk01.rank00000000.bin'
            cases.append(rhs_state(case/'diagnostics'/f'state.full.{suffix}','fp64')/(1/n**2/64))
            strengths.append(rhs_state(case/'diagnostics'/f'state.h1_strength.{suffix}','fp64')[...,0])
        selector=[slice(None),slice(None),slice(None)]
        selector[1-axis]=n//2
        mask=np.zeros(reference.shape[:-1],bool)
        # The old 421 wall is not a nonconstant-pressure closure under test here.
        bands=[slice(1,4),slice(n-3,n)] if axis==0 else [slice(n-3,n)]
        for band in bands:
            selector[axis]=band;mask[tuple(selector)]=True
        error=max(float(abs(q-reference)[mask].max()) for q in cases)
        parity=float(abs(cases[0]-cases[1])[mask].max())
        activation=max(float(s[mask].max()) for s in strengths)
        assert activation>0,'nonconstant-pressure test must actually activate the blend'
        assert parity<=2e-10,parity
        errors.append(error);reports.append(dict(n=n,error=error,parity=parity,max_strength=activation))
    orders=np.log2(np.array(errors[:-1])/errors[1:])
    report=dict(axis=axis,scope='physical near-boundary interior rows; compressive velocity and nonconstant pressure',
                cases=reports,errors=errors,orders=orders.tolist())
    (tmp_path/'convergence.json').write_text(json.dumps(report,indent=2)+'\n')
    assert np.all(orders>=5.5),report
