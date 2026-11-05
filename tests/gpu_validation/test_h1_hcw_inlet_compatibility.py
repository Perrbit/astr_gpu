"""Read-only native decomposition before selecting a new inlet closure."""
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from analyze_h1_hcw_sensitivity import rank_rhs,native_q,native_rhs
from test_high_order_boundary_runtime import EXE,H1_VARIANT,launch_hcw

pytestmark=pytest.mark.skipif(not EXE or H1_VARIANT!='admissible-filter',
    reason='Select the current executable and the existing admissible-filter candidate')
EVIDENCE=Path(os.environ.get('ASTR_H1_HCW_EVIDENCE',
    Path(__file__).parent/'out/h1_hcw_long_window_20261101')).resolve()


def test_native_high_flux_decomposition(tmp_path):
    source=next((EVIDENCE/'dense840_to1260_cpu/outdat/new/fields').glob(
        'segment*/step000000001000/data.h5'))
    case=tmp_path/'diagnostic'
    launch_hcw(case,'cpu',steps=1,field_source=source,diagnostics=True,inlet_flux=True)
    values=rank_rhs(case,'h1_inlet_flux_endpoints')
    assert values.shape[-1]==26,'native endpoint diagnostic must retain the four-interface nullspace direction'
    actual=values[:4,1:-1,:-1,:5]
    baseline=values[:4,1:-1,:-1,11:16]
    correction=values[:4,1:-1,:-1,16:21]
    np.testing.assert_allclose(actual,baseline+correction,rtol=2e-14,atol=2e-14)
    assert np.max(abs(correction))>0
    np.testing.assert_allclose(correction[:3],-np.array([1/8,1/4,1/8])[:,None,None,None]*
        values[:3,1:-1,:-1,21:26],rtol=2e-14,atol=2e-14)


@pytest.mark.parametrize('step',[840,1000])
def test_observed_error_direction_and_diagnostic_isolation(tmp_path,step):
    cases=[];samples=[]
    sources={840:dict(cpu=EVIDENCE/'long_windows/test_hcw_two_flowthrough_windo0/cpu',
                     gpu=EVIDENCE/'gpu_early_prefix'),
             1000:{name:EVIDENCE/f'dense840_to1260_{name}' for name in ('cpu','gpu')}}
    inputs={name:next((sources[step][name]/'outdat/new/fields').glob(
        f'segment*/step{step:012d}/data.h5')) for name in ('cpu','gpu')}
    for source in ('cpu','gpu'):
        case=tmp_path/source
        launch_hcw(case,'cpu',steps=1,field_source=inputs[source],diagnostics=True,inlet_flux=True)
        cases.append(case);samples.append(rank_rhs(case,'h1_inlet_flux_endpoints')[:4,1:-1,:-1])
    off=tmp_path/'off'
    launch_hcw(off,'cpu',steps=1,field_source=inputs['cpu'],diagnostics=True,inlet_flux=False)
    ending='outdat/new/checkpoints/step000000000001/state.h5'
    with h5py.File(cases[0]/ending) as on,h5py.File(off/ending) as without:
        for c in range(1,12):np.testing.assert_array_equal(on[f'q{c:04d}'][...],without[f'q{c:04d}'][...])
    with h5py.File(cases[0]/'outdat/new/resources/geometry.h5') as geometry:
        weight=geometry['q0004'][...].transpose(2,1,0)[1:4,1:-1,:-1,None]
    base=native_q(cases[0])[1:4,1:-1,:-1]
    delta=native_q(cases[1])[1:4,1:-1,:-1]-base
    norm=np.sum(weight*delta**2)
    high,low,theta,polynomial,correction=[[s[...,part] for s in samples] for part in
        (slice(0,5),slice(5,10),slice(10,11),slice(11,16),slice(16,21))]
    fluxes=dict(polynomial=polynomial[1]-polynomial[0],delta7=correction[1]-correction[0],
        high=high[1]-high[0],frozen_polynomial=(1-theta[0])*(polynomial[1]-polynomial[0]),
        frozen_delta7=(1-theta[0])*(correction[1]-correction[0]),
        frozen_limited=theta[0]*(low[1]-low[0]),feedback=(theta[1]-theta[0])*(low[1]-high[1]))
    rates={name:(flux[:-1]-flux[1:])/weight for name,flux in fluxes.items()}
    actual=-(native_rhs(cases[1],'conv_after_x')-native_rhs(cases[0],'conv_after_x'))[1:4,1:-1,:-1]/weight
    combined=sum(rates[name] for name in ('frozen_polynomial','frozen_delta7','frozen_limited','feedback'))
    residual=float(np.max(abs(combined-actual)))
    assert residual<=2e-10
    report=dict(scope='native CPU operator on two observed states; first three inlet columns, unique z nodes',
        norm='Jacobian-weighted Euclidean five-component error; not physical energy or an eigenvalue proof',
        projections={name:float(np.sum(weight*delta*rate)/norm) for name,rate in rates.items()},
        actual_x_projection=float(np.sum(weight*delta*actual)/norm),
        reconstruction_residual_maxabs=residual,diagnostic_changes_physical_state=False)
    pressure=.4*(base[...,4]-.5*np.sum(base[...,1:4]**2,axis=-1)/base[...,0])
    mach=(base[...,1]/base[...,0])/np.sqrt(1.4*pressure/base[...,0])
    local_norm=np.sum(weight*delta**2,axis=-1)
    local_growth=np.sum(weight*delta*actual,axis=-1)
    report['normal_mach_groups']={name:dict(nodes=int(mask.sum()),
        error_norm_fraction=float(local_norm[mask].sum()/norm),
        growth_contribution=float(local_growth[mask].sum()/norm))
        for name,mask in [('subsonic',mach<1),('supersonic',mach>=1)]}
    y_growth=local_growth.sum(axis=(0,2))/norm
    report['largest_y_contributions']=[dict(j=int(j+1),growth_contribution=float(y_growth[j]),
        mean_normal_mach=float(mach[:,j].mean())) for j in np.argsort(abs(y_growth))[-8:][::-1]]
    report['scalar_transient_diagnostic']=scalar_transient_diagnostic()
    from analyze_h1_inlet_closure_design import CURRENT,PROPOSED
    change=np.array([float(b-a) for a,b in zip(CURRENT,PROPOSED)])[:,None,None,None]
    candidate=[high[i]-change*samples[i][...,21:26] for i in range(2)]
    candidate_flux=[candidate[i]+theta[i]*(low[i]-candidate[i]) for i in range(2)]
    candidate_rate=((candidate_flux[1]-candidate_flux[0])[:-1]-(candidate_flux[1]-candidate_flux[0])[1:])/weight
    report['proposed_frozen_flux_holdout']=dict(
        scope='affine flux evaluation on native packets; not a native candidate RK step or positivity validation',
        coefficient_selection='frozen before this holdout; not fitted to this observed error direction',
        strengths=list(map(str,PROPOSED)),
        x_projection=float(np.sum(weight*delta*candidate_rate)/norm))
    (tmp_path/f'high_terms{step}.json').write_text(json.dumps(report,indent=2)+'\n')


def scalar_transient_diagnostic():
    from test_high_order_boundary_closures import stabilized_mp_matrix
    cases=[]
    for n in (32,64,128):
        a=-stabilized_mp_matrix(n,.1)[1:,1:]
        z=.1*a;step=np.eye(n)+z+z@z/2+z@z@z/6
        _,singular,right=np.linalg.svd(step)
        cases.append(dict(intervals=n,spectral_abscissa_h=float(np.linalg.eigvals(a).real.max()),
            numerical_abscissa_h=float(np.linalg.eigvalsh((a+a.T)/2).max()),
            rk3_spectral_radius=float(abs(np.linalg.eigvals(step)).max()),
            rk3_operator_2norm=float(singular[0]),
            right_singular_inlet8_fraction=float(np.sum(abs(right[0,:8])**2))))
    return dict(scope='existing scalar trial closure; matrix diagnostic only, no CFD propagation or new gate',
        scalar_cfl=.1,mp_ld_weight=.1,cases=cases)
