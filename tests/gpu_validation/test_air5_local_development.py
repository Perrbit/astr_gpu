"""Approved development-window plumbing and represented-state checks, no CFD."""
import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import run_air5_local_development as window
from air5_radau_reference import Air5RadauReference


@pytest.mark.parametrize('variant', ['baseline','combined'])
def test_preparation_preserves_approved_physics(tmp_path,variant):
    case=tmp_path/variant
    env,record=window.prepare_window(case,variant,100,True)
    primary=(case/'datin/input.air5_c4').read_text()
    output=(case/'datin/input.output').read_text()
    assert record['intervals']==[63,127,15] and record['dt']==2.5e-10
    assert 'f,t,t,f,f,f,f,t,t' in primary
    assert 'exact_restart=t' in output and 'keep=2,initial_frame=t' in output
    assert record['report_stride']==1
    assert record['physics']['lfilter'] is True
    assert not (case/'datin/monitor.dat').exists()
    assert (case/'outdat/new').is_dir()
    assert env['ASTR_GPU_HALO_TRANSPORT']=='pinned'
    assert env['ASTR_AIR5_FILTER_MODE']=='conservative-flux'
    assert env['ASTR_AIR5_SOURCE_EVALUATION']==('joint' if variant=='baseline' else 'source-only')
    profile=np.loadtxt(case/'datin/air5_hbl_profile.dat')
    np.testing.assert_array_equal(profile[-1,6:8],[1500.,1500.])
    np.testing.assert_array_equal(profile[0,6:8],[3000.,3000.])
    assert profile[-1,5]==20000.


def checkpoint(path):
    model=Air5RadauReference(window.ROOT/'chemMech/air5_kimjo12.json')
    rho=.05
    species=rho*np.array([.767,.233,0.,0.,0.])
    momentum=np.array([rho*100.,0.,0.])
    ev=model.ev_from_tv(species,1500.)
    q=np.r_[rho,momentum,model.q5_from_state(rho,momentum,species,ev,1500.),species,ev]
    caches=np.r_[rho,momentum/rho,model.pressure(species,1500.),1500.,1500.,species/rho]
    with h5py.File(path,'w') as file:
        for m,value in enumerate(np.r_[q,np.zeros(11),caches],1):
            file[f'q{m:04}']=np.full((1,1,1),value)


def test_final_state_checks_compensated_representation(tmp_path):
    path=tmp_path/'state.h5'
    checkpoint(path)
    result=window.final_state(path,dict(domain=[.03,.0045,.00075]))
    assert result['represented_states']['q']['final_uniform_cartesian_cfl_upper_bound']<.5
    assert result['represented_states']['q-minus-carry']['transport_sum_violations']==0
    with h5py.File(path,'r+') as file:
        file['q0017'][:]=1.
    with pytest.raises(ValueError,match='positivity/closure'):
        window.final_state(path,dict(domain=[.03,.0045,.00075]))


def test_long_window_requires_preflight(tmp_path,monkeypatch):
    monkeypatch.setattr(window,'CAMPAIGN',tmp_path)
    case=tmp_path/'window'
    case.mkdir()
    (case/'result.json').write_text(json.dumps(dict(status='failed')))
    with pytest.raises(ValueError,match='matched 100-step preflight'):
        window.run(SimpleNamespace(output=case,stage='long'))
