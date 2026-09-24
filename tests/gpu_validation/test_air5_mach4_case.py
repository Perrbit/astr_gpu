import json

import numpy as np
import pytest

from prepare_air5_mach4_case import prepare, seed_profile, rescale_inlet_profile


def test_seed_matches_approved_state_and_retains_zero_species():
    rows, meta = seed_profile()
    assert rows[0, 2] == 0.
    np.testing.assert_allclose(rows[[0,-1],6:8], [[3000.,3000.],[1500.,1500.]])
    assert np.all(rows[:,5] == 20000.)
    np.testing.assert_array_equal(rows[:,10:], 0.)
    np.testing.assert_allclose(rows[:,8:10], np.broadcast_to([.767,.233],(len(rows),2)))
    assert meta['deflection_deg'] == pytest.approx(12.93744917198)
    assert meta['pressure'][1] == pytest.approx(63346.3128585)
    assert meta['temperature'][1] == pytest.approx(2177.25977099)
    assert meta['max_scaled_normal_flux_residual'] < 2e-12
    assert meta['top_x'] < meta['geometric_wall_intersection_x'] < meta['domain'][0]
    displacement = np.trapezoid(1.-rows[:,1]*rows[:,2]/(rows[-1,1]*rows[-1,2]),rows[:,0])
    assert displacement == pytest.approx(meta['ref_len'],rel=1e-10)


def test_precursor_has_no_incident_forcing(tmp_path):
    case = tmp_path/'case'
    text = prepare(case).read_text()
    assert 'air5hbl\n' in text and 'air5sbli\n' not in text
    assert '41,3000.d0' in text
    assert '3,f,0.3d0,0.05d0' in text
    assert 'f,t,f,f,f,f,t,t,t' in text
    assert not (case/'datin/air5_incident_shock.dat').exists()
    assert 'HTR' not in (case/'datin/air5_hbl_profile.dat').read_text()
    meta = json.loads((case/'mach4_case_metadata.json').read_text())
    np.testing.assert_allclose(np.loadtxt(case/'datin/air5_hbl_domain.dat',skiprows=1),meta['domain'])
    assert not meta['incident_shock_enabled']
    with pytest.raises(FileExistsError):
        prepare(case)


@pytest.mark.parametrize('grid,dt', [('2,31,7','1e-9'),('31,31,7','nan'),('31,31,7','0')])
def test_invalid_input_does_not_create_case(tmp_path,grid,dt):
    with pytest.raises(ValueError):
        prepare(tmp_path/'invalid',grid,deltat=dt)
    assert not (tmp_path/'invalid').exists()


@pytest.mark.parametrize('scale',[.9,1.1])
def test_inlet_scale_preserves_states_and_top(tmp_path,scale):
    prepare(tmp_path/'case')
    path=tmp_path/'case/datin/air5_hbl_profile.dat'
    before=np.loadtxt(path)
    rescale_inlet_profile(path,scale)
    after=np.loadtxt(path)
    np.testing.assert_array_equal(after[:,1:],before[:,1:])
    np.testing.assert_allclose(after[:-1,0],before[:-1,0]*scale)
    assert after[-1,0]==before[-1,0]
    assert after[0,0]==0.
    saved=path.read_bytes()
    with pytest.raises(ValueError):
        rescale_inlet_profile(path,1000.)
    assert path.read_bytes()==saved
