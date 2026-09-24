from pathlib import Path

import numpy as np
import pytest

from air5_radau_reference import Air5RadauReference
from run_air5_sbli_long import state_metrics, checkpoint_time, precursor_profiles


def fields():
    shape = (3, 4, 5)
    result = {key:np.full(shape, value) for key,value in
        dict(ro=1.,u1=2905.07,u2=0.,u3=0.,p=101325.,t=450.,tv=450.,
             sp001=.77,sp002=.23,sp003=0.,sp004=0.,sp005=0.).items()}
    return result


def metrics(data):
    thermo = Air5RadauReference(Path(__file__).resolve().parents[2]/'chemMech/air5_kimjo12.json')
    return state_metrics(data, thermo, 2905.07, 101325.)


def test_spanwise_warning_is_recorded_without_waiving_state_gates():
    data = fields()
    data['u3'][0] = 1.
    result = metrics(data)
    assert result['max_spanwise_velocity_over_uinf'] == pytest.approx(1/2905.07)
    assert result['spanwise_pressure_rms_over_pinf'] == 0.


@pytest.mark.parametrize('key,value', [('sp003',-1e-100),('ro',0.),('t',299.),
    ('tv',8001.),('p',1.01e6),('u1',np.nan),('sp001',.8)])
def test_invalid_states_stop_without_clipping(key, value):
    data = fields()
    data[key][0,0,0] = value
    with pytest.raises(ValueError):
        metrics(data)


def test_restart_clock_uses_stored_time_when_dt_changes():
    assert checkpoint_time(1000,2e-6,1100,1e-9) == pytest.approx(2.1e-6)
    assert checkpoint_time(1000,2e-6,1000,1e-9) == 2e-6
    with pytest.raises(ValueError):
        checkpoint_time(1000,2e-6,999,1e-9)


def test_profile_diagnostic_uses_density_weighted_displacement():
    data = fields()
    y = np.linspace(0,1,4)
    data['u1'][:] = y[None,:,None]*2905.07
    data['ro'][:] = (2.-y)[None,:,None]
    data['ro'][0] *= 2.
    data['u1'][0] *= (1.+y)[:,None]
    flux = (data['ro'][:-1]*data['u1'][:-1]).mean(axis=0)[:,0]
    expected = np.trapezoid(1.-flux/flux[-1],y)
    result = precursor_profiles(data,[2.,1.,1.])
    assert len(result['stations']) == 3
    for sample in result['stations']:
        assert sample['displacement_thickness'] == pytest.approx(expected)
        assert sample['wall_pressure'] == 101325.
    data['u1'][:,-1] = 0.
    with pytest.raises(ValueError):
        precursor_profiles(data,[2.,1.,1.])
