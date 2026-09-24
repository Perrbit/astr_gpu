from pathlib import Path

import numpy as np
import pytest

from air5_postshock_reference import Air5PostShockReference
from air5_radau_reference import Air5RadauReference
from generate_air5_oblique_shock_states import frozen_oblique_jump, jump_metadata, normal_flux, top_state

ROOT = Path(__file__).resolve().parents[2]


def make_jump(**kwargs):
    arguments = dict(mach=8., temperature=500., tv=500., pressure=5000., shock_angle_deg=30.)
    arguments.update(kwargs)
    return frozen_oblique_jump(Air5RadauReference(ROOT/'chemMech/air5_kimjo12.json'), **arguments)


@pytest.mark.parametrize('beta', [10., 20., 30., 60., 90.])
def test_all_eleven_normal_fluxes_are_conserved(beta):
    jump = make_jump(shock_angle_deg=beta)
    a = normal_flux(jump.upstream_q, jump.pressure[0], jump.normal)
    b = normal_flux(jump.downstream_q, jump.pressure[1], jump.normal)
    np.testing.assert_allclose(a, b, rtol=2e-14, atol=1e-10)


def test_tangential_speed_species_and_specific_vibration_are_frozen():
    jump = make_jump()
    a, b = jump.upstream_q, jump.downstream_q
    np.testing.assert_allclose(a[5:]/a[0], b[5:]/b[0], rtol=2e-15)
    assert np.dot(a[1:4]/a[0], jump.tangent) == pytest.approx(np.dot(b[1:4]/b[0], jump.tangent), rel=2e-15)
    assert b[2] < 0
    assert b[1] > 0


def test_normal_limit_matches_existing_reference():
    jump = make_jump(shock_angle_deg=90.)
    ref = Air5PostShockReference(Air5RadauReference(ROOT/'chemMech/air5_kimjo12.json'))
    np.testing.assert_allclose(jump.downstream_q, ref.conservative_state(ref.postshock), rtol=3e-15, atol=1e-10)


def test_rotated_inflow_preserves_flux_and_deflection():
    a, b = make_jump(), make_jump(inflow_angle_deg=2.)
    alpha = np.deg2rad(2.)
    rotation = np.array([[np.cos(alpha), -np.sin(alpha), 0.],
                         [np.sin(alpha), np.cos(alpha), 0.], [0., 0., 1.]])
    np.testing.assert_allclose(b.downstream_q[1:4], rotation@a.downstream_q[1:4], rtol=3e-15)
    np.testing.assert_allclose(normal_flux(b.upstream_q, b.pressure[0], b.normal),
                               normal_flux(b.downstream_q, b.pressure[1], b.normal), rtol=2e-14, atol=1e-10)
    x = jump_metadata(a, top_x=.004, top_y=.004)
    y = jump_metadata(b, top_x=.004, top_y=.004)
    assert x['deflection_deg'] == pytest.approx(y['deflection_deg'], rel=3e-15)
    assert y['geometric_wall_intersection_x'] == pytest.approx(.004+.004/np.tan(np.deg2rad(28.)))


def test_geometric_top_switch_and_wall_intersection():
    jump = make_jump()
    data = jump_metadata(jump, top_x=.004, top_y=.004)
    assert data['geometric_wall_intersection_x'] == pytest.approx(.004+.004/np.tan(np.pi/6))
    np.testing.assert_array_equal(top_state(jump, .003, .004), jump.upstream_q)
    np.testing.assert_array_equal(top_state(jump, .004, .004), jump.downstream_q)
    np.testing.assert_array_equal(top_state(jump, .005, .004), jump.downstream_q)
    copy = top_state(jump, .005, .004)
    copy[:] = 0
    assert np.any(jump.downstream_q != 0)


def test_theta_beta_m_relation_uses_mixture_gamma_not_fixed_air_gamma():
    jump = make_jump(mass_fraction=np.array([.6, .2, .1, .05, .05]))
    beta, mach, gamma = np.pi/6, 8., jump.gamma_tr
    expected = 2/np.tan(beta)*(mach**2*np.sin(beta)**2-1)/(mach**2*(gamma+np.cos(2*beta))+2)
    v = jump.downstream_q[1:4]/jump.downstream_q[0]
    assert -v[1]/v[0] == pytest.approx(expected, rel=2e-14)
    assert abs(gamma-1.4) > .01


@pytest.mark.parametrize('kwargs', [dict(mach=1.), dict(mach=np.nan),
    dict(shock_angle_deg=1.), dict(shock_angle_deg=91.), dict(temperature=299.),
    dict(tv=8001.), dict(pressure=999.), dict(temperature=4000., shock_angle_deg=90.),
    dict(pressure=1e6), dict(inflow_angle_deg=31.),
    dict(mass_fraction=np.array([.8, .2, -.01, .01, 0.]))])
def test_invalid_jump_fails_without_clipping(kwargs):
    with pytest.raises(ValueError):
        make_jump(**kwargs)
