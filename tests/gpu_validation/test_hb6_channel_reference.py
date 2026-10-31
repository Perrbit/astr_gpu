"""Reference self-checks independent of ASTR's wall/filter coefficient tables."""
import numpy as np

from hb6_channel_reference import (FORCE, REYNOLDS, MACH, GAMMA, PRANDTL,
                                   initial_fields, steady_reference)


def test_constant_property_analytic_limit():
    reference = steady_reference(variable=False)
    y = np.linspace(-1., 1., 257)
    _, u, temperature = reference.primitive(y)
    exact_u = FORCE*REYNOLDS*(1-y*y)/2
    exact_t = 1 + PRANDTL*(GAMMA-1)*MACH**2*FORCE**2*REYNOLDS**2*(1-y**4)/12
    np.testing.assert_allclose(u, exact_u, rtol=0, atol=2e-10)
    np.testing.assert_allclose(temperature, exact_t, rtol=0, atol=2e-10)


def test_sutherland_mass_force_and_heat():
    checks = steady_reference().checks()
    assert checks['minimum_temperature'] >= 1
    assert abs(checks['mean_density']-1) <= 2e-13
    for name in ('force_shear_residual', 'power_heat_residual', 'maximum_shear_error'):
        assert abs(checks[name]) <= 2e-10, checks
    assert checks['maximum_bvp_residual'] <= 1e-10


def test_three_dimensional_initialization_and_wall_compatibility():
    coordinates, base = initial_fields((32, 32, 16))
    _, perturbed = initial_fields((32, 32, 16), perturbation=1e-4)
    for name in ('ro', 't'):
        np.testing.assert_array_equal(base[name], perturbed[name])
    for name in ('u1', 'u2', 'u3'):
        np.testing.assert_array_equal(perturbed[name][:, (0, -1), :], 0.)
        np.testing.assert_allclose(perturbed[name][0], perturbed[name][-1], rtol=0, atol=1e-14)
        np.testing.assert_allclose(perturbed[name][..., 0], perturbed[name][..., -1], rtol=0, atol=1e-14)
    assert np.max(np.abs(perturbed['u1']-base['u1'])) <= 1e-4
    assert np.ptp(perturbed['u2']) > 0
    assert np.ptp(perturbed['u2'][0, 16]) > 1e-6
    assert coordinates[1].min() == -1 and coordinates[1].max() == 1


def test_kinetic_deviation_quadrature_constant_field():
    from compare_hb6_channel_windows import kinetic_deviation
    grid=(8,8,4)
    rho=np.full(tuple(n+1 for n in grid),2.)
    base=np.zeros(rho.shape+(3,))
    velocity=np.broadcast_to(np.array([1.,2.,3.]),base.shape)
    expected=.5*2*14*(2*np.pi*2*np.pi)
    np.testing.assert_allclose(kinetic_deviation(rho,velocity,base,grid),expected,rtol=0,atol=2e-10)
