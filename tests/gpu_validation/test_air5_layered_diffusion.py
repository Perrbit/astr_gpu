"""Local flux algebra only; ASTR remains the sole flow time integrator."""
import numpy as np


def test_rotated_slab_canonicalization():
    from run_air5_layered_stress_gate import canonical_field
    original = np.arange(8*3*4*11).reshape(8, 3, 4, 11)
    for axis in range(3):
        rotated = np.swapaxes(original, 0, axis).copy()
        rotated[..., [1, 1+axis]] = rotated[..., [1+axis, 1]]
        np.testing.assert_array_equal(canonical_field(rotated, axis), original)
        parts = np.split(rotated, 2, axis=axis)
        assembled = np.concatenate([canonical_field(p, axis) for p in parts], axis=0)
        np.testing.assert_array_equal(assembled, original)
        scalar = original[..., 0]
        np.testing.assert_array_equal(
            canonical_field(np.swapaxes(scalar, 0, axis), axis, momentum=False), scalar)

from test_air5_species_convection_limiter import (
    admissible_face_ratio, interior_ratio, state_is_admissible,
)


def mix(flux, carried, alpha):
    result = flux.copy()
    result[5:10] *= alpha
    if alpha != 1:
        result[[4, 10]] += (alpha-1)*carried[[4, 10]]
    return result


def test_species_layer_leaves_stress_and_scales_carried_energy():
    flux = np.array([0., 2., -3., 4., 100., .1, -.2, .05, .03, .02, 30.])
    carried = np.zeros(11)
    carried[4], carried[10] = 60., 20.
    for alpha in (0., .2, 1.):
        limited = mix(flux, carried, alpha)
        np.testing.assert_array_equal(limited[1:4], flux[1:4])
        np.testing.assert_allclose(limited[4], 40.+alpha*60.)
        np.testing.assert_allclose(limited[10], 10.+alpha*20.)
        np.testing.assert_allclose(limited[5:10].sum(), 0., atol=1e-16)
    np.testing.assert_array_equal(mix(flux, carried, 1.), flux)


def test_stress_fixtures_preserve_eos_and_periodic_endpoints():
    from air5_radau_reference import Air5RadauReference
    from run_air5_layered_stress_gate import fixture_fields, ROOT
    thermo = Air5RadauReference(ROOT/'chemMech/air5_kimjo12.json')
    for kind in ('energy', 'species'):
        species, density, tv = fixture_fields(kind, thermo)
        assert species.shape == (49, 5) and species.min() >= 0
        np.testing.assert_allclose(species.sum(axis=1), 1., atol=1e-15)
        np.testing.assert_array_equal(species[0], species[-1])
        assert tv[0] == tv[-1] and tv.min() >= 300 and tv.max() <= 8000
        np.testing.assert_allclose(density*(species@thermo.gas_constant)*4000, 1e5)
        if kind == 'species':
            assert species[24, 2] == 0 and species[26, 2] > 0


def test_energy_layer_can_still_limit_stress():
    cv = np.ones(5)
    zeros = np.zeros(5)
    base = np.array([1., 0., 0., 0., 310., .7, .2, .1, 0., 0., 1.])
    face = np.zeros(11)
    face[1] = 100.
    beta = admissible_face_ratio(base, face, cv, zeros, zeros, True)
    assert 0 < beta < 1
    assert state_is_admissible(base+6*beta*face, cv, zeros, zeros)


def test_separate_budgets_allow_independent_neighbor_restrictions():
    rng = np.random.default_rng(20260924)
    cv = np.array([742., 650., 890., 780., 690.])
    formation = np.array([-3.1e5, -2.7e5, 3.33e7, 1.52e7, 2.72e6])
    ev = np.array([2000., 1500., 0., 0., 1000.])
    thermal_limited = False
    for trace in (0., 1e-25, .003):
        base = np.array([1., 2500., 20., 0., 0., .78, .21, trace, .002, .008-trace, 0.])
        base[10] = base[5:10]@ev+1000
        base[4] = base[1:4]@base[1:4]/2+base[10]+base[5:10]@formation+500*(base[5:10]@cv)
        for _ in range(20):
            faces = rng.normal(size=(6, 11))
            faces[:, 0] = 0
            faces[:, 5:10] *= np.maximum(base[5:10], 1e-26)
            faces[:, 5] = -faces[:, 6:10].sum(axis=1)
            faces[:, 1:4] *= 1000
            faces[:, 4] *= 1e6
            faces[:, 10] *= 1000
            carried = np.zeros_like(faces)
            carried[:, 4] = faces[:, 5:10]@(800*cv+formation+ev)
            carried[:, 10] = faces[:, 5:10]@ev
            negative = np.minimum(faces[:, 5:10], 0).sum(axis=0)
            alpha = min([1.]+[interior_ratio(base[5+s]/-negative[s])
                for s in range(5) if negative[s] < 0])
            mixed = np.array([mix(f, h, a) for f, h, a in
                zip(faces, carried, rng.uniform(0, alpha, 6))])
            beta = min(admissible_face_ratio(base, f, cv, formation, ev, True) for f in mixed)
            thermal_limited |= beta < 1
            final = base+(rng.uniform(0, beta, 6)[:, None]*mixed).sum(axis=0)
            assert state_is_admissible(final, cv, formation, ev)
            np.testing.assert_allclose(final[5:10].sum(), final[0], atol=1e-15)
    assert thermal_limited
