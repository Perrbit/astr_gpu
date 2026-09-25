"""Local flux algebra, not an independent flow integrator."""
import numpy as np

from test_air5_species_convection_limiter import (
    admissible_face_ratio, interior_ratio, state_is_admissible,
)


def targets(high, low, beta):
    fluid = high.copy()
    fluid[6:10] = low[6:10]
    fluid[5] = fluid[0] - fluid[6:10].sum()
    base = low + beta * (fluid - low)
    if beta == 1.:
        base[[0, 1, 2, 3, 4, 10]] = high[[0, 1, 2, 3, 4, 10]]
    base[5] = base[0] - base[6:10].sum()
    target = base.copy()
    target[6:10] = high[6:10]
    target[5] = target[0] - target[6:10].sum()
    return fluid, base, target


def flux(values):
    result = np.asarray(values, dtype=float)
    result[5] = result[0] - result[6:10].sum()
    return result


def test_sequential_flux_closure_and_high_order_endpoint():
    high = flux([3., 8., 9., 10., 50., 0., .4, .1, .2, .3, 4.])
    low = flux([2., 2., 3., 4., 30., 0., .3, .2, .1, 0., 2.])
    for beta in (0., .3, 1.):
        fluid, base, target = targets(high, low, beta)
        np.testing.assert_array_equal(fluid[6:10], low[6:10])
        np.testing.assert_array_equal((target-base)[[0, 1, 2, 3, 4, 10]], 0.)
        for alpha in (0., .1, 1.):
            final = base + alpha*(target-base)
            np.testing.assert_allclose(final[5:10].sum(), final[0], atol=2e-15)
            if alpha == beta == 1.:
                np.testing.assert_array_equal(final, high)


def test_trace_limiting_does_not_change_fixed_fluid_flux():
    high = flux([3., 8., 9., 10., 50., 0., .4, .1, .2, 1e-18, 4.])
    low = flux([2., 2., 3., 4., 30., 0., .3, .2, .1, 0., 2.])
    _, base, target = targets(high, low, 1.)
    for alpha in (0., 1e-8, .5, 1.):
        final = base + alpha*(target-base)
        np.testing.assert_array_equal(final[[0, 1, 2, 3, 4, 10]],
                                      high[[0, 1, 2, 3, 4, 10]])


def species_ratio(state, corrections, cv, formation, ev_floor):
    ratio = 1.
    budget = np.minimum(corrections[:, 5:10], 0.).sum(axis=0)
    for s in range(5):
        if budget[s] < 0.:
            ratio = min(ratio, interior_ratio(state[5+s]/-budget[s]))
    for correction in corrections:
        ratio = min(ratio, admissible_face_ratio(
            state, correction, cv, formation, ev_floor, species_budget=True))
    return ratio


def test_species_redistribution_preserves_energy_and_nonnegativity():
    rng = np.random.default_rng(7304)
    cv = np.array([742., 650., 890., 780., 690.])
    formation = np.array([-3.1e5, -2.7e5, 3.33e7, 1.52e7, 2.72e6])
    ev_floor = np.array([2000., 1500., 0., 0., 1000.])
    for trace in (0., 1e-25, .001):
        state = np.zeros(11)
        state[5:10] = [.78-trace, .2, .01, .01, trace]
        state[0] = state[5:10].sum()
        state[1:4] = [2500., 20., 0.]
        state[10] = state[5:10]@ev_floor + 100.
        state[4] = (state[1:4]@state[1:4]/(2*state[0]) + state[10]
                    + state[5:10]@formation + 500*(state[5:10]@cv))
        for _ in range(30):
            correction = np.zeros((6, 11))
            correction[:, 6:10] = rng.normal(size=(6, 4))*.03
            correction[:, 5] = -correction[:, 6:10].sum(axis=1)
            ratio = species_ratio(state, correction, cv, formation, ev_floor)
            # Each neighbor can reduce its common face coefficient independently.
            theta = rng.uniform(0., ratio, 6)
            final = state+(theta[:, None]*correction).sum(axis=0)
            assert state_is_admissible(final, cv, formation, ev_floor)
            np.testing.assert_allclose(final[5:10].sum(), final[0], atol=2e-15)
            np.testing.assert_array_equal(final[[0, 1, 2, 3, 4, 10]],
                                          state[[0, 1, 2, 3, 4, 10]])


def test_species_energy_constraint_remains_active():
    cv = np.ones(5)
    formation = np.array([0., 0., 1e6, 0., 0.])
    zero = np.zeros(5)
    state = np.array([1., 0., 0., 0., 401., .8, .2, 0., 0., 0., 1.])
    correction = np.zeros((6, 11))
    correction[0, 5] = -.01
    correction[0, 7] = .01
    ratio = species_ratio(state, correction, cv, formation, zero)
    assert 0. < ratio < .01
    assert state_is_admissible(state+ratio*correction.sum(axis=0), cv, formation, zero)


def test_shared_face_flux_conserves_each_component():
    high = flux([3., 8., 9., 10., 50., 0., .4, .1, .2, .3, 4.])
    low = flux([2., 2., 3., 4., 30., 0., .3, .2, .1, 0., 2.])
    _, base, target = targets(high, low, min(.2, .8))
    final = base + min(.1, .9)*(target-base)
    np.testing.assert_array_equal(-final+final, np.zeros(11))
