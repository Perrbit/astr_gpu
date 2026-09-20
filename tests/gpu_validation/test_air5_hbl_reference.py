from pathlib import Path

import numpy as np

from tests.gpu_validation.air5_hbl_reference import (
    Air5BoundaryLayerReference,
    Air5BoundaryLayerState,
    NUM_PARABOLIC_EQUATIONS,
)


ROOT = Path(__file__).resolve().parents[2]
MECHANISM = ROOT / "chemMech/air5_kimjo12.json"


def _state() -> Air5BoundaryLayerState:
    return Air5BoundaryLayerState(
        pressure=5000.0,
        velocity=np.array([2900.0, 3.0, 0.0]),
        temperature=4200.0,
        tv=3100.0,
        mass_fraction=np.array([0.74, 0.20, 0.01, 0.02, 0.03]),
    )


def test_air5_hbl_streamwise_flux_has_frozen_equation_order() -> None:
    reference = Air5BoundaryLayerReference(MECHANISM)
    state = _state()
    q = reference.conservative_state(state)
    flux = reference.streamwise_flux(state)

    assert flux.shape == (NUM_PARABOLIC_EQUATIONS,)
    np.testing.assert_allclose(flux[0], q[0] * state.velocity[0])
    np.testing.assert_allclose(
        flux[1], q[0] * state.velocity[0] ** 2 + state.pressure
    )
    np.testing.assert_allclose(flux[2:6], state.velocity[0] * q[6:10])
    np.testing.assert_allclose(flux[6], state.velocity[0] * (q[4] + state.pressure))
    np.testing.assert_allclose(flux[7], state.velocity[0] * q[10])


def test_air5_hbl_wall_normal_flux_uses_closed_species_diffusion() -> None:
    reference = Air5BoundaryLayerReference(MECHANISM)
    state = _state()
    gradient = np.array([12.0, -7.0, -2.0, -1.0, -2.0])
    flux = reference.wall_normal_flux(
        state,
        grad_velocity_y=np.array([2.0e6, 0.0, 0.0]),
        grad_temperature_y=-8.0e6,
        grad_tv_y=-3.0e6,
        grad_mass_fraction_y=gradient,
    )

    assert flux.shape == (NUM_PARABOLIC_EQUATIONS,)
    assert np.all(np.isfinite(flux))
    density = reference.density(state)
    grad_velocity = np.zeros((3, 3))
    grad_velocity[:, 1] = np.array([2.0e6, 0.0, 0.0])
    grad_mass_fraction = np.zeros((5, 3))
    grad_mass_fraction[:, 1] = gradient
    diffusion = reference.transport.diffusive_flux(
        rho=density,
        velocity=state.velocity,
        temperature=state.temperature,
        tv=state.tv,
        pressure=state.pressure,
        grad_velocity=grad_velocity,
        grad_temperature=np.array([0.0, -8.0e6, 0.0]),
        grad_tv=np.array([0.0, -3.0e6, 0.0]),
        mass_fraction=state.mass_fraction,
        grad_mass_fraction=grad_mass_fraction,
    )
    np.testing.assert_allclose(
        np.sum(diffusion.species_flux[:, 1]), 0.0, atol=2.0e-14
    )
    np.testing.assert_allclose(
        flux[2:6],
        density * state.velocity[1] * state.mass_fraction[1:5]
        + diffusion.species_flux[1:5, 1],
        rtol=2.0e-12,
        atol=2.0e-12,
    )


def test_air5_hbl_source_preserves_total_mass_elements_and_energy() -> None:
    reference = Air5BoundaryLayerReference(MECHANISM)
    state = _state()
    q = reference.conservative_state(state)
    local = reference.chemistry.rhs(
        0.0, np.r_[q[5:10], q[10]], q[0], q[1:4], q[4]
    )
    source = reference.source(state)

    assert source.shape == (NUM_PARABOLIC_EQUATIONS,)
    np.testing.assert_allclose(source[2:6], local[1:5])
    np.testing.assert_allclose(source[6], 0.0)
    np.testing.assert_allclose(source[7], local[5])
    np.testing.assert_allclose(np.sum(local[:5]), 0.0, atol=2.0e-10)
    element_source = reference.chemistry.atom_counts.T @ (
        local[:5] / reference.chemistry.molar_mass
    )
    np.testing.assert_allclose(element_source, 0.0, atol=2.0e-8)


def test_air5_hbl_reference_rejects_unclosed_composition() -> None:
    reference = Air5BoundaryLayerReference(MECHANISM)
    state = _state()
    invalid = Air5BoundaryLayerState(
        pressure=state.pressure,
        velocity=state.velocity,
        temperature=state.temperature,
        tv=state.tv,
        mass_fraction=state.mass_fraction * 0.99,
    )

    try:
        reference.streamwise_flux(invalid)
    except ValueError as error:
        assert "close" in str(error)
    else:
        raise AssertionError("unclosed air5 composition must fail closed")
