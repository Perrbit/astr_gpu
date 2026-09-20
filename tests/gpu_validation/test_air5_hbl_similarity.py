from pathlib import Path

import numpy as np

from tests.gpu_validation.air5_hbl_similarity import Air5FrozenSimilarity
from tests.gpu_validation.air5_hbl_marcher import (
    _derivative_matrix,
    _differentiate,
)
from tests.gpu_validation.air5_hbl_reference import Air5BoundaryLayerState


ROOT = Path(__file__).resolve().parents[2]
MECHANISM = ROOT / "chemMech/air5_kimjo12.json"
EDGE_MASS_FRACTION = np.array(
    [0.766907344, 0.233092656, 1.0e-60, 1.0e-60, 1.0e-60]
)


def _reference() -> Air5FrozenSimilarity:
    return Air5FrozenSimilarity(
        MECHANISM,
        pressure=101325.0,
        edge_velocity=2905.07,
        edge_temperature=450.0,
        wall_temperature=2925.0,
        mass_fraction=EDGE_MASS_FRACTION,
    )


def test_similarity_profile_satisfies_wall_and_edge_contract() -> None:
    reference = _reference()
    y = np.r_[0.0, np.geomspace(2.5e-7, 2.0e-4, 64)]

    profile = reference.profile(1.0612769542744582e-3, y).primitive

    np.testing.assert_allclose(profile[0, 0:2], 0.0, atol=1.0e-14)
    np.testing.assert_allclose(profile[0, 2:4], 2925.0, atol=1.0e-12)
    assert np.all(np.isfinite(profile))
    assert np.all(profile[:, 2:4] > 0.0)
    assert np.min(profile[:, 0]) >= -1.0e-10
    assert np.max(profile[:, 0]) <= 2905.07 * (1.0 + 1.0e-10)
    np.testing.assert_allclose(profile[:, 3], profile[:, 2])
    np.testing.assert_allclose(
        profile[:, 4:8],
        np.broadcast_to(EDGE_MASS_FRACTION[None, 1:5], (y.size, 4)),
    )
    np.testing.assert_allclose(profile[-1, 0], 2905.07, rtol=1.0e-8)
    np.testing.assert_allclose(profile[-1, 2], 450.0, rtol=1.0e-8)


def test_similarity_bvp_reports_converged_fp64_diagnostics() -> None:
    reference = _reference()

    diagnostics = reference.diagnostics

    assert diagnostics.status == 0
    assert diagnostics.iterations > 0
    assert diagnostics.nodes >= 241
    assert diagnostics.maximum_boundary_residual <= 1.0e-8
    assert diagnostics.maximum_rms_residual <= 1.0e-7


def test_similarity_wall_quantities_have_physical_signs_and_scaling() -> None:
    reference = _reference()

    wall_a = reference.wall_quantities(1.0e-3)
    wall_b = reference.wall_quantities(4.0e-3)

    assert wall_a.skin_friction > 0.0
    assert wall_a.shear_stress > 0.0
    assert wall_a.conductive_energy_flux > 0.0
    np.testing.assert_allclose(
        wall_b.skin_friction / wall_a.skin_friction, 0.5, rtol=2.0e-12
    )
    np.testing.assert_allclose(
        wall_b.conductive_energy_flux / wall_a.conductive_energy_flux,
        0.5,
        rtol=2.0e-12,
    )


def test_similarity_profiles_converge_in_discrete_boundary_layer_equations() -> None:
    reference = _reference()
    streamwise_coordinate = 1.0612769542744582e-3
    streamwise_step = 1.0e-6
    residuals = []
    for points in (33, 65, 129):
        normalized_y = np.linspace(0.0, 1.0, points)
        y = 2.0e-4 * normalized_y**1.25
        previous = reference.profile(streamwise_coordinate, y).primitive
        current = reference.profile(
            streamwise_coordinate + streamwise_step, y
        ).primitive
        derivative = _derivative_matrix(y)
        velocity_gradient = _differentiate(derivative, current[:, 0:2])
        temperature_gradient = _differentiate(derivative, current[:, 2])
        previous_flux = []
        current_flux = []
        normal_flux = []
        for index in range(points):
            previous_state = Air5BoundaryLayerState(
                pressure=reference.pressure,
                velocity=np.array(
                    [previous[index, 0], previous[index, 1], 0.0]
                ),
                temperature=previous[index, 2],
                tv=previous[index, 3],
                mass_fraction=reference.mass_fraction,
            )
            current_state = Air5BoundaryLayerState(
                pressure=reference.pressure,
                velocity=np.array([current[index, 0], current[index, 1], 0.0]),
                temperature=current[index, 2],
                tv=current[index, 3],
                mass_fraction=reference.mass_fraction,
            )
            previous_flux.append(
                reference.reference.streamwise_flux(previous_state)[[0, 1, 6]]
            )
            current_flux.append(
                reference.reference.streamwise_flux(current_state)[[0, 1, 6]]
            )
            normal_flux.append(
                reference.reference.wall_normal_flux(
                    current_state,
                    grad_velocity_y=np.array(
                        [
                            velocity_gradient[index, 0],
                            velocity_gradient[index, 1],
                            0.0,
                        ]
                    ),
                    grad_temperature_y=temperature_gradient[index],
                    grad_tv_y=temperature_gradient[index],
                    grad_mass_fraction_y=np.zeros(5),
                )[[0, 1, 6]]
            )
        previous_flux = np.asarray(previous_flux)
        current_flux = np.asarray(current_flux)
        residual = (
            (current_flux - previous_flux) / streamwise_step
            + _differentiate(derivative, np.asarray(normal_flux))
        )
        scale = np.max(np.abs(current_flux), axis=0) / streamwise_coordinate
        residuals.append(np.max(np.abs(residual[3:-3]), axis=0) / scale)

    assert np.all(residuals[1] < residuals[0])
    assert np.all(residuals[2] < residuals[1])
    assert np.all(residuals[2] / residuals[1] < 0.4)
