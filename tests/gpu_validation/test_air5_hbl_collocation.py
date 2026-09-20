import os
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

from tests.gpu_validation.air5_hbl_collocation import (
    Air5HblCollocation,
    Air5HblCollocationBoundary,
    _derivative_matrix,
    _solve_square_hybrid,
)
from tests.gpu_validation.air5_hbl_reference import Air5BoundaryLayerState
from tests.gpu_validation.air5_hbl_similarity import Air5FrozenSimilarity


ROOT = Path(__file__).resolve().parents[2]
MECHANISM = ROOT / "chemMech/air5_kimjo12.json"


def test_square_hybrid_closes_nonlinear_system_to_explicit_residual_gate() -> None:
    solution, residual_max, evaluations = _solve_square_hybrid(
        lambda values: np.array(
            [values[0] ** 2 - 2.0, values[1] ** 3 - 27.0]
        ),
        np.array([1.0, 2.0]),
        residual_tolerance=1.0e-12,
        max_function_evaluations=200,
    )

    np.testing.assert_allclose(solution, np.array([np.sqrt(2.0), 3.0]))
    assert residual_max <= 1.0e-12
    assert evaluations <= 200


def test_five_point_derivative_is_exact_for_quartic_on_stretched_grid() -> None:
    coordinate = np.linspace(0.0, 1.0, 13) ** 1.25
    values = 1.0 + 2.0 * coordinate - 3.0 * coordinate**2 + coordinate**4
    expected = 2.0 - 6.0 * coordinate + 4.0 * coordinate**3

    derivative = _derivative_matrix(coordinate, stencil_width=5) @ values

    np.testing.assert_allclose(derivative, expected, atol=2.0e-12, rtol=0.0)


def test_vectorized_frozen_fluxes_match_pointwise_fp64_oracle() -> None:
    solver, _, _, _, _ = _uniform_problem()
    temperature = np.array([[1800.0, 2200.0, 2600.0], [1950.0, 2350.0, 2750.0]])
    primitive = np.empty(temperature.shape + (8,))
    primitive[:, :, 0] = np.array([[300.0, 900.0, 1500.0], [450.0, 1050.0, 1650.0]])
    primitive[:, :, 1] = np.array([[2.0, 3.0, 4.0], [2.5, 3.5, 4.5]])
    primitive[:, :, 2] = temperature
    primitive[:, :, 3] = temperature
    primitive[:, :, 4:8] = solver.boundary.mass_fraction[None, None, 1:5]
    velocity_gradient_y = np.array(
        [
            [[1.0e5, -2.0e3], [1.2e5, -1.5e3], [1.4e5, -1.0e3]],
            [[1.1e5, -1.8e3], [1.3e5, -1.3e3], [1.5e5, -0.8e3]],
        ]
    )
    temperature_gradient_y = np.array(
        [[2.0e6, 1.5e6, 1.0e6], [1.8e6, 1.3e6, 0.8e6]]
    )

    streamwise, normal = solver._source_off_single_temperature_fluxes(
        primitive, velocity_gradient_y, temperature_gradient_y
    )

    expected_streamwise = np.empty_like(streamwise)
    expected_normal = np.empty_like(normal)
    for index in np.ndindex(temperature.shape):
        state = Air5BoundaryLayerState(
            pressure=solver.boundary.pressure,
            velocity=np.array([primitive[index][0], primitive[index][1], 0.0]),
            temperature=temperature[index],
            tv=temperature[index],
            mass_fraction=solver.boundary.mass_fraction,
        )
        expected_streamwise[index] = solver.reference.streamwise_flux(state)[[1, 6]]
        expected_normal[index] = solver.reference.wall_normal_flux(
            state,
            grad_velocity_y=np.array(
                [velocity_gradient_y[index][0], velocity_gradient_y[index][1], 0.0]
            ),
            grad_temperature_y=temperature_gradient_y[index],
            grad_tv_y=temperature_gradient_y[index],
            grad_mass_fraction_y=np.zeros(5),
        )[[1, 6]]

    np.testing.assert_allclose(streamwise, expected_streamwise, rtol=2.0e-15)
    np.testing.assert_allclose(normal, expected_normal, rtol=2.0e-15)


def test_vectorized_two_temperature_fluxes_match_pointwise_fp64_oracle() -> None:
    solver, _, _, _, _ = _uniform_problem()
    temperature = np.array([[1800.0, 2400.0], [2100.0, 2700.0]])
    tv = np.array([[1200.0, 1600.0], [1400.0, 1900.0]])
    primitive = np.empty(temperature.shape + (8,))
    primitive[:, :, 0] = np.array([[400.0, 1200.0], [700.0, 1700.0]])
    primitive[:, :, 1] = np.array([[2.0, 4.0], [3.0, 5.0]])
    primitive[:, :, 2] = temperature
    primitive[:, :, 3] = tv
    primitive[:, :, 4:8] = solver.boundary.mass_fraction[None, None, 1:5]
    velocity_gradient_y = np.array(
        [[[1.0e5, -2.0e3], [1.4e5, -1.0e3]],
         [[1.2e5, -1.5e3], [1.6e5, -0.5e3]]]
    )
    temperature_gradient_y = np.array([[2.0e6, 1.0e6], [1.5e6, 0.5e6]])
    tv_gradient_y = np.array([[0.8e6, 0.4e6], [0.6e6, 0.2e6]])

    streamwise, normal = solver._frozen_two_temperature_fluxes(
        primitive,
        velocity_gradient_y,
        temperature_gradient_y,
        tv_gradient_y,
    )

    expected_streamwise = np.empty_like(streamwise)
    expected_normal = np.empty_like(normal)
    for index in np.ndindex(temperature.shape):
        state = Air5BoundaryLayerState(
            pressure=solver.boundary.pressure,
            velocity=np.array([primitive[index][0], primitive[index][1], 0.0]),
            temperature=temperature[index],
            tv=tv[index],
            mass_fraction=solver.boundary.mass_fraction,
        )
        expected_streamwise[index] = solver.reference.streamwise_flux(state)[
            [1, 6, 7]
        ]
        expected_normal[index] = solver.reference.wall_normal_flux(
            state,
            grad_velocity_y=np.array(
                [velocity_gradient_y[index][0], velocity_gradient_y[index][1], 0.0]
            ),
            grad_temperature_y=temperature_gradient_y[index],
            grad_tv_y=tv_gradient_y[index],
            grad_mass_fraction_y=np.zeros(5),
        )[[1, 6, 7]]

    np.testing.assert_allclose(streamwise, expected_streamwise, rtol=2.0e-15)
    np.testing.assert_allclose(normal, expected_normal, rtol=4.0e-15)


def test_direct_vt_source_matches_general_fp64_oracle() -> None:
    solver, _, _, _, _ = _uniform_problem()
    primitive = np.empty((2, 2, 8))
    primitive[:, :, 0] = np.array([[400.0, 1200.0], [700.0, 1700.0]])
    primitive[:, :, 1] = 0.0
    primitive[:, :, 2] = np.array([[1800.0, 2400.0], [2100.0, 2700.0]])
    primitive[:, :, 3] = np.array([[1200.0, 1600.0], [1400.0, 1900.0]])
    primitive[:, :, 4:8] = solver.boundary.mass_fraction[None, None, 1:5]

    direct = solver._frozen_vt_source(primitive)

    expected = np.empty((2, 2))
    for index in np.ndindex(expected.shape):
        state = Air5BoundaryLayerState(
            pressure=solver.boundary.pressure,
            velocity=np.array([primitive[index][0], 0.0, 0.0]),
            temperature=primitive[index][2],
            tv=primitive[index][3],
            mass_fraction=solver.boundary.mass_fraction,
        )
        expected[index] = solver.reference.source(state, mode="vt")[7]

    np.testing.assert_allclose(direct, expected, rtol=2.0e-15)


def _uniform_problem() -> tuple[
    Air5HblCollocation,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    mass_fraction = np.array([0.765, 0.234, 2.0e-4, 3.0e-4, 5.0e-4])
    boundary = Air5HblCollocationBoundary(
        pressure=8000.0,
        edge_velocity=2500.0,
        edge_temperature=1800.0,
        wall_velocity=2500.0,
        wall_temperature=1800.0,
        mass_fraction=mass_fraction,
    )
    solver = Air5HblCollocation(MECHANISM, boundary)
    x = np.linspace(1.0e-3, 1.4e-3, 4)
    y = np.linspace(0.0, 2.0e-3, 9)
    density = boundary.pressure / (
        np.dot(mass_fraction, solver.reference.chemistry.gas_constant)
        * boundary.edge_temperature
    )
    streamfunction = np.broadcast_to(
        density * boundary.edge_velocity * y[None, :], (x.size, y.size)
    ).copy()
    temperature = np.full_like(streamfunction, boundary.edge_temperature)
    return solver, x, y, streamfunction, temperature


def test_global_source_off_residual_preserves_uniform_flow() -> None:
    solver, x, y, streamfunction, temperature = _uniform_problem()

    residual = solver.source_off_single_temperature_residual(
        streamfunction, temperature, x, y
    )

    assert np.max(np.abs(residual)) < 1.0e-12


def test_global_vt_residual_preserves_uniform_equilibrium_flow() -> None:
    solver, x, y, streamfunction, temperature = _uniform_problem()

    residual = solver.frozen_vt_residual(
        streamfunction, temperature, temperature, x, y
    )

    assert np.max(np.abs(residual)) < 1.0e-12


def test_global_vt_solver_does_not_drift_uniform_equilibrium_flow() -> None:
    solver, x, y, streamfunction, temperature = _uniform_problem()

    result = solver.solve_frozen_vt(
        x,
        y,
        initial_streamfunction=streamfunction,
        initial_temperature=temperature,
        initial_tv=temperature,
    )

    assert result.scaled_residual_max < 1.0e-12
    np.testing.assert_allclose(
        result.mass_streamfunction, streamfunction, atol=1.0e-12, rtol=0.0
    )
    np.testing.assert_allclose(
        result.primitive[:, :, 2], temperature, atol=1.0e-12, rtol=0.0
    )
    np.testing.assert_allclose(
        result.primitive[:, :, 3], temperature, atol=1.0e-12, rtol=0.0
    )


def test_frozen_ev_predictor_does_not_drift_uniform_equilibrium_flow() -> None:
    solver, x, y, streamfunction, temperature = _uniform_problem()

    predicted_tv = solver._predict_frozen_tv(
        streamfunction, temperature, temperature, x, y
    )

    np.testing.assert_allclose(predicted_tv, temperature, atol=1.0e-12, rtol=0.0)


def test_global_vt_continuation_does_not_drift_uniform_equilibrium_flow() -> None:
    solver, x, y, streamfunction, temperature = _uniform_problem()

    result = solver.solve_frozen_vt_continuation(
        x,
        y,
        initial_streamfunction=streamfunction,
        initial_temperature=temperature,
        initial_tv=temperature,
    )

    assert result.scaled_residual_max < 1.0e-12
    np.testing.assert_allclose(result.primitive[:, :, 2], temperature)
    np.testing.assert_allclose(result.primitive[:, :, 3], temperature)

    wall = solver.wall_quantities(result)
    np.testing.assert_allclose(wall.skin_friction, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(wall.translational_heat_flux, 0.0, atol=2.0e-10)
    np.testing.assert_allclose(wall.vibrational_heat_flux, 0.0, atol=1.0e-10)
    np.testing.assert_allclose(wall.total_heat_flux, 0.0, atol=2.0e-10)


def test_global_source_off_solver_does_not_drift_uniform_flow() -> None:
    solver, x, y, streamfunction, temperature = _uniform_problem()

    result = solver.solve_source_off_single_temperature(
        x,
        y,
        initial_streamfunction=streamfunction,
        initial_temperature=temperature,
    )

    assert result.scaled_residual_max < 1.0e-12
    np.testing.assert_allclose(
        result.mass_streamfunction, streamfunction, atol=1.0e-12, rtol=0.0
    )
    np.testing.assert_allclose(
        result.primitive[:, :, 2], temperature, atol=1.0e-12, rtol=0.0
    )
    np.testing.assert_allclose(result.primitive[:, :, 3], temperature)
    np.testing.assert_allclose(
        result.primitive[:, :, 4:8],
        np.broadcast_to(
            solver.boundary.mass_fraction[None, None, 1:5],
            (x.size, y.size, 4),
        ),
    )


def test_global_source_off_residual_converges_to_dorodnitsyn_solution() -> None:
    mass_fraction = np.array(
        [0.766907344, 0.233092656, 1.0e-60, 1.0e-60, 1.0e-60]
    )
    similarity = Air5FrozenSimilarity(
        MECHANISM,
        pressure=101325.0,
        edge_velocity=2905.07,
        edge_temperature=450.0,
        wall_temperature=2925.0,
        mass_fraction=mass_fraction,
    )
    boundary = Air5HblCollocationBoundary(
        pressure=similarity.pressure,
        edge_velocity=similarity.edge_velocity,
        edge_temperature=similarity.edge_temperature,
        wall_velocity=0.0,
        wall_temperature=similarity.wall_temperature,
        mass_fraction=mass_fraction,
    )
    solver = Air5HblCollocation(MECHANISM, boundary)
    residual_maxima = []
    for x_points, y_points in ((4, 13), (5, 17), (7, 25)):
        x = np.linspace(
            1.0612769542744582e-3, 1.4612769542744582e-3, x_points
        )
        normalized_y = np.linspace(0.0, 1.0, y_points)
        y = 2.0e-4 * normalized_y**1.25
        profiles = [similarity.profile(station, y) for station in x]
        streamfunction = np.stack(
            [profile.mass_streamfunction for profile in profiles]
        )
        temperature = np.stack(
            [profile.primitive[:, 2] for profile in profiles]
        )

        residual = solver.source_off_single_temperature_residual(
            streamfunction,
            temperature,
            x,
            y,
            inlet_streamfunction=streamfunction[0],
            inlet_temperature=temperature[0],
        )
        residual_maxima.append(np.max(np.abs(residual)))

    assert residual_maxima[1] < residual_maxima[0]
    assert residual_maxima[2] < residual_maxima[1]
    assert residual_maxima[2] / residual_maxima[0] < 0.5


def test_global_source_off_solver_converges_to_dorodnitsyn_profiles() -> None:
    mass_fraction = np.array(
        [0.766907344, 0.233092656, 1.0e-60, 1.0e-60, 1.0e-60]
    )
    similarity = Air5FrozenSimilarity(
        MECHANISM,
        pressure=101325.0,
        edge_velocity=2905.07,
        edge_temperature=450.0,
        wall_temperature=2925.0,
        mass_fraction=mass_fraction,
    )
    solver = Air5HblCollocation(
        MECHANISM,
        Air5HblCollocationBoundary(
            pressure=similarity.pressure,
            edge_velocity=similarity.edge_velocity,
            edge_temperature=similarity.edge_temperature,
            wall_velocity=0.0,
            wall_temperature=similarity.wall_temperature,
            mass_fraction=mass_fraction,
        ),
    )
    x = np.linspace(1.0612769542744582e-3, 1.2612769542744582e-3, 3)
    velocity_errors = []
    temperature_errors = []
    final_result = None
    final_reference = None
    final_reference_streamfunction = None
    for y_points in (13, 17, 25):
        y = 2.0e-4 * np.linspace(0.0, 1.0, y_points) ** 1.25
        profiles = [similarity.profile(station, y) for station in x]
        reference_primitive = np.stack(
            [profile.primitive for profile in profiles]
        )
        reference_streamfunction = np.stack(
            [profile.mass_streamfunction for profile in profiles]
        )

        result = solver.solve_source_off_single_temperature(
            x,
            y,
            initial_streamfunction=reference_streamfunction,
            initial_temperature=reference_primitive[:, :, 2],
            residual_tolerance=1.0e-10,
            max_function_evaluations=300,
        )

        assert result.scaled_residual_max <= 1.0e-10
        assert np.all(np.isfinite(result.primitive))
        assert np.all(result.primitive[:, :, 2:4] > 0.0)
        velocity_errors.append(
            np.linalg.norm(
                result.primitive[1:, :, 0] - reference_primitive[1:, :, 0]
            )
            / np.linalg.norm(reference_primitive[1:, :, 0])
        )
        temperature_errors.append(
            np.linalg.norm(
                result.primitive[1:, :, 2] - reference_primitive[1:, :, 2]
            )
            / np.linalg.norm(reference_primitive[1:, :, 2])
        )
        final_result = result
        final_reference = reference_primitive
        final_reference_streamfunction = reference_streamfunction

    assert velocity_errors[1] < velocity_errors[0]
    assert velocity_errors[2] < velocity_errors[1]
    assert temperature_errors[1] < temperature_errors[0]
    assert temperature_errors[2] < temperature_errors[1]
    assert velocity_errors[-1] < 1.0e-2
    assert temperature_errors[-1] < 1.1e-2
    assert final_result is not None
    assert final_reference is not None
    assert final_reference_streamfunction is not None
    np.testing.assert_allclose(
        final_result.primitive[:, :, 3], final_result.primitive[:, :, 2]
    )
    np.testing.assert_allclose(
        final_result.mass_streamfunction[0],
        final_reference_streamfunction[0],
        atol=2.0e-14,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        final_result.primitive[0, :, 2],
        final_reference[0, :, 2],
        atol=2.0e-10,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        final_result.primitive[1:, 0, 0], 0.0, atol=1.0e-10
    )
    np.testing.assert_allclose(
        final_result.primitive[1:, -1, 0], similarity.edge_velocity, rtol=1.0e-10
    )
    np.testing.assert_allclose(
        final_result.primitive[1:, 0, 2], similarity.wall_temperature, rtol=1.0e-12
    )
    np.testing.assert_allclose(
        final_result.primitive[1:, -1, 2], similarity.edge_temperature, rtol=1.0e-12
    )


def test_global_frozen_vt_continuation_closes_nonuniform_flat_plate() -> None:
    mass_fraction = np.array(
        [0.766907344, 0.233092656, 1.0e-60, 1.0e-60, 1.0e-60]
    )
    similarity = Air5FrozenSimilarity(
        MECHANISM,
        pressure=101325.0,
        edge_velocity=2905.07,
        edge_temperature=450.0,
        wall_temperature=2925.0,
        mass_fraction=mass_fraction,
    )
    solver = Air5HblCollocation(
        MECHANISM,
        Air5HblCollocationBoundary(
            pressure=similarity.pressure,
            edge_velocity=similarity.edge_velocity,
            edge_temperature=similarity.edge_temperature,
            wall_velocity=0.0,
            wall_temperature=similarity.wall_temperature,
            mass_fraction=mass_fraction,
        ),
    )
    x = np.linspace(1.0612769542744582e-3, 1.2612769542744582e-3, 3)
    y = 2.0e-4 * np.linspace(0.0, 1.0, 13) ** 1.25
    profiles = [similarity.profile(station, y) for station in x]
    reference_primitive = np.stack(
        [profile.primitive for profile in profiles]
    )
    reference_streamfunction = np.stack(
        [profile.mass_streamfunction for profile in profiles]
    )
    single_temperature = solver.solve_source_off_single_temperature(
        x,
        y,
        initial_streamfunction=reference_streamfunction,
        initial_temperature=reference_primitive[:, :, 2],
        max_function_evaluations=300,
    )

    result = solver.solve_frozen_vt_continuation(
        x,
        y,
        initial_streamfunction=single_temperature.mass_streamfunction,
        initial_temperature=single_temperature.primitive[:, :, 2],
        initial_tv=single_temperature.primitive[:, :, 2],
    )

    temperature = result.primitive[:, :, 2]
    tv = result.primitive[:, :, 3]
    temperature_gap = np.max(np.abs(temperature - tv))
    assert result.scaled_residual_max <= 1.0e-10
    assert result.function_evaluations < 1000
    assert np.all(np.isfinite(result.primitive))
    assert np.all(temperature > 0.0)
    assert np.all(tv > 0.0)
    assert 100.0 < temperature_gap < 300.0
    assert np.max(np.abs(solver._frozen_vt_source(result.primitive))) > 0.0
    np.testing.assert_allclose(result.primitive[1:, 0, 0], 0.0, atol=1.0e-10)
    np.testing.assert_allclose(
        result.primitive[1:, -1, 0], similarity.edge_velocity, rtol=1.0e-10
    )
    np.testing.assert_allclose(
        result.primitive[1:, 0, 2:4], similarity.wall_temperature, rtol=1.0e-12
    )
    np.testing.assert_allclose(
        result.primitive[1:, -1, 2:4], similarity.edge_temperature, rtol=1.0e-12
    )


@pytest.mark.skipif(
    os.environ.get("ASTR_RUN_EXPENSIVE_REFERENCE") != "1",
    reason="set ASTR_RUN_EXPENSIVE_REFERENCE=1 for the 65/97-point reference gate",
)
def test_global_frozen_vt_flat_plate_is_grid_converged() -> None:
    mass_fraction = np.array(
        [0.766907344, 0.233092656, 1.0e-60, 1.0e-60, 1.0e-60]
    )
    similarity = Air5FrozenSimilarity(
        MECHANISM,
        pressure=101325.0,
        edge_velocity=2905.07,
        edge_temperature=450.0,
        wall_temperature=2925.0,
        mass_fraction=mass_fraction,
    )
    solver = Air5HblCollocation(
        MECHANISM,
        Air5HblCollocationBoundary(
            pressure=similarity.pressure,
            edge_velocity=similarity.edge_velocity,
            edge_temperature=similarity.edge_temperature,
            wall_velocity=0.0,
            wall_temperature=similarity.wall_temperature,
            mass_fraction=mass_fraction,
        ),
    )
    x = np.linspace(1.0612769542744582e-3, 1.2612769542744582e-3, 3)
    solutions = []
    for y_points in (65, 97):
        y = 2.0e-4 * np.linspace(0.0, 1.0, y_points) ** 1.25
        profiles = [similarity.profile(station, y) for station in x]
        initial_primitive = np.stack([profile.primitive for profile in profiles])
        initial_streamfunction = np.stack(
            [profile.mass_streamfunction for profile in profiles]
        )
        result = solver.solve_frozen_vt_continuation(
            x,
            y,
            initial_streamfunction=initial_streamfunction,
            initial_temperature=initial_primitive[:, :, 2],
            initial_tv=initial_primitive[:, :, 2],
        )
        assert result.scaled_residual_max <= 1.0e-10
        assert np.all(np.isfinite(result.primitive))
        assert np.all(result.primitive[:, :, 2:4] > 0.0)
        solutions.append((result, solver.wall_quantities(result)))

    (coarse, coarse_wall), (fine, fine_wall) = solutions
    for field, scale in (
        (0, similarity.edge_velocity),
        (2, similarity.wall_temperature),
        (3, similarity.wall_temperature),
    ):
        interpolated = PchipInterpolator(
            coarse.y, coarse.primitive[-1, :, field]
        )(fine.y)
        difference = interpolated - fine.primitive[-1, :, field]
        assert np.linalg.norm(difference) / np.linalg.norm(
            fine.primitive[-1, :, field]
        ) < 5.0e-4
        assert np.max(np.abs(difference)) / scale < 2.0e-3

    coarse_wall_values = np.array(
        [
            coarse_wall.skin_friction[-1],
            coarse_wall.translational_heat_flux[-1],
            coarse_wall.vibrational_heat_flux[-1],
            coarse_wall.total_heat_flux[-1],
        ]
    )
    fine_wall_values = np.array(
        [
            fine_wall.skin_friction[-1],
            fine_wall.translational_heat_flux[-1],
            fine_wall.vibrational_heat_flux[-1],
            fine_wall.total_heat_flux[-1],
        ]
    )
    np.testing.assert_allclose(
        coarse_wall_values, fine_wall_values, rtol=5.0e-4, atol=0.0
    )
    assert 200.0 < np.max(
        np.abs(fine.primitive[:, :, 2] - fine.primitive[:, :, 3])
    ) < 300.0
