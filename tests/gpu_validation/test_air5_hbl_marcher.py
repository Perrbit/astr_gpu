import os
from pathlib import Path

import numpy as np
import pytest

from tests.gpu_validation.air5_htr_profile import (
    load_htr_similarity_profile,
    map_htr_profile_to_air5,
)
from tests.gpu_validation.air5_hbl_marcher import (
    Air5HblBoundaryConditions,
    Air5HblMarchResult,
    Air5HblMarcher,
    _derivative_matrix,
    _differentiate,
)


ROOT = Path(__file__).resolve().parents[2]
MECHANISM = ROOT / "chemMech/air5_kimjo12.json"
HTR_PROFILE = (
    ROOT / "tests/gpu_validation/data/htr_multispecies_tbl_mach6_similarity.dat"
)


def _uniform_contract() -> tuple[np.ndarray, np.ndarray, Air5HblBoundaryConditions]:
    y = np.geomspace(1.0e-8, 2.0e-3, 9)
    y[0] = 0.0
    mass_fraction = np.array([0.765, 0.234, 2.0e-4, 3.0e-4, 5.0e-4])
    primitive = np.tile(
        np.r_[2500.0, 0.0, 1800.0, 1800.0, mass_fraction[1:5]],
        (y.size, 1),
    )
    boundary = Air5HblBoundaryConditions(
        pressure=8000.0,
        edge_velocity=np.array([2500.0, 0.0, 0.0]),
        edge_temperature=1800.0,
        edge_tv=1800.0,
        edge_mass_fraction=mass_fraction,
        wall_velocity=2500.0,
        wall_temperature=1800.0,
        wall_tv=1800.0,
    )
    return primitive, y, boundary


def _htr_contract(points: int = 9) -> tuple[
    np.ndarray, np.ndarray, Air5HblBoundaryConditions
]:
    mapped, _ = map_htr_profile_to_air5(
        load_htr_similarity_profile(HTR_PROFILE), pressure=101325.0
    )
    indices = np.linspace(0, len(mapped) - 1, points).round().astype(int)
    selected = [mapped[index] for index in indices]
    y = np.asarray([row.y for row in selected])
    primitive = np.asarray(
        [
            [
                row.u,
                row.v,
                row.temperature,
                row.tv,
                *row.mass_fraction[1:5],
            ]
            for row in selected
        ]
    )
    edge = selected[-1]
    wall = selected[0]
    boundary = Air5HblBoundaryConditions(
        pressure=101325.0,
        edge_velocity=np.array([edge.u, edge.v, edge.w]),
        edge_temperature=edge.temperature,
        edge_tv=edge.tv,
        edge_mass_fraction=np.asarray(edge.mass_fraction),
        wall_velocity=wall.u,
        wall_temperature=wall.temperature,
        wall_tv=wall.tv,
    )
    return primitive, y, boundary


def _htr_stretched_contract(points: int) -> tuple[
    np.ndarray, np.ndarray, Air5HblBoundaryConditions
]:
    mapped, _ = map_htr_profile_to_air5(
        load_htr_similarity_profile(HTR_PROFILE), pressure=101325.0
    )
    source_y = np.asarray([row.y for row in mapped])
    source_primitive = np.asarray(
        [
            [
                row.u,
                row.v,
                row.temperature,
                row.tv,
                *row.mass_fraction[1:5],
            ]
            for row in mapped
        ]
    )
    y = source_y[-1] * np.linspace(0.0, 1.0, points) ** 2
    primitive = np.column_stack(
        [
            np.interp(y, source_y, source_primitive[:, column])
            for column in range(8)
        ]
    )
    edge = mapped[-1]
    wall = mapped[0]
    boundary = Air5HblBoundaryConditions(
        pressure=101325.0,
        edge_velocity=np.array([edge.u, edge.v, edge.w]),
        edge_temperature=edge.temperature,
        edge_tv=edge.tv,
        edge_mass_fraction=np.asarray(edge.mass_fraction),
        wall_velocity=wall.u,
        wall_temperature=wall.temperature,
        wall_tv=wall.tv,
    )
    return primitive, y, boundary


def test_positive_implicit_station_closes_coupled_htr_residual() -> None:
    primitive, y, boundary = _htr_contract()
    marcher = Air5HblMarcher(MECHANISM)

    result = marcher.march_positive_station(
        primitive,
        y,
        1.0e-8,
        boundary,
        source_mode="coupled",
        max_function_evaluations=100,
    )
    mass_fraction = marcher._mass_fraction(result.primitive)

    assert result.scaled_residual_max <= 1.0e-10
    assert np.all(np.isfinite(result.primitive))
    assert np.all(result.primitive[:, 2:4] > 0.0)
    assert np.all(mass_fraction >= 0.0)
    np.testing.assert_allclose(np.sum(mass_fraction, axis=1), 1.0, atol=2.0e-14)


def test_positive_implicit_station_enforces_discrete_noncatalytic_wall() -> None:
    primitive, y, boundary = _htr_contract()
    marcher = Air5HblMarcher(MECHANISM)

    result = marcher.march_positive_station(
        primitive,
        y,
        1.0e-6,
        boundary,
        source_mode="coupled",
        max_function_evaluations=500,
    )
    wall_gradient = _differentiate(
        _derivative_matrix(y), marcher._mass_fraction(result.primitive)
    )[0]

    assert np.max(np.abs(wall_gradient)) <= 1.0e-10


def test_positive_coordinates_preserve_exact_zero_independent_species() -> None:
    primitive, _, _ = _uniform_contract()
    primitive[:, 5] = 0.0

    coordinates = Air5HblMarcher._positive_coordinates(primitive)
    reconstructed = Air5HblMarcher._primitive_from_positive_coordinates(coordinates)

    np.testing.assert_allclose(reconstructed, primitive, atol=2.0e-16, rtol=0.0)
    assert np.all(Air5HblMarcher._mass_fraction(reconstructed) >= 0.0)


def test_positive_implicit_station_preserves_exact_zero_species() -> None:
    primitive, y, original_boundary = _uniform_contract()
    primitive[:, 5] = 0.0
    composition = Air5HblMarcher._mass_fraction(primitive)[0]
    boundary = Air5HblBoundaryConditions(
        pressure=original_boundary.pressure,
        edge_velocity=original_boundary.edge_velocity,
        edge_temperature=original_boundary.edge_temperature,
        edge_tv=original_boundary.edge_tv,
        edge_mass_fraction=composition,
        wall_velocity=original_boundary.wall_velocity,
        wall_temperature=original_boundary.wall_temperature,
        wall_tv=original_boundary.wall_tv,
    )
    marcher = Air5HblMarcher(MECHANISM)

    result = marcher.march_positive_station(
        primitive,
        y,
        1.0e-4,
        boundary,
        source_mode="off",
    )

    np.testing.assert_allclose(result.primitive, primitive, atol=2.0e-12, rtol=0.0)
    assert np.all(marcher._mass_fraction(result.primitive) >= 0.0)


def test_positive_implicit_profile_marches_multiple_coupled_htr_stations() -> None:
    primitive, y, boundary = _htr_contract()
    marcher = Air5HblMarcher(MECHANISM)
    x = 1.0612769542744582e-3 + np.arange(3) * 1.0e-6

    result = marcher.march_positive(
        primitive,
        y,
        x,
        boundary,
        source_mode="coupled",
        max_function_evaluations=500,
    )
    mass_fraction = np.asarray(
        [marcher._mass_fraction(profile) for profile in result.primitive]
    )

    assert result.primitive.shape == (x.size, y.size, 8)
    assert result.scaled_linear_residual_max <= 1.0e-10
    assert np.all(result.primitive[:, :, 2:4] > 0.0)
    assert np.all(mass_fraction >= 0.0)
    assert not np.array_equal(result.primitive[-1], primitive)


def test_positive_implicit_coupled_station_activates_finite_rate_sources() -> None:
    primitive, y, boundary = _htr_contract()
    marcher = Air5HblMarcher(MECHANISM)

    source_off = marcher.march_positive_station(
        primitive,
        y,
        1.0e-6,
        boundary,
        source_mode="off",
        max_function_evaluations=500,
    ).primitive
    coupled = marcher.march_positive_station(
        primitive,
        y,
        1.0e-6,
        boundary,
        source_mode="coupled",
        max_function_evaluations=500,
    ).primitive

    species_difference = np.max(
        np.abs(
            marcher._mass_fraction(coupled)
            - marcher._mass_fraction(source_off)
        )
    )
    assert species_difference > 1.0e-13
    assert np.max(np.abs(coupled[:, 3] - source_off[:, 3])) > 1.0e-6


def test_positive_implicit_uniform_profile_has_zero_wall_transport() -> None:
    primitive, y, boundary = _uniform_contract()
    marcher = Air5HblMarcher(MECHANISM)
    result = marcher.march_positive(
        primitive,
        y,
        np.linspace(0.0, 2.0e-4, 3),
        boundary,
        source_mode="off",
        single_temperature=True,
    )

    wall = marcher.wall_quantities(result, y, boundary)

    np.testing.assert_allclose(wall.skin_friction, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(wall.translational_heat_flux, 0.0, atol=1.0e-9)
    np.testing.assert_allclose(wall.vibrational_heat_flux, 0.0, atol=1.0e-9)
    np.testing.assert_allclose(wall.species_enthalpy_flux, 0.0, atol=1.0e-9)
    np.testing.assert_allclose(wall.total_heat_flux, 0.0, atol=1.0e-9)


def test_wall_heat_flux_is_positive_from_lower_wall_into_fluid() -> None:
    primitive, y, original_boundary = _uniform_contract()
    primitive[:, 2] = 1800.0 + 100.0 * y / y[-1]
    primitive[:, 3] = primitive[:, 2]
    boundary = Air5HblBoundaryConditions(
        pressure=original_boundary.pressure,
        edge_velocity=original_boundary.edge_velocity,
        edge_temperature=1900.0,
        edge_tv=1900.0,
        edge_mass_fraction=original_boundary.edge_mass_fraction,
        wall_velocity=original_boundary.wall_velocity,
        wall_temperature=1800.0,
        wall_tv=1800.0,
    )
    result = Air5HblMarchResult(
        x=np.array([0.0, 1.0]),
        primitive=np.stack((primitive, primitive)),
        scaled_linear_residual_max=0.0,
    )

    wall = Air5HblMarcher(MECHANISM).wall_quantities(result, y, boundary)

    assert np.all(wall.translational_heat_flux < 0.0)
    assert np.all(wall.vibrational_heat_flux < 0.0)
    assert np.all(wall.total_heat_flux < 0.0)


@pytest.mark.skipif(
    os.getenv("ASTR_RUN_EXPENSIVE_REFERENCE") != "1",
    reason="finite-rate HBL step-convergence gate is explicitly opt-in",
)
def test_positive_implicit_coupled_htr_profile_is_step_converged() -> None:
    primitive, y, boundary = _htr_contract()
    marcher = Air5HblMarcher(MECHANISM)
    x_origin = 1.0612769542744582e-3
    length = 2.0e-6
    downstream = []
    residual_maxima = []
    for intervals in (1, 2, 4):
        result = marcher.march_positive(
            primitive,
            y,
            np.linspace(x_origin, x_origin + length, intervals + 1),
            boundary,
            source_mode="coupled",
            max_function_evaluations=500,
        )
        downstream.append(result.primitive[-1])
        residual_maxima.append(result.scaled_linear_residual_max)

    scale = np.array(
        [
            boundary.edge_velocity[0],
            max(np.max(np.abs(primitive[:, 1])), 1.0),
            boundary.edge_temperature,
            boundary.edge_tv,
            1.0,
            1.0,
            1.0,
            1.0,
        ]
    )
    coarse_difference = (downstream[0] - downstream[1]) / scale
    fine_difference = (downstream[1] - downstream[2]) / scale

    assert max(residual_maxima) <= 1.0e-10
    assert np.sqrt(np.mean(fine_difference**2)) < 0.25 * np.sqrt(
        np.mean(coarse_difference**2)
    )
    assert np.max(np.abs(fine_difference)) < 0.25 * np.max(
        np.abs(coarse_difference)
    )


@pytest.mark.skipif(
    os.getenv("ASTR_RUN_EXPENSIVE_REFERENCE") != "1",
    reason="finite-rate HBL wall-normal convergence gate is explicitly opt-in",
)
def test_positive_implicit_coupled_htr_profile_is_wall_normal_converged() -> None:
    marcher = Air5HblMarcher(MECHANISM)
    solutions = []
    wall_values = []
    boundary = None
    for points in (17, 33, 65):
        primitive, y, boundary = _htr_stretched_contract(points)
        station = marcher.march_positive_station(
            primitive,
            y,
            1.0e-6,
            boundary,
            source_mode="coupled",
            max_function_evaluations=1200,
        )
        result = Air5HblMarchResult(
            x=np.array([0.0, 1.0e-6]),
            primitive=np.stack((primitive, station.primitive)),
            scaled_linear_residual_max=station.scaled_residual_max,
        )
        wall = marcher.wall_quantities(result, y, boundary)
        solutions.append((y, station.primitive, station.scaled_residual_max))
        wall_values.append((wall.skin_friction[-1], wall.total_heat_flux[-1]))

    assert boundary is not None
    scale = np.array(
        [
            boundary.edge_velocity[0],
            max(np.max(np.abs(solutions[-1][1][:, 1])), 1.0),
            boundary.edge_temperature,
            boundary.edge_tv,
            1.0,
            1.0,
            1.0,
            1.0,
        ]
    )
    differences = []
    for (coarse_y, coarse, _), (fine_y, fine, _) in zip(
        solutions, solutions[1:]
    ):
        fine_on_coarse = np.column_stack(
            [np.interp(coarse_y, fine_y, fine[:, column]) for column in range(8)]
        )
        differences.append((coarse - fine_on_coarse) / scale)

    assert max(item[2] for item in solutions) <= 1.0e-10
    assert np.sqrt(np.mean(differences[1] ** 2)) < 0.3 * np.sqrt(
        np.mean(differences[0] ** 2)
    )
    assert np.max(np.abs(differences[1])) < 0.3 * np.max(
        np.abs(differences[0])
    )
    assert abs(wall_values[1][0] - wall_values[2][0]) / abs(
        wall_values[2][0]
    ) < 2.0e-3
    assert abs(wall_values[1][1] - wall_values[2][1]) / abs(
        wall_values[2][1]
    ) < 2.0e-3


@pytest.mark.skipif(
    os.getenv("ASTR_RUN_LONG_REFERENCE") != "1",
    reason="finite-rate HBL accumulated march gate is explicitly opt-in",
)
def test_positive_implicit_coupled_htr_profile_survives_accumulated_march() -> None:
    primitive, y, boundary = _htr_stretched_contract(33)
    marcher = Air5HblMarcher(MECHANISM)
    x_origin = 1.0612769542744582e-3
    result = marcher.march_positive(
        primitive,
        y,
        x_origin + np.arange(11) * 1.0e-5,
        boundary,
        source_mode="coupled",
        max_function_evaluations=1200,
    )
    mass_fraction = np.asarray(
        [marcher._mass_fraction(profile) for profile in result.primitive]
    )
    wall = marcher.wall_quantities(result, y, boundary)

    assert result.scaled_linear_residual_max <= 1.0e-10
    assert np.all(mass_fraction >= 0.0)
    np.testing.assert_allclose(
        np.sum(mass_fraction, axis=2), 1.0, atol=3.0e-14, rtol=0.0
    )
    assert 100.0 < np.max(
        np.abs(result.primitive[:, :, 2] - result.primitive[:, :, 3])
    ) < 300.0
    assert np.all(np.isfinite(wall.skin_friction))
    assert np.all(np.isfinite(wall.total_heat_flux))
    assert np.all(wall.skin_friction > 0.0)


def test_uniform_single_temperature_station_has_machine_zero_residual() -> None:
    primitive, y, boundary = _uniform_contract()
    marcher = Air5HblMarcher(MECHANISM)

    residual = marcher.scaled_residual(
        primitive.ravel(),
        primitive,
        y,
        1.0e-4,
        boundary,
        source_mode="off",
        single_temperature=True,
    )

    assert np.max(np.abs(residual)) < 1.0e-12


def test_uniform_single_temperature_station_marches_without_drift() -> None:
    primitive, y, boundary = _uniform_contract()
    marcher = Air5HblMarcher(MECHANISM)

    result = marcher.march_station(
        primitive,
        y,
        1.0e-4,
        boundary,
        source_mode="off",
        single_temperature=True,
    )

    assert result.scaled_residual_max < 1.0e-12
    np.testing.assert_allclose(result.primitive, primitive, atol=1.0e-12, rtol=0.0)


def test_uniform_parabolic_derivative_is_machine_zero() -> None:
    primitive, y, boundary = _uniform_contract()
    marcher = Air5HblMarcher(MECHANISM)

    result = marcher.streamwise_derivative(
        primitive,
        y,
        boundary,
        source_mode="off",
        single_temperature=True,
    )

    assert result.scaled_linear_residual_max < 1.0e-12
    np.testing.assert_allclose(result.dynamic_derivative, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(result.normal_velocity, 0.0, atol=1.0e-12)


def test_uniform_parabolic_profile_marches_without_drift() -> None:
    primitive, y, boundary = _uniform_contract()
    marcher = Air5HblMarcher(MECHANISM)

    result = marcher.march(
        primitive,
        y,
        np.linspace(0.0, 1.0e-3, 5),
        boundary,
        source_mode="off",
        single_temperature=True,
    )

    assert result.scaled_linear_residual_max < 1.0e-12
    np.testing.assert_allclose(
        result.primitive,
        np.repeat(primitive[None, :, :], result.x.size, axis=0),
        atol=1.0e-12,
        rtol=0.0,
    )


def test_uniform_radau_profile_marches_without_drift() -> None:
    primitive, y, boundary = _uniform_contract()
    marcher = Air5HblMarcher(MECHANISM)

    result = marcher.march_radau(
        primitive,
        y,
        np.linspace(0.0, 1.0e-3, 5),
        boundary,
        source_mode="off",
        single_temperature=True,
        frozen_species=True,
    )

    assert result.scaled_linear_residual_max < 1.0e-12
    np.testing.assert_allclose(
        result.primitive,
        np.repeat(primitive[None, :, :], result.x.size, axis=0),
        atol=2.0e-12,
        rtol=0.0,
    )


def test_uniform_frozen_station_marches_without_drift() -> None:
    primitive, y, boundary = _uniform_contract()
    marcher = Air5HblMarcher(MECHANISM)

    result = marcher.march_single_temperature_frozen_station(
        primitive,
        y,
        1.0e-4,
        boundary,
    )

    assert result.scaled_residual_max < 1.0e-12
    np.testing.assert_allclose(result.primitive, primitive, atol=1.0e-12, rtol=0.0)


def test_uniform_frozen_profile_marches_without_drift() -> None:
    primitive, y, boundary = _uniform_contract()
    marcher = Air5HblMarcher(MECHANISM)

    result = marcher.march_single_temperature_frozen(
        primitive,
        y,
        np.linspace(0.0, 1.0e-3, 5),
        boundary,
    )

    assert result.scaled_linear_residual_max < 1.0e-12
    np.testing.assert_allclose(
        result.primitive,
        np.repeat(primitive[None, :, :], result.x.size, axis=0),
        atol=1.0e-12,
        rtol=0.0,
    )


def test_no_slip_frozen_station_is_physical_and_converged() -> None:
    primitive, _, uniform_boundary = _uniform_contract()
    y = np.r_[0.0, np.geomspace(2.5e-7, 2.0e-3, 8)]
    primitive = np.repeat(primitive[:1], y.size, axis=0)
    boundary = Air5HblBoundaryConditions(
        pressure=uniform_boundary.pressure,
        edge_velocity=uniform_boundary.edge_velocity,
        edge_temperature=uniform_boundary.edge_temperature,
        edge_tv=uniform_boundary.edge_tv,
        edge_mass_fraction=uniform_boundary.edge_mass_fraction,
        wall_velocity=0.0,
        wall_temperature=2400.0,
        wall_tv=2400.0,
    )
    eta = y / y[-1]
    primitive[:, 0] = boundary.edge_velocity[0] * np.tanh(8.0 * eta)
    primitive[:, 1] = 0.0
    primitive[:, 2] = boundary.wall_temperature + (
        boundary.edge_temperature - boundary.wall_temperature
    ) * np.tanh(6.0 * eta)
    primitive[:, 3] = primitive[:, 2]
    primitive[0, 0] = boundary.wall_velocity
    primitive[0, 2:4] = boundary.wall_temperature
    primitive[-1, 0] = boundary.edge_velocity[0]
    primitive[-1, 2:4] = boundary.edge_temperature
    marcher = Air5HblMarcher(MECHANISM)

    result = marcher.march_single_temperature_frozen_station(
        primitive,
        y,
        1.0e-6,
        boundary,
    )
    mass_fraction = marcher._mass_fraction(result.primitive)

    assert result.scaled_residual_max <= 1.0e-10
    assert np.all(np.isfinite(result.primitive))
    assert np.all(result.primitive[:, 2:4] > 0.0)
    assert np.all(mass_fraction >= 0.0)
    np.testing.assert_allclose(np.sum(mass_fraction, axis=1), 1.0, atol=2.0e-14)
    np.testing.assert_allclose(
        result.primitive[:, 4:8],
        np.broadcast_to(boundary.edge_mass_fraction[None, 1:5], (y.size, 4)),
    )
    np.testing.assert_allclose(result.primitive[:, 3], result.primitive[:, 2])
    wall_error = np.array(
        [
            (result.primitive[0, 0] - boundary.wall_velocity)
            / boundary.edge_velocity[0],
            result.primitive[0, 1] / boundary.edge_velocity[0],
            (result.primitive[0, 2] - boundary.wall_temperature)
            / boundary.edge_temperature,
            (result.primitive[0, 3] - boundary.wall_temperature)
            / boundary.edge_temperature,
        ]
    )
    edge_error = np.array(
        [
            (result.primitive[-1, 0] - boundary.edge_velocity[0])
            / boundary.edge_velocity[0],
            (result.primitive[-1, 2] - boundary.edge_temperature)
            / boundary.edge_temperature,
            (result.primitive[-1, 3] - boundary.edge_temperature)
            / boundary.edge_temperature,
        ]
    )
    assert np.max(np.abs(wall_error)) <= 1.0e-10
    assert np.max(np.abs(edge_error)) <= 1.0e-10


def test_marcher_rejects_nonphysical_closed_species_state() -> None:
    primitive, y, boundary = _uniform_contract()
    primitive[:, 4:8] = 0.3
    marcher = Air5HblMarcher(MECHANISM)

    residual = marcher.scaled_residual(
        primitive.ravel(), primitive, y, 1.0e-4, boundary
    )
    assert np.all(residual == 1.0e6)


def test_marcher_jacobian_sparsity_covers_five_station_operator() -> None:
    sparsity = Air5HblMarcher._jacobian_sparsity(9)

    assert sparsity.shape == (72, 72)
    interior = sparsity[4 * 8 : 5 * 8].nonzero()[1]
    assert set(interior // 8) == {2, 3, 4, 5, 6}
    wall = sparsity[0:8].nonzero()[1]
    assert set(wall // 8) == {0, 1}
    edge = sparsity[-8:].nonzero()[1]
    assert set(edge // 8) == {8}


def test_analytic_streamwise_flux_jacobian_matches_finite_difference() -> None:
    primitive, _, boundary = _uniform_contract()
    primitive[4, 0] = 2200.0
    primitive[4, 2] = 2400.0
    primitive[4, 3] = 1900.0
    primitive[4, 4:8] = np.array([0.20, 0.01, 0.02, 0.03])
    marcher = Air5HblMarcher(MECHANISM)
    dynamic = primitive[4, [0, 2, 3, 4, 5, 6, 7]]
    scale = marcher._dynamic_scale(primitive, boundary)
    analytic = marcher._streamwise_flux_jacobian(
        dynamic, boundary.pressure, scale
    )

    numerical = np.empty_like(analytic)
    for column in range(7):
        step = 2.0e-5
        fluxes = []
        for multiplier in (-2.0, -1.0, 1.0, 2.0):
            trial = dynamic.copy()
            trial[column] += multiplier * step * scale[column]
            state = marcher._states(
                marcher._primitive_from_dynamic(trial[None, :], np.zeros(1)),
                boundary.pressure,
            )[0]
            fluxes.append(marcher.reference.streamwise_flux(state))
        numerical[:, column] = (
            fluxes[0] - 8.0 * fluxes[1] + 8.0 * fluxes[2] - fluxes[3]
        ) / (12.0 * step)

    np.testing.assert_allclose(analytic, numerical, rtol=2.0e-9, atol=5.0e-4)
