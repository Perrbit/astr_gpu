from pathlib import Path

import numpy as np

from tests.gpu_validation.air5_hbl_marcher import (
    Air5HblBoundaryConditions,
    Air5HblMarcher,
)


ROOT = Path(__file__).resolve().parents[2]
MECHANISM = ROOT / "chemMech/air5_kimjo12.json"


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
