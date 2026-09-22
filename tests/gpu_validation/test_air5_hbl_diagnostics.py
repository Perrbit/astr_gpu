from __future__ import annotations

from pathlib import Path

import numpy as np

from tests.gpu_validation.air5_hbl_diagnostics import (
    analyze_cartesian_hbl,
    load_active_q_snapshot,
    save_diagnostics,
)
from tests.gpu_validation.air5_radau_reference import Air5RadauReference
from tests.gpu_validation.air5_transport_reference import Air5TransportReference


ROOT = Path(__file__).resolve().parents[2]
MECHANISM = ROOT / "chemMech/air5_kimjo12.json"
VALIDATION_IO = ROOT / "src/validation_io.F90"


def _primitive_q(
    model: Air5RadauReference,
    density: float,
    velocity: np.ndarray,
    temperature: float,
    tv: float,
    mass_fraction: np.ndarray,
) -> np.ndarray:
    rho_species = density * mass_fraction
    momentum = density * velocity
    ev = model.ev_from_tv(rho_species, tv)
    result = np.empty(11)
    result[0] = density
    result[1:4] = momentum
    result[4] = model.q5_from_state(
        density, momentum, rho_species, ev, temperature
    )
    result[5:10] = rho_species
    result[10] = ev
    return result


def _linear_wall_field() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model = Air5RadauReference(MECHANISM)
    x = np.linspace(0.0, 2.0e-3, 5)
    y = np.linspace(0.0, 8.0e-5, 9)
    z = np.linspace(0.0, 1.0e-4, 3)
    q = np.empty((x.size, y.size, z.size, 11), order="F")
    mass_fraction_wall = np.array([0.76, 0.23, 0.002, 0.005, 0.003])
    mass_fraction_gradient = np.array([20.0, -10.0, -4.0, -3.0, -3.0])
    for i, _ in enumerate(x):
        for j, y_value in enumerate(y):
            mass_fraction = mass_fraction_wall + mass_fraction_gradient * y_value
            for k, _ in enumerate(z):
                q[i, j, k] = _primitive_q(
                    model,
                    density=0.20,
                    velocity=np.array([4.0e6 * y_value, 0.0, 0.0]),
                    temperature=3000.0 - 1.0e7 * y_value,
                    tv=2800.0 - 5.0e6 * y_value,
                    mass_fraction=mass_fraction,
                )
    return q, x, y, z


def test_cartesian_hbl_wall_diagnostics_match_linear_state() -> None:
    q, x, y, z = _linear_wall_field()
    model = Air5RadauReference(MECHANISM)
    transport = Air5TransportReference(MECHANISM)

    result = analyze_cartesian_hbl(
        q,
        x,
        y,
        z,
        model=model,
        transport=transport,
        reference_density=0.78103376204034,
        reference_velocity=2905.07,
        profile_stations=(x[1], x[3]),
    )

    wall_y = np.array([0.76, 0.23, 0.002, 0.005, 0.003])
    viscosity, conductivity_tr, conductivity_v, _ = transport.transport_properties(
        3000.0, 2800.0, model.pressure(0.20 * wall_y, 3000.0), wall_y
    )
    expected_cf = (
        2.0 * viscosity * 4.0e6 / (0.78103376204034 * 2905.07**2)
    )
    np.testing.assert_allclose(result.skin_friction, expected_cf, rtol=2.0e-12)
    np.testing.assert_allclose(
        result.heat_flux_translational, conductivity_tr * 1.0e7, rtol=2.0e-12
    )
    np.testing.assert_allclose(
        result.heat_flux_vibrational, conductivity_v * 5.0e6, rtol=2.0e-12
    )
    np.testing.assert_allclose(result.heat_flux_species, 0.0, atol=0.0)
    np.testing.assert_allclose(
        result.heat_flux_total,
        result.heat_flux_translational
        + result.heat_flux_vibrational
        + result.heat_flux_species,
        rtol=2.0e-12,
    )


def test_cartesian_hbl_profiles_preserve_two_temperature_and_species() -> None:
    q, x, y, z = _linear_wall_field()
    result = analyze_cartesian_hbl(
        q,
        x,
        y,
        z,
        model=Air5RadauReference(MECHANISM),
        transport=Air5TransportReference(MECHANISM),
        reference_density=0.78103376204034,
        reference_velocity=2905.07,
        profile_stations=(x[1], x[3]),
    )

    assert result.profile_temperature.shape == (2, y.size)
    assert result.profile_tv.shape == (2, y.size)
    assert result.profile_mass_fraction.shape == (2, y.size, 5)
    np.testing.assert_allclose(result.profile_temperature[:, 0], 3000.0)
    np.testing.assert_allclose(result.profile_tv[:, 0], 2800.0)
    np.testing.assert_allclose(
        np.sum(result.profile_mass_fraction, axis=2), 1.0, atol=2.0e-14
    )
    np.testing.assert_allclose(result.profile_x, np.array([x[1], x[3]]))


def test_cartesian_hbl_diagnostics_reject_nonuniform_spanwise_state() -> None:
    q, x, y, z = _linear_wall_field()
    q[:, :, 1, 1] *= 1.001

    try:
        analyze_cartesian_hbl(
            q,
            x,
            y,
            z,
            model=Air5RadauReference(MECHANISM),
            transport=Air5TransportReference(MECHANISM),
            reference_density=0.78103376204034,
            reference_velocity=2905.07,
            profile_stations=(x[2],),
        )
    except ValueError as error:
        assert "spanwise" in str(error)
    else:
        raise AssertionError("the laminar A1 gate must reject a spanwise-varying field")


def test_cartesian_hbl_diagnostics_accept_explicit_spanwise_tolerance() -> None:
    q, x, y, z = _linear_wall_field()
    q[:, :, 1, 1] *= 1.0 + 5.0e-8

    result = analyze_cartesian_hbl(
        q,
        x,
        y,
        z,
        model=Air5RadauReference(MECHANISM),
        transport=Air5TransportReference(MECHANISM),
        reference_density=0.78103376204034,
        reference_velocity=2905.07,
        profile_stations=(x[2],),
        spanwise_tolerance=1.0e-7,
    )

    assert np.all(np.isfinite(result.skin_friction))


def test_cartesian_hbl_long_time_gate_accepts_roundoff_spanwise_momentum() -> None:
    q, x, y, z = _linear_wall_field()
    reference_density = 0.78103376204034
    reference_velocity = 2905.07
    q[:, 1:, :, 3] = 0.5e-10 * reference_density * reference_velocity

    result = analyze_cartesian_hbl(
        q,
        x,
        y,
        z,
        model=Air5RadauReference(MECHANISM),
        transport=Air5TransportReference(MECHANISM),
        reference_density=reference_density,
        reference_velocity=reference_velocity,
        profile_stations=(x[2],),
        spanwise_tolerance=2.0e-12,
        spanwise_momentum_relative_tolerance=1.0e-10,
    )

    assert np.all(np.isfinite(result.skin_friction))


def test_cartesian_hbl_long_time_gate_rejects_excess_spanwise_momentum() -> None:
    q, x, y, z = _linear_wall_field()
    reference_density = 0.78103376204034
    reference_velocity = 2905.07
    q[:, 1:, :, 3] = 1.1e-10 * reference_density * reference_velocity

    try:
        analyze_cartesian_hbl(
            q,
            x,
            y,
            z,
            model=Air5RadauReference(MECHANISM),
            transport=Air5TransportReference(MECHANISM),
            reference_density=reference_density,
            reference_velocity=reference_velocity,
            profile_stations=(x[2],),
            spanwise_tolerance=2.0e-12,
            spanwise_momentum_relative_tolerance=1.0e-10,
        )
    except ValueError as error:
        assert "spanwise momentum" in str(error)
    else:
        raise AssertionError("the long-time A1 gate must bound spanwise momentum")


def test_cartesian_hbl_long_time_gate_uses_spanwise_mean_momentum() -> None:
    q, x, y, z = _linear_wall_field()
    reference_density = 0.78103376204034
    reference_velocity = 2905.07
    perturbation = 2.0e-10 * reference_density * reference_velocity
    q[:, 1:, 0, 3] = perturbation
    q[:, 1:, 2, 3] = -perturbation

    result = analyze_cartesian_hbl(
        q,
        x,
        y,
        z,
        model=Air5RadauReference(MECHANISM),
        transport=Air5TransportReference(MECHANISM),
        reference_density=reference_density,
        reference_velocity=reference_velocity,
        profile_stations=(x[2],),
        spanwise_tolerance=2.0e-12,
        spanwise_momentum_relative_tolerance=1.0e-10,
    )

    assert np.all(np.isfinite(result.skin_friction))


def test_snapshot_loader_and_npz_export_preserve_diagnostics(tmp_path: Path) -> None:
    q, x, y, z = _linear_wall_field()
    snapshot = tmp_path / "air5.bin"
    with snapshot.open("wb") as stream:
        np.asarray(
            [x.size - 1, y.size - 1, z.size - 1, 0, 11], dtype=np.int32
        ).tofile(stream)
        np.asarray(q, order="F").ravel(order="F").tofile(stream)

    loaded = load_active_q_snapshot(snapshot)
    np.testing.assert_array_equal(loaded, q)
    result = analyze_cartesian_hbl(
        loaded,
        x,
        y,
        z,
        model=Air5RadauReference(MECHANISM),
        transport=Air5TransportReference(MECHANISM),
        reference_density=0.78103376204034,
        reference_velocity=2905.07,
        profile_stations=(x[2],),
    )
    archive = tmp_path / "diagnostics.npz"
    report = tmp_path / "diagnostics.txt"
    save_diagnostics(result, archive, report)

    with np.load(archive) as saved:
        assert set(saved.files) == {
            "wall_x",
            "skin_friction",
            "heat_flux_total",
            "heat_flux_translational",
            "heat_flux_vibrational",
            "heat_flux_species",
            "profile_x",
            "profile_y",
            "profile_velocity",
            "profile_temperature",
            "profile_tv",
            "profile_mass_fraction",
        }
        np.testing.assert_allclose(saved["heat_flux_total"], result.heat_flux_total)
    text = report.read_text(encoding="ascii")
    assert "heat_flux_definition: positive from lower wall into fluid" in text
    assert "max_heat_decomposition_residual:" in text


def test_validation_snapshot_request_is_gated_before_gpu_copyback() -> None:
    source = "".join(VALIDATION_IO.read_text(encoding="utf-8").lower().split())

    assert "astr_validation_rhs_step" in source
    assert "validation_step=0" in source
    assert "astr_validation_rhs_step_secondary" in source
    assert "validation_step_secondary=-1" in source
    assert "rhs_validation_requested=validation_snapshot_requested(nstep)" in source
    assert "if(.not.validation_snapshot_requested(step_value).or.stage_value<1)return" in source
