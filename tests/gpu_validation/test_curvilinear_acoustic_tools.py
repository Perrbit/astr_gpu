#!/usr/bin/env python3
"""Unit contracts for the curved-boundary acoustic validation tools."""

from __future__ import annotations

from pathlib import Path
import sys

import h5py
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
VALIDATION = ROOT / "tests" / "gpu_validation"
sys.path.insert(0, str(VALIDATION))

from analyze_curvilinear_acoustic_reflection import (  # noqa: E402
    acoustic_invariants,
    figure_paths,
    integrate_characteristic_energy,
    surface_geometry,
)
from check_curvilinear_nscbc_geometry import geometry_metrics  # noqa: E402
from generate_curvilinear_acoustic_pulse import (  # noqa: E402
    build_acoustic_state,
    compact_support_intervals,
)
from generate_curvilinear_tgv_grid import mapped_grid  # noqa: E402


def test_outgoing_wave_has_zero_incoming_invariant() -> None:
    p_prime = np.array([1.0e-4])
    rho0 = 1.0
    c0 = 1.0
    un_prime = p_prime / (rho0 * c0)
    w_plus, w_minus = acoustic_invariants(p_prime, un_prime, rho0, c0)
    assert np.allclose(w_plus, 2.0 * p_prime)
    assert np.allclose(w_minus, 0.0)


def test_incoming_wave_has_zero_outgoing_invariant() -> None:
    p_prime = np.array([2.0e-4])
    rho0 = 1.3
    c0 = 0.8
    un_prime = -p_prime / (rho0 * c0)
    w_plus, w_minus = acoustic_invariants(p_prime, un_prime, rho0, c0)
    assert np.allclose(w_plus, 0.0)
    assert np.allclose(w_minus, 2.0 * p_prime)


def test_acoustic_state_obeys_astr_equation_of_state() -> None:
    x = np.linspace(0.0, 1.0, 9)[None, None, :]
    y = np.linspace(0.0, 1.0, 7)[None, :, None]
    z = np.linspace(0.0, 1.0, 5)[:, None, None]
    x, y, z = np.broadcast_arrays(x, y, z)
    gamma = 1.4
    mach = 0.3
    state = build_acoustic_state(
        (x, y, z),
        rho0=1.0,
        p0=1.0 / (gamma * mach**2),
        velocity0=(0.0, 0.0, 0.0),
        gamma=gamma,
        mach=mach,
        amplitude=1.0e-4,
        center=(0.5, 0.5, 0.5),
        width=0.08,
        direction=(0.0, 1.0, 0.0),
        clearance_widths=2.0,
    )
    reconstructed_p = state["ro"] * state["t"] / (gamma * mach**2)
    assert np.allclose(reconstructed_p, state["p"], rtol=1.0e-14, atol=1.0e-14)
    assert np.all(state["ro"] > 0.0)
    assert np.all(state["t"] > 0.0)


def test_upper_y_ramp_grid_is_cartesian_below_ramp_and_curved_at_boundary() -> None:
    im, jm, km = 16, 24, 20
    ramp_start = 0.72
    x, y, z, jacobian = mapped_grid(
        im,
        jm,
        km,
        0.15,
        "y-upper-ramp",
        ramp_start_fraction=ramp_start,
    )
    eta = np.linspace(0.0, 2.0 * np.pi, jm + 1)
    cartesian = eta <= ramp_start * 2.0 * np.pi
    expected = np.broadcast_to(eta[None, :, None], y.shape)
    assert np.array_equal(y[:, cartesian, :], expected[:, cartesian, :])
    assert np.ptp(y[:, -1, :]) > 0.25
    assert np.min(jacobian) > 0.0
    assert np.array_equal(x, np.broadcast_to(x[:, :1, :], x.shape))
    assert np.array_equal(z, np.broadcast_to(z[:1, :1, :], z.shape))


def test_upper_y_ramp_geometry_has_flat_probe_and_valid_boundary(
    tmp_path: Path,
) -> None:
    im, jm, km = 16, 24, 20
    amplitude = 0.15
    ramp_start = 0.72
    x, y, z, _ = mapped_grid(
        im,
        jm,
        km,
        amplitude,
        "y-upper-ramp",
        ramp_start_fraction=ramp_start,
    )
    grid = tmp_path / "grid.h5"
    with h5py.File(grid, "w") as handle:
        for name, values in zip(("x", "y", "z"), (x, y, z)):
            handle.create_dataset(name, data=values.transpose(2, 1, 0))
    metrics = geometry_metrics(
        grid,
        amplitude,
        mapping="y-upper-ramp",
        ramp_start_fraction=ramp_start,
    )
    normal, weights = surface_geometry(grid, 2 * jm // 3)
    assert metrics["mapping_error_max"] < 1.0e-12
    assert metrics["analytic_jacobian_min"] > 0.0
    assert np.max(np.abs(normal[..., 0])) < 1.0e-12
    assert np.max(np.abs(normal[..., 1] - 1.0)) < 1.0e-12
    assert np.max(np.abs(normal[..., 2])) < 1.0e-12
    assert np.min(weights) > 0.0


def test_plane_y_packet_is_pure_outgoing_on_cartesian_probe() -> None:
    x = np.linspace(0.0, 1.0, 9)[None, None, :]
    y = np.linspace(0.0, 1.0, 17)[None, :, None]
    z = np.linspace(0.0, 1.0, 5)[:, None, None]
    x, y, z = np.broadcast_arrays(x, y, z)
    rho0 = 1.0
    gamma = 1.4
    mach = 0.3
    p0 = 1.0 / (gamma * mach**2)
    c0 = np.sqrt(gamma * p0 / rho0)
    state = build_acoustic_state(
        (x, y, z),
        rho0=rho0,
        p0=p0,
        velocity0=(0.0, 0.0, 0.0),
        gamma=gamma,
        mach=mach,
        amplitude=1.0e-4,
        center=(0.5, 0.5, 0.5),
        width=0.08,
        direction=(0.0, 1.0, 0.0),
        clearance_widths=2.0,
        profile="plane-y",
    )
    p_prime = state["p"][:, 8, :] - p0
    _, w_minus = acoustic_invariants(
        p_prime,
        state["u2"][:, 8, :],
        rho0,
        c0,
    )
    assert np.max(np.abs(w_minus)) < 1.0e-15
    assert np.ptp(state["p"][:, 8, :]) == 0.0


def test_compact_plane_packet_is_pure_and_zero_outside_support() -> None:
    x = np.linspace(0.0, 2.0 * np.pi, 33)[None, None, :]
    y = np.linspace(0.0, 2.0 * np.pi, 25)[None, :, None]
    z = np.linspace(0.0, 2.0 * np.pi, 33)[:, None, None]
    x, y, z = np.broadcast_arrays(x, y, z)
    rho0 = 1.0
    gamma = 1.4
    mach = 0.3
    p0 = 1.0 / (gamma * mach**2)
    c0 = np.sqrt(gamma * p0 / rho0)
    state = build_acoustic_state(
        (x, y, z),
        rho0=rho0,
        p0=p0,
        velocity0=(0.0, 0.0, 0.0),
        gamma=gamma,
        mach=mach,
        amplitude=1.0e-4,
        center=(np.pi, np.pi, np.pi),
        width=1.0,
        direction=(0.0, 1.0, 0.0),
        clearance_widths=2.5,
        profile="plane-y-compact",
    )
    support = np.abs(y - np.pi) < 1.0
    assert np.all(state["p"][~support] == p0)
    center_index = int(np.argmin(np.abs(y[0, :, 0] - np.pi)))
    assert state["p"][0, center_index, 0] - p0 == pytest.approx(1.0e-4)
    p_prime = state["p"] - p0
    _, w_minus = acoustic_invariants(p_prime, state["u2"], rho0, c0)
    assert np.max(np.abs(w_minus)) < 1.0e-15


def test_compact_support_resolution_uses_the_largest_local_y_spacing() -> None:
    x = np.linspace(0.0, 2.0 * np.pi, 33)[None, None, :]
    y = np.linspace(0.0, 2.0 * np.pi, 49)[None, :, None]
    z = np.linspace(0.0, 2.0 * np.pi, 33)[:, None, None]
    coordinates = np.broadcast_arrays(x, y, z)

    intervals = compact_support_intervals(coordinates[1], np.pi, 1.0)

    assert intervals == pytest.approx(48.0 / np.pi)


def test_plane_y_packet_rejects_oblique_direction() -> None:
    coordinates = tuple(np.zeros((3, 3, 3)) for _ in range(3))
    with pytest.raises(ValueError, match="positive y direction"):
        build_acoustic_state(
            coordinates,
            rho0=1.0,
            p0=1.0,
            velocity0=(0.0, 0.0, 0.0),
            gamma=1.4,
            mach=0.3,
            amplitude=1.0e-4,
            center=(0.0, 0.0, 0.0),
            width=0.1,
            direction=(0.1, 0.99, 0.0),
            clearance_widths=0.0,
            profile="plane-y",
        )


def test_acoustic_driver_uses_plane_packet_before_upper_geometry_ramp() -> None:
    script = (
        ROOT / "tests/gpu_validation/run_curvilinear_nscbc52_acoustic_compare.sh"
    ).read_text(encoding="ascii")
    assert 'GRID_MAPPING="${GRID_MAPPING:-y-upper-ramp}"' in script
    assert 'GRID_LEVELS="${GRID_LEVELS:-64,48,64;80,60,80;96,72,96}"' in script
    assert 'RAMP_START_FRACTION="${RAMP_START_FRACTION:-0.72}"' in script
    assert 'PROBE_FRACTION="${PROBE_FRACTION:-2,3}"' in script
    assert 'ACOUSTIC_PROFILE="${ACOUSTIC_PROFILE:-plane-y-compact}"' in script
    assert 'PULSE_WIDTH="${PULSE_WIDTH:-1.0}"' in script
    assert 'MIN_SUPPORT_INTERVALS="${MIN_SUPPORT_INTERVALS:-15.0}"' in script
    assert 'INCIDENT_WINDOW="${INCIDENT_WINDOW:-0.04,0.72}"' in script
    assert 'REFLECTED_WINDOW="${REFLECTED_WINDOW:-1.15,2.10}"' in script
    assert 'INPUT_NAME="${INPUT_NAME:-input.tgv}"' in script
    assert 'prepare_tgv_case.py' in script
    assert 'prepare_s1_flatplate_case.py' not in script
    assert '--homogeneous t,f,t' in script
    assert '--bctype "1;1;41,1.0;52;1;1"' in script
    assert '--ninit 3' in script
    assert '--lwsequ t --feqwsequ "$FEQCHKPT"' in script
    assert script.count('--mach "$MACH"') >= 2
    assert '--profile "$ACOUSTIC_PROFILE"' in script
    assert '--minimum-support-intervals "$MIN_SUPPORT_INTERVALS"' in script
    assert "probe_fraction < ramp_start" in script


@pytest.mark.parametrize(
    "amplitude,direction",
    ((1.0, (0.0, 1.0, 0.0)), (1.0e-4, (1.0, 0.0, 0.0))),
)
def test_acoustic_state_rejects_nonlinear_or_nonupward_packet(
    amplitude: float, direction: tuple[float, float, float]
) -> None:
    coordinates = tuple(np.zeros((3, 3, 3)) for _ in range(3))
    with pytest.raises(ValueError):
        build_acoustic_state(
            coordinates,
            rho0=1.0,
            p0=1.0,
            velocity0=(0.0, 0.0, 0.0),
            gamma=1.4,
            mach=0.3,
            amplitude=amplitude,
            center=(0.0, 0.0, 0.0),
            width=0.1,
            direction=direction,
            clearance_widths=0.0,
        )


def test_reflection_energy_uses_separate_time_windows() -> None:
    times = np.arange(6, dtype=np.float64)
    w_plus = np.zeros((6, 2, 2), dtype=np.float64)
    w_minus = np.zeros_like(w_plus)
    w_plus[0:3] = 2.0
    w_minus[3:6] = 0.2
    result = integrate_characteristic_energy(
        times,
        w_plus,
        w_minus,
        np.ones((2, 2), dtype=np.float64),
        rho0=1.0,
        c0=1.0,
        incident_window=(0.0, 2.0),
        reflected_window=(3.0, 5.0),
    )
    assert result["reflection"] == pytest.approx(0.1)


def test_reflection_windows_include_roundoff_shifted_endpoints() -> None:
    times = np.array(
        [
            0.0,
            0.02,
            0.04,
            np.nextafter(0.06, 0.0),
            0.08,
            np.nextafter(0.10, 1.0),
        ]
    )
    w_plus = np.ones((6, 1, 1), dtype=np.float64)
    w_minus = 0.1 * w_plus
    result = integrate_characteristic_energy(
        times,
        w_plus,
        w_minus,
        np.ones((1, 1), dtype=np.float64),
        rho0=1.0,
        c0=1.0,
        incident_window=(0.0, 0.04),
        reflected_window=(0.06, 0.10),
    )
    assert result["reflection"] == pytest.approx(0.1)


def test_reflection_energy_rejects_invalid_inputs() -> None:
    times = np.arange(4, dtype=np.float64)
    fields = np.zeros((4, 2, 2), dtype=np.float64)
    weights = np.ones((2, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="overlap"):
        integrate_characteristic_energy(
            times,
            fields,
            fields,
            weights,
            rho0=1.0,
            c0=1.0,
            incident_window=(0.0, 2.0),
            reflected_window=(1.0, 3.0),
        )
    with pytest.raises(ValueError, match="incident energy"):
        integrate_characteristic_energy(
            times,
            fields,
            fields,
            weights,
            rho0=1.0,
            c0=1.0,
            incident_window=(0.0, 1.0),
            reflected_window=(2.0, 3.0),
        )
    bad = fields.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        integrate_characteristic_energy(
            times,
            bad,
            fields,
            weights,
            rho0=1.0,
            c0=1.0,
            incident_window=(0.0, 1.0),
            reflected_window=(2.0, 3.0),
        )


def test_reflection_figure_paths_are_eps_and_jpeg(tmp_path: Path) -> None:
    eps, jpeg = figure_paths(tmp_path)
    assert eps == tmp_path / "reflection.eps"
    assert jpeg == tmp_path / "reflection.jpeg"


def test_acoustic_gate_uses_plane_packet_and_flat_probe_region() -> None:
    runner = (VALIDATION / "run_curvilinear_nscbc52_acoustic_compare.sh").read_text(
        encoding="ascii"
    )
    assert 'GRID_MAPPING="${GRID_MAPPING:-y-upper-ramp}"' in runner
    assert 'RAMP_START_FRACTION="${RAMP_START_FRACTION:-0.72}"' in runner
    assert 'ACOUSTIC_PROFILE="${ACOUSTIC_PROFILE:-plane-y-compact}"' in runner
    assert 'PROBE_FRACTION="${PROBE_FRACTION:-2,3}"' in runner
    assert '--mapping "$GRID_MAPPING"' in runner
    assert '--profile "$ACOUSTIC_PROFILE"' in runner
    assert "probe_index=$((probe_numerator * jm / probe_denominator))" in runner
