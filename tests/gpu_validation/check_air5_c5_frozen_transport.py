#!/usr/bin/env python3
"""Validate the C5 frozen-air5 transport cases against an independent oracle."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from air5_radau_reference import Air5RadauReference
from air5_transport_reference import Air5TransportReference, derivative_sixth_order
from compare_q_validation_snapshots import QSnapshot, read_q_snapshot


NUMQ = 11
SPECIES = slice(5, 10)
ROUNDOFF_MULTIPLIER = 256.0


@dataclass(frozen=True)
class RhsSnapshot:
    header: tuple[int, int, int, int]
    values: np.ndarray


@dataclass(frozen=True)
class FrozenTransportMetrics:
    max_rhs_abs: float
    max_rhs_scaled: float
    max_extrusion_error: float
    max_rhs_extrusion_scaled: float
    max_species_mass_closure: float
    max_conservation_relative_change: float
    correction_velocity_max: float
    active_species_rhs_count: int
    exact_translation_scaled_error: float
    variance_relative_change: float
    invariant_scaled_error: float


def derivative_from_halo(
    values: np.ndarray, halo: int, active_extent: int, spacing: float
) -> np.ndarray:
    """Differentiate active indices 0:active_extent from a centered halo line."""
    values = np.asarray(values, dtype=float)
    if values.shape[0] < active_extent + 2 * halo + 1 or halo < 3:
        raise ValueError("insufficient halo width for a sixth-order derivative")
    centers = np.arange(halo, halo + active_extent + 1)
    return (
        0.75 * (values[centers + 1] - values[centers - 1])
        - 0.15 * (values[centers + 2] - values[centers - 2])
        + (values[centers + 3] - values[centers - 3]) / 60.0
    ) / spacing


def read_rhs_snapshot(path: Path) -> RhsSnapshot:
    path = Path(path)
    with path.open("rb") as stream:
        header_array = np.fromfile(stream, dtype=np.int32, count=4)
        values = np.fromfile(stream, dtype=np.float64)
    if header_array.size != 4:
        raise ValueError(f"{path}: incomplete RHS snapshot header")
    header = tuple(int(value) for value in header_array)
    im, jm, km, numq = header
    expected = (im + 1) * (jm + 1) * (km + 1) * numq
    if min(im, jm, km) < 0 or numq != NUMQ or values.size != expected:
        raise ValueError(f"{path}: invalid RHS snapshot shape {header}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path}: non-finite RHS value")
    return RhsSnapshot(header=header, values=values)


def _rank_files(prefix: Path, label: str, rk: int) -> dict[int, Path]:
    pattern = f"{prefix.name}.{label}.step00000000.rk{rk:02d}.rank*.bin"
    files: dict[int, Path] = {}
    for path in sorted(prefix.parent.glob(pattern)):
        rank = int(path.stem.rsplit("rank", 1)[1])
        files[rank] = path
    if not files:
        raise ValueError(f"no {label} RK{rk} snapshots found for {prefix}")
    return files


def _q_array(snapshot: QSnapshot) -> np.ndarray:
    im, jm, km, hm, numq = snapshot.header
    return snapshot.values.reshape(
        (im + 2 * hm + 1, jm + 2 * hm + 1, km + 2 * hm + 1, numq),
        order="F",
    )


def _rhs_array(snapshot: RhsSnapshot) -> np.ndarray:
    im, jm, km, numq = snapshot.header
    return snapshot.values.reshape((im + 1, jm + 1, km + 1, numq), order="F")


def _rank_coordinates(rank: int, topology: tuple[int, int, int]) -> tuple[int, int, int]:
    tx, ty, _ = topology
    irk = rank % tx
    jrk = (rank // tx) % ty
    krk = rank // (tx * ty)
    return irk, jrk, krk


def _load_q_set(files: dict[int, Path]) -> dict[int, tuple[QSnapshot, np.ndarray]]:
    result: dict[int, tuple[QSnapshot, np.ndarray]] = {}
    for rank, path in files.items():
        snapshot = read_q_snapshot(path)
        result[rank] = snapshot, _q_array(snapshot)
    return result


def _assemble_global_x(
    snapshots: dict[int, tuple[QSnapshot, np.ndarray]],
    topology: tuple[int, int, int],
    global_x: int,
) -> tuple[np.ndarray, dict[int, int]]:
    tx, _, _ = topology
    pieces: list[np.ndarray] = []
    offsets: dict[int, int] = {}
    offset = 0
    for irk in range(tx):
        rank = irk
        if rank not in snapshots:
            raise ValueError(f"rank {rank} is missing from the x-line assembly")
        snapshot, q = snapshots[rank]
        im, _, _, hm, _ = snapshot.header
        offsets[irk] = offset
        pieces.append(q[hm : hm + im, hm, hm, :])
        offset += im
    assembled = np.concatenate(pieces, axis=0)
    if offset != global_x or assembled.shape != (global_x, NUMQ):
        raise ValueError(
            f"global x assembly mismatch: expected {global_x}, found {assembled.shape[0]}"
        )
    return assembled, offsets


def _recover_primitives(
    q: np.ndarray, thermo: Air5RadauReference
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rho = q[:, 0]
    momentum = q[:, 1:4]
    rho_species = q[:, SPECIES]
    ev = q[:, 10]
    if np.any(rho <= 0.0) or np.any(rho_species < 0.0):
        raise ValueError("nonphysical conservative state in frozen-transport snapshot")
    velocity = momentum / rho[:, None]
    kinetic = np.sum(momentum * momentum, axis=1) / (2.0 * rho)
    cv_density = rho_species @ thermo.cv_tr
    temperature = (
        q[:, 4]
        - kinetic
        - ev
        - rho_species @ thermo.formation_energy
    ) / cv_density
    mass_fraction = rho_species / rho[:, None]
    tv = np.asarray(
        [thermo.tv_from_ev(rho_species[index], ev[index]) for index in range(q.shape[0])]
    )
    pressure = np.sum(rho_species * thermo.gas_constant[None, :], axis=1) * temperature
    if not np.all(np.isfinite([temperature, tv, pressure])):
        raise ValueError("non-finite primitive state recovered from snapshot")
    return rho, velocity, temperature, tv, pressure, mass_fraction


def _independent_rhs(
    q: np.ndarray,
    spacing: float,
    diffusion: bool,
    thermo: Air5RadauReference,
    transport: Air5TransportReference,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rho, velocity, temperature, tv, pressure, mass_fraction = _recover_primitives(q, thermo)
    convective_flux = q * velocity[:, 0, None]
    convective_flux[:, 1] += pressure
    convective_flux[:, 4] += pressure * velocity[:, 0]
    rhs = -derivative_sixth_order(convective_flux, spacing)
    flux_rate_scale = np.max(np.abs(convective_flux), axis=0) / spacing
    # Cartesian metrics are themselves differentiated numerically.  Their
    # roundoff-level off-diagonal entries can project pressure into either
    # transverse momentum equation even when the analytic flux is zero.
    flux_rate_scale[1:4] = np.maximum(
        flux_rate_scale[1:4], np.max(np.abs(pressure)) / spacing
    )
    correction_velocity = np.zeros(q.shape[0])
    species_flux = np.zeros((q.shape[0], 5))
    if not diffusion:
        roundoff_floor = ROUNDOFF_MULTIPLIER * np.finfo(float).eps * flux_rate_scale
        return rhs, correction_velocity, species_flux, roundoff_floor

    grad_velocity = derivative_sixth_order(velocity, spacing)
    grad_temperature = derivative_sixth_order(temperature, spacing)
    grad_tv = derivative_sixth_order(tv, spacing)
    grad_mass_fraction = derivative_sixth_order(mass_fraction, spacing)
    diffusive_flux = np.zeros_like(q)
    for index in range(q.shape[0]):
        grad_u = np.zeros((3, 3))
        grad_u[:, 0] = grad_velocity[index]
        grad_y = np.zeros((5, 3))
        grad_y[:, 0] = grad_mass_fraction[index]
        result = transport.diffusive_flux(
            rho=rho[index],
            velocity=velocity[index],
            temperature=temperature[index],
            tv=tv[index],
            pressure=pressure[index],
            grad_velocity=grad_u,
            grad_temperature=np.asarray([grad_temperature[index], 0.0, 0.0]),
            grad_tv=np.asarray([grad_tv[index], 0.0, 0.0]),
            mass_fraction=mass_fraction[index],
            grad_mass_fraction=grad_y,
        )
        diffusive_flux[index, 1:4] = result.momentum_flux[:, 0]
        diffusive_flux[index, 4] = result.energy_flux[0]
        diffusive_flux[index, SPECIES] = result.species_flux[:, 0]
        diffusive_flux[index, 10] = result.ev_flux[0]
        correction_velocity[index] = result.correction_velocity[0]
        species_flux[index] = result.species_flux[:, 0]

    divergence = derivative_sixth_order(diffusive_flux, spacing)
    flux_rate_scale += np.max(np.abs(diffusive_flux), axis=0) / spacing
    rhs[:, 1:5] += divergence[:, 1:5]
    rhs[:, SPECIES] -= divergence[:, SPECIES]
    rhs[:, 10] += divergence[:, 10]
    roundoff_floor = ROUNDOFF_MULTIPLIER * np.finfo(float).eps * flux_rate_scale
    return rhs, correction_velocity, species_flux, roundoff_floor


def _scaled_max_error(
    actual: np.ndarray,
    expected: np.ndarray,
    atol: float,
    rtol: float,
    roundoff_floor: np.ndarray,
) -> float:
    scale = atol + roundoff_floor + rtol * np.abs(expected)
    return float(np.max(np.abs(actual - expected) / scale))


def _global_owned_state(
    snapshots: dict[int, tuple[QSnapshot, np.ndarray]],
    topology: tuple[int, int, int],
) -> np.ndarray:
    pieces: list[np.ndarray] = []
    for rank in sorted(snapshots):
        snapshot, q = snapshots[rank]
        im, jm, km, hm, _ = snapshot.header
        pieces.append(q[hm : hm + im, hm : hm + jm, hm : hm + km, :].reshape(-1, NUMQ))
    return np.concatenate(pieces, axis=0)


def analyze(
    prefix: Path,
    mechanism: Path,
    case: str,
    grid: tuple[int, int, int],
    topology: tuple[int, int, int],
    deltat: float,
    ref_len: float,
    rhs_atol: float,
    rhs_rtol: float,
) -> FrozenTransportMetrics:
    if case not in {"advection-wave", "diffusion-layer", "ev-pulse"}:
        raise ValueError(f"unsupported frozen-transport case: {case}")
    if np.prod(topology) <= 0 or tuple(grid[index] % topology[index] for index in range(3)) != (0, 0, 0):
        raise ValueError("grid must be divisible by the requested topology")

    pre_files = _rank_files(prefix, "pre_rhs", 1)
    final_files = _rank_files(prefix, "post_update", 3)
    rhs_files = _rank_files(prefix, "full", 1)
    expected_ranks = int(np.prod(topology))
    if not (len(pre_files) == len(final_files) == len(rhs_files) == expected_ranks):
        raise ValueError("snapshot rank count does not match topology")
    pre = _load_q_set(pre_files)
    final = _load_q_set(final_files)
    global_pre, x_offsets = _assemble_global_x(pre, topology, grid[0])
    global_final, _ = _assemble_global_x(final, topology, grid[0])

    thermo = Air5RadauReference(mechanism)
    transport = Air5TransportReference(mechanism)
    spacing = 2.0 * np.pi * ref_len / grid[0]
    expected_rhs, correction_velocity, species_flux, roundoff_floor = _independent_rhs(
        global_pre, spacing, case != "advection-wave", thermo, transport
    )

    max_rhs_abs = 0.0
    max_rhs_scaled = 0.0
    max_extrusion_error = 0.0
    max_rhs_extrusion_scaled = 0.0
    for rank in sorted(rhs_files):
        irk, _, _ = _rank_coordinates(rank, topology)
        rhs_snapshot = read_rhs_snapshot(rhs_files[rank])
        rhs = _rhs_array(rhs_snapshot)
        q_snapshot, q = pre[rank]
        im, jm, km, hm, _ = q_snapshot.header
        if rhs_snapshot.header != (im, jm, km, NUMQ):
            raise ValueError(f"rank {rank}: q/RHS header mismatch")
        indices = (x_offsets[irk] + np.arange(im + 1)) % grid[0]
        expected = expected_rhs[indices]
        actual = rhs / (spacing * (2.0 * np.pi * ref_len / grid[1]) * (2.0 * np.pi * ref_len / grid[2]))
        expected_4d = expected[:, None, None, :]
        max_rhs_abs = max(max_rhs_abs, float(np.max(np.abs(actual - expected_4d))))
        max_rhs_scaled = max(
            max_rhs_scaled,
            _scaled_max_error(
                actual,
                expected_4d,
                rhs_atol,
                rhs_rtol,
                roundoff_floor[None, None, None, :],
            ),
        )
        active_q = q[hm : hm + im + 1, hm : hm + jm + 1, hm : hm + km + 1, :]
        max_extrusion_error = max(
            max_extrusion_error,
            float(np.max(np.abs(active_q - active_q[:, :1, :1, :]))),
        )
        rhs_extrusion = np.abs(actual - actual[:, :1, :1, :])
        rhs_scale = (
            rhs_atol
            + roundoff_floor[None, None, None, :]
            + rhs_rtol * np.maximum(np.abs(actual), np.abs(expected_4d))
        )
        max_rhs_extrusion_scaled = max(
            max_rhs_extrusion_scaled,
            float(np.max(rhs_extrusion / rhs_scale)),
        )

    initial_owned = _global_owned_state(pre, topology)
    final_owned = _global_owned_state(final, topology)
    mass_closure = max(
        float(np.max(np.abs(np.sum(initial_owned[:, SPECIES], axis=1) - initial_owned[:, 0]))),
        float(np.max(np.abs(np.sum(final_owned[:, SPECIES], axis=1) - final_owned[:, 0]))),
    )
    initial_totals = np.sum(initial_owned, axis=0)
    final_totals = np.sum(final_owned, axis=0)
    conservation_relative = float(
        np.max(np.abs(final_totals - initial_totals) / np.maximum(np.abs(initial_totals), 1.0))
    )

    exact_error = 0.0
    variance_change = 0.0
    invariant_error = 0.0
    if case == "advection-wave":
        frequency = 2.0 * np.pi * np.fft.fftfreq(grid[0], d=spacing)
        shifted = np.fft.ifft(
            np.fft.fft(global_pre, axis=0)
            * np.exp(-1j * frequency[:, None] * 100.0 * deltat),
            axis=0,
        ).real
        scale = np.maximum(np.ptp(global_pre, axis=0), 1.0)
        exact_error = float(np.max(np.abs(global_final - shifted) / scale[None, :]))
    elif case == "diffusion-layer":
        initial_variance = float(np.var(global_pre[:, 5]))
        final_variance = float(np.var(global_final[:, 5]))
        variance_change = (final_variance - initial_variance) / initial_variance
    else:
        initial_variance = float(np.var(global_pre[:, 10]))
        final_variance = float(np.var(global_final[:, 10]))
        variance_change = (final_variance - initial_variance) / initial_variance
        unchanged = np.r_[0:4, 5:10]
        unchanged_scale = np.maximum(np.abs(global_pre[:, unchanged]), 1.0)
        unchanged_error = np.max(
            np.abs(global_final[:, unchanged] - global_pre[:, unchanged]) / unchanged_scale
        )
        initial_difference = global_pre[:, 4] - global_pre[:, 10]
        final_difference = global_final[:, 4] - global_final[:, 10]
        difference_scale = np.maximum(np.abs(initial_difference), 1.0)
        invariant_error = float(max(unchanged_error, np.max(np.abs(final_difference - initial_difference) / difference_scale)))

    active_species_rhs_count = int(
        np.count_nonzero(np.max(np.abs(expected_rhs[:, SPECIES]), axis=0) > rhs_atol)
    )
    return FrozenTransportMetrics(
        max_rhs_abs=max_rhs_abs,
        max_rhs_scaled=max_rhs_scaled,
        max_extrusion_error=max_extrusion_error,
        max_rhs_extrusion_scaled=max_rhs_extrusion_scaled,
        max_species_mass_closure=mass_closure,
        max_conservation_relative_change=conservation_relative,
        correction_velocity_max=float(np.max(np.abs(correction_velocity))),
        active_species_rhs_count=active_species_rhs_count,
        exact_translation_scaled_error=exact_error,
        variance_relative_change=variance_change,
        invariant_scaled_error=invariant_error,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True, type=Path)
    parser.add_argument("--mechanism", required=True, type=Path)
    parser.add_argument("--case", required=True, choices=("advection-wave", "diffusion-layer", "ev-pulse"))
    parser.add_argument("--grid", default="48,6,6")
    parser.add_argument("--topology", default="1,1,1")
    parser.add_argument("--deltat", type=float, default=1.0e-6)
    parser.add_argument("--ref-len", type=float, default=1.0e-2)
    parser.add_argument("--rhs-atol", type=float, default=1.0e-8)
    parser.add_argument("--rhs-rtol", type=float, default=2.0e-10)
    parser.add_argument("--structural-tol", type=float, default=2.0e-11)
    parser.add_argument("--translation-tol", type=float, default=2.0e-9)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    grid = tuple(int(value) for value in args.grid.split(","))
    topology = tuple(int(value) for value in args.topology.split(","))
    if len(grid) != 3 or len(topology) != 3:
        raise ValueError("grid and topology must each contain three integers")

    metrics = analyze(
        args.prefix,
        args.mechanism,
        args.case,
        grid,
        topology,
        args.deltat,
        args.ref_len,
        args.rhs_atol,
        args.rhs_rtol,
    )
    passed = metrics.max_rhs_scaled <= 1.0
    passed &= metrics.max_extrusion_error <= args.structural_tol
    passed &= metrics.max_rhs_extrusion_scaled <= 1.0
    passed &= metrics.max_species_mass_closure <= args.structural_tol
    passed &= metrics.max_conservation_relative_change <= args.structural_tol
    if args.case == "advection-wave":
        passed &= metrics.exact_translation_scaled_error <= args.translation_tol
    elif args.case == "diffusion-layer":
        passed &= metrics.correction_velocity_max > 1.0e-12
        passed &= metrics.active_species_rhs_count == 5
        passed &= metrics.variance_relative_change < 0.0
    else:
        passed &= metrics.variance_relative_change < 0.0
        passed &= metrics.invariant_scaled_error <= args.structural_tol

    lines = [
        f"status: {'pass' if passed else 'fail'}",
        f"case: {args.case}",
        f"max_rhs_abs: {metrics.max_rhs_abs:.16e}",
        f"max_rhs_scaled: {metrics.max_rhs_scaled:.16e}",
        f"max_extrusion_error: {metrics.max_extrusion_error:.16e}",
        f"max_rhs_extrusion_scaled: {metrics.max_rhs_extrusion_scaled:.16e}",
        f"max_species_mass_closure: {metrics.max_species_mass_closure:.16e}",
        f"max_conservation_relative_change: {metrics.max_conservation_relative_change:.16e}",
        f"correction_velocity_max: {metrics.correction_velocity_max:.16e}",
        f"active_species_rhs_count: {metrics.active_species_rhs_count}",
        f"exact_translation_scaled_error: {metrics.exact_translation_scaled_error:.16e}",
        f"variance_relative_change: {metrics.variance_relative_change:.16e}",
        f"invariant_scaled_error: {metrics.invariant_scaled_error:.16e}",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
