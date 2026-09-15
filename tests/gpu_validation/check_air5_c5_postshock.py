#!/usr/bin/env python3
"""Validate one fixed-air5 post-shock relaxation line against its ODE profile."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from air5_postshock_reference import Air5FlowPoint, Air5RelaxationProfile
from air5_radau_reference import Air5RadauReference
from compare_q_validation_snapshots import read_q_snapshot


NUMQ = 11
SPECIES = slice(5, 10)


@dataclass(frozen=True)
class PostShockLineMetrics:
    initial_q_max_abs: float
    rho_max_abs: float
    u_max_abs: float
    temperature_max_abs: float
    tv_max_abs: float
    mass_fraction_max_abs: float
    species_mass_closure_max_abs: float
    minimum_species_density: float


def load_global_x_line(
    prefix: Path,
    *,
    label: str,
    stage: int,
    topology: tuple[int, int, int],
    point_count: int,
) -> np.ndarray:
    """Load the jrk=krk=0 line while assigning shared interfaces to the right rank."""
    prefix = Path(prefix)
    tx, ty, tz = topology
    if min(topology) < 1 or point_count < 2:
        raise ValueError("topology and point_count must be positive")
    pattern = (
        f"{prefix.name}.{label}.step00000000.rk{stage:02d}.rank*.bin"
    )
    files = sorted(prefix.parent.glob(pattern))
    expected_ranks = tx * ty * tz
    if len(files) != expected_ranks:
        raise ValueError(
            f"{pattern}: expected {expected_ranks} ranks, found {len(files)}"
        )
    by_rank = {
        int(path.stem.rsplit("rank", 1)[1]): path
        for path in files
    }
    if set(by_rank) != set(range(expected_ranks)):
        raise ValueError("snapshot rank identifiers are not contiguous")

    pieces: list[np.ndarray] = []
    for irk in range(tx):
        rank = irk  # jrk=0 and krk=0 under ASTR's Cartesian rank ordering.
        snapshot = read_q_snapshot(by_rank[rank])
        im, jm, km, hm, numq = snapshot.header
        if numq != NUMQ or min(im, jm, km) < 0:
            raise ValueError(f"rank {rank}: invalid air5 snapshot header")
        shape = (im + 2 * hm + 1, jm + 2 * hm + 1, km + 2 * hm + 1, numq)
        q = snapshot.values.reshape(shape, order="F")
        line = q[hm : hm + im + 1, hm, hm, :]
        pieces.append(line if irk == tx - 1 else line[:-1])
    result = np.concatenate(pieces, axis=0)
    if result.shape != (point_count, NUMQ):
        raise ValueError(
            f"assembled x line has shape {result.shape}, expected ({point_count}, {NUMQ})"
        )
    return result


def snapshot_extrusion_max_abs(
    prefix: Path,
    *,
    label: str,
    stage: int,
    topology: tuple[int, int, int],
    reference_line: np.ndarray,
) -> float:
    """Measure all active nodes, including shared interfaces, against one x line."""
    prefix = Path(prefix)
    tx, ty, tz = topology
    reference = np.asarray(reference_line, dtype=np.float64)
    if min(topology) < 1 or reference.ndim != 2 or reference.shape[1] != NUMQ:
        raise ValueError("invalid topology or post-shock reference line")
    pattern = (
        f"{prefix.name}.{label}.step00000000.rk{stage:02d}.rank*.bin"
    )
    paths = sorted(prefix.parent.glob(pattern))
    expected_ranks = tx * ty * tz
    if len(paths) != expected_ranks:
        raise ValueError(
            f"{pattern}: expected {expected_ranks} ranks, found {len(paths)}"
        )
    by_rank = {
        int(path.stem.rsplit("rank", 1)[1]): path
        for path in paths
    }
    if set(by_rank) != set(range(expected_ranks)):
        raise ValueError("snapshot rank identifiers are not contiguous")

    x_offsets: dict[int, int] = {}
    x_extents: dict[int, int] = {}
    offset = 0
    for irk in range(tx):
        snapshot = read_q_snapshot(by_rank[irk])
        im = snapshot.header[0]
        x_offsets[irk] = offset
        x_extents[irk] = im
        offset += im
    if offset + 1 != reference.shape[0]:
        raise ValueError("snapshot x decomposition does not match the reference line")

    maximum = 0.0
    for rank in range(expected_ranks):
        irk = rank % tx
        snapshot = read_q_snapshot(by_rank[rank])
        im, jm, km, hm, numq = snapshot.header
        if numq != NUMQ or im != x_extents[irk]:
            raise ValueError(f"rank {rank}: inconsistent air5 snapshot decomposition")
        shape = (im + 2 * hm + 1, jm + 2 * hm + 1, km + 2 * hm + 1, numq)
        q = snapshot.values.reshape(shape, order="F")
        active = q[
            hm : hm + im + 1,
            hm : hm + jm + 1,
            hm : hm + km + 1,
            :,
        ]
        expected = reference[
            x_offsets[irk] : x_offsets[irk] + im + 1,
            None,
            None,
            :,
        ]
        maximum = max(maximum, float(np.max(np.abs(active - expected))))
    return maximum


def _reference_q(profile: Air5RelaxationProfile, chemistry: Air5RadauReference) -> np.ndarray:
    values = np.empty((len(profile.points), NUMQ), dtype=np.float64)
    for index, point in enumerate(profile.points):
        rho_species = point.density * point.mass_fraction
        momentum = np.asarray([point.density * point.speed, 0.0, 0.0])
        ev = point.density * point.ev_specific
        values[index, 0] = point.density
        values[index, 1:4] = momentum
        values[index, 4] = chemistry.q5_from_state(
            point.density, momentum, rho_species, ev, point.temperature
        )
        values[index, SPECIES] = rho_species
        values[index, 10] = ev
    return values


def _validate_q(name: str, values: np.ndarray, point_count: int) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (point_count, NUMQ):
        raise ValueError(
            f"{name} must have shape ({point_count}, {NUMQ}), found {result.shape}"
        )
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def analyze_profile_line(
    initial_q: np.ndarray,
    final_q: np.ndarray,
    profile: Air5RelaxationProfile,
    chemistry: Air5RadauReference,
    *,
    boundary_cut: int,
) -> PostShockLineMetrics:
    """Compare one numerical x line with an independent steady relaxation profile."""
    point_count = len(profile.points)
    if profile.x.shape != (point_count,) or point_count < 1:
        raise ValueError("post-shock profile is empty or inconsistent")
    if boundary_cut < 0 or 2 * boundary_cut >= point_count:
        raise ValueError("boundary_cut leaves no interior profile points")

    initial = _validate_q("initial_q", initial_q, point_count)
    final = _validate_q("final_q", final_q, point_count)
    reference_q = _reference_q(profile, chemistry)
    interior = slice(boundary_cut, point_count - boundary_cut)
    numerical = final[interior]
    points = profile.points[interior]

    rho = numerical[:, 0]
    rho_species = numerical[:, SPECIES]
    ev = numerical[:, 10]
    if np.any(rho <= 0.0) or np.any(rho_species < 0.0) or np.any(ev < 0.0):
        raise ValueError("post-shock line contains a nonphysical conservative state")

    momentum = numerical[:, 1:4]
    velocity = momentum / rho[:, None]
    temperature = np.asarray(
        [
            chemistry.temperature_from_q5(
                rho[index], momentum[index], rho_species[index], ev[index], numerical[index, 4]
            )
            for index in range(numerical.shape[0])
        ]
    )
    tv = np.asarray(
        [
            chemistry.tv_from_ev(rho_species[index], ev[index])
            for index in range(numerical.shape[0])
        ]
    )
    mass_fraction = rho_species / rho[:, None]

    reference_rho = np.asarray([point.density for point in points])
    reference_u = np.asarray([point.speed for point in points])
    reference_temperature = np.asarray([point.temperature for point in points])
    reference_tv = np.asarray([point.tv for point in points])
    reference_mass_fraction = np.asarray([point.mass_fraction for point in points])

    return PostShockLineMetrics(
        initial_q_max_abs=float(np.max(np.abs(initial - reference_q))),
        rho_max_abs=float(np.max(np.abs(rho - reference_rho))),
        u_max_abs=float(np.max(np.abs(velocity[:, 0] - reference_u))),
        temperature_max_abs=float(np.max(np.abs(temperature - reference_temperature))),
        tv_max_abs=float(np.max(np.abs(tv - reference_tv))),
        mass_fraction_max_abs=float(
            np.max(np.abs(mass_fraction - reference_mass_fraction))
        ),
        species_mass_closure_max_abs=float(
            np.max(np.abs(rho - np.sum(rho_species, axis=1)))
        ),
        minimum_species_density=float(np.min(rho_species)),
    )


def load_profile(path: Path, chemistry: Air5RadauReference) -> Air5RelaxationProfile:
    rows = np.loadtxt(path, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows[None, :]
    if rows.ndim != 2 or rows.shape[1] != 13 or rows.shape[0] < 2:
        raise ValueError("post-shock profile must contain at least two 13-column rows")
    if not np.all(np.isfinite(rows)) or np.any(np.diff(rows[:, 0]) <= 0.0):
        raise ValueError("post-shock profile coordinates or values are invalid")

    points: list[Air5FlowPoint] = []
    for row in rows:
        density, speed, pressure, temperature, tv = row[1:6]
        mass_fraction = row[6:11]
        ev = row[11]
        if density <= 0.0 or speed <= 0.0 or pressure <= 0.0:
            raise ValueError("post-shock profile contains non-positive flow values")
        if np.any(mass_fraction < 0.0) or abs(np.sum(mass_fraction) - 1.0) > 2.0e-12:
            raise ValueError("post-shock profile mass fractions are invalid")
        rho_species = density * mass_fraction
        momentum = np.asarray([density * speed, 0.0, 0.0])
        computed_q5 = chemistry.q5_from_state(
            density, momentum, rho_species, ev, temperature
        )
        q5_scale = max(abs(row[12]), 1.0)
        if abs(computed_q5 - row[12]) > 2.0e-12 * q5_scale:
            raise ValueError("post-shock profile q5 is inconsistent")
        gas_constant = float(np.dot(mass_fraction, chemistry.gas_constant))
        cv = float(np.dot(mass_fraction, chemistry.cv_tr))
        gamma = (cv + gas_constant) / cv
        mach = speed / np.sqrt(gamma * gas_constant * temperature)
        points.append(
            Air5FlowPoint(
                density=float(density),
                speed=float(speed),
                pressure=float(pressure),
                temperature=float(temperature),
                tv=float(tv),
                mass_fraction=mass_fraction.copy(),
                ev_specific=float(ev / density),
                mach=float(mach),
            )
        )
    return Air5RelaxationProfile(x=rows[:, 0].copy(), points=tuple(points))


def _parse_topology(text: str) -> tuple[int, int, int]:
    try:
        values = tuple(int(value.strip()) for value in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("topology must contain three integers") from error
    if len(values) != 3 or min(values) < 1:
        raise argparse.ArgumentTypeError("topology must contain three positive integers")
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--mechanism", required=True, type=Path)
    parser.add_argument("--topology", required=True, type=_parse_topology)
    parser.add_argument("--boundary-cut", type=int, default=6)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--initial-q-atol", type=float, default=1.0e-8)
    parser.add_argument("--rho-atol", type=float, default=1.0e-7)
    parser.add_argument("--u-atol", type=float, default=1.0e-3)
    parser.add_argument("--temperature-atol", type=float, default=1.0e-3)
    parser.add_argument("--tv-atol", type=float, default=1.0e-3)
    parser.add_argument("--mass-fraction-atol", type=float, default=2.0e-6)
    parser.add_argument("--mass-closure-atol", type=float, default=1.0e-11)
    parser.add_argument("--extrusion-atol", type=float, default=1.0e-8)
    parser.add_argument("--minimum-species-density", type=float, default=0.0)
    args = parser.parse_args()
    limits = (
        args.initial_q_atol,
        args.rho_atol,
        args.u_atol,
        args.temperature_atol,
        args.tv_atol,
        args.mass_fraction_atol,
        args.mass_closure_atol,
        args.extrusion_atol,
        args.minimum_species_density,
    )
    if min(limits) < 0.0:
        raise ValueError("post-shock validation limits must be non-negative")

    try:
        chemistry = Air5RadauReference(args.mechanism)
        profile = load_profile(args.profile, chemistry)
        initial_q = load_global_x_line(
            args.prefix,
            label="pre_chemistry",
            stage=1,
            topology=args.topology,
            point_count=len(profile.points),
        )
        final_q = load_global_x_line(
            args.prefix,
            label="post_chemistry",
            stage=2,
            topology=args.topology,
            point_count=len(profile.points),
        )
        metrics = analyze_profile_line(
            initial_q,
            final_q,
            profile,
            chemistry,
            boundary_cut=args.boundary_cut,
        )
        max_extrusion_abs = snapshot_extrusion_max_abs(
            args.prefix,
            label="post_chemistry",
            stage=2,
            topology=args.topology,
            reference_line=final_q,
        )
        passed = (
            metrics.initial_q_max_abs <= args.initial_q_atol
            and metrics.rho_max_abs <= args.rho_atol
            and metrics.u_max_abs <= args.u_atol
            and metrics.temperature_max_abs <= args.temperature_atol
            and metrics.tv_max_abs <= args.tv_atol
            and metrics.mass_fraction_max_abs <= args.mass_fraction_atol
            and metrics.species_mass_closure_max_abs <= args.mass_closure_atol
            and max_extrusion_abs <= args.extrusion_atol
            and metrics.minimum_species_density > args.minimum_species_density
        )
        lines = [
            f"status: {'pass' if passed else 'fail'}",
            f"initial_q_max_abs: {metrics.initial_q_max_abs:.16e}",
            f"rho_max_abs: {metrics.rho_max_abs:.16e}",
            f"u_max_abs: {metrics.u_max_abs:.16e}",
            f"temperature_max_abs: {metrics.temperature_max_abs:.16e}",
            f"tv_max_abs: {metrics.tv_max_abs:.16e}",
            f"mass_fraction_max_abs: {metrics.mass_fraction_max_abs:.16e}",
            f"species_mass_closure_max_abs: {metrics.species_mass_closure_max_abs:.16e}",
            f"minimum_species_density: {metrics.minimum_species_density:.16e}",
            f"max_extrusion_abs: {max_extrusion_abs:.16e}",
            f"boundary_cut: {args.boundary_cut}",
            f"initial_q_limit: {args.initial_q_atol:.16e}",
            f"rho_limit: {args.rho_atol:.16e}",
            f"u_limit: {args.u_atol:.16e}",
            f"temperature_limit: {args.temperature_atol:.16e}",
            f"tv_limit: {args.tv_atol:.16e}",
            f"mass_fraction_limit: {args.mass_fraction_atol:.16e}",
            f"mass_closure_limit: {args.mass_closure_atol:.16e}",
            f"extrusion_limit: {args.extrusion_atol:.16e}",
        ]
    except (OSError, KeyError, TypeError, ValueError) as error:
        passed = False
        lines = ["status: fail", f"error: {error}"]

    output = "\n".join(lines) + "\n"
    print(output, end="")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(output, encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
