#!/usr/bin/env python3
"""Check positivity, closure, conservation, and sensor activation for numq11."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from air5_radau_reference import Air5RadauReference
from compare_q_validation_snapshots import QSnapshot, read_q_snapshot


NUMQ = 11
SPECIES = slice(5, 10)


@dataclass(frozen=True)
class ShockTubeMetrics:
    minimum_density: float
    minimum_species_density: float
    minimum_temperature: float
    minimum_pressure: float
    mass_closure: float
    conservation_relative_change: float
    extrusion_error: float
    marked_shock_interfaces: int


def q_array(snapshot: QSnapshot) -> np.ndarray:
    im, jm, km, hm, numq = snapshot.header
    if numq != NUMQ:
        raise ValueError(f"expected numq={NUMQ}, found {numq}")
    return snapshot.values.reshape(
        (im + 2 * hm + 1, jm + 2 * hm + 1, km + 2 * hm + 1, numq),
        order="F",
    )


def owned_state(snapshot: QSnapshot) -> np.ndarray:
    im, jm, km, hm, _ = snapshot.header
    values = q_array(snapshot)
    return values[hm : hm + im, hm : hm + jm, hm : hm + km, :]


def read_sensor_mask(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as stream:
        header = np.fromfile(stream, dtype=np.int32, count=3)
        if header.size != 3:
            raise ValueError(f"{path}: incomplete sensor header")
        im, jm, km = (int(value) for value in header)
        count = (im + 1) * (jm + 1) * (km + 1)
        sensor = np.fromfile(stream, dtype=np.float64, count=count)
        mask = np.fromfile(stream, dtype=np.int8, count=count)
    if sensor.size != count or mask.size != count:
        raise ValueError(f"{path}: incomplete sensor payload")
    if not np.all(np.isfinite(sensor)):
        raise ValueError(f"{path}: non-finite sensor value")
    shape = (im + 1, jm + 1, km + 1)
    return sensor.reshape(shape, order="F"), mask.reshape(shape, order="F")


def snapshot_rank(path: Path) -> int:
    return int(path.stem.rsplit("rank", 1)[1])


def snapshot_step(path: Path) -> int:
    return int(path.name.split(".step", 1)[1].split(".", 1)[0])


def select_q_files(prefix: Path, label: str, latest: bool) -> dict[int, Path]:
    files = sorted(prefix.parent.glob(f"{prefix.name}.{label}.step*.rk*.rank*.bin"))
    if not files:
        raise ValueError(f"no {label} snapshots found for {prefix}")
    target_step = (max if latest else min)(snapshot_step(path) for path in files)
    stage_files = [path for path in files if snapshot_step(path) == target_step]
    target_stage = (max if latest else min)(
        int(path.name.split(".rk", 1)[1].split(".", 1)[0]) for path in stage_files
    )
    selected = {
        snapshot_rank(path): path
        for path in stage_files
        if int(path.name.split(".rk", 1)[1].split(".", 1)[0]) == target_stage
    }
    return selected


def analyze(prefix: Path, mechanism: Path) -> ShockTubeMetrics:
    initial_files = select_q_files(prefix, "pre_rhs", latest=False)
    final_files = select_q_files(prefix, "post_update", latest=True)
    if initial_files.keys() != final_files.keys():
        raise ValueError("initial/final rank sets differ")

    initial_parts = []
    final_parts = []
    extrusion_error = 0.0
    for rank in sorted(initial_files):
        initial_snapshot = read_q_snapshot(initial_files[rank])
        final_snapshot = read_q_snapshot(final_files[rank])
        if initial_snapshot.header != final_snapshot.header:
            raise ValueError(f"rank {rank}: initial/final headers differ")
        initial = owned_state(initial_snapshot)
        final = owned_state(final_snapshot)
        initial_parts.append(initial.reshape(-1, NUMQ))
        final_parts.append(final.reshape(-1, NUMQ))
        component_scale = np.maximum(
            np.max(np.abs(final), axis=(0, 1, 2)),
            1.0,
        )
        extrusion_error = max(
            extrusion_error,
            float(
                np.max(
                    np.abs(final - final[:, :1, :1, :])
                    / component_scale[None, None, None, :]
                )
            ),
        )

    initial = np.concatenate(initial_parts, axis=0)
    final = np.concatenate(final_parts, axis=0)
    thermo = Air5RadauReference(mechanism)
    density = final[:, 0]
    species_density = final[:, SPECIES]
    momentum = final[:, 1:4]
    vibrational_energy = final[:, 10]
    kinetic_energy = np.sum(momentum * momentum, axis=1) / (2.0 * density)
    cv_density = species_density @ thermo.cv_tr
    temperature = (
        final[:, 4]
        - kinetic_energy
        - vibrational_energy
        - species_density @ thermo.formation_energy
    ) / cv_density
    pressure = (species_density @ thermo.gas_constant) * temperature

    mass_closure = float(np.max(np.abs(np.sum(species_density, axis=1) - density)))
    initial_totals = np.sum(initial, axis=0)
    final_totals = np.sum(final, axis=0)
    conservation_scale = np.maximum(
        np.maximum(np.sum(np.abs(initial), axis=0), np.sum(np.abs(final), axis=0)),
        1.0,
    )
    conservation_relative_change = float(
        np.max(np.abs(final_totals - initial_totals) / conservation_scale)
    )

    sensor_files = sorted(prefix.parent.glob(f"{prefix.name}.sensor.step*.rk01.rank*.bin"))
    if not sensor_files:
        raise ValueError(f"no shock-sensor snapshots found for {prefix}")
    marked_shock_interfaces = 0
    for path in sensor_files:
        _, mask = read_sensor_mask(path)
        marked_shock_interfaces += int(np.count_nonzero(mask[:-1, :-1, :-1]))

    return ShockTubeMetrics(
        minimum_density=float(np.min(density)),
        minimum_species_density=float(np.min(species_density)),
        minimum_temperature=float(np.min(temperature)),
        minimum_pressure=float(np.min(pressure)),
        mass_closure=mass_closure,
        conservation_relative_change=conservation_relative_change,
        extrusion_error=extrusion_error,
        marked_shock_interfaces=marked_shock_interfaces,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True, type=Path)
    parser.add_argument("--mechanism", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--closure-tol", type=float, default=2.0e-12)
    parser.add_argument("--conservation-tol", type=float, default=2.0e-12)
    parser.add_argument("--extrusion-tol", type=float, default=2.0e-11)
    args = parser.parse_args()

    metrics = analyze(args.prefix, args.mechanism)
    passed = (
        metrics.minimum_density > 0.0
        and metrics.minimum_species_density >= -args.closure_tol
        and metrics.minimum_temperature > 0.0
        and metrics.minimum_pressure > 0.0
        and metrics.mass_closure <= args.closure_tol
        and metrics.conservation_relative_change <= args.conservation_tol
        and metrics.extrusion_error <= args.extrusion_tol
        and metrics.marked_shock_interfaces > 0
    )
    lines = [
        f"status: {'pass' if passed else 'fail'}",
        f"minimum_density: {metrics.minimum_density:.16e}",
        f"minimum_species_density: {metrics.minimum_species_density:.16e}",
        f"minimum_temperature: {metrics.minimum_temperature:.16e}",
        f"minimum_pressure: {metrics.minimum_pressure:.16e}",
        f"mass_closure: {metrics.mass_closure:.16e}",
        f"conservation_relative_change: {metrics.conservation_relative_change:.16e}",
        f"extrusion_error: {metrics.extrusion_error:.16e}",
        f"marked_shock_interfaces: {metrics.marked_shock_interfaces}",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
