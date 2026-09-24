#!/usr/bin/env python3
"""Check the short reacting normal-shock CPU/GPU gate."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from air5_radau_reference import Air5RadauReference
from check_air5_numq11_shock_tube import (
    NUMQ,
    SPECIES,
    owned_state,
    read_sensor_mask,
    select_q_files,
)
from compare_q_validation_snapshots import read_q_snapshot


def pressure_outlet_error(prefix: Path, mechanism: Path, topology: tuple[int, int, int],
                          target: float) -> float:
    """Check the physical outlet after chemistry and boundary preparation."""
    files = select_q_files(prefix, "post_chemistry", latest=True)
    tx, ty, tz = topology
    if set(files) != set(range(tx*ty*tz)):
        raise ValueError("pressure-outlet snapshot ranks do not match topology")
    thermo = Air5RadauReference(mechanism)
    maximum = 0.0
    for rank, path in files.items():
        if rank % tx != tx-1:
            continue
        snapshot = read_q_snapshot(path)
        im, jm, km, hm, numq = snapshot.header
        q = snapshot.values.reshape(
            (im+2*hm+1, jm+2*hm+1, km+2*hm+1, numq), order="F")
        face = q[hm+im, hm:hm+jm+1, hm:hm+km+1, :]
        _, _, _, pressure, mach = primitive_metrics(face, thermo)
        if not np.isfinite(pressure).all() or np.any(mach <= 0.0) or np.any(mach >= 1.0):
            raise ValueError("pressure-outlet gate requires outward subsonic flow")
        maximum = max(maximum, float(np.max(np.abs(pressure-target))) / target)
    return maximum


@dataclass(frozen=True)
class NormalShockMetrics:
    minimum_density: float
    minimum_species_density: float
    minimum_temperature: float
    minimum_pressure: float
    mass_closure: float
    extrusion_error: float
    initial_flux_relative_spread: float
    upstream_mach: float
    downstream_mach: float
    marked_shock_interfaces: int


def primitive_metrics(q: np.ndarray, thermo: Air5RadauReference):
    density = q[..., 0]
    momentum = q[..., 1:4]
    species_density = q[..., SPECIES]
    vibrational_energy = q[..., 10]
    kinetic_energy = np.sum(momentum * momentum, axis=-1) / (2.0 * density)
    cv_density = species_density @ thermo.cv_tr
    temperature = (
        q[..., 4]
        - kinetic_energy
        - vibrational_energy
        - species_density @ thermo.formation_energy
    ) / cv_density
    pressure = (species_density @ thermo.gas_constant) * temperature
    mass_fraction = species_density / density[..., None]
    gas_constant = mass_fraction @ thermo.gas_constant
    cv = mass_fraction @ thermo.cv_tr
    gamma = (cv + gas_constant) / cv
    speed_of_sound = np.sqrt(gamma * gas_constant * temperature)
    mach = momentum[..., 0] / density / speed_of_sound
    return density, species_density, temperature, pressure, mach


def analyze(prefix: Path, mechanism: Path) -> NormalShockMetrics:
    initial_files = select_q_files(prefix, "pre_chemistry", latest=False)
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
        scale = np.maximum(np.max(np.abs(final), axis=(0, 1, 2)), 1.0)
        extrusion_error = max(
            extrusion_error,
            float(
                np.max(
                    np.abs(final - final[:, :1, :1, :])
                    / scale[None, None, None, :]
                )
            ),
        )

    initial = np.concatenate(initial_parts, axis=0)
    final = np.concatenate(final_parts, axis=0)
    thermo = Air5RadauReference(mechanism)
    density, species_density, temperature, pressure, _ = primitive_metrics(
        final, thermo
    )
    initial_density, _, _, initial_pressure, initial_mach = primitive_metrics(
        initial, thermo
    )
    initial_u = initial[:, 1] / initial_density
    initial_fluxes = np.column_stack(
        (
            initial[:, 1],
            initial[:, 1] * initial_u + initial_pressure,
            initial_u * (initial[:, 4] + initial_pressure),
        )
    )
    flux_scale = np.maximum(np.median(np.abs(initial_fluxes), axis=0), 1.0)
    flux_spread = float(
        np.max(
            np.abs(initial_fluxes - np.median(initial_fluxes, axis=0))
            / flux_scale
        )
    )

    sensor_files = sorted(
        prefix.parent.glob(f"{prefix.name}.sensor.step*.rk01.rank*.bin")
    )
    if not sensor_files:
        raise ValueError(f"no shock-sensor snapshots found for {prefix}")
    marked = 0
    for path in sensor_files:
        _, mask = read_sensor_mask(path)
        marked += int(np.count_nonzero(mask[:-1, :-1, :-1]))

    return NormalShockMetrics(
        minimum_density=float(np.min(density)),
        minimum_species_density=float(np.min(species_density)),
        minimum_temperature=float(np.min(temperature)),
        minimum_pressure=float(np.min(pressure)),
        mass_closure=float(
            np.max(np.abs(np.sum(species_density, axis=1) - density))
        ),
        extrusion_error=extrusion_error,
        initial_flux_relative_spread=flux_spread,
        upstream_mach=float(np.max(initial_mach)),
        downstream_mach=float(np.min(initial_mach)),
        marked_shock_interfaces=marked,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--mechanism", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--closure-tol", type=float, default=2.0e-12)
    parser.add_argument("--extrusion-tol", type=float, default=2.0e-11)
    parser.add_argument("--flux-tol", type=float, default=2.0e-12)
    parser.add_argument("--boundary-states", type=Path)
    parser.add_argument("--topology", default="1,1,1")
    args = parser.parse_args()

    metrics = analyze(args.prefix, args.mechanism)
    passed = (
        metrics.minimum_density > 0.0
        and metrics.minimum_species_density >= -args.closure_tol
        and metrics.minimum_temperature > 0.0
        and metrics.minimum_pressure > 0.0
        and metrics.mass_closure <= args.closure_tol
        and metrics.extrusion_error <= args.extrusion_tol
        and metrics.initial_flux_relative_spread <= args.flux_tol
        and metrics.upstream_mach > 1.0
        and 0.0 < metrics.downstream_mach < 1.0
        and metrics.marked_shock_interfaces > 0
    )
    lines = [f"status: {'pass' if passed else 'fail'}"]
    lines.extend(
        f"{name}: {value:.16e}"
        for name, value in (
            ("minimum_density", metrics.minimum_density),
            ("minimum_species_density", metrics.minimum_species_density),
            ("minimum_temperature", metrics.minimum_temperature),
            ("minimum_pressure", metrics.minimum_pressure),
            ("mass_closure", metrics.mass_closure),
            ("extrusion_error", metrics.extrusion_error),
            ("initial_flux_relative_spread", metrics.initial_flux_relative_spread),
            ("upstream_mach", metrics.upstream_mach),
            ("downstream_mach", metrics.downstream_mach),
        )
    )
    lines.append(f"marked_shock_interfaces: {metrics.marked_shock_interfaces}")
    if args.boundary_states is not None:
        targets = [float(line.split("=", 1)[1])
                   for line in args.boundary_states.read_text().splitlines()
                   if line.startswith("# outlet_pressure_pa=")]
        if targets:
            if len(targets) != 1 or not np.isfinite(targets[0]) or targets[0] <= 0.0:
                raise ValueError("invalid pressure-outlet metadata")
            topology = tuple(int(x) for x in args.topology.split(","))
            if len(topology) != 3 or min(topology) < 1:
                raise ValueError("invalid pressure-outlet topology")
            error = pressure_outlet_error(args.prefix, args.mechanism, topology, targets[0])
            passed = passed and error <= 2.0e-11
            lines.append(f"outlet_pressure_relative_error: {error:.16e}")
            lines[0] = f"status: {'pass' if passed else 'fail'}"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
