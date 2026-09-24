#!/usr/bin/env python3
"""Validate a captured reacting normal shock against a steady air5 reference."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import h5py
from scipy.integrate import quad

from air5_postshock_reference import Air5PostShockReference
from air5_radau_reference import Air5RadauReference
from check_air5_c5_normal_shock import primitive_metrics
from check_air5_c5_postshock import (
    load_global_x_line,
    snapshot_extrusion_max_abs,
)


@dataclass(frozen=True)
class CapturedNormalShockMetrics:
    shock_interface: int
    shock_x: float
    comparison_points: int
    mass_flux_relative_error: float
    momentum_flux_relative_error: float
    energy_flux_relative_error: float
    rho_relative_error: float
    u_relative_error: float
    temperature_relative_error: float
    tv_relative_error: float
    mass_fraction_max_abs: float
    species_mass_closure_max_abs: float
    minimum_species_density: float
    terminal_temperature_drop: float
    terminal_tv_rise: float
    terminal_atomic_oxygen_rise: float


def analyze_captured_normal_shock_line(
    q: np.ndarray,
    x: np.ndarray,
    reference: Air5PostShockReference,
    *,
    shock_skip_cells: int,
) -> CapturedNormalShockMetrics:
    """Align on the captured pressure jump and compare only downstream nodes."""
    values = np.asarray(q, dtype=np.float64)
    coordinates = np.asarray(x, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 11:
        raise ValueError("captured normal-shock q line must have shape (n, 11)")
    if coordinates.shape != (values.shape[0],) or values.shape[0] < 6:
        raise ValueError("captured normal-shock coordinates are inconsistent")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(coordinates)):
        raise ValueError("captured normal-shock line contains non-finite values")
    if np.any(np.diff(coordinates) <= 0.0):
        raise ValueError("captured normal-shock coordinates must increase")
    if shock_skip_cells < 0:
        raise ValueError("shock_skip_cells must be non-negative")

    density, species_density, temperature, pressure, _ = primitive_metrics(
        values, reference.chemistry
    )
    if np.any(density <= 0.0) or np.any(temperature <= 0.0) or np.any(pressure <= 0.0):
        raise ValueError("captured normal-shock line has a non-positive bulk state")
    shock_interface = int(np.argmax(np.abs(np.diff(pressure))))
    shock_x = float(
        0.5 * (coordinates[shock_interface] + coordinates[shock_interface + 1])
    )
    first = shock_interface + 1 + shock_skip_cells
    if first >= values.shape[0] - 1:
        raise ValueError("too few downstream nodes remain after excluding the shock")
    downstream_distance = coordinates[first:] - shock_x
    if downstream_distance[0] <= 0.0:
        raise ValueError("downstream comparison coordinates are not positive")

    solution = reference.integrate(float(downstream_distance[-1]), rtol=1.0e-10)
    profile = reference.sample(solution, downstream_distance)
    numerical = values[first:]
    numerical_density = density[first:]
    numerical_species = species_density[first:]
    numerical_temperature = temperature[first:]
    numerical_pressure = pressure[first:]
    numerical_u = numerical[:, 1] / numerical_density
    numerical_y = numerical_species / numerical_density[:, None]
    numerical_tv = np.asarray(
        [
            reference.chemistry.tv_from_ev(numerical_species[index], numerical[index, 10])
            for index in range(numerical.shape[0])
        ]
    )

    reference_density = np.asarray([point.density for point in profile.points])
    reference_u = np.asarray([point.speed for point in profile.points])
    reference_temperature = profile.temperature
    reference_tv = profile.tv
    reference_y = profile.mass_fraction

    mass_flux = numerical[:, 1]
    momentum_flux = numerical[:, 1] * numerical_u + numerical_pressure
    energy_flux = numerical_u * (numerical[:, 4] + numerical_pressure)
    target_flux = reference.conservative_fluxes(reference.upstream)

    def relative_error(actual: np.ndarray, target: np.ndarray) -> float:
        scale = np.maximum(np.abs(target), np.finfo(np.float64).tiny)
        return float(np.max(np.abs(actual - target) / scale))

    frozen = reference.postshock
    return CapturedNormalShockMetrics(
        shock_interface=shock_interface,
        shock_x=shock_x,
        comparison_points=numerical.shape[0],
        mass_flux_relative_error=relative_error(mass_flux, target_flux[0]),
        momentum_flux_relative_error=relative_error(momentum_flux, target_flux[1]),
        energy_flux_relative_error=relative_error(energy_flux, target_flux[2]),
        rho_relative_error=relative_error(numerical_density, reference_density),
        u_relative_error=relative_error(numerical_u, reference_u),
        temperature_relative_error=relative_error(
            numerical_temperature, reference_temperature
        ),
        tv_relative_error=relative_error(numerical_tv, reference_tv),
        mass_fraction_max_abs=float(np.max(np.abs(numerical_y - reference_y))),
        species_mass_closure_max_abs=float(
            np.max(np.abs(numerical_density - np.sum(numerical_species, axis=1)))
        ),
        minimum_species_density=float(np.min(numerical_species)),
        terminal_temperature_drop=float(frozen.temperature - numerical_temperature[-1]),
        terminal_tv_rise=float(numerical_tv[-1] - frozen.tv),
        terminal_atomic_oxygen_rise=float(
            numerical_y[-1, 3] - frozen.mass_fraction[3]
        ),
    )


def _parse_topology(text: str) -> tuple[int, int, int]:
    try:
        values = tuple(int(value.strip()) for value in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("topology must contain three integers") from error
    if len(values) != 3 or min(values) < 1:
        raise argparse.ArgumentTypeError("topology must contain three positive integers")
    return values


def reference_transit_time(reference: Air5PostShockReference, length: float) -> float:
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("reference relaxation length must be finite and positive")
    solution = reference.integrate(length, rtol=1.0e-10)
    value, _ = quad(
        lambda x: 1.0 / reference.reconstruct(solution.sol(x)).speed,
        0.0, length, epsabs=1.0e-13, epsrel=1.0e-8,
    )
    return float(value)


def checkpoint_stationarity(
    previous: Path, current: Path, transit: float, elapsed: float
) -> float:
    """Maximum primitive change per reference transit, at matched checkpoint phases."""
    fields = ("ro", "u1", "u2", "u3", "t", "tv",
              "sp001", "sp002", "sp003", "sp004", "sp005")
    maximum = 0.0
    with h5py.File(previous, "r") as old, h5py.File(current, "r") as new:
        t0 = float(np.asarray(old["time"]).item())
        t1 = float(np.asarray(new["time"]).item())
        if not np.isfinite([t0, t1, transit, elapsed]).all() or transit <= 0.0:
            raise ValueError("invalid stationarity time")
        interval = t1 - t0
        if interval <= 0.0 or elapsed < t1 or elapsed - t1 > interval:
            raise ValueError("stationarity checkpoints have stale or nonincreasing times")
        for field in fields:
            a, b = old[field][()], new[field][()]
            if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError(f"invalid stationarity field: {field}")
            scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), 1.0)
            maximum = max(maximum, float(np.max(np.abs(a-b))) / scale)
    return maximum * transit / interval


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True, type=Path)
    parser.add_argument("--mechanism", required=True, type=Path)
    parser.add_argument("--topology", required=True, type=_parse_topology)
    parser.add_argument("--point-count", required=True, type=int)
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--domain-length", type=float, default=0.02)
    parser.add_argument("--elapsed-time", required=True, type=float)
    parser.add_argument("--minimum-flowthroughs", type=float, default=1.0)
    parser.add_argument("--shock-skip-cells", type=int, default=3)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--flux-relative-tol", type=float, default=5.0e-2)
    parser.add_argument("--field-relative-tol", type=float, default=5.0e-2)
    parser.add_argument("--mass-fraction-atol", type=float, default=1.0e-2)
    parser.add_argument("--mass-closure-atol", type=float, default=2.0e-11)
    parser.add_argument("--extrusion-atol", type=float, default=2.0e-11)
    parser.add_argument("--minimum-species-density", type=float, default=-1.0e-13)
    parser.add_argument("--reference-shock-x", type=float, default=0.01)
    parser.add_argument("--previous-checkpoint", type=Path, required=True)
    parser.add_argument("--current-checkpoint", type=Path, required=True)
    parser.add_argument("--stationarity-tol", type=float, default=1.0e-3)
    args = parser.parse_args()
    if (
        args.point_count < 6
        or args.step < 0
        or args.domain_length <= 0.0
        or args.elapsed_time <= 0.0
        or args.minimum_flowthroughs <= 0.0
        or not 0.0 < args.reference_shock_x < args.domain_length
        or not np.isfinite(args.stationarity_tol) or args.stationarity_tol <= 0.0
    ):
        raise ValueError("point count, step, or domain length is invalid")

    try:
        chemistry = Air5RadauReference(args.mechanism)
        reference = Air5PostShockReference(chemistry)
        line = load_global_x_line(
            args.prefix,
            label="post_chemistry",
            step=args.step,
            stage=2,
            topology=args.topology,
            point_count=args.point_count,
        )
        coordinates = np.linspace(0.0, args.domain_length, args.point_count)
        metrics = analyze_captured_normal_shock_line(
            line,
            coordinates,
            reference,
            shock_skip_cells=args.shock_skip_cells,
        )
        extrusion = snapshot_extrusion_max_abs(
            args.prefix,
            label="post_chemistry",
            step=args.step,
            stage=2,
            topology=args.topology,
            reference_line=line,
            component_scaled=True,
        )
        shock_fraction = metrics.shock_x / args.domain_length
        downstream_flowthrough_time = reference_transit_time(
            reference, args.domain_length - args.reference_shock_x
        )
        completed_flowthroughs = args.elapsed_time / downstream_flowthrough_time
        stationarity = checkpoint_stationarity(
            args.previous_checkpoint, args.current_checkpoint,
            downstream_flowthrough_time, args.elapsed_time,
        )
        passed = (
            0.25 <= shock_fraction <= 0.75
            and completed_flowthroughs >= args.minimum_flowthroughs
            and stationarity <= args.stationarity_tol
            and metrics.comparison_points >= 3
            and metrics.mass_flux_relative_error <= args.flux_relative_tol
            and metrics.momentum_flux_relative_error <= args.flux_relative_tol
            and metrics.energy_flux_relative_error <= args.flux_relative_tol
            and metrics.rho_relative_error <= args.field_relative_tol
            and metrics.u_relative_error <= args.field_relative_tol
            and metrics.temperature_relative_error <= args.field_relative_tol
            and metrics.tv_relative_error <= args.field_relative_tol
            and metrics.mass_fraction_max_abs <= args.mass_fraction_atol
            and metrics.species_mass_closure_max_abs <= args.mass_closure_atol
            and metrics.minimum_species_density >= args.minimum_species_density
            and extrusion <= args.extrusion_atol
            and metrics.terminal_temperature_drop > 0.0
            and metrics.terminal_tv_rise > 0.0
            and metrics.terminal_atomic_oxygen_rise > 0.0
        )
        rows = [
            f"status: {'pass' if passed else 'fail'}",
            f"step: {args.step}",
            f"shock_interface: {metrics.shock_interface}",
            f"shock_x: {metrics.shock_x:.16e}",
            f"shock_domain_fraction: {shock_fraction:.16e}",
            f"elapsed_time: {args.elapsed_time:.16e}",
            f"downstream_flowthrough_time: {downstream_flowthrough_time:.16e}",
            f"completed_flowthroughs: {completed_flowthroughs:.16e}",
            f"max_primitive_change_per_reference_transit: {stationarity:.16e}",
            f"stationarity_tolerance: {args.stationarity_tol:.16e}",
            f"comparison_points: {metrics.comparison_points}",
        ]
        for name in (
            "mass_flux_relative_error",
            "momentum_flux_relative_error",
            "energy_flux_relative_error",
            "rho_relative_error",
            "u_relative_error",
            "temperature_relative_error",
            "tv_relative_error",
            "mass_fraction_max_abs",
            "species_mass_closure_max_abs",
            "minimum_species_density",
            "terminal_temperature_drop",
            "terminal_tv_rise",
            "terminal_atomic_oxygen_rise",
        ):
            rows.append(f"{name}: {getattr(metrics, name):.16e}")
        rows.append(f"max_component_scaled_extrusion: {extrusion:.16e}")
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        passed = False
        rows = ["status: fail", f"error: {error}"]

    output = "\n".join(rows) + "\n"
    print(output, end="")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(output, encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
