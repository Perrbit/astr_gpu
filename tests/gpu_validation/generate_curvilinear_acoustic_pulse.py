#!/usr/bin/env python3
"""Generate a weak upward acoustic packet for the curved NSCBC gate."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def parse_vector(value: str) -> tuple[float, float, float]:
    try:
        result = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected three comma-separated values") from exc
    if len(result) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated values")
    return result


def read_coordinates(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        missing = [name for name in ("x", "y", "z") if name not in handle]
        if missing:
            raise ValueError(f"grid is missing datasets: {', '.join(missing)}")
        coordinates = tuple(np.asarray(handle[name], dtype=np.float64) for name in ("x", "y", "z"))
    if len({array.shape for array in coordinates}) != 1:
        raise ValueError("grid coordinate shapes do not match")
    if not all(np.all(np.isfinite(array)) for array in coordinates):
        raise ValueError("grid coordinates must be finite")
    return coordinates


def compact_support_intervals(
    y: np.ndarray,
    center_y: float,
    width: float,
) -> float:
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 3 or not np.all(np.isfinite(y)):
        raise ValueError("y coordinates must be a finite three-dimensional array")
    if not np.isfinite(center_y) or width <= 0.0:
        raise ValueError("compact support center and width are invalid")
    spacing = np.abs(np.diff(y, axis=1))
    midpoint = 0.5 * (y[:, 1:, :] + y[:, :-1, :])
    support = np.abs(midpoint - center_y) < width
    if not np.any(support) or np.any(spacing[support] <= 0.0):
        raise ValueError("compact support has invalid local y spacing")
    return float(2.0 * width / np.max(spacing[support]))


def build_acoustic_state(
    coordinates: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    rho0: float,
    p0: float,
    velocity0: tuple[float, float, float],
    gamma: float,
    mach: float,
    amplitude: float,
    center: tuple[float, float, float],
    width: float,
    direction: tuple[float, float, float],
    clearance_widths: float = 3.0,
    profile: str = "spherical",
) -> dict[str, np.ndarray]:
    x, y, z = coordinates
    if x.shape != y.shape or x.shape != z.shape:
        raise ValueError("grid coordinate shapes do not match")
    if not all(np.all(np.isfinite(array)) for array in coordinates):
        raise ValueError("grid coordinates must be finite")
    if rho0 <= 0.0 or p0 <= 0.0:
        raise ValueError("base density and pressure must be positive")
    if gamma <= 1.0 or mach <= 0.0:
        raise ValueError("gamma must exceed one and Mach must be positive")
    if amplitude <= 0.0 or amplitude / p0 > 1.0e-3:
        raise ValueError("acoustic amplitude must satisfy 0 < amplitude/p0 <= 1e-3")
    if width <= 0.0 or clearance_widths < 0.0:
        raise ValueError("width and clearance constraint are invalid")

    direction_array = np.asarray(direction, dtype=np.float64)
    direction_norm = np.linalg.norm(direction_array)
    if not np.isfinite(direction_norm) or direction_norm <= 0.0:
        raise ValueError("acoustic direction must be finite and nonzero")
    direction_array /= direction_norm
    if direction_array[1] <= 0.0:
        raise ValueError("acoustic direction must have a positive upward component")
    if profile not in ("spherical", "plane-y", "plane-y-compact"):
        raise ValueError(f"unsupported acoustic profile: {profile}")
    if profile in ("plane-y", "plane-y-compact") and not np.allclose(
        direction_array,
        np.array((0.0, 1.0, 0.0)),
        rtol=0.0,
        atol=16.0 * np.finfo(np.float64).eps,
    ):
        raise ValueError("plane-y acoustic profile requires the positive y direction")

    center_array = np.asarray(center, dtype=np.float64)
    if center_array.shape != (3,) or not np.all(np.isfinite(center_array)):
        raise ValueError("acoustic center must contain three finite values")
    clearance = clearance_widths * width
    physical_coordinates = (x, y) if profile == "spherical" else (y,)
    center_components = center_array[:2] if profile == "spherical" else center_array[1:2]
    for center_component, coordinate in zip(center_components, physical_coordinates):
        lower = float(np.min(coordinate))
        upper = float(np.max(coordinate))
        if center_component - lower < clearance or upper - center_component < clearance:
            raise ValueError("acoustic packet is too close to a physical boundary")

    if profile == "spherical":
        z_min = float(np.min(z))
        z_max = float(np.max(z))
        z_period = z_max - z_min
        if z_period <= 0.0:
            raise ValueError("periodic z extent must be positive")
        dz = np.abs(z - center_array[2])
        dz = np.minimum(dz, z_period - np.minimum(dz, z_period))
        radius2 = (x - center_array[0]) ** 2 + (y - center_array[1]) ** 2 + dz**2
    else:
        radius2 = (y - center_array[1]) ** 2

    if profile == "plane-y-compact":
        normalized_distance=np.sqrt(radius2)/width
        p_prime=np.zeros_like(y)
        support=normalized_distance<1.0
        p_prime[support]=amplitude*np.cos(0.5*np.pi*normalized_distance[support])**4
    else:
        p_prime = amplitude * np.exp(-radius2 / width**2)
    c0 = np.sqrt(gamma * p0 / rho0)
    rho = rho0 + p_prime / c0**2
    pressure = p0 + p_prime
    velocity_scale = p_prime / (rho0 * c0)
    velocity = [
        velocity0[axis] + velocity_scale * direction_array[axis]
        for axis in range(3)
    ]
    temperature = gamma * mach**2 * pressure / rho

    fields = {
        "ro": rho,
        "u1": velocity[0],
        "u2": velocity[1],
        "u3": velocity[2],
        "p": pressure,
        "t": temperature,
    }
    if not all(np.all(np.isfinite(field)) for field in fields.values()):
        raise ValueError("acoustic state contains non-finite values")
    if np.min(rho) <= 0.0 or np.min(pressure) <= 0.0 or np.min(temperature) <= 0.0:
        raise ValueError("acoustic state contains a non-positive thermodynamic value")
    return fields


def write_initial_field(
    output: Path,
    fields: dict[str, np.ndarray],
    metadata: dict[str, object],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as handle:
        for name in ("ro", "u1", "u2", "u3", "t"):
            handle.create_dataset(name, data=np.asarray(fields[name], dtype=np.float64))
        for name, value in metadata.items():
            handle.attrs[name] = value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rho0", type=float, default=1.0)
    parser.add_argument("--p0", required=True, type=float)
    parser.add_argument("--velocity0", type=parse_vector, default=(0.0, 0.0, 0.0))
    parser.add_argument("--gamma", type=float, default=1.4)
    parser.add_argument("--mach", type=float, default=0.3)
    parser.add_argument("--amplitude", type=float, default=1.0e-4)
    parser.add_argument("--center", required=True, type=parse_vector)
    parser.add_argument("--width", required=True, type=float)
    parser.add_argument("--direction", type=parse_vector, default=(0.0, 1.0, 0.0))
    parser.add_argument("--clearance-widths", type=float, default=3.0)
    parser.add_argument(
        "--profile",
        choices=("spherical", "plane-y", "plane-y-compact"),
        default="spherical",
    )
    parser.add_argument("--minimum-support-intervals", type=float, default=0.0)
    args = parser.parse_args()

    coordinates = read_coordinates(args.grid)
    if args.minimum_support_intervals < 0.0:
        raise ValueError("minimum support intervals must be non-negative")
    if args.profile == "plane-y-compact" and args.minimum_support_intervals > 0.0:
        support_intervals = compact_support_intervals(
            coordinates[1], args.center[1], args.width
        )
        if support_intervals < args.minimum_support_intervals:
            raise ValueError(
                "compact acoustic support is under-resolved: "
                f"{support_intervals:.6g} < {args.minimum_support_intervals:.6g} intervals"
            )
        print(f"compact_support_intervals: {support_intervals:.16e}")
    fields = build_acoustic_state(
        coordinates,
        rho0=args.rho0,
        p0=args.p0,
        velocity0=args.velocity0,
        gamma=args.gamma,
        mach=args.mach,
        amplitude=args.amplitude,
        center=args.center,
        width=args.width,
        direction=args.direction,
        clearance_widths=args.clearance_widths,
        profile=args.profile,
    )
    c0 = np.sqrt(args.gamma * args.p0 / args.rho0)
    write_initial_field(
        args.output,
        fields,
        {
            "rho0": args.rho0,
            "p0": args.p0,
            "velocity0": args.velocity0,
            "gamma": args.gamma,
            "mach": args.mach,
            "c0": c0,
            "amplitude": args.amplitude,
            "center": args.center,
            "width": args.width,
            "direction": args.direction,
            "profile": args.profile,
        },
    )
    print(f"acoustic_initial_field: {args.output}")
    print(f"amplitude_ratio: {args.amplitude / args.p0:.16e}")
    print(f"sound_speed: {c0:.16e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
