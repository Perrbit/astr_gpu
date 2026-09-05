#!/usr/bin/env python3
"""Check C19 curved y-wall slip and wall-blowing invariants."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def flowfield_path(path: Path) -> Path:
    return path / "outdat" / "flowfield.h5" if path.is_dir() else path


def read_velocity(path: Path) -> np.ndarray:
    with h5py.File(flowfield_path(path), "r") as handle:
        return np.stack(
            [np.asarray(handle[name], dtype=np.float64) for name in ("u1", "u2", "u3")],
            axis=-1,
        )


def read_y_normals(path: Path, side: int) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        metric = np.stack(
            [
                np.asarray(handle[f"dxi2{component}"], dtype=np.float64).transpose(2, 1, 0)
                for component in range(1, 4)
            ],
            axis=-1,
        )
    face = np.take(metric, 0 if side == 0 else metric.shape[1] - 1, axis=1)
    face = np.swapaxes(face, 0, 1)
    normal = face / np.linalg.norm(face, axis=-1, keepdims=True)
    return normal if side == 0 else -normal


def read_lower_coordinates(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        xcoord = np.asarray(handle["x"], dtype=np.float64)[:, 0, :]
        zcoord = np.asarray(handle["z"], dtype=np.float64)[:, 0, :]
    return xcoord, zcoord


def wall_blowing(
    xcoord: np.ndarray,
    zcoord: np.ndarray,
    amplitude: float,
    uinf: float,
    xa: float,
    xb: float,
    xc: float,
    nmod_z: int,
) -> np.ndarray:
    result = np.zeros_like(xcoord)
    nk, ni = result.shape
    gi = np.broadcast_to(np.arange(ni, dtype=np.int64), (nk, ni))
    gk = np.broadcast_to(np.arange(nk, dtype=np.int64)[:, None], (nk, ni))
    hash_value = 104729 * (gi + 1) + 130363 * (gk + 1) + 433494437
    hash_value -= (hash_value // 2147483629) * 2147483629
    fluctuation = 0.1 * (2.0 * hash_value.astype(np.float64) / 2147483629.0 - 1.0)
    length_z = float(np.max(zcoord) - np.min(zcoord))

    first = (xcoord >= xa) & (xcoord <= xb)
    theta = 2.0 * np.pi * (xcoord[first] - xa) / (xb - xa)
    fx = 4.0 * np.sin(theta) * (1.0 - np.cos(theta)) / np.sqrt(27.0)
    gz = np.sin(2.0 * nmod_z * np.pi * zcoord[first] / length_z)
    result[first] = amplitude * uinf * fx * gz * (1.0 + fluctuation[first])

    second = (xcoord >= xb) & (xcoord <= xc)
    theta = 2.0 * np.pi * (xcoord[second] - xb) / (xc - xb)
    fx = 4.0 * np.sin(theta) * (1.0 - np.cos(theta)) / np.sqrt(27.0)
    gz = np.sin(2.0 * nmod_z * np.pi * zcoord[second] / length_z + 0.5 * np.pi)
    result[second] = amplitude * uinf * fx * gz * (1.0 + fluctuation[second])
    return result


def max_abs(array: np.ndarray) -> float:
    return float(np.max(np.abs(array))) if array.size else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--geometry", required=True, type=Path)
    parser.add_argument("--grid", required=True, type=Path)
    parser.add_argument("--kind", required=True, type=int, choices=(41, 42, 411, 421))
    parser.add_argument("--xslip", type=float, default=np.pi)
    parser.add_argument("--wall-amplitude", type=float, default=0.0)
    parser.add_argument("--uinf", type=float, default=1.0)
    parser.add_argument("--xa", type=float, default=0.5)
    parser.add_argument("--xb", type=float, default=3.0)
    parser.add_argument("--xc", type=float, default=5.8)
    parser.add_argument("--nmod-z", type=int, default=2)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    parser.add_argument("--min-slip-speed", type=float, default=1.0e-4)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    velocity = read_velocity(args.input)
    if not np.all(np.isfinite(velocity)):
        raise ValueError("wall velocity contains non-finite values")
    lower_velocity = velocity[:, 0, :, :]
    upper_velocity = velocity[:, -1, :, :]
    lower_normal = read_y_normals(args.geometry, 0)
    upper_normal = read_y_normals(args.geometry, 1)
    xcoord, zcoord = read_lower_coordinates(args.grid)
    blowing = wall_blowing(
        xcoord,
        zcoord,
        args.wall_amplitude,
        args.uinf,
        args.xa,
        args.xb,
        args.xc,
        args.nmod_z,
    )

    lower_normal_velocity = np.sum(lower_velocity * lower_normal, axis=-1)
    upper_normal_velocity = np.sum(upper_velocity * upper_normal, axis=-1)
    lower_tangent = lower_velocity - lower_normal_velocity[..., None] * lower_normal
    upper_tangent = upper_velocity - upper_normal_velocity[..., None] * upper_normal
    lower_target = blowing.copy()
    if args.kind == 411:
        lower_target[xcoord <= args.xslip] = 0.0

    # x and z are periodic in this slice. ASTR stores duplicate endpoint planes
    # whose post-boundary halo refresh does not retain a unique forcing index.
    unique_periodic = np.s_[1:-1, :-1]
    errors: dict[str, float] = {
        "lower_normal_target": max_abs(
            (lower_normal_velocity - lower_target)[unique_periodic]
        ),
        "upper_normal_zero": max_abs(upper_normal_velocity),
    }
    if args.kind in (41, 42):
        errors["lower_tangent_zero"] = max_abs(lower_tangent)
        errors["upper_velocity_zero"] = max_abs(upper_velocity)
    elif args.kind == 411:
        nonslip = xcoord > args.xslip
        # The last i plane is the duplicate endpoint of the periodic x axis.
        # Halo refresh copies the i=0 slip state there after wall application.
        nonslip[:, -1] = False
        slip = ~nonslip
        errors["lower_nonslip_tangent_zero"] = max_abs(lower_tangent[nonslip])
        errors["upper_velocity_zero"] = max_abs(upper_velocity)
        slip_speed = max_abs(lower_tangent[slip])
        errors["lower_slip_speed"] = slip_speed
    else:
        slip_speed = min(max_abs(lower_tangent), max_abs(upper_tangent))
        errors["minimum_face_slip_speed"] = slip_speed

    failures = [
        name
        for name, value in errors.items()
        if name not in ("lower_slip_speed", "minimum_face_slip_speed")
        and (not np.isfinite(value) or value > args.atol)
    ]
    for name in ("lower_slip_speed", "minimum_face_slip_speed"):
        if name in errors and errors[name] < args.min_slip_speed:
            failures.append(name)
    if abs(args.wall_amplitude) > 0.0 and max_abs(blowing) <= 1.0e-8:
        failures.append("wall_blowing_excitation")

    lines = [
        f"status: {'fail' if failures else 'pass'}",
        f"kind: {args.kind}",
        f"atol: {args.atol:.16e}",
        f"expected_blowing_linf: {max_abs(blowing):.16e}",
    ]
    lines.extend(f"{name}: {value:.16e}" for name, value in errors.items())
    if failures:
        lines.append("failures: " + ",".join(failures))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="ascii")
    print("\n".join(lines))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
