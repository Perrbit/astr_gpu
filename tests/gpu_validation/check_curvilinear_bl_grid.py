#!/usr/bin/env python3
"""Check the controlled CURVE-C7 mapping independently of ASTR metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--x-min", type=float, default=-1.0)
    parser.add_argument("--x-max", type=float, default=10.0)
    parser.add_argument("--y-stretch", type=float, default=5.0)
    parser.add_argument("--z-length", type=float, default=0.25)
    parser.add_argument("--warp-x", type=float, required=True)
    parser.add_argument("--warp-y", type=float, required=True)
    parser.add_argument("--boundary-atol", type=float, default=1.0e-14)
    args = parser.parse_args()

    with h5py.File(args.grid, "r") as handle:
        x = np.asarray(handle["x"], dtype=np.float64).transpose(2, 1, 0)
        y = np.asarray(handle["y"], dtype=np.float64).transpose(2, 1, 0)
        z = np.asarray(handle["z"], dtype=np.float64).transpose(2, 1, 0)

    im, jm, km = (size - 1 for size in x.shape)
    s = np.linspace(0.0, 1.0, im + 1)[:, None, None]
    eta = np.linspace(0.0, 1.0, jm + 1)[None, :, None]
    xline = args.x_min + (args.x_max - args.x_min) * s
    yline = np.expm1(args.y_stretch * eta) / np.expm1(args.y_stretch)
    zline = np.linspace(0.0, args.z_length, km + 1)[None, None, :]

    boundary_error = max(
        float(np.max(np.abs(x[0, :, :] - xline[0, :, :]))),
        float(np.max(np.abs(x[-1, :, :] - xline[-1, :, :]))),
        float(np.max(np.abs(x[:, 0, :] - xline[:, 0, :]))),
        float(np.max(np.abs(x[:, -1, :] - xline[:, 0, :]))),
        float(np.max(np.abs(y[0, :, :] - yline[0, :, :]))),
        float(np.max(np.abs(y[-1, :, :] - yline[0, :, :]))),
        float(np.max(np.abs(y[:, 0, :] - yline[:, 0, :]))),
        float(np.max(np.abs(y[:, -1, :] - yline[:, -1, :]))),
        float(np.max(np.abs(z - zline))),
    )

    y_base = yline
    dy_deta = args.y_stretch * np.exp(args.y_stretch * eta) / np.expm1(args.y_stretch)
    wall_shape = y_base * (1.0 - y_base)
    wall_shape_eta = (1.0 - 2.0 * y_base) * dy_deta
    dx_di = (
        args.x_max - args.x_min
        + args.warp_x * np.pi * np.cos(np.pi * s) * wall_shape
    ) / im
    dx_dj = args.warp_x * np.sin(np.pi * s) * wall_shape_eta / jm
    dy_di = args.warp_y * 2.0 * np.pi * np.cos(2.0 * np.pi * s) * wall_shape / im
    dy_dj = dy_deta * (
        1.0 + args.warp_y * np.sin(2.0 * np.pi * s) * (1.0 - 2.0 * y_base)
    ) / jm
    dz_dk = args.z_length / km
    jacobian = (dx_di * dy_dj - dx_dj * dy_di) * dz_dk
    max_cross = max(float(np.max(np.abs(dx_dj))), float(np.max(np.abs(dy_di))))

    finite = bool(np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and np.all(np.isfinite(z)))
    positive = bool(np.all(jacobian > 0.0))
    status = finite and positive and boundary_error <= args.boundary_atol and max_cross > 0.0
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        "\n".join(
            (
                f"finite: {str(finite).lower()}",
                f"analytic_jacobian_min: {np.min(jacobian):.16e}",
                f"analytic_jacobian_max: {np.max(jacobian):.16e}",
                f"max_cross_metric_derivative: {max_cross:.16e}",
                f"physical_boundary_coordinate_error: {boundary_error:.16e}",
                f"status: {'pass' if status else 'fail'}",
            )
        )
        + "\n",
        encoding="ascii",
    )
    if not status:
        raise SystemExit("CURVE-C7 grid check failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
