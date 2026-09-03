#!/usr/bin/env python3
"""Compare CURVE-C7 ASTR metrics with the analytic nonorthogonal mapping."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def read_metrics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        jacobian = np.asarray(handle["jacob"], dtype=np.float64).transpose(2, 1, 0)
        inverse = np.empty(jacobian.shape + (3, 3), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                inverse[..., i, j] = np.asarray(
                    handle[f"dxi{i + 1}{j + 1}"], dtype=np.float64
                ).transpose(2, 1, 0)
    return jacobian, inverse


def analytic_metrics(
    grid: tuple[int, int, int],
    x_min: float,
    x_max: float,
    y_stretch: float,
    z_length: float,
    warp_x: float,
    warp_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    im, jm, km = grid
    s = np.linspace(0.0, 1.0, im + 1)[:, None, None]
    eta = np.linspace(0.0, 1.0, jm + 1)[None, :, None]
    y = np.expm1(y_stretch * eta) / np.expm1(y_stretch)
    dy_deta = y_stretch * np.exp(y_stretch * eta) / np.expm1(y_stretch)
    shape = y * (1.0 - y)
    shape_eta = (1.0 - 2.0 * y) * dy_deta
    dx_di = (x_max - x_min + warp_x * np.pi * np.cos(np.pi * s) * shape) / im
    dx_dj = warp_x * np.sin(np.pi * s) * shape_eta / jm
    dy_di = warp_y * 2.0 * np.pi * np.cos(2.0 * np.pi * s) * shape / im
    dy_dj = dy_deta * (1.0 + warp_y * np.sin(2.0 * np.pi * s) * (1.0 - 2.0 * y)) / jm
    dz_dk = z_length / km
    jacobian_2d = dx_di * dy_dj - dx_dj * dy_di
    jacobian = np.broadcast_to(jacobian_2d * dz_dk, (im + 1, jm + 1, km + 1)).copy()
    inverse = np.zeros(jacobian.shape + (3, 3), dtype=np.float64)
    inverse[..., 0, 0] = np.broadcast_to(dy_dj / jacobian_2d, jacobian.shape)
    inverse[..., 0, 1] = np.broadcast_to(-dx_dj / jacobian_2d, jacobian.shape)
    inverse[..., 1, 0] = np.broadcast_to(-dy_di / jacobian_2d, jacobian.shape)
    inverse[..., 1, 1] = np.broadcast_to(dx_di / jacobian_2d, jacobian.shape)
    inverse[..., 2, 2] = 1.0 / dz_dk
    return jacobian, inverse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--grid", default="64,64,8")
    parser.add_argument("--x-min", type=float, default=-1.0)
    parser.add_argument("--x-max", type=float, default=10.0)
    parser.add_argument("--y-stretch", type=float, default=5.0)
    parser.add_argument("--z-length", type=float, default=0.25)
    parser.add_argument("--warp-x", type=float, required=True)
    parser.add_argument("--warp-y", type=float, required=True)
    parser.add_argument("--interior-trim", type=int, default=3)
    parser.add_argument("--jacobian-atol", type=float, default=1.0e-8)
    parser.add_argument("--inverse-atol", type=float, default=1.0e-4)
    args = parser.parse_args()
    grid = tuple(int(value) for value in args.grid.split(","))
    if len(grid) != 3 or min(grid) < 8:
        raise ValueError("--grid must contain three dimensions of at least eight")

    numerical_jacobian, numerical_inverse = read_metrics(args.input)
    exact_jacobian, exact_inverse = analytic_metrics(
        grid,
        args.x_min,
        args.x_max,
        args.y_stretch,
        args.z_length,
        args.warp_x,
        args.warp_y,
    )
    if numerical_jacobian.shape != exact_jacobian.shape:
        raise ValueError(f"metric shape {numerical_jacobian.shape} does not match {exact_jacobian.shape}")

    if args.interior_trim < 3 or 2 * args.interior_trim >= min(grid[:2]):
        raise ValueError("--interior-trim must retain an x/y interior and cover the sixth-order stencil")
    jacobian_difference = np.abs(numerical_jacobian - exact_jacobian)
    inverse_difference = np.abs(numerical_inverse - exact_inverse)
    jacobian_error = float(np.max(jacobian_difference))
    inverse_error = float(np.max(inverse_difference))
    interior = (
        slice(args.interior_trim, -args.interior_trim),
        slice(args.interior_trim, -args.interior_trim),
        slice(None),
    )
    interior_jacobian_error = float(np.max(jacobian_difference[interior]))
    interior_inverse_error = float(
        np.max(inverse_difference[interior + (slice(None), slice(None))])
    )
    off_diagonal = max(
        float(np.max(np.abs(numerical_inverse[..., 0, 1]))),
        float(np.max(np.abs(numerical_inverse[..., 1, 0]))),
    )
    finite = bool(np.all(np.isfinite(numerical_jacobian)) and np.all(np.isfinite(numerical_inverse)))
    positive = bool(np.min(numerical_jacobian) > 0.0)
    status = (
        finite
        and positive
        and off_diagonal > 0.0
        and interior_jacobian_error <= args.jacobian_atol
        and interior_inverse_error <= args.inverse_atol
    )
    lines = (
        f"status: {'pass' if status else 'fail'}",
        f"finite: {str(finite).lower()}",
        f"jacobian_min: {np.min(numerical_jacobian):.16e}",
        f"jacobian_max: {np.max(numerical_jacobian):.16e}",
        f"full_jacobian_linf: {jacobian_error:.16e}",
        f"full_inverse_linf: {inverse_error:.16e}",
        f"interior_trim: {args.interior_trim}",
        f"interior_jacobian_linf: {interior_jacobian_error:.16e}",
        f"interior_inverse_linf: {interior_inverse_error:.16e}",
        f"off_diagonal_inverse_max: {off_diagonal:.16e}",
        f"jacobian_atol: {args.jacobian_atol:.16e}",
        f"inverse_atol: {args.inverse_atol:.16e}",
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="ascii")
    print("\n".join(lines))
    return 0 if status else 1


if __name__ == "__main__":
    raise SystemExit(main())
