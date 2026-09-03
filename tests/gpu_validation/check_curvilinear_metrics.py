#!/usr/bin/env python3
"""Compare ASTR curvilinear metrics with the analytic periodic mapping."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def computational_coordinates(grid: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    im, jm, km = grid
    xi = np.linspace(0.0, 2.0 * np.pi, im + 1)
    eta = np.linspace(0.0, 2.0 * np.pi, jm + 1)
    zeta = np.linspace(0.0, 2.0 * np.pi, km + 1)
    return np.meshgrid(xi, eta, zeta, indexing="ij")


def mapping_derivatives(grid: tuple[int, int, int], amplitude: float) -> np.ndarray:
    xi, eta, zeta = computational_coordinates(grid)
    derivatives = np.zeros(xi.shape + (3, 3), dtype=np.float64)
    derivatives[..., 0, 0] = 1.0 + amplitude * np.cos(xi) * np.sin(eta)
    derivatives[..., 0, 1] = amplitude * np.sin(xi) * np.cos(eta)
    derivatives[..., 1, 1] = 1.0 + amplitude * np.cos(eta) * np.sin(zeta)
    derivatives[..., 1, 2] = amplitude * np.sin(eta) * np.cos(zeta)
    derivatives[..., 2, 0] = amplitude * np.sin(zeta) * np.cos(xi)
    derivatives[..., 2, 2] = 1.0 + amplitude * np.cos(zeta) * np.sin(xi)
    return derivatives


def index_mapping_derivatives(grid: tuple[int, int, int], amplitude: float) -> np.ndarray:
    """Return physical-coordinate derivatives with respect to ASTR's integer indices."""
    derivatives = mapping_derivatives(grid, amplitude)
    spacing = 2.0 * np.pi / np.asarray(grid, dtype=np.float64)
    return derivatives * spacing


def analytic_metrics(grid: tuple[int, int, int], amplitude: float) -> tuple[np.ndarray, np.ndarray]:
    derivatives = index_mapping_derivatives(grid, amplitude)
    jacobian = np.linalg.det(derivatives)
    inverse = np.linalg.inv(derivatives)
    return jacobian, inverse


def periodic_derivative(field: np.ndarray, axis: int) -> np.ndarray:
    return (
        -np.roll(field, 3, axis=axis)
        + 9.0 * np.roll(field, 2, axis=axis)
        - 45.0 * np.roll(field, 1, axis=axis)
        + 45.0 * np.roll(field, -1, axis=axis)
        - 9.0 * np.roll(field, -2, axis=axis)
        + np.roll(field, -3, axis=axis)
    ) / 60.0


def metric_identity_residual(jacobian: np.ndarray, inverse: np.ndarray) -> np.ndarray:
    unique = tuple(slice(0, size - 1) for size in jacobian.shape)
    jacobian_unique = jacobian[unique]
    inverse_unique = inverse[unique]
    residual = np.zeros(jacobian_unique.shape + (3,), dtype=np.float64)
    for physical_direction in range(3):
        for computational_direction in range(3):
            cofactor = jacobian_unique * inverse_unique[..., computational_direction, physical_direction]
            residual[..., physical_direction] += periodic_derivative(cofactor, computational_direction)
    return residual


def parse_grid(value: str) -> tuple[int, int, int]:
    try:
        grid = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--grid must have the form im,jm,km") from exc
    if len(grid) != 3 or min(grid) < 8:
        raise argparse.ArgumentTypeError("all grid dimensions must be at least eight")
    return grid


def read_astr_metrics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        names = ["jacob", *(f"dxi{i}{j}" for i in range(1, 4) for j in range(1, 4))]
        missing = [name for name in names if name not in handle]
        if missing:
            raise KeyError(f"{path}: missing datasets {missing}")
        jacobian = handle["jacob"][...].transpose(2, 1, 0)
        inverse = np.empty(jacobian.shape + (3, 3), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                inverse[..., i, j] = handle[f"dxi{i + 1}{j + 1}"][...].transpose(2, 1, 0)
    return jacobian, inverse


def norm_line(name: str, numerical: np.ndarray, exact: np.ndarray) -> tuple[str, float]:
    difference = numerical - exact
    linf = float(np.max(np.abs(difference)))
    l2 = float(np.sqrt(np.mean(difference * difference)))
    return f"{name} linf={linf:.16e} l2={l2:.16e}", linf


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--grid", required=True, type=parse_grid)
    parser.add_argument("--amplitude", type=float, default=0.15)
    parser.add_argument("--identity-atol", type=float, default=1.0e-10)
    args = parser.parse_args()

    numerical_jacobian, numerical_inverse = read_astr_metrics(args.input)
    exact_jacobian, exact_inverse = analytic_metrics(args.grid, args.amplitude)
    if numerical_jacobian.shape != exact_jacobian.shape:
        raise ValueError(
            f"metric shape {numerical_jacobian.shape} does not match grid {exact_jacobian.shape}"
        )

    lines = ["status: pass"]
    status = 0
    line, jacobian_error = norm_line("jacob", numerical_jacobian, exact_jacobian)
    lines.append(line)
    inverse_errors: list[float] = []
    for i in range(3):
        for j in range(3):
            line, error = norm_line(
                f"dxi{i + 1}{j + 1}", numerical_inverse[..., i, j], exact_inverse[..., i, j]
            )
            lines.append(line)
            inverse_errors.append(error)

    identity = metric_identity_residual(numerical_jacobian, numerical_inverse)
    identity_max = 0.0
    for direction in range(3):
        value = float(np.max(np.abs(identity[..., direction])))
        identity_max = max(identity_max, value)
        lines.append(f"metric_identity_{direction + 1} linf={value:.16e}")

    minimum_jacobian = float(np.min(numerical_jacobian))
    maximum_jacobian = float(np.max(numerical_jacobian))
    lines.extend(
        (
            f"jacob_min={minimum_jacobian:.16e}",
            f"jacob_max={maximum_jacobian:.16e}",
            f"jacob_error_max={jacobian_error:.16e}",
            f"dxi_error_max={max(inverse_errors):.16e}",
            f"metric_identity_max={identity_max:.16e}",
        )
    )
    if (
        minimum_jacobian <= 0.0
        or not np.all(np.isfinite(numerical_jacobian))
        or not np.all(np.isfinite(numerical_inverse))
        or not np.all(np.isfinite(identity))
        or identity_max > args.identity_atol
    ):
        status = 1
        lines[0] = "status: fail"

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
