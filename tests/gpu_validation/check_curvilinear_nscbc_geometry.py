#!/usr/bin/env python3
"""Validate the analytic y-wavy grid used by the curved NSCBC gate."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def read_grid(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        missing = [name for name in ("x", "y", "z") if name not in handle]
        if missing:
            raise KeyError(f"{path}: missing datasets {missing}")
        return tuple(np.asarray(handle[name][...]).transpose(2, 1, 0) for name in ("x", "y", "z"))


def geometry_metrics(
    path: Path,
    amplitude: float,
    *,
    mapping: str = "y-wavy",
    ramp_start_fraction: float = 0.72,
) -> dict[str, float]:
    x, y, z = read_grid(path)
    if x.shape != y.shape or x.shape != z.shape or min(x.shape) < 2:
        raise ValueError("grid coordinate arrays must share a three-dimensional shape")
    if not all(np.all(np.isfinite(field)) for field in (x, y, z)):
        raise ValueError("grid contains non-finite coordinates")

    im, jm, km = (extent - 1 for extent in x.shape)
    xi = np.linspace(0.0, 2.0 * np.pi, im + 1)[:, None, None]
    eta = np.linspace(0.0, 2.0 * np.pi, jm + 1)[None, :, None]
    zeta = np.linspace(0.0, 2.0 * np.pi, km + 1)[None, None, :]
    expected_x = np.broadcast_to(xi, x.shape)
    expected_z = np.broadcast_to(zeta, z.shape)
    if mapping == "y-wavy":
        ramp = np.ones_like(eta)
        dramp_deta = np.zeros_like(eta)
    elif mapping == "y-upper-ramp":
        if not 0.0 < ramp_start_fraction < 1.0:
            raise ValueError("ramp_start_fraction must lie in (0, 1)")
        tau = np.clip(
            (eta / (2.0 * np.pi) - ramp_start_fraction)
            / (1.0 - ramp_start_fraction),
            0.0,
            1.0,
        )
        ramp = tau**3 * (10.0 - 15.0 * tau + 6.0 * tau**2)
        dramp_deta = (
            30.0 * tau**2 * (1.0 - tau) ** 2
            / (2.0 * np.pi * (1.0 - ramp_start_fraction))
        )
    else:
        raise ValueError(f"unsupported NSCBC geometry mapping: {mapping}")
    expected_y = eta + amplitude * ramp * np.sin(xi) * np.sin(zeta)
    mapping_error = max(
        float(np.max(np.abs(x - expected_x))),
        float(np.max(np.abs(y - expected_y))),
        float(np.max(np.abs(z - expected_z))),
    )
    if mapping_error > 1.0e-12:
        raise ValueError(f"grid is not the requested {mapping} mapping: error={mapping_error:.16e}")

    jacobian = 1.0 + amplitude * dramp_deta * np.sin(xi) * np.sin(zeta)
    upper_x = x[:, -1, :]
    upper_z = z[:, -1, :]
    upper_ramp = float(ramp[0, -1, 0])
    upper_dramp = float(dramp_deta[0, -1, 0])
    upper_dy_deta = (
        1.0
        + amplitude
        * upper_dramp
        * np.sin(upper_x)
        * np.sin(upper_z)
    )
    grad_eta = np.empty(upper_x.shape + (3,), dtype=np.float64)
    grad_eta[..., 0] = (
        -amplitude * upper_ramp * np.cos(upper_x) * np.sin(upper_z)
        / upper_dy_deta
    )
    grad_eta[..., 1] = 1.0 / upper_dy_deta
    grad_eta[..., 2] = (
        -amplitude * upper_ramp * np.sin(upper_x) * np.cos(upper_z)
        / upper_dy_deta
    )
    upper = np.stack((x[:, -1, :], y[:, -1, :], z[:, -1, :]), axis=-1)
    interior = np.stack((x[:, -2, :], y[:, -2, :], z[:, -2, :]), axis=-1)
    metric_norm = np.linalg.norm(grad_eta, axis=-1)
    orientation = np.sum(grad_eta * (upper - interior), axis=-1)

    metrics = {
        "mapping_error_max": mapping_error,
        "analytic_jacobian_min": float(np.min(jacobian)),
        "analytic_jacobian_max": float(np.max(jacobian)),
        "upper_metric_norm_min": float(np.min(metric_norm)),
        "upper_metric_norm_max": float(np.max(metric_norm)),
        "upper_orientation_min": float(np.min(orientation)),
        "upper_orientation_max": float(np.max(orientation)),
    }
    if metrics["analytic_jacobian_min"] <= 0.0:
        raise ValueError("analytic Jacobian is not positive")
    if metrics["upper_metric_norm_min"] <= 0.0:
        raise ValueError("upper eta metric has zero norm")
    if metrics["upper_orientation_min"] <= 0.0:
        raise ValueError("upper eta metric points into the computational domain")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--amplitude", type=float, default=0.15)
    parser.add_argument(
        "--mapping",
        choices=("y-wavy", "y-upper-ramp"),
        default="y-wavy",
    )
    parser.add_argument("--ramp-start-fraction", type=float, default=0.72)
    args = parser.parse_args()
    if not 0.0 < args.amplitude < 0.5:
        raise ValueError("--amplitude must lie in (0, 0.5)")
    metrics = geometry_metrics(
        args.grid,
        args.amplitude,
        mapping=args.mapping,
        ramp_start_fraction=args.ramp_start_fraction,
    )
    lines = ["geometry: PASS", *(f"{key}: {value:.16e}" for key, value in metrics.items())]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="ascii")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
