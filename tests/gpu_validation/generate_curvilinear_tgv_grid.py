#!/usr/bin/env python3
"""Generate a smooth periodic three-dimensional curvilinear TGV grid."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def parse_grid(value: str) -> tuple[int, int, int]:
    try:
        im, jm, km = (int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--grid must have the form im,jm,km") from exc
    if min(im, jm, km) < 10:
        raise argparse.ArgumentTypeError("each grid dimension must be at least 10")
    return im, jm, km


def mapped_grid(
    im: int,
    jm: int,
    km: int,
    amplitude: float,
    mapping: str = "periodic",
    *,
    ramp_start_fraction: float = 0.72,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xi = np.linspace(0.0, 2.0 * np.pi, im + 1)
    eta = np.linspace(0.0, 2.0 * np.pi, jm + 1)
    zeta = np.linspace(0.0, 2.0 * np.pi, km + 1)
    xi3, eta3, zeta3 = np.meshgrid(xi, eta, zeta, indexing="ij")

    derivatives = np.empty(xi3.shape + (3, 3), dtype=np.float64)
    derivatives.fill(0.0)
    if mapping == "periodic":
        x = xi3 + amplitude * np.sin(xi3) * np.sin(eta3)
        y = eta3 + amplitude * np.sin(eta3) * np.sin(zeta3)
        z = zeta3 + amplitude * np.sin(zeta3) * np.sin(xi3)
        derivatives[..., 0, 0] = 1.0 + amplitude * np.cos(xi3) * np.sin(eta3)
        derivatives[..., 0, 1] = amplitude * np.sin(xi3) * np.cos(eta3)
        derivatives[..., 1, 1] = 1.0 + amplitude * np.cos(eta3) * np.sin(zeta3)
        derivatives[..., 1, 2] = amplitude * np.sin(eta3) * np.cos(zeta3)
        derivatives[..., 2, 0] = amplitude * np.sin(zeta3) * np.cos(xi3)
        derivatives[..., 2, 2] = 1.0 + amplitude * np.cos(zeta3) * np.sin(xi3)
    elif mapping == "x-wavy":
        x = xi3 + amplitude * np.sin(eta3) * np.sin(zeta3)
        y = eta3
        z = zeta3
        derivatives[..., 0, 0] = 1.0
        derivatives[..., 0, 1] = amplitude * np.cos(eta3) * np.sin(zeta3)
        derivatives[..., 0, 2] = amplitude * np.sin(eta3) * np.cos(zeta3)
        derivatives[..., 1, 1] = 1.0
        derivatives[..., 2, 2] = 1.0
    elif mapping == "y-wavy":
        x = xi3
        y = eta3 + amplitude * np.sin(xi3) * np.sin(zeta3)
        z = zeta3
        derivatives[..., 0, 0] = 1.0
        derivatives[..., 1, 0] = amplitude * np.cos(xi3) * np.sin(zeta3)
        derivatives[..., 1, 1] = 1.0
        derivatives[..., 1, 2] = amplitude * np.sin(xi3) * np.cos(zeta3)
        derivatives[..., 2, 2] = 1.0
    elif mapping == "y-upper-ramp":
        if not 0.0 < ramp_start_fraction < 1.0:
            raise ValueError("ramp_start_fraction must lie in (0, 1)")
        eta_fraction = eta3 / (2.0 * np.pi)
        tau = np.clip(
            (eta_fraction - ramp_start_fraction) / (1.0 - ramp_start_fraction),
            0.0,
            1.0,
        )
        ramp = tau**3 * (10.0 - 15.0 * tau + 6.0 * tau**2)
        dramp_deta = (
            30.0 * tau**2 * (1.0 - tau) ** 2
            / (2.0 * np.pi * (1.0 - ramp_start_fraction))
        )
        x = xi3
        y = eta3 + amplitude * ramp * np.sin(xi3) * np.sin(zeta3)
        z = zeta3
        derivatives[..., 0, 0] = 1.0
        derivatives[..., 1, 0] = amplitude * ramp * np.cos(xi3) * np.sin(zeta3)
        derivatives[..., 1, 1] = (
            1.0 + amplitude * dramp_deta * np.sin(xi3) * np.sin(zeta3)
        )
        derivatives[..., 1, 2] = amplitude * ramp * np.sin(xi3) * np.cos(zeta3)
        derivatives[..., 2, 2] = 1.0
    elif mapping == "z-wavy":
        x = xi3
        y = eta3
        z = zeta3 + amplitude * np.sin(xi3) * np.sin(eta3)
        derivatives[..., 0, 0] = 1.0
        derivatives[..., 1, 1] = 1.0
        derivatives[..., 2, 0] = amplitude * np.cos(xi3) * np.sin(eta3)
        derivatives[..., 2, 1] = amplitude * np.sin(xi3) * np.cos(eta3)
        derivatives[..., 2, 2] = 1.0
    else:
        raise ValueError(f"unsupported mapping: {mapping}")
    return x, y, z, np.linalg.det(derivatives)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--grid", type=parse_grid, default=(32, 32, 32))
    parser.add_argument("--amplitude", type=float, default=0.15)
    parser.add_argument(
        "--mapping",
        choices=("periodic", "x-wavy", "y-wavy", "y-upper-ramp", "z-wavy"),
        default="periodic",
    )
    parser.add_argument("--ramp-start-fraction", type=float, default=0.72)
    args = parser.parse_args()

    im, jm, km = args.grid
    if not 0.0 < args.amplitude < 0.5:
        raise ValueError("--amplitude must lie in (0, 0.5)")

    x, y, z, jacobian = mapped_grid(
        im,
        jm,
        km,
        args.amplitude,
        args.mapping,
        ramp_start_fraction=args.ramp_start_fraction,
    )
    min_jacobian = float(np.min(jacobian))
    max_jacobian = float(np.max(jacobian))
    if min_jacobian <= 0.0:
        raise RuntimeError(f"mapping is inverted: min analytic Jacobian={min_jacobian:.16e}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as handle:
        handle.create_dataset("x", data=np.asfortranarray(x).transpose(2, 1, 0))
        handle.create_dataset("y", data=np.asfortranarray(y).transpose(2, 1, 0))
        handle.create_dataset("z", data=np.asfortranarray(z).transpose(2, 1, 0))

    if args.mapping == "periodic":
        mapping_description = (
            "x=xi+a*sin(xi)*sin(eta), y=eta+a*sin(eta)*sin(zeta), "
            "z=zeta+a*sin(zeta)*sin(xi)"
        )
    elif args.mapping == "x-wavy":
        mapping_description = "x=xi+a*sin(eta)*sin(zeta), y=eta, z=zeta"
    elif args.mapping == "y-wavy":
        mapping_description = "x=xi, y=eta+a*sin(xi)*sin(zeta), z=zeta"
    elif args.mapping == "y-upper-ramp":
        mapping_description = (
            "x=xi, y=eta+a*s(eta)*sin(xi)*sin(zeta), z=zeta, "
            f"s=quintic-ramp-from-{args.ramp_start_fraction:.8f}"
        )
    else:
        mapping_description = "x=xi, y=eta, z=zeta+a*sin(xi)*sin(eta)"
    cross_derivative = args.amplitude
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        "\n".join(
            (
                f"mapping: {mapping_description}",
                f"grid: {im},{jm},{km}",
                f"amplitude: {args.amplitude:.16e}",
                f"analytic_jacobian_min: {min_jacobian:.16e}",
                f"analytic_jacobian_max: {max_jacobian:.16e}",
                f"max_cross_derivative: {cross_derivative:.16e}",
            )
        )
        + "\n"
    )
    print(args.report.read_text(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
