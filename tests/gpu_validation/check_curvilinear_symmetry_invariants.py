#!/usr/bin/env python3
"""Check curvilinear symmetry projection against discrete and analytic normals."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def flowfield_path(path: Path) -> Path:
    return path / "outdat" / "flowfield.h5" if path.is_dir() else path


def analytic_normals(axis: str, shape: tuple[int, int, int], amplitude: float) -> np.ndarray:
    nk, nj, ni = shape
    xi = np.linspace(0.0, 2.0 * np.pi, ni)
    eta = np.linspace(0.0, 2.0 * np.pi, nj)
    zeta = np.linspace(0.0, 2.0 * np.pi, nk)
    if axis == "x":
        eta2, zeta2 = np.meshgrid(eta, zeta, indexing="xy")
        normal = np.stack(
            (
                np.ones_like(eta2),
                -amplitude * np.cos(eta2) * np.sin(zeta2),
                -amplitude * np.sin(eta2) * np.cos(zeta2),
            ),
            axis=-1,
        )
    elif axis == "y":
        xi2, zeta2 = np.meshgrid(xi, zeta, indexing="xy")
        normal = np.stack(
            (
                -amplitude * np.cos(xi2) * np.sin(zeta2),
                np.ones_like(xi2),
                -amplitude * np.sin(xi2) * np.cos(zeta2),
            ),
            axis=-1,
        )
    else:
        xi2, eta2 = np.meshgrid(xi, eta, indexing="xy")
        normal = np.stack(
            (
                -amplitude * np.cos(xi2) * np.sin(eta2),
                -amplitude * np.sin(xi2) * np.cos(eta2),
                np.ones_like(xi2),
            ),
            axis=-1,
        )
    return normal / np.linalg.norm(normal, axis=-1, keepdims=True)


def discrete_normals(path: Path, axis: str, side: int) -> np.ndarray:
    metric_axis = {"x": 1, "y": 2, "z": 3}[axis]
    with h5py.File(path, "r") as handle:
        components = [
            np.asarray(handle[f"dxi{metric_axis}{physical_axis}"], dtype=np.float64).transpose(2, 1, 0)
            for physical_axis in range(1, 4)
        ]
    metric = np.stack(components, axis=-1)
    face_axis = {"x": 0, "y": 1, "z": 2}[axis]
    index = 0 if side == 0 else metric.shape[face_axis] - 1
    face = np.take(metric, index, axis=face_axis)
    # HDF flow fields are ordered k,j,i after reading.
    face = np.swapaxes(face, 0, 1)
    return face / np.linalg.norm(face, axis=-1, keepdims=True)


def boundary_velocity(fields: dict[str, np.ndarray], axis: str, side: int) -> np.ndarray:
    hdf_axis = {"x": 2, "y": 1, "z": 0}[axis]
    index = 0 if side == 0 else fields["u1"].shape[hdf_axis] - 1
    return np.stack(
        [np.take(fields[name], index, axis=hdf_axis) for name in ("u1", "u2", "u3")],
        axis=-1,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--axis", required=True, choices=("x", "y", "z"))
    parser.add_argument("--amplitude", required=True, type=float)
    parser.add_argument("--geometry", type=Path)
    parser.add_argument("--projection-atol", type=float, default=1.0e-12)
    parser.add_argument("--analytic-atol", type=float, default=1.0e-6)
    parser.add_argument("--min-curved-component", type=float, default=1.0e-6)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    with h5py.File(flowfield_path(args.input), "r") as handle:
        fields = {name: handle[name][...] for name in ("u1", "u2", "u3")}
    if not all(np.all(np.isfinite(field)) for field in fields.values()):
        raise ValueError("symmetry velocity field contains non-finite values")

    analytic_normal = analytic_normals(args.axis, fields["u1"].shape, args.amplitude)
    component = {"x": 0, "y": 1, "z": 2}[args.axis]
    analytic_residuals: list[float] = []
    projection_residuals: list[float] = []
    curved_components: list[float] = []
    for side in (0, 1):
        velocity = boundary_velocity(fields, args.axis, side)
        analytic_residuals.append(
            float(np.max(np.abs(np.sum(velocity * analytic_normal, axis=-1))))
        )
        if args.geometry is not None:
            normal = discrete_normals(args.geometry, args.axis, side)
            projection_residuals.append(
                float(np.max(np.abs(np.sum(velocity * normal, axis=-1))))
            )
        curved_components.append(float(np.max(np.abs(velocity[..., component]))))

    passed = (
        max(analytic_residuals) <= args.analytic_atol
        and min(curved_components) >= args.min_curved_component
        and (
            args.geometry is None
            or max(projection_residuals) <= args.projection_atol
        )
    )
    lines = [
        f"status: {'pass' if passed else 'fail'}",
        f"axis: {args.axis}",
        f"analytic_atol: {args.analytic_atol:.16e}",
        f"lower_analytic_normal_velocity_linf: {analytic_residuals[0]:.16e}",
        f"upper_analytic_normal_velocity_linf: {analytic_residuals[1]:.16e}",
        f"lower_cartesian_normal_component_linf: {curved_components[0]:.16e}",
        f"upper_cartesian_normal_component_linf: {curved_components[1]:.16e}",
    ]
    if args.geometry is not None:
        lines.extend(
            (
                f"projection_atol: {args.projection_atol:.16e}",
                f"lower_discrete_normal_velocity_linf: {projection_residuals[0]:.16e}",
                f"upper_discrete_normal_velocity_linf: {projection_residuals[1]:.16e}",
            )
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="ascii")
    print("\n".join(lines))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
