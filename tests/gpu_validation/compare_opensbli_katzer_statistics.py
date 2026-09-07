#!/usr/bin/env python3
"""Compare ASTR wall diagnostics with the OpenSBLI Katzer SBLI archive."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import zipfile

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401


GAMMA = 1.4
MACH = 2.0
REYNOLDS = 950.0
REFERENCE_TEMPERATURE_K = 288.0
SUTHERLAND_TEMPERATURE_K = 110.4
REFERENCE_TIME = 13000.0


def derivative_weights(nodes: np.ndarray, evaluation_point: float) -> np.ndarray:
    """Return polynomial first-derivative weights for arbitrary 1-D nodes."""
    nodes = np.asarray(nodes, dtype=np.float64)
    if nodes.ndim != 1 or nodes.size < 2 or not np.all(np.isfinite(nodes)):
        raise ValueError("derivative nodes must be a finite one-dimensional array")
    offsets = nodes - float(evaluation_point)
    vandermonde = np.vstack([offsets**degree for degree in range(nodes.size)])
    rhs = np.zeros(nodes.size, dtype=np.float64)
    rhs[1] = 1.0
    return np.linalg.solve(vandermonde, rhs)


def directed_zero_crossings(x: np.ndarray, values: np.ndarray) -> tuple[list[float], list[float]]:
    """Return positive-to-negative and negative-to-positive linear zero crossings."""
    x = np.asarray(x, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or values.shape != x.shape:
        raise ValueError("zero-crossing inputs must be one-dimensional and shape-matched")
    separation: list[float] = []
    reattachment: list[float] = []
    for index in range(x.size - 1):
        left = values[index]
        right = values[index + 1]
        if not np.isfinite(left) or not np.isfinite(right) or left == right:
            continue
        if left * right < 0.0:
            root = float(x[index] - left * (x[index + 1] - x[index]) / (right - left))
            if left > 0.0 and right < 0.0:
                separation.append(root)
            elif left < 0.0 and right > 0.0:
                reattachment.append(root)
    return separation, reattachment


def read_ops_dataset(group: h5py.Group, name: str) -> np.ndarray:
    dataset = group[name]
    halo = np.abs(dataset.attrs.get("d_m", np.zeros(dataset.ndim, dtype=np.int32)))
    slices = tuple(slice(int(width), size - int(width)) for width, size in zip(halo, dataset.shape))
    return np.asarray(dataset[slices], dtype=np.float64)


def read_reference(reference_zip: Path) -> dict[str, np.ndarray]:
    with zipfile.ZipFile(reference_zip) as archive:
        matches = [name for name in archive.namelist() if Path(name).name == "opensbli.h5"]
        if len(matches) != 1:
            raise ValueError(f"expected one opensbli.h5 in {reference_zip}, found {matches}")
        payload = io.BytesIO(archive.read(matches[0]))
    with h5py.File(payload, "r") as handle:
        group = handle["opensbliblock00"]
        x = read_ops_dataset(group, "x0_B0")[0, :]
        y = read_ops_dataset(group, "x1_B0")
        rho = read_ops_dataset(group, "rho_B0")
        u = read_ops_dataset(group, "rhou0_B0") / rho
        v = read_ops_dataset(group, "rhou1_B0") / rho
        energy = read_ops_dataset(group, "rhoE_B0")
        pressure = (GAMMA - 1.0) * (energy - 0.5 * rho * (u * u + v * v))
        cf = np.asarray(group["Cf_B0"][...], dtype=np.float64).reshape(-1)
    if cf.shape != x.shape:
        raise ValueError(f"reference Cf shape {cf.shape} does not match x shape {x.shape}")
    temperature = GAMMA * MACH**2 * pressure / rho
    sutherland_ratio = SUTHERLAND_TEMPERATURE_K / REFERENCE_TEMPERATURE_K
    mu_wall = temperature[0, :] ** 1.5 * (1.0 + sutherland_ratio) / (
        temperature[0, :] + sutherland_ratio
    )
    wall_weights = derivative_weights(y[:6, 0], y[0, 0])
    reconstructed_cf = 2.0 * mu_wall * (wall_weights @ u[:6, :]) / REYNOLDS
    return {
        "x": x,
        "cf": cf,
        "reconstructed_cf": reconstructed_cf,
        "pressure_ratio": pressure[0, :] / pressure[0, 0],
    }


def read_astr(flow_path: Path, grid_path: Path) -> dict[str, np.ndarray | float | int | dict[str, float]]:
    with h5py.File(flow_path, "r") as flow, h5py.File(grid_path, "r") as grid:
        required_flow = ("ro", "u1", "u2", "u3", "p", "t", "nstep", "time")
        missing = [name for name in required_flow if name not in flow]
        if missing:
            raise KeyError(f"{flow_path}: missing datasets {missing}")
        x3 = np.asarray(grid["x"][...], dtype=np.float64)
        y3 = np.asarray(grid["y"][...], dtype=np.float64)
        fields = {name: np.asarray(flow[name][...], dtype=np.float64) for name in required_flow[:6]}
        nstep = int(np.asarray(flow["nstep"][...]).reshape(-1)[0])
        time = float(np.asarray(flow["time"][...]).reshape(-1)[0])
    if any(values.shape != x3.shape for values in fields.values()) or y3.shape != x3.shape:
        raise ValueError("ASTR grid and flow arrays do not have a common shape")
    if x3.ndim != 3 or x3.shape[1] < 6:
        raise ValueError(f"expected ASTR (z,y,x) arrays with at least six y points, got {x3.shape}")
    arrays = [x3, y3, *fields.values()]
    if not all(np.all(np.isfinite(values)) for values in arrays):
        raise FloatingPointError("ASTR grid or flow contains non-finite values")
    if np.min(fields["ro"]) <= 0.0 or np.min(fields["p"]) <= 0.0 or np.min(fields["t"]) <= 0.0:
        raise FloatingPointError("ASTR density, pressure, or temperature is non-positive")

    z_spread = {
        name: float(np.max(np.abs(values - np.mean(values, axis=0, keepdims=True))))
        for name, values in fields.items()
    }
    x = np.mean(x3[:, 0, :], axis=0)
    y_nodes = np.mean(y3[:, :6, :], axis=(0, 2))
    if np.any(np.diff(y_nodes) <= 0.0):
        raise ValueError("the first six wall-normal grid points are not strictly increasing")
    weights = derivative_weights(y_nodes, y_nodes[0])
    u_mean = np.mean(fields["u1"], axis=0)
    dudy_wall = weights @ u_mean[:6, :]
    temperature_wall = np.mean(fields["t"][:, 0, :], axis=0)
    sutherland_ratio = SUTHERLAND_TEMPERATURE_K / REFERENCE_TEMPERATURE_K
    mu_wall = temperature_wall**1.5 * (1.0 + sutherland_ratio) / (
        temperature_wall + sutherland_ratio
    )
    cf = 2.0 * mu_wall * dudy_wall / REYNOLDS
    pressure_wall = np.mean(fields["p"][:, 0, :], axis=0)
    return {
        "x": x,
        "cf": cf,
        "pressure_ratio": pressure_wall / pressure_wall[0],
        "nstep": nstep,
        "time": time,
        "z_spread": z_spread,
    }


def error_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    difference = candidate - reference
    reference_scale = float(np.max(np.abs(reference)))
    return {
        "linf": float(np.max(np.abs(difference))),
        "l2": float(np.sqrt(np.mean(difference * difference))),
        "reference_scale": reference_scale,
        "relative_linf_by_reference_peak": float(np.max(np.abs(difference)) / reference_scale),
    }


def separation_metrics(x: np.ndarray, cf: np.ndarray) -> dict[str, float | None | list[float]]:
    separation, reattachment = directed_zero_crossings(x, cf)
    result: dict[str, float | None | list[float]] = {
        "separation_candidates": separation,
        "reattachment_candidates": reattachment,
        "x_separation": None,
        "x_reattachment": None,
        "separation_length": None,
    }
    if separation:
        first_separation = separation[0]
        later_reattachment = [root for root in reattachment if root > first_separation]
        if later_reattachment:
            result["x_separation"] = first_separation
            result["x_reattachment"] = later_reattachment[0]
            result["separation_length"] = later_reattachment[0] - first_separation
    return result


def write_comparison_csv(path: Path, x: np.ndarray, astr: dict, reference: dict) -> None:
    with path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("x", "astr_cf", "reference_cf", "astr_pw_p1", "reference_pw_p1"))
        writer.writerows(
            zip(x, astr["cf"], reference["cf"], astr["pressure_ratio"], reference["pressure_ratio"])
        )


def write_plots(output_dir: Path, x: np.ndarray, astr: dict, reference: dict) -> None:
    plt.style.use(["science", "ieee", "std-colors"])
    plt.rcParams["axes.grid"] = False
    plt.rcParams["grid.alpha"] = 0.0
    plt.rcParams.update({
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    })
    for name, ylabel, astr_values, reference_values in (
        ("skin_friction", r"$C_f$", astr["cf"], reference["cf"]),
        ("wall_pressure", r"$p_w/p_1$", astr["pressure_ratio"], reference["pressure_ratio"]),
    ):
        fig, axis = plt.subplots(figsize=(7.2, 4.6))
        axis.plot(x, reference_values, color="#d62728", linewidth=1.7, label="OpenSBLI")
        axis.plot(x, astr_values, color="#1f77b4", linewidth=1.5, label="ASTR")
        if name == "skin_friction":
            axis.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
            axis.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        axis.set_xlabel(r"$x$")
        axis.set_ylabel(ylabel)
        axis.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / f"{name}.eps", bbox_inches="tight")
        fig.savefig(output_dir / f"{name}.jpeg", dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--astr-flow", required=True, type=Path)
    parser.add_argument("--astr-grid", required=True, type=Path)
    parser.add_argument("--reference-zip", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-time", type=float, default=REFERENCE_TIME)
    parser.add_argument("--time-atol", type=float, default=1.0e-8)
    args = parser.parse_args()

    astr = read_astr(args.astr_flow, args.astr_grid)
    reference = read_reference(args.reference_zip)
    if astr["x"].shape != reference["x"].shape:
        raise ValueError(f"x-grid shape mismatch: ASTR {astr['x'].shape}, reference {reference['x'].shape}")
    x_error = float(np.max(np.abs(astr["x"] - reference["x"])))
    if x_error > 1.0e-12:
        raise ValueError(f"x grids differ by {x_error:.16e}; interpolation was not authorized")

    astr_separation = separation_metrics(astr["x"], astr["cf"])
    reference_separation = separation_metrics(reference["x"], reference["cf"])
    reached_expected_time = abs(float(astr["time"]) - args.expected_time) <= args.time_atol
    report = {
        "status": "time_reached" if reached_expected_time else "incomplete_time",
        "reference_doi": "10.5258/SOTON/D0458",
        "reference_time": REFERENCE_TIME,
        "expected_time": args.expected_time,
        "astr_time": astr["time"],
        "astr_nstep": astr["nstep"],
        "comparison_contract": {
            "reference_discretization": "two-dimensional fifth-order WENO-Z",
            "astr_discretization": "explicit MP reconstruction on a nine-point periodic extrusion",
            "astr_prandtl": 0.72,
            "prandtl_limit": "the archive has no Pr metadata; 0.72 is taken from the accompanying public source",
            "cf_definition": "2*mu_wall/Re*(du/dy)_wall with a six-point physical-y derivative",
            "pressure_definition": "wall pressure normalized by its inlet value",
        },
        "x_grid_linf": x_error,
        "z_uniformity_linf": astr["z_spread"],
        "reference_cf_reconstruction_error": error_metrics(
            reference["reconstructed_cf"], reference["cf"]
        ),
        "skin_friction_error": error_metrics(astr["cf"], reference["cf"]),
        "wall_pressure_ratio_error": error_metrics(
            astr["pressure_ratio"], reference["pressure_ratio"]
        ),
        "astr_separation": astr_separation,
        "reference_separation": reference_separation,
    }
    if astr_separation["separation_length"] is not None:
        reference_length = float(reference_separation["separation_length"])
        report["separation_length_error"] = {
            "absolute": float(astr_separation["separation_length"] - reference_length),
            "relative": float(
                (astr_separation["separation_length"] - reference_length) / reference_length
            ),
        }
    else:
        report["separation_length_error"] = None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_comparison_csv(args.output_dir / "wall_comparison.csv", astr["x"], astr, reference)
    write_plots(args.output_dir, astr["x"], astr, reference)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="ascii"
    )
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if reached_expected_time else 2


if __name__ == "__main__":
    raise SystemExit(main())
