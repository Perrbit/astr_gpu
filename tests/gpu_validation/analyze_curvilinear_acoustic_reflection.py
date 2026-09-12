#!/usr/bin/env python3
"""Measure outgoing and reflected acoustic characteristics on a curved surface."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401


OUTPUT_DIR = Path("tests/gpu_validation/out/curvilinear_nscbc52_acoustic")

plt.style.use(["science", "ieee", "std-colors"])
plt.rcParams["axes.grid"] = False
plt.rcParams["grid.alpha"] = 0.0
plt.rcParams.update(
    {
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    }
)


def parse_window(value: str) -> tuple[float, float]:
    try:
        window = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("window must have the form start,end") from exc
    if len(window) != 2:
        raise argparse.ArgumentTypeError("window must have the form start,end")
    return window


def parse_vector(value: str) -> tuple[float, float, float]:
    try:
        result = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected three comma-separated values") from exc
    if len(result) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated values")
    return result


def acoustic_invariants(
    p_prime: np.ndarray,
    un_prime: np.ndarray,
    rho0: float,
    c0: float,
) -> tuple[np.ndarray, np.ndarray]:
    if rho0 <= 0.0 or c0 <= 0.0:
        raise ValueError("base density and sound speed must be positive")
    if p_prime.shape != un_prime.shape:
        raise ValueError("pressure and normal-velocity shapes do not match")
    if not np.all(np.isfinite(p_prime)) or not np.all(np.isfinite(un_prime)):
        raise ValueError("acoustic fields must be finite")
    impedance_velocity = rho0 * c0 * un_prime
    return p_prime + impedance_velocity, p_prime - impedance_velocity


def _window_mask(times: np.ndarray, window: tuple[float, float], label: str) -> np.ndarray:
    start, end = window
    if not np.isfinite(start) or not np.isfinite(end) or start >= end:
        raise ValueError(f"invalid {label} window")
    scale = max(1.0, abs(start), abs(end), float(np.max(np.abs(times))))
    endpoint_tolerance = 8.0 * np.finfo(np.float64).eps * scale
    mask = (times >= start - endpoint_tolerance) & (
        times <= end + endpoint_tolerance
    )
    if np.count_nonzero(mask) < 2:
        raise ValueError(f"{label} window must contain at least two samples")
    return mask


def integrate_characteristic_energy(
    times: np.ndarray,
    w_plus: np.ndarray,
    w_minus: np.ndarray,
    surface_weights: np.ndarray,
    *,
    rho0: float,
    c0: float,
    incident_window: tuple[float, float],
    reflected_window: tuple[float, float],
) -> dict[str, float | np.ndarray]:
    times = np.asarray(times, dtype=np.float64)
    w_plus = np.asarray(w_plus, dtype=np.float64)
    w_minus = np.asarray(w_minus, dtype=np.float64)
    surface_weights = np.asarray(surface_weights, dtype=np.float64)
    if times.ndim != 1 or times.size < 2 or not np.all(np.diff(times) > 0.0):
        raise ValueError("times must be a strictly increasing one-dimensional array")
    if w_plus.shape != w_minus.shape or w_plus.ndim != 3:
        raise ValueError("characteristic histories must have shape time,k,i")
    if w_plus.shape[0] != times.size or w_plus.shape[1:] != surface_weights.shape:
        raise ValueError("time, characteristic, and surface-weight shapes do not match")
    if not all(
        np.all(np.isfinite(values))
        for values in (times, w_plus, w_minus, surface_weights)
    ):
        raise ValueError("characteristic integration inputs must be finite")
    if np.any(surface_weights <= 0.0):
        raise ValueError("physical surface weights must be positive")
    if incident_window[1] >= reflected_window[0]:
        raise ValueError("incident and reflected windows overlap")

    incident_mask = _window_mask(times, incident_window, "incident")
    reflected_mask = _window_mask(times, reflected_window, "reflected")
    scale = 1.0 / (4.0 * rho0 * c0**2)
    outgoing_power = scale * np.sum(w_plus**2 * surface_weights[None, :, :], axis=(1, 2))
    incoming_power = scale * np.sum(w_minus**2 * surface_weights[None, :, :], axis=(1, 2))
    incident_energy = float(np.trapezoid(outgoing_power[incident_mask], times[incident_mask]))
    reflected_energy = float(np.trapezoid(incoming_power[reflected_mask], times[reflected_mask]))
    if not np.isfinite(incident_energy) or incident_energy <= 0.0:
        raise ValueError("incident energy must be finite and positive")
    if not np.isfinite(reflected_energy) or reflected_energy < 0.0:
        raise ValueError("reflected energy must be finite and non-negative")
    return {
        "incident_energy": incident_energy,
        "reflected_energy": reflected_energy,
        "reflection": float(np.sqrt(reflected_energy / incident_energy)),
        "outgoing_power": outgoing_power,
        "incoming_power": incoming_power,
    }


def figure_paths(output_dir: Path) -> tuple[Path, Path]:
    return output_dir / "reflection.eps", output_dir / "reflection.jpeg"


def surface_geometry(
    grid_path: Path, probe_index: int
) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(grid_path, "r") as handle:
        coordinates = [np.asarray(handle[name], dtype=np.float64) for name in ("x", "y", "z")]
    if len({array.shape for array in coordinates}) != 1 or len(coordinates[0].shape) != 3:
        raise ValueError("grid coordinates must be matching three-dimensional arrays")
    _, nj, _ = coordinates[0].shape
    if probe_index <= 0 or probe_index >= nj - 1:
        raise ValueError("probe index must be strictly inside the y domain")
    surface = np.stack([array[:, probe_index, :] for array in coordinates], axis=-1)
    tangent_i = np.gradient(surface, axis=1, edge_order=2)
    tangent_k = np.gradient(surface, axis=0, edge_order=2)
    area_vector = np.cross(tangent_k, tangent_i)
    area = np.linalg.norm(area_vector, axis=-1)
    if not np.all(np.isfinite(area)) or np.min(area) <= 0.0:
        raise ValueError("probe surface has invalid physical measure")
    normal = area_vector / area[..., None]
    if np.min(normal[..., 1]) <= 0.0:
        raise ValueError("probe surface normal is not consistently upward")
    trapezoid = np.ones_like(area)
    trapezoid[:, (0, -1)] *= 0.5
    trapezoid[(0, -1), :] *= 0.5
    return normal, area * trapezoid


def read_history(
    paths: list[Path],
    probe_index: int,
    normal: np.ndarray,
    *,
    p0: float,
    velocity0: tuple[float, float, float],
    rho0: float,
    c0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not paths:
        raise ValueError("no complete-RK field snapshots were provided")
    times: list[float] = []
    plus: list[np.ndarray] = []
    minus: list[np.ndarray] = []
    for path in paths:
        with h5py.File(path, "r") as handle:
            missing = [name for name in ("p", "u1", "u2", "u3", "time") if name not in handle]
            if missing:
                raise ValueError(f"{path} is missing datasets: {', '.join(missing)}")
            times.append(float(np.asarray(handle["time"]).reshape(-1)[0]))
            pressure = np.asarray(handle["p"], dtype=np.float64)[:, probe_index, :]
            velocity = np.stack(
                [np.asarray(handle[f"u{axis + 1}"], dtype=np.float64)[:, probe_index, :] for axis in range(3)],
                axis=-1,
            )
        if pressure.shape != normal.shape[:2] or velocity.shape != normal.shape:
            raise ValueError(f"{path} does not match the grid probe surface")
        p_prime = pressure - p0
        un_prime = np.sum((velocity - np.asarray(velocity0)) * normal, axis=-1)
        w_plus, w_minus = acoustic_invariants(p_prime, un_prime, rho0, c0)
        plus.append(w_plus)
        minus.append(w_minus)
    order = np.argsort(times)
    return (
        np.asarray(times, dtype=np.float64)[order],
        np.asarray(plus, dtype=np.float64)[order],
        np.asarray(minus, dtype=np.float64)[order],
    )


def write_outputs(
    output_dir: Path,
    times: np.ndarray,
    result: dict[str, float | np.ndarray],
    metadata: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    outgoing = np.asarray(result["outgoing_power"])
    incoming = np.asarray(result["incoming_power"])
    with (output_dir / "characteristic_history.csv").open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle)
        writer.writerow(("time", "outgoing_power", "incoming_power"))
        writer.writerows(zip(times, outgoing, incoming))
    metrics = {
        **metadata,
        "incident_energy": float(result["incident_energy"]),
        "reflected_energy": float(result["reflected_energy"]),
        "reflection": float(result["reflection"]),
    }
    (output_dir / "reflection_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(times, np.sqrt(outgoing), label=r"$E_+^{1/2}$")
    ax.plot(times, np.sqrt(incoming), label=r"$E_-^{1/2}$")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("Characteristic amplitude")
    ax.legend()
    fig.tight_layout()
    eps, jpeg = figure_paths(output_dir)
    fig.savefig(eps, format="eps", bbox_inches="tight")
    fig.savefig(jpeg, format="jpeg", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", required=True, type=Path)
    parser.add_argument("--snapshots", required=True, nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--probe-index", required=True, type=int)
    parser.add_argument("--rho0", required=True, type=float)
    parser.add_argument("--p0", required=True, type=float)
    parser.add_argument("--c0", required=True, type=float)
    parser.add_argument("--velocity0", type=parse_vector, default=(0.0, 0.0, 0.0))
    parser.add_argument("--incident-window", required=True, type=parse_window)
    parser.add_argument("--reflected-window", required=True, type=parse_window)
    args = parser.parse_args()

    normal, weights = surface_geometry(args.grid, args.probe_index)
    times, w_plus, w_minus = read_history(
        args.snapshots,
        args.probe_index,
        normal,
        p0=args.p0,
        velocity0=args.velocity0,
        rho0=args.rho0,
        c0=args.c0,
    )
    result = integrate_characteristic_energy(
        times,
        w_plus,
        w_minus,
        weights,
        rho0=args.rho0,
        c0=args.c0,
        incident_window=args.incident_window,
        reflected_window=args.reflected_window,
    )
    write_outputs(
        args.output_dir,
        times,
        result,
        {
            "probe_index": args.probe_index,
            "rho0": args.rho0,
            "p0": args.p0,
            "c0": args.c0,
            "velocity0": args.velocity0,
            "incident_window": args.incident_window,
            "reflected_window": args.reflected_window,
            "snapshot_count": len(args.snapshots),
        },
    )
    print(f"reflection: {float(result['reflection']):.16e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
