#!/usr/bin/env python3
"""Diagnose a curvilinear Mach-5 laminar boundary-layer snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.integrate import cumulative_trapezoid

from generate_compressible_blasius_profile import (
    GAMMA,
    PRANDTL,
    solve_profile,
    sutherland_ratio,
)


FIELD_NAMES = ("ro", "u1", "u2", "u3", "t")


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def flowfield_path(path: Path) -> Path:
    if path.is_dir():
        return path / "outdat" / "flowfield.h5"
    return path


def read_hdf(path: Path, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        missing = [name for name in names if name not in handle]
        if missing:
            raise KeyError(f"{path}: missing datasets {missing}")
        result = {name: np.asarray(handle[name][...], dtype=np.float64) for name in names}
    shapes = {array.shape for array in result.values()}
    if len(shapes) != 1:
        raise RuntimeError(f"{path}: inconsistent dataset shapes {sorted(shapes)}")
    if not all(np.all(np.isfinite(array)) for array in result.values()):
        raise RuntimeError(f"{path}: non-finite values present")
    return result


def finite_difference_weights(coordinate: np.ndarray, target: float) -> np.ndarray:
    """Return first-derivative weights exact for polynomials through degree n-1."""
    offset = np.asarray(coordinate, dtype=np.float64) - target
    order = offset.size
    if order < 2 or np.unique(offset).size != order:
        raise ValueError("finite-difference coordinates must be distinct")
    vandermonde = np.vstack([offset**power for power in range(order)])
    rhs = np.zeros(order)
    rhs[1] = 1.0
    return np.linalg.solve(vandermonde, rhs)


def derivative_matrix(size: int, stencil: int = 7) -> np.ndarray:
    if size < stencil:
        raise ValueError(f"at least {stencil} points are required")
    matrix = np.zeros((size, size), dtype=np.float64)
    coordinate = np.arange(size, dtype=np.float64)
    radius = stencil // 2
    for index in range(size):
        start = min(max(index - radius, 0), size - stencil)
        nodes = np.arange(start, start + stencil)
        matrix[index, nodes] = finite_difference_weights(coordinate[nodes], coordinate[index])
    return matrix


def wall_derivatives(
    grid: dict[str, np.ndarray], fields: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Recover lower-wall physical gradients from computational-line derivatives."""
    shape = grid["x"].shape
    if fields["u1"].shape != shape or len(shape) != 3:
        raise RuntimeError("grid and field datasets must share a three-dimensional (k,j,i) shape")
    nk, nj, ni = shape
    if ni < 7 or nj < 7 or nk < 2:
        raise RuntimeError("wall diagnostics require at least 7x7x2 points")

    di = derivative_matrix(ni)
    dj0 = finite_difference_weights(np.arange(7, dtype=np.float64), 0.0)

    def wall_i(array: np.ndarray) -> np.ndarray:
        return array[:, 0, :] @ di.T

    def wall_j(array: np.ndarray) -> np.ndarray:
        return np.einsum("kji,j->ki", array[:, :7, :], dj0)

    x_i = wall_i(grid["x"])
    x_j = wall_j(grid["x"])
    y_i = wall_i(grid["y"])
    y_j = wall_j(grid["y"])
    determinant = x_i * y_j - x_j * y_i
    scale = np.maximum(np.abs(x_i * y_j) + np.abs(x_j * y_i), 1.0)
    if np.any(~np.isfinite(determinant)) or np.any(np.abs(determinant) <= 1.0e-13 * scale):
        raise RuntimeError("singular or non-finite lower-wall mapping Jacobian")

    tangent_norm = np.hypot(x_i, y_i)
    if np.any(tangent_norm <= 0.0):
        raise RuntimeError("degenerate lower-wall streamwise tangent")
    tangent_x = x_i / tangent_norm
    tangent_y = y_i / tangent_norm
    normal_x = -tangent_y
    normal_y = tangent_x
    inward_x = grid["x"][:, 1, :] - grid["x"][:, 0, :]
    inward_y = grid["y"][:, 1, :] - grid["y"][:, 0, :]
    orientation = np.where(normal_x * inward_x + normal_y * inward_y >= 0.0, 1.0, -1.0)
    normal_x *= orientation
    normal_y *= orientation

    gradients: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in ("u1", "u2", "t"):
        derivative_i = wall_i(fields[name])
        derivative_j = wall_j(fields[name])
        derivative_x = (y_j * derivative_i - y_i * derivative_j) / determinant
        derivative_y = (-x_j * derivative_i + x_i * derivative_j) / determinant
        gradients[name] = (derivative_x, derivative_y)

    du_dx, du_dy = gradients["u1"]
    dv_dx, dv_dy = gradients["u2"]
    dt_dx, dt_dy = gradients["t"]
    strain_normal_x = 2.0 * du_dx * normal_x + (du_dy + dv_dx) * normal_y
    strain_normal_y = (du_dy + dv_dx) * normal_x + 2.0 * dv_dy * normal_y
    tangent_shear_rate = tangent_x * strain_normal_x + tangent_y * strain_normal_y
    normal_temperature_gradient = dt_dx * normal_x + dt_dy * normal_y

    return {
        "determinant": determinant,
        "tangent_x": tangent_x,
        "tangent_y": tangent_y,
        "normal_x": normal_x,
        "normal_y": normal_y,
        "du_dy": du_dy,
        "dtemperature_dy": dt_dy,
        "tangent_shear_rate": tangent_shear_rate,
        "normal_temperature_gradient": normal_temperature_gradient,
    }


def spanwise_average(values: np.ndarray, zwall: np.ndarray) -> np.ndarray:
    if values.shape != zwall.shape:
        raise ValueError("spanwise values and coordinates must have the same shape")
    span = zwall[-1, :] - zwall[0, :]
    if np.any(span <= 0.0):
        raise RuntimeError("spanwise wall coordinates must increase")
    return np.trapezoid(values, zwall, axis=0) / span


def similarity_solution(
    mach: float,
    reynolds: float,
    reference_temperature: float,
    wall_temperature: float,
    eta_max: float = 20.0,
    points: int = 2001,
) -> dict[str, np.ndarray]:
    eta, streamfunction, velocity, velocity_eta, temperature = solve_profile(
        mach, reference_temperature, wall_temperature, eta_max, points
    )
    return {
        "eta": eta,
        "streamfunction": streamfunction,
        "velocity": velocity,
        "velocity_eta": velocity_eta,
        "temperature": temperature,
        "height": cumulative_trapezoid(temperature, eta, initial=0.0),
    }


def similarity_at(
    x: np.ndarray,
    y: np.ndarray,
    solution: dict[str, np.ndarray],
    reynolds: float,
    virtual_leading_edge: float,
) -> tuple[np.ndarray, np.ndarray]:
    station = x - virtual_leading_edge
    if np.any(station <= 0.0):
        raise RuntimeError("all analysis points must lie downstream of the virtual leading edge")
    coordinate = y / np.sqrt(2.0 * station / reynolds)
    u = np.interp(coordinate, solution["height"], solution["velocity"], right=1.0)
    temperature = np.interp(
        coordinate, solution["height"], solution["temperature"], right=1.0
    )
    return u, temperature


def reference_wall_curves(
    wall_x: np.ndarray,
    solution: dict[str, np.ndarray],
    mach: float,
    reynolds: float,
    reference_temperature: float,
    wall_temperature: float,
    virtual_leading_edge: float,
) -> tuple[np.ndarray, np.ndarray]:
    station = wall_x - virtual_leading_edge
    if np.any(station <= 0.0):
        raise RuntimeError("wall station lies upstream of the virtual leading edge")
    scale = wall_temperature * np.sqrt(2.0 * station / reynolds)
    du_dy = solution["velocity_eta"][0] / scale
    temperature_eta = finite_difference_weights(
        solution["eta"][:7], solution["eta"][0]
    ) @ solution["temperature"][:7]
    dt_dy = temperature_eta / scale
    wall_mu = float(sutherland_ratio(np.array([wall_temperature]), reference_temperature)[0])
    cf = 2.0 * wall_mu * du_dy / reynolds
    conductivity = wall_mu / (reynolds * PRANDTL * (GAMMA - 1.0) * mach**2)
    qw = conductivity * dt_dy
    return cf, qw


def station_profile(
    station: float, grid: dict[str, np.ndarray], fields: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nk, nj, _ = grid["x"].shape
    y_values = np.empty((nk, nj))
    u_values = np.empty((nk, nj))
    t_values = np.empty((nk, nj))
    for k in range(nk):
        for j in range(nj):
            xline = grid["x"][k, j, :]
            if np.any(np.diff(xline) <= 0.0) or not xline[0] <= station <= xline[-1]:
                raise RuntimeError(f"station x={station:g} is outside a monotone grid line")
            y_values[k, j] = np.interp(station, xline, grid["y"][k, j, :])
            u_values[k, j] = np.interp(station, xline, fields["u1"][k, j, :])
            t_values[k, j] = np.interp(station, xline, fields["t"][k, j, :])
    return np.mean(y_values, axis=0), np.mean(u_values, axis=0), np.mean(t_values, axis=0)


def relative_l2(value: np.ndarray, reference: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference))
    if denominator == 0.0:
        return float(np.linalg.norm(value - reference))
    return float(np.linalg.norm(value - reference) / denominator)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--field", required=True, type=Path)
    parser.add_argument("--grid", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--npz", required=True, type=Path)
    parser.add_argument("--wall-csv", required=True, type=Path)
    parser.add_argument("--profile-dir", required=True, type=Path)
    parser.add_argument("--mach", type=positive_float, default=5.0)
    parser.add_argument("--reynolds", type=positive_float, default=1.83052e6)
    parser.add_argument("--reference-temperature", type=positive_float, default=226.65)
    parser.add_argument("--wall-temperature", type=positive_float, default=5.191440547760865)
    parser.add_argument("--virtual-leading-edge", type=float, default=-2.0)
    parser.add_argument("--analysis-x-min", type=float, default=0.0)
    parser.add_argument("--analysis-x-max", type=float, default=8.0)
    parser.add_argument("--stations", default="0,3,7")
    args = parser.parse_args()

    grid = read_hdf(args.grid, ("x", "y", "z"))
    fields = read_hdf(flowfield_path(args.field), FIELD_NAMES)
    if grid["x"].shape != fields["u1"].shape:
        raise RuntimeError("grid and field shapes differ")
    wall_y = grid["y"][:, 0, :]
    if float(np.ptp(wall_y)) > 1.0e-12:
        raise RuntimeError("Blasius comparison requires a flat lower wall")

    derivative = wall_derivatives(grid, fields)
    wall_x = spanwise_average(grid["x"][:, 0, :], grid["z"][:, 0, :])
    wall_temperature = fields["t"][:, 0, :]
    viscosity = sutherland_ratio(wall_temperature, args.reference_temperature) / args.reynolds
    conductivity = viscosity / (PRANDTL * (GAMMA - 1.0) * args.mach**2)
    cf_astr = spanwise_average(2.0 * viscosity * derivative["du_dy"], grid["z"][:, 0, :])
    cf_geometric = spanwise_average(
        2.0 * viscosity * derivative["tangent_shear_rate"], grid["z"][:, 0, :]
    )
    qw_astr = spanwise_average(
        conductivity * derivative["dtemperature_dy"], grid["z"][:, 0, :]
    )
    qw_geometric = spanwise_average(
        conductivity * derivative["normal_temperature_gradient"], grid["z"][:, 0, :]
    )

    solution = similarity_solution(
        args.mach, args.reynolds, args.reference_temperature, args.wall_temperature
    )
    cf_reference, qw_reference = reference_wall_curves(
        wall_x,
        solution,
        args.mach,
        args.reynolds,
        args.reference_temperature,
        args.wall_temperature,
        args.virtual_leading_edge,
    )
    analysis_mask = (wall_x >= args.analysis_x_min) & (wall_x <= args.analysis_x_max)
    if np.count_nonzero(analysis_mask) < 3:
        raise RuntimeError("analysis interval contains fewer than three wall stations")

    stations = np.array([float(value) for value in args.stations.split(",")], dtype=np.float64)
    if stations.size == 0 or np.any(np.diff(stations) <= 0.0):
        raise RuntimeError("profile stations must be a strictly increasing comma-separated list")
    profiles_y = []
    profiles_u = []
    profiles_t = []
    profiles_u_reference = []
    profiles_t_reference = []
    profile_lines = []
    args.profile_dir.mkdir(parents=True, exist_ok=True)
    for station in stations:
        yline, uline, tline = station_profile(float(station), grid, fields)
        uref, tref = similarity_at(
            np.full_like(yline, station), yline, solution, args.reynolds, args.virtual_leading_edge
        )
        profiles_y.append(yline)
        profiles_u.append(uline)
        profiles_t.append(tline)
        profiles_u_reference.append(uref)
        profiles_t_reference.append(tref)
        profile_path = args.profile_dir / f"profile_x{station:g}.csv"
        np.savetxt(
            profile_path,
            np.column_stack((yline, uline, uref, tline, tref)),
            delimiter=",",
            header="y,u,u_blasius,T,T_blasius",
            comments="",
        )
        profile_lines.append(
            f"profile_x={station:.8g} u_rel_l2={relative_l2(uline, uref):.16e} "
            f"T_rel_l2={relative_l2(tline, tref):.16e} csv={profile_path}"
        )

    profiles_y_array = np.stack(profiles_y)
    profiles_u_array = np.stack(profiles_u)
    profiles_t_array = np.stack(profiles_t)
    profiles_u_reference_array = np.stack(profiles_u_reference)
    profiles_t_reference_array = np.stack(profiles_t_reference)

    args.wall_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        args.wall_csv,
        np.column_stack(
            (wall_x, cf_geometric, cf_astr, cf_reference, qw_geometric, qw_astr, qw_reference)
        ),
        delimiter=",",
        header="x,Cf_geometric,Cf_astr,Cf_blasius,qw_geometric,qw_astr,qw_blasius",
        comments="",
    )
    args.npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.npz,
        x=wall_x,
        analysis_mask=analysis_mask,
        cf_geometric=cf_geometric,
        cf_astr=cf_astr,
        cf_reference=cf_reference,
        qw_geometric=qw_geometric,
        qw_astr=qw_astr,
        qw_reference=qw_reference,
        stations=stations,
        profile_y=profiles_y_array,
        profile_u=profiles_u_array,
        profile_temperature=profiles_t_array,
        profile_u_reference=profiles_u_reference_array,
        profile_temperature_reference=profiles_t_reference_array,
    )

    mask = analysis_mask
    wall_tangent_y = float(np.max(np.abs(derivative["tangent_y"])))
    cf_definition_delta = float(np.max(np.abs(cf_geometric[mask] - cf_astr[mask])))
    qw_definition_delta = float(np.max(np.abs(qw_geometric[mask] - qw_astr[mask])))
    lines = [
        "status: complete",
        "scope: transient compressible-Blasius consistency diagnostic; not a steady DNS validation",
        "Cf_geometric: 2 mu/Re times t dot (grad(u)+grad(u)^T) dot n",
        "Cf_astr: 2 mu/Re times du/dy, matching the continuous src/statistic.F90 fbcxbl normalization",
        "qw_geometric: mu/[Re Pr (gamma-1) Mach^2] times grad(T) dot n",
        "qw_astr: mu/[Re Pr (gamma-1) Mach^2] times dT/dy",
        f"field: {flowfield_path(args.field)}",
        f"grid: {args.grid}",
        f"analysis_x: [{args.analysis_x_min:.8g}, {args.analysis_x_max:.8g}]",
        f"max_abs_wall_tangent_y: {wall_tangent_y:.16e}",
        f"max_abs_Cf_geometric_minus_astr: {cf_definition_delta:.16e}",
        f"max_abs_qw_geometric_minus_astr: {qw_definition_delta:.16e}",
        f"Cf_vs_blasius_rel_l2: {relative_l2(cf_geometric[mask], cf_reference[mask]):.16e}",
        f"qw_vs_blasius_rel_l2: {relative_l2(qw_geometric[mask], qw_reference[mask]):.16e}",
        f"wall_csv: {args.wall_csv}",
        f"npz: {args.npz}",
        *profile_lines,
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="ascii")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
