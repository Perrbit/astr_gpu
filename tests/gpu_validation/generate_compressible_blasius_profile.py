#!/usr/bin/env python3
"""Generate an isothermal compressible Blasius inlet for the S1 M5 gate."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

try:
    from scipy.integrate import cumulative_trapezoid, solve_bvp
except ImportError as error:
    raise RuntimeError("scipy is required to generate the compressible Blasius profile") from error


GAMMA = 1.4
PRANDTL = 0.72


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def sutherland_ratio(temperature: np.ndarray, reference_temperature: float) -> np.ndarray:
    sutherland_temperature = 110.3 / reference_temperature
    return temperature * np.sqrt(temperature) * (1.0 + sutherland_temperature) / (
        temperature + sutherland_temperature
    )


def oblique_shock_ratios(mach: float, shock_angle_deg: float) -> tuple[float, float, float, np.ndarray]:
    beta = np.deg2rad(shock_angle_deg)
    minimum_beta = np.arcsin(1.0 / mach)
    if not minimum_beta < beta < 0.5 * np.pi:
        raise RuntimeError("shock angle must be between the Mach angle and 90 degrees")
    mach_normal = mach * np.sin(beta)
    pressure_ratio = 1.0 + 2.0 * GAMMA / (GAMMA + 1.0) * (mach_normal**2 - 1.0)
    density_ratio = (GAMMA + 1.0) * mach_normal**2 / ((GAMMA - 1.0) * mach_normal**2 + 2.0)
    temperature_ratio = pressure_ratio / density_ratio
    tangent = np.array([np.cos(beta), -np.sin(beta)])
    normal = np.array([np.sin(beta), np.cos(beta)])
    downstream_velocity = np.cos(beta) * tangent + (np.sin(beta) / density_ratio) * normal
    return pressure_ratio, density_ratio, temperature_ratio, downstream_velocity


def solve_profile(
    mach: float,
    reference_temperature: float,
    wall_temperature: float,
    eta_max: float,
    points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    eta = np.linspace(0.0, eta_max, points)
    sutherland_temperature = 110.3 / reference_temperature

    def rhs(_: np.ndarray, state: np.ndarray) -> np.ndarray:
        f, velocity, velocity_eta, temperature, temperature_eta = state
        temperature_safe = np.maximum(temperature, 1.0e-12)
        first = temperature_eta / (2.0 * temperature_safe) - temperature_eta / (
            temperature_safe + sutherland_temperature
        )
        second = (temperature_safe + sutherland_temperature) / (
            np.sqrt(temperature_safe) * (1.0 + sutherland_temperature)
        )
        return np.vstack(
            (
                velocity,
                velocity_eta,
                -velocity_eta * (first + f * second),
                temperature_eta,
                -temperature_eta**2
                * (0.5 / temperature_safe - 1.0 / (temperature_safe + sutherland_temperature))
                - PRANDTL * f * temperature_eta * second
                - (GAMMA - 1.0) * PRANDTL * mach**2 * velocity_eta**2,
            )
        )

    def boundary(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.array((left[0], left[1], left[3] - wall_temperature, right[1] - 1.0, right[3] - 1.0))

    velocity = 1.0 - np.exp(-eta)
    temperature = 1.0 + (wall_temperature - 1.0) * np.exp(-eta)
    initial = np.vstack(
        (
            eta - 1.0 + np.exp(-eta),
            velocity,
            np.exp(-eta),
            temperature,
            -(wall_temperature - 1.0) * np.exp(-eta),
        )
    )
    solution = solve_bvp(rhs, boundary, eta, initial, tol=1.0e-8, max_nodes=100000, verbose=0)
    if solution.status != 0:
        raise RuntimeError(f"compressible Blasius solve failed: {solution.message}")

    state = solution.sol(eta)
    residual = boundary(state[:, 0], state[:, -1])
    if np.max(np.abs(residual)) > 1.0e-7:
        raise RuntimeError(f"compressible Blasius residual is too large: {np.max(np.abs(residual)):.3e}")
    if np.min(state[3]) <= 0.0 or np.min(np.diff(state[1])) < -1.0e-8:
        raise RuntimeError("compressible Blasius profile is not physically monotone")
    return eta, state[0], state[1], state[2], state[3]


def interpolate_at(value: float, coordinate: np.ndarray, profile: np.ndarray) -> float:
    return float(np.interp(value, profile, coordinate))


def map_similarity_profile(
    yline: np.ndarray,
    eta: np.ndarray,
    streamfunction: np.ndarray,
    velocity: np.ndarray,
    temperature: np.ndarray,
    reynolds: float,
    station_x: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scale = np.sqrt(2.0 * station_x / reynolds)
    y_similarity = cumulative_trapezoid(temperature, eta, initial=0.0) * scale
    normal_velocity = -temperature * (streamfunction - velocity * eta) / np.sqrt(2.0 * reynolds * station_x)
    if yline[-1] < y_similarity[-1]:
        raise RuntimeError("grid top is inside the similarity-solution integration interval")
    velocity_line = np.interp(yline, y_similarity, velocity, right=1.0)
    temperature_line = np.interp(yline, y_similarity, temperature, right=1.0)
    normal_velocity_line = np.interp(yline, y_similarity, normal_velocity, right=normal_velocity[-1])
    density_line = 1.0 / temperature_line
    return density_line, velocity_line, normal_velocity_line, temperature_line, y_similarity


def write_profile(
    output: Path,
    grid: Path,
    mach: float,
    reynolds: float,
    reference_temperature: float,
    wall_temperature: float,
    station_x: float,
    density_mode: str,
    pressure_mode: str,
    profile_oblique_shock: bool,
    profile_shock_y_min: float,
    shock_angle_deg: float,
    eta_max: float,
    points: int,
) -> None:
    with h5py.File(grid, "r") as handle:
        yline = np.asarray(handle["y"][0, :, 0], dtype=np.float64)
    if yline.ndim != 1 or yline.size < 4 or not np.all(np.diff(yline) > 0.0):
        raise RuntimeError("grid y coordinates must be a strictly increasing one-dimensional line")

    eta, streamfunction, velocity, velocity_eta, temperature = solve_profile(
        mach, reference_temperature, wall_temperature, eta_max, points
    )
    density_line, velocity_line, normal_velocity_line, temperature_line, y_similarity = map_similarity_profile(
        yline, eta, streamfunction, velocity, temperature, reynolds, station_x
    )
    pressure_line = density_line * temperature_line / (GAMMA * mach**2)

    if profile_oblique_shock:
        if pressure_mode != "provided":
            raise RuntimeError("--profile-oblique-shock requires --pressure-mode provided")
        if profile_shock_y_min < 0.0:
            raise RuntimeError("--profile-shock-y-min must be non-negative")
        pressure_ratio, density_ratio, temperature_ratio, downstream_velocity = oblique_shock_ratios(
            mach, shock_angle_deg
        )
        mask = yline >= profile_shock_y_min
        density_line[mask] *= density_ratio
        temperature_line[mask] *= temperature_ratio
        pressure_line[mask] *= pressure_ratio
        velocity_line[mask] *= downstream_velocity[0]
        normal_velocity_line[mask] += downstream_velocity[1]

    delta99 = interpolate_at(0.99, y_similarity, velocity)
    density_similarity = 1.0 / temperature
    displacement = np.trapezoid(1.0 - density_similarity * velocity, y_similarity)
    momentum = np.trapezoid(density_similarity * velocity * (1.0 - velocity), y_similarity)
    wall_mu = float(sutherland_ratio(np.array([temperature[0]]), reference_temperature)[0])
    wall_du_dy = velocity_eta[0] / (temperature[0] * np.sqrt(2.0 * station_x / reynolds))
    friction_velocity = np.sqrt(wall_mu * wall_du_dy / (reynolds * density_similarity[0]))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="ascii") as handle:
        pressure_header = f" pressure={pressure_mode}" if pressure_mode == "provided" else ""
        handle.write(f"S1-C1 compressible Blasius similarity profile density={density_mode}{pressure_header}\n")
        handle.write("delta delta_star theta u_tau\n")
        handle.write(f"{delta99:.16e} {displacement:.16e} {momentum:.16e} {friction_velocity:.16e}\n")
        if pressure_mode == "provided":
            handle.write("rho u v temperature pressure\n")
            for density, u_value, v_value, temperature_value, pressure in zip(
                density_line, velocity_line, normal_velocity_line, temperature_line, pressure_line
            ):
                handle.write(
                    f"{density:.16e} {u_value:.16e} {v_value:.16e} "
                    f"{temperature_value:.16e} {pressure:.16e}\n"
                )
        else:
            handle.write("rho u v temperature\n")
            for density, u_value, v_value, temperature_value in zip(
                density_line, velocity_line, normal_velocity_line, temperature_line
            ):
                handle.write(f"{density:.16e} {u_value:.16e} {v_value:.16e} {temperature_value:.16e}\n")


def write_similarity_initial_field(
    output: Path,
    grid: Path,
    mach: float,
    reynolds: float,
    reference_temperature: float,
    wall_temperature: float,
    virtual_leading_edge: float,
    eta_max: float,
    points: int,
    oblique_shock: bool,
    shock_angle_deg: float,
    shock_x0: float,
    shock_y0: float,
    shock_y_min: float,
) -> None:
    with h5py.File(grid, "r") as handle:
        xfield = np.asarray(handle["x"][:], dtype=np.float64)
        yfield = np.asarray(handle["y"][:], dtype=np.float64)
    yline = yfield[0, :, 0]
    xline = xfield[0, 0, :]
    if not np.all(np.diff(xline) > 0.0):
        raise RuntimeError("grid x coordinates must be strictly increasing for similarity-field initialization")

    eta, streamfunction, velocity, _, temperature = solve_profile(
        mach, reference_temperature, wall_temperature, eta_max, points
    )
    station = xfield - virtual_leading_edge
    if np.min(station) <= 0.0:
        raise RuntimeError("all x coordinates must lie downstream of the virtual leading edge")

    similarity_height = cumulative_trapezoid(temperature, eta, initial=0.0)
    similarity_coordinate = yfield / np.sqrt(2.0 * station / reynolds)
    flat_coordinate = similarity_coordinate.ravel()
    u1 = np.interp(flat_coordinate, similarity_height, velocity, right=1.0).reshape(xfield.shape)
    tmp = np.interp(flat_coordinate, similarity_height, temperature, right=1.0).reshape(xfield.shape)
    normal_numerator = -temperature * (streamfunction - velocity * eta)
    u2 = np.interp(
        flat_coordinate,
        similarity_height,
        normal_numerator,
        right=normal_numerator[-1],
    ).reshape(xfield.shape) / np.sqrt(2.0 * reynolds * station)
    rho = 1.0 / tmp
    u3 = np.zeros_like(xfield)

    if oblique_shock:
        if shock_y_min < 0.0:
            raise RuntimeError("--shock-y-min must be non-negative")
        pressure_ratio, density_ratio, temperature_ratio, downstream_velocity = oblique_shock_ratios(
            mach, shock_angle_deg
        )
        beta = np.deg2rad(shock_angle_deg)
        normal = np.array([np.sin(beta), np.cos(beta)])
        signed_distance = (xfield - shock_x0) * normal[0] + (yfield - shock_y0) * normal[1]
        mask = (signed_distance >= 0.0) & (yfield >= shock_y_min)
        rho[mask] *= density_ratio
        tmp[mask] *= temperature_ratio
        u1[mask] *= downstream_velocity[0]
        u2[mask] += downstream_velocity[1]

    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as handle:
        handle.create_dataset("ro", data=rho)
        handle.create_dataset("u1", data=u1)
        handle.create_dataset("u2", data=u2)
        handle.create_dataset("u3", data=u3)
        handle.create_dataset("t", data=tmp)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mach", required=True, type=positive_float)
    parser.add_argument("--reynolds", required=True, type=positive_float)
    parser.add_argument("--reference-temperature", required=True, type=positive_float)
    parser.add_argument("--wall-temperature", required=True, type=positive_float)
    parser.add_argument("--station-x", required=True, type=positive_float)
    parser.add_argument("--field-output", type=Path)
    parser.add_argument("--virtual-leading-edge", type=float)
    parser.add_argument("--field-oblique-shock", action="store_true")
    parser.add_argument("--shock-angle-deg", type=positive_float, default=35.0)
    parser.add_argument("--shock-x0", type=float, default=2.0)
    parser.add_argument("--shock-y0", type=float, default=0.18)
    parser.add_argument("--shock-y-min", type=float, default=0.02)
    parser.add_argument("--density-mode", choices=("provided", "reconstruct"), default="provided")
    parser.add_argument("--pressure-mode", choices=("provided", "reconstruct"), default="reconstruct")
    parser.add_argument("--profile-oblique-shock", action="store_true")
    parser.add_argument("--profile-shock-y-min", type=float, default=0.18)
    parser.add_argument("--eta-max", type=positive_float, default=20.0)
    parser.add_argument("--points", type=int, default=801)
    args = parser.parse_args()
    if args.points < 101:
        raise ValueError("--points must be at least 101")
    if args.wall_temperature < 0.1:
        raise ValueError("--wall-temperature is too small for this perfect-gas similarity solver")
    write_profile(
        args.output,
        args.grid,
        args.mach,
        args.reynolds,
        args.reference_temperature,
        args.wall_temperature,
        args.station_x,
        args.density_mode,
        args.pressure_mode,
        args.profile_oblique_shock,
        args.profile_shock_y_min,
        args.shock_angle_deg,
        args.eta_max,
        args.points,
    )
    if args.field_output is not None:
        if args.virtual_leading_edge is None:
            raise ValueError("--virtual-leading-edge is required with --field-output")
        write_similarity_initial_field(
            args.field_output,
            args.grid,
            args.mach,
            args.reynolds,
            args.reference_temperature,
            args.wall_temperature,
            args.virtual_leading_edge,
            args.eta_max,
            args.points,
            args.field_oblique_shock,
            args.shock_angle_deg,
            args.shock_x0,
            args.shock_y0,
            args.shock_y_min,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
