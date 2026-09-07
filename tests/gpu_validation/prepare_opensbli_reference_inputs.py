#!/usr/bin/env python3
"""Generate isolated Katzer input assets, never runnable legacy SBLI inputs."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.integrate import cumulative_trapezoid

from generate_compressible_blasius_profile import map_similarity_profile, solve_profile


def prepare(destination, nx=609, ny=255, nz=9, prandtl=0.72):
    destination = Path(destination)
    if min(nx, ny, nz) < 2 or not np.isfinite(prandtl) or prandtl <= 0.0:
        raise ValueError("Need at least two points per axis and finite positive Pr")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing input assets: {destination}")
    mach, reynolds, reference_temperature, wall_temperature = 2.0, 950.0, 288.0, 1.676194
    eta, f, u, du, temperature = solve_profile(
        mach, reference_temperature, wall_temperature, 20.0, 4001,
        prandtl=prandtl, sutherland_temperature=110.4,
    )
    height = cumulative_trapezoid(temperature, eta, initial=0.0)
    displacement_scale = np.trapezoid(temperature - u, eta)
    station = reynolds / (2.0 * displacement_scale**2)
    yline = 115.0 * np.sinh(5.0 * np.linspace(0.0, 1.0, ny)) / np.sinh(5.0)
    rho, velocity, normal, temp, y_similarity = map_similarity_profile(
        yline, eta, f, u, temperature, reynolds, station
    )
    delta_star = np.trapezoid(temperature - u, eta) / displacement_scale
    delta99 = np.interp(0.99, u, y_similarity)
    theta = np.trapezoid(u * (1.0 - u), eta) / displacement_scale
    suth = 110.4 / reference_temperature
    muwall = temp[0]**1.5 * (1.0 + suth) / (temp[0] + suth)
    dudywall = du[0] * displacement_scale / temp[0]
    utau = np.sqrt(muwall * dudywall / (reynolds * rho[0]))
    shape = (nz, ny, nx)
    z, y, x = np.meshgrid(np.linspace(0.0, 1.0, nz), yline,
                          np.linspace(0.0, 400.0, nx), indexing="ij")
    arrays = {"ro": rho, "u1": velocity, "u2": normal, "u3": np.zeros_like(rho), "t": temp}
    if not all(np.isfinite(value).all() for value in arrays.values()) or min(rho.min(), temp.min()) <= 0:
        raise RuntimeError("Invalid primitive initial conditions")
    report = {
        "status": "INPUT_ONLY_NOT_RUNNABLE",
        "reference_doi": "10.5258/SOTON/D0458",
        "physical_points_xyz": [nx, ny, nz], "astr_im_jm_km": [nx-1, ny-1, nz-1],
        "mach": mach, "reynolds_delta_star": reynolds, "prandtl": prandtl,
        "prandtl_provenance": "candidate; thesis table 0.71 versus later source 0.72 unresolved",
        "sutherland_temperature_K": 110.4, "reference_temperature_K": reference_temperature,
        "wall_temperature_ratio": wall_temperature, "domain_xyz": [400.0, 115.0, 1.0],
        "initialization": "same laminar inlet at every x; no preseeded shock",
        "inlet_station_from_virtual_leading_edge": float(station),
        "similarity_delta_star": float(delta_star),
        "grid_quadrature_delta_star": float(np.trapezoid(1.0-rho*velocity, yline)),
        "outer_v": float(normal[-1]),
        "upper_switch_x": 40.0,
        "upper_q_left": [1.00000596004, 1.00000268202, 0.00565001630205, 0.0, 0.94644428042],
        "upper_q_right": [1.129734572, 1.0921171, -0.058866065, 0.0, 1.0590824],
        "boundary_config_file": "conservative_boundary.nml",
        "pending": ["reference boundary CPU/GPU mainloop integration", "boundary RHS evolution",
                    "MPI corner and halo contract", "CPU/GPU field comparison", "time/grid convergence"],
    }
    destination.mkdir(parents=True)
    with h5py.File(destination / "grid.h5", "w") as handle:
        for name, values in (("x", x), ("y", y), ("z", z)):
            handle[name] = values
    with h5py.File(destination / "initial.h5", "w") as handle:
        for name, values in arrays.items():
            handle[name] = np.broadcast_to(values[None, :, None], shape)
    header = ("OpenSBLI candidate similarity inlet density=provided pressure=provided\n"
              "delta delta_star theta u_tau\n"
              f"{delta99:.16e} {delta_star:.16e} {theta:.16e} {utau:.16e}\n"
              "rho u v temperature pressure")
    pressure = rho * temp / (1.4 * mach**2)
    np.savetxt(destination / "profile.dat", np.column_stack((rho, velocity, normal, temp, pressure)),
               header=header, comments="", fmt="%.16e")
    boundary = ("&conservative_boundary\n schema=1,\n"
                f" split_x={report['upper_switch_x']:.17e},\n"
                " q_left=" + ",".join(f"{x:.17e}" for x in report["upper_q_left"]) + ",\n"
                " q_right=" + ",".join(f"{x:.17e}" for x in report["upper_q_right"]) + "\n/\n")
    (destination / "conservative_boundary.nml").write_text(boundary, encoding="ascii")
    (destination / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--nx", type=int, default=609)
    parser.add_argument("--ny", type=int, default=255)
    parser.add_argument("--nz", type=int, default=9)
    parser.add_argument("--prandtl", type=float, default=0.72)
    args = parser.parse_args()
    print(json.dumps(prepare(args.destination, args.nx, args.ny, args.nz, args.prandtl), indent=2))


if __name__ == "__main__":
    main()
