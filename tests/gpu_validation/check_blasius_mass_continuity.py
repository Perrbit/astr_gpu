#!/usr/bin/env python3
"""Fail fast when the existing inlet map violates steady mass continuity.

This is an isolated diagnostic, not an ASTR or OpenSBLI physical validation.
Run from any directory; no solver inputs or reference fields are modified.
"""

import json

import numpy as np
from scipy.integrate import cumulative_trapezoid

from generate_compressible_blasius_profile import (
    map_similarity_profile,
    solve_profile,
    trapezoidal_integral,
)


def check_resolution(points):
    reynolds = 950.0
    eta, f, u, _, temperature = solve_profile(2.0, 288.0, 1.676194, 20.0, points)
    integral_temperature = cumulative_trapezoid(temperature, eta, initial=0.0)
    displacement_scale = trapezoidal_integral(temperature - u, eta)
    station = reynolds / (2.0 * displacement_scale**2)
    y = integral_temperature * np.sqrt(2.0 * station / reynolds)
    # One extra freestream point satisfies the production map's top-height check.
    yline = np.append(y, y[-1] + 1.0)
    rho, velocity, normal, _, _ = map_similarity_profile(
        yline, eta, f, u, temperature, reynolds, station
    )
    rho, velocity, normal = rho[:-1], velocity[:-1], normal[:-1]
    expected_v = (u * integral_temperature - temperature * f) / np.sqrt(
        2.0 * reynolds * station
    )
    # At fixed physical y, similarity scaling gives d(rho*u)/dx below.
    dx_mass_flux = -y / (2.0 * station) * np.gradient(rho * velocity, y, edge_order=2)
    old_residual = dx_mass_flux + np.gradient(rho * normal, y, edge_order=2)
    reference_residual = dx_mass_flux + np.gradient(rho * expected_v, y, edge_order=2)
    interior = slice(10, -10)
    result = {
        "points": points,
        "station_x": float(station),
        "existing_outer_v": float(normal[-1]),
        "continuity_outer_v": float(expected_v[-1]),
        "max_v_difference": float(np.max(np.abs(normal - expected_v))),
        "existing_mass_residual": float(np.max(np.abs(old_residual[interior]))),
        "continuity_mass_residual": float(np.max(np.abs(reference_residual[interior]))),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise RuntimeError("Non-finite similarity diagnostic")
    return result


def main():
    rows = [check_resolution(points) for points in (2001, 4001)]
    if rows[1]["continuity_mass_residual"] >= 0.4 * rows[0]["continuity_mass_residual"]:
        raise RuntimeError("Independent continuity check did not converge under refinement")
    passed = all(row["existing_mass_residual"] < 1.0e-6 for row in rows)
    print(json.dumps({"passed": passed, "resolutions": rows}, indent=2))
    if not passed:
        raise SystemExit("FAIL: existing compressible inlet v violates mass continuity; stop CFD setup")


if __name__ == "__main__":
    main()
