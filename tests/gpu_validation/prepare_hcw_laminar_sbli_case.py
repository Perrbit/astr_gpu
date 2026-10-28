#!/usr/bin/env python3
"""Prepare, but never launch, the HCW-inspired laminar SBLI review case."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from generate_compressible_blasius_profile import (
    GAMMA, oblique_shock_ratios, sutherland_ratio, write_profile, write_similarity_initial_field,
)
from prepare_s1_flatplate_case import write_controller, write_input


MACH = 7.0
TEMPERATURE_K = 240.0
PRESSURE_PA = 400.0
WALL_TEMPERATURE_K = 300.0
CONE_HALF_ANGLE_DEG = 9.0
REFERENCE_LENGTH_M = 1.0
IMPINGEMENT_X_M = 0.8625
REYNOLDS_AT_INLET = 5.0e4
REYNOLDS_AT_IMPINGEMENT = 2.0e5
DOMAIN_HEIGHT_M = 0.03
OUTLET_X_M = 1.0
SPAN_M = 0.002
Y_STRETCH = 3.5
CFL_TARGET = 0.5
DIFFUSIVE_NUMBER_TARGET = 0.25
COMPLETE_STEPS = 30000
CONVECTION_SCHEME = "743e"
# commvar::hm=5. This minimum uses periodic wrap of a strict 2D extrusion, ksize=1.
MIN_PERIODIC_INTERVALS = 5


def time_integration_plan(parameters: dict, dt: float) -> dict:
    flow, geometry = parameters["flow"], parameters["geometry"]
    physical_dt = dt * flow["reference_time_s"]
    flowthrough = (geometry["outlet_x_m"] - geometry["inlet_x_m"]) / flow["velocity_m_s"]
    duration = COMPLETE_STEPS * physical_dt
    return {"starting_step": 0, "controller_maxstep": COMPLETE_STEPS - 1,
            "complete_steps": COMPLETE_STEPS, "dt_nondimensional": dt,
            "dt_s": physical_dt, "duration_nondimensional": COMPLETE_STEPS * dt,
            "duration_s": duration, "freestream_flowthrough_time_s": flowthrough,
            "freestream_flowthroughs": duration / flowthrough, "executed": False}


def coordinate_derivative_643e(values: np.ndarray) -> np.ndarray:
    """ASTR diff6ec, ntype=4, on a complete nonperiodic coordinate line."""
    derivative = np.empty_like(values)
    derivative[0] = -1.5 * values[0] + 2.0 * values[1] - 0.5 * values[2]
    derivative[1] = 0.5 * (values[2] - values[0])
    derivative[2] = 2.0 / 3.0 * (values[3] - values[1]) - (values[4] - values[0]) / 12.0
    derivative[3:-3] = (0.75 * (values[4:-2] - values[2:-4])
                       - 0.15 * (values[5:-1] - values[1:-5])
                       + (values[6:] - values[:-6]) / 60.0)
    derivative[-3] = 2.0 / 3.0 * (values[-2] - values[-4]) - (values[-1] - values[-5]) / 12.0
    derivative[-2] = 0.5 * (values[-1] - values[-3])
    derivative[-1] = 0.5 * values[-3] - 2.0 * values[-2] + 1.5 * values[-1]
    return derivative


def initial_grid_diagnostics(parameters: dict, xline: np.ndarray, yline: np.ndarray,
                             zline: np.ndarray, field: dict,
                             inlet_friction_velocity: float) -> dict:
    """Static initial-state CFL/diffusion estimates and Blasius wall units."""
    flow, geometry = parameters["flow"], parameters["geometry"]
    dx = coordinate_derivative_643e(xline)[None, :]
    dy = coordinate_derivative_643e(yline)[:, None]
    dz = (zline[-1] - zline[0]) / (zline.size - 1)
    if np.any(dx <= 0.0) or np.any(dy <= 0.0) or dz <= 0.0:
        raise ValueError("Grid metrics must be positive for the Cartesian CFL estimate")
    sound = np.sqrt(field["t"]) / flow["mach"]
    rates = [(np.abs(field["u1"]) + sound) / dx,
             (np.abs(field["u2"]) + sound) / dy,
             (np.abs(field["u3"]) + sound) / dz]
    maxima = [float(rate.max()) for rate in rates]
    acoustic_rate = sum(maxima)  # Same directional-max upper bound as commcal::cflcal.
    viscosity = sutherland_ratio(field["t"], flow["temperature_K"])
    kinematic = viscosity / (flow["reynolds_reference"] * field["ro"])
    diffusion = max(4.0 / 3.0, flow["gamma"] / flow["prandtl"]) * kinematic
    diffusive_rate = float(np.max(diffusion * (dx**-2 + dy**-2 + dz**-2)))
    acoustic_dt = CFL_TARGET / acoustic_rate
    diffusive_dt = DIFFUSIVE_NUMBER_TARGET / diffusive_rate
    selected_dt = min(acoustic_dt, diffusive_dt)

    wall_temperature = flow["wall_temperature_K"] / flow["temperature_K"]
    wall_mu = float(sutherland_ratio(np.array([wall_temperature]), flow["temperature_K"])[0])
    wall_density = 1.0 / wall_temperature
    wall_nu = wall_mu / (flow["reynolds_reference"] * wall_density)
    inlet_age = geometry["inlet_x_m"] - geometry["virtual_leading_edge_x_m"]
    first_y = float(yline[1] - yline[0])
    wall_units = []
    for name, station in (("inlet", geometry["inlet_x_m"]),
                          ("nominal_impingement", geometry["nominal_impingement_x_m"]),
                          ("outlet_unshocked_reference", geometry["outlet_x_m"])):
        age = station - geometry["virtual_leading_edge_x_m"]
        utau = inlet_friction_velocity * (inlet_age / age)**0.25
        wall_units.append({"station": name, "x_m": station,
                           "u_tau_m_s": utau * flow["velocity_m_s"],
                           "tau_wall_Pa": wall_density * flow["density_kg_m3"]
                           * (utau * flow["velocity_m_s"])**2,
                           "first_offwall_node_yplus": first_y * utau / wall_nu,
                           "first_interval_midpoint_yplus": 0.5 * first_y * utau / wall_nu})
    return {"scope": "static unshocked initial field; not a time integration stability certificate",
            "metric_model": "separable grid with ASTR 643e coordinate-derivative closures",
            "cfl_target": CFL_TARGET, "directional_acoustic_rates": maxima,
            "acoustic_rate_upper_bound": acoustic_rate,
            "acoustic_rate_local_sum_max": float(np.max(sum(rates))),
            "dt_from_acoustic_cfl": acoustic_dt,
            "diffusive_number_target": DIFFUSIVE_NUMBER_TARGET,
            "diffusion_model": "max(4/3,gamma/Pr)*mu/(Re*rho)*sum(metric_spacing^-2)",
            "diffusion_limit_scope": "conservative estimate, not the exact RK3/NSCBC stability boundary",
            "maximum_diffusive_rate": diffusive_rate, "dt_from_diffusion": diffusive_dt,
            "selected_dt": selected_dt, "selected_initial_acoustic_cfl": selected_dt * acoustic_rate,
            "selected_initial_diffusive_number": selected_dt * diffusive_rate,
            "limiting_estimate": "diffusion" if diffusive_dt < acoustic_dt else "acoustic_cfl",
            "wall_density_kg_m3": wall_density * flow["density_kg_m3"],
            "wall_viscosity_Pa_s": wall_mu * flow["viscosity_Pa_s"],
            "wall_kinematic_viscosity_m2_s": wall_nu * flow["velocity_m_s"] * flow["reference_length_m"],
            "first_offwall_node_y_m": first_y * flow["reference_length_m"],
            "wall_units_scope": "unshocked Blasius estimate using wall density and viscosity",
            "wall_units": wall_units}


def cone_surface_angle(mach: float, beta: float) -> float:
    """Return the cone angle at v_theta=0; velocities use maximum speed."""
    mn2 = (mach * math.sin(beta)) ** 2
    density_ratio = (GAMMA + 1.0) * mn2 / ((GAMMA - 1.0) * mn2 + 2.0)
    speed = 1.0 / math.sqrt(1.0 + 2.0 / ((GAMMA - 1.0) * mach**2))

    def rhs(angle, velocity):
        vr, vt = velocity
        sound2 = 0.5 * (GAMMA - 1.0) * (1.0 - vr**2 - vt**2)
        return (vt, (vr * vt**2 - sound2 * (2.0 * vr + vt / math.tan(angle)))
                / (sound2 - vt**2))

    def wall(angle, velocity):
        return velocity[1]

    wall.terminal = True
    solution = solve_ivp(
        rhs, (beta, 1.0e-6),
        (speed * math.cos(beta), -speed * math.sin(beta) / density_ratio),
        events=wall, rtol=1.0e-10, atol=1.0e-12, max_step=0.005,
    )
    if not solution.success or solution.t_events[0].size != 1:
        raise RuntimeError("Taylor-Maccoll integration did not reach the cone surface")
    return float(solution.t_events[0][0])


def cone_shock_angle(mach: float, half_angle_deg: float) -> float:
    """Shoot the weak attached Taylor-Maccoll solution; return degrees."""
    if mach <= 1.0 or not 0.0 < half_angle_deg < 45.0:
        raise ValueError("Need supersonic flow and a slender positive cone angle")
    target = math.radians(half_angle_deg)
    lower = math.asin(1.0 / mach) + 1.0e-6
    upper = lower
    while cone_surface_angle(mach, upper) < target:
        upper += math.radians(0.5)
        if upper >= math.radians(60.0):
            raise ValueError("No weak attached cone shock found in the supported search range")
    beta = brentq(lambda b: cone_surface_angle(mach, b) - target,
                  lower, upper, xtol=1.0e-12)
    return math.degrees(beta)


def reflected_shock(mach: float, turning: float) -> tuple[float, float]:
    """Weak regular reflection: beta relative to state 2 and p3/p2."""
    def deflection(beta):
        return math.atan(2.0 / math.tan(beta) * ((mach * math.sin(beta))**2 - 1.0)
                         / (mach**2 * (GAMMA + math.cos(2.0 * beta)) + 2.0))

    lower = math.asin(1.0 / mach) + 1.0e-8
    upper = lower
    while deflection(upper) < turning:
        upper += math.radians(0.5)
        if upper >= math.radians(60.0):
            raise ValueError("No weak regular-reflection shock found")
    beta = brentq(lambda b: deflection(b) - turning, lower, upper, xtol=1.0e-12)
    pressure, _, _, _ = oblique_shock_ratios(mach, math.degrees(beta))
    return beta, float(pressure)


def case_parameters() -> dict:
    gas_constant = 287.1
    density = PRESSURE_PA / (gas_constant * TEMPERATURE_K)
    velocity = MACH * math.sqrt(GAMMA * gas_constant * TEMPERATURE_K)
    viscosity = (1.716e-5 * (TEMPERATURE_K / 273.15)**1.5
                 * (273.15 + 110.3) / (TEMPERATURE_K + 110.3))
    reynolds = density * velocity * REFERENCE_LENGTH_M / viscosity
    beta_deg = cone_shock_angle(MACH, CONE_HALF_ANGLE_DEG)
    beta = math.radians(beta_deg)
    p2, rho2, t2, uv2 = oblique_shock_ratios(MACH, beta_deg)
    turning = math.atan2(-uv2[1], uv2[0])
    mach2 = MACH * float(np.linalg.norm(uv2)) / math.sqrt(t2)
    reflection_beta, p3_p2 = reflected_shock(mach2, turning)
    reflection_angle = reflection_beta - turning
    virtual_le = IMPINGEMENT_X_M - REYNOLDS_AT_IMPINGEMENT / reynolds
    inlet = virtual_le + REYNOLDS_AT_INLET / reynolds
    shock_entry = IMPINGEMENT_X_M - DOMAIN_HEIGHT_M / math.tan(beta)
    reflected_exit = IMPINGEMENT_X_M + DOMAIN_HEIGHT_M / math.tan(reflection_angle)
    if not inlet < shock_entry < IMPINGEMENT_X_M < OUTLET_X_M < reflected_exit:
        raise ValueError("Domain conflicts with the two-state incident-shock top target")
    return {
        "status": "PARAMETERS_FOR_REVIEW_NOT_PHYSICALLY_VALIDATED",
        "source": {"doi": "10.6052/0459-1879-16-036", "pages": "878-879",
                   "body_length_m": 1.0, "cone_half_angle_deg": CONE_HALF_ANGLE_DEG,
                   "original_wing_leading_edge_x_m": IMPINGEMENT_X_M,
                   "original_wing_trailing_edge_x_m": 1.2, "original_wing_thickness_m": 0.002},
        "flow": {"mach": MACH, "gamma": GAMMA, "gas_constant_J_kg_K": gas_constant,
                 "temperature_K": TEMPERATURE_K, "pressure_Pa": PRESSURE_PA,
                 "wall_temperature_K": WALL_TEMPERATURE_K, "density_kg_m3": density,
                 "velocity_m_s": velocity, "viscosity_Pa_s": viscosity,
                 "reference_length_m": REFERENCE_LENGTH_M, "reynolds_reference": reynolds,
                 "prandtl": 0.72, "sutherland_temperature_K": 110.3,
                 "reference_time_s": REFERENCE_LENGTH_M / velocity,
                 "perfect_gas_total_temperature_K": TEMPERATURE_K * (1.0 + 0.2 * MACH**2)},
        "shock": {"beta_deg": beta_deg, "turning_deg": math.degrees(turning),
                  "p2_p1": float(p2), "rho2_rho1": float(rho2), "T2_T1": float(t2),
                  "u2_Uinf": float(uv2[0]), "v2_solver_Uinf": float(uv2[1]),
                  "mach2": mach2, "reflected_beta_relative_deg": math.degrees(reflection_beta),
                  "reflected_angle_to_plate_deg": math.degrees(reflection_angle),
                  "ideal_regular_reflection_p3_p1": float(p2 * p3_p2)},
        "geometry": {"virtual_leading_edge_x_m": virtual_le, "inlet_x_m": inlet,
                     "nominal_impingement_x_m": IMPINGEMENT_X_M, "outlet_x_m": OUTLET_X_M,
                     "domain_height_m": DOMAIN_HEIGHT_M, "span_m": SPAN_M,
                     "physical_plate_height_m": IMPINGEMENT_X_M * math.tan(beta),
                     "top_incident_shock_entry_x_m": shock_entry,
                     "ideal_reflected_shock_top_crossing_x_m": reflected_exit,
                     "physical_coordinate_map": "X=x; Y=plate_height-y; Z=-z"},
        "boundary_layer": {"reynolds_at_inlet": REYNOLDS_AT_INLET,
                           "reynolds_at_nominal_impingement": REYNOLDS_AT_IMPINGEMENT,
                           "model": "compressible isothermal Blasius, Pr=0.72, Sutherland viscosity",
                           "initialization": "x-dependent unshocked similarity field; no forcing of upstream interior"},
        "normalization": {"pressure_scale_Pa": density * velocity**2,
                          "freestream_pressure": 1.0 / (GAMMA * MACH**2),
                          "wall_temperature": WALL_TEMPERATURE_K / TEMPERATURE_K},
        "boundaries": {"codes": [11, 22, 41, 52, 1, 1], "inlet": "prof, complete_state",
                       "upper_mode": "sbli_grcbc",
                       "outlet_pressure_ratio": float(p2 * p3_p2)},
        "numerics": {"convection": CONVECTION_SCHEME, "diffusion": "643e",
                     "recon_schem": 5, "reconstruction": "MP7-LD",
                     "characteristic_decomposition": True, "time_scheme": "rk3",
                     "precision": "fp64", "shock_threshold": 0.001,
                     "bfacmpld": 0.1, "main_filter": False, "sponge": False},
        "scope": ["local planar approximation of the conical incident shock",
                  "virtual leading edge sets boundary-layer age; not the original HCW leading edge",
                  "regular reflection is a boundary-placement and pressure-target estimate, not a CFD result",
                  "no wing thickness, leading-edge shock, conical postshock variation or spanwise dynamics",
                  "grid convergence, boundary sensitivity and developed SBLI remain unverified"],
    }


def prepare_case(destination: Path, use_gpu: str = "t", im: int = 1024,
                 jm: int = 256, km: int = MIN_PERIODIC_INTERVALS) -> dict:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite case: {destination}")
    if min(im, jm) < 16 or km < MIN_PERIODIC_INTERVALS:
        raise ValueError("Need x/y >=16 intervals and periodic z >=5 intervals for the fixed halo")
    parameters = case_parameters()
    flow, geometry, shock = (parameters[key] for key in ("flow", "geometry", "shock"))
    datin = destination / "datin"
    datin.mkdir(parents=True)
    (destination / "outdat/new").mkdir(parents=True)
    xline = np.linspace(geometry["inlet_x_m"], OUTLET_X_M, im + 1)
    yline = DOMAIN_HEIGHT_M * np.sinh(Y_STRETCH * np.linspace(0.0, 1.0, jm + 1)) / np.sinh(Y_STRETCH)
    zline = np.linspace(0.0, SPAN_M, km + 1)
    z, y, x = np.meshgrid(zline, yline, xline, indexing="ij")
    grid = datin / "grid.flatplate.h5"
    with h5py.File(grid, "w") as handle:
        for name, values in (("x", x), ("y", y), ("z", z)):
            handle[name] = values
    write_input(
        datin / "input.hcw", im, jm, km, use_gpu, "t", "f", CONVECTION_SCHEME, "t", 0.001,
        flow["reynolds_reference"], MACH, TEMPERATURE_K, WALL_TEMPERATURE_K / TEMPERATURE_K,
        52, 11, 3, 0, "prof", 1, x_max_bctype=22,
        outlet_pressure=shock["ideal_regular_reflection_p3_p1"],
    )
    write_profile(
        datin / "inlet.prof", grid, MACH, flow["reynolds_reference"], TEMPERATURE_K,
        WALL_TEMPERATURE_K / TEMPERATURE_K,
        geometry["inlet_x_m"] - geometry["virtual_leading_edge_x_m"],
        "provided", "provided", False, 0.0, shock["beta_deg"], 20.0, 2001,
    )
    inlet_delta, _, _, inlet_utau = map(float, (datin / "inlet.prof").read_text().splitlines()[2].split())
    parameters["boundary_layer"]["similarity_delta99_inlet_m"] = inlet_delta
    parameters["boundary_layer"]["similarity_delta99_nominal_impingement_m"] = (
        inlet_delta * math.sqrt(REYNOLDS_AT_IMPINGEMENT / REYNOLDS_AT_INLET))
    write_similarity_initial_field(
        datin / "flowini3d.h5", grid, MACH, flow["reynolds_reference"], TEMPERATURE_K,
        WALL_TEMPERATURE_K / TEMPERATURE_K, geometry["virtual_leading_edge_x_m"],
        20.0, 2001, False, shock["beta_deg"], geometry["top_incident_shock_entry_x_m"],
        DOMAIN_HEIGHT_M, 0.0,
    )
    with h5py.File(datin / "flowini3d.h5") as handle:
        initial_plane = {name: handle[name][0] for name in ("ro", "u1", "u2", "u3", "t")}
    diagnostics = initial_grid_diagnostics(parameters, xline, yline, zline, initial_plane, inlet_utau)
    dt = diagnostics["selected_dt"]
    # ASTR starts at nstep=0 and advances while nstep<=maxstep.
    write_controller(datin / "controller", COMPLETE_STEPS - 1, 10, dt)
    parameters["static_diagnostics"] = diagnostics
    (datin / "input.output").write_text(
        "&output\n directory='outdat/new',exact_restart=t,checkpoint_statistics=t,"
        "host_budget_bytes=-1,device_budget_bytes=-1,buffer_bytes=4194304\n/\n"
        "&checkpoint\n enabled=t,interval_steps=3000,keep=2,initial_frame=f\n/\n"
        "&volume\n enabled=f\n/\n&slices\n enabled=f\n/\n", encoding="ascii",
    )
    runtime = {"ASTR_PROFILE_INFLOW_MODE": "complete_state",
               "ASTR_NSCBC_FARFIELD_MODE": parameters["boundaries"]["upper_mode"],
               "ASTR_NSCBC_FARFIELD_SHOCK_X": geometry["top_incident_shock_entry_x_m"],
               "ASTR_NSCBC_FARFIELD_SHOCK_ANGLE_DEG": shock["beta_deg"],
               "ASTR_NSCBC_FARFIELD_RHO": 1.0, "ASTR_NSCBC_FARFIELD_U": 1.0,
               "ASTR_NSCBC_FARFIELD_V": 0.0, "ASTR_NSCBC_FARFIELD_W": 0.0,
               "ASTR_NSCBC_FARFIELD_T": 1.0, "ASTR_GPU_SYNC_MODE": "explicit",
               "ASTR_PERFECT_GAS_PRANDTL": 0.72, "ASTR_SUTHERLAND_TEMPERATURE_K": 110.3,
               "ASTR_GPU_MP_POSITIVITY": "off", "ASTR_CFL_DIAGNOSTICS": "on"}
    (destination / "runtime.env").write_text(
        "".join(f"export {key}={value}\n" for key, value in runtime.items()), encoding="ascii",
    )
    parameters["status"] = "PREPARED_FOR_REVIEW_NOT_RUN"
    parameters["input"] = "datin/input.hcw"
    parameters["grid"] = {"astr_intervals": [im, jm, km], "nodes": [im + 1, jm + 1, km + 1],
                          "dx_m": float(xline[1] - xline[0]), "dy_wall_m": float(yline[1]),
                          "y_sinh_stretch": Y_STRETCH, "dz_m": float(zline[1] - zline[0]),
                          "required_z_partitions": 1,
                          "spanwise_scope": "strict 2D extrusion; minimum does not establish z accuracy"}
    parameters["run_plan"] = time_integration_plan(parameters, dt)
    (destination / "case_manifest.json").write_text(
        json.dumps(parameters, indent=2) + "\n", encoding="ascii",
    )
    return parameters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--use-gpu", choices=("t", "f"), default="t")
    parser.add_argument("--im", type=int, default=1024)
    parser.add_argument("--jm", type=int, default=256)
    parser.add_argument("--km", type=int, default=MIN_PERIODIC_INTERVALS)
    args = parser.parse_args()
    result = case_parameters() if args.destination is None else prepare_case(**vars(args))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
