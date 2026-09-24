#!/usr/bin/env python3
"""Frozen air5 oblique jump for C5-6B2 preparation, not a flow integrator.

The upstream flow defaults to +x. The shock descends from (top_x, top_y),
turning the downstream flow toward -y. Species and specific vibrational
energy are frozen across the jump; relaxation belongs to the ASTR solver.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from air5_postshock_reference import Air5PostShockReference
from air5_radau_reference import Air5RadauReference


@dataclass(frozen=True)
class ObliqueJump:
    upstream_q: np.ndarray
    downstream_q: np.ndarray
    normal: np.ndarray
    tangent: np.ndarray
    shock_angle_deg: float
    inflow_angle_deg: float
    pressure: tuple[float, float]
    temperature: tuple[float, float]
    tv: float
    gamma_tr: float
    sound_speed: tuple[float, float]


def normal_flux(q: np.ndarray, pressure: float, normal: np.ndarray) -> np.ndarray:
    speed = float(np.dot(q[1:4], normal)/q[0])
    flux = q*speed
    flux[1:4] += pressure*normal
    flux[4] += pressure*speed
    return flux


def frozen_oblique_jump(chemistry: Air5RadauReference, *, mach: float,
                        temperature: float, tv: float, pressure: float,
                        shock_angle_deg: float,
                        inflow_angle_deg: float = 0.,
                        mass_fraction: np.ndarray | None = None) -> ObliqueJump:
    parameters = (mach, temperature, tv, pressure, shock_angle_deg, inflow_angle_deg)
    if not np.isfinite(parameters).all():
        raise ValueError('shock parameters must be finite')
    if mach <= 1 or not 0 < shock_angle_deg <= 90:
        raise ValueError('require Mach > 1 and shock angle in (0,90] degrees')
    if not 0 < shock_angle_deg-inflow_angle_deg <= 90 or abs(inflow_angle_deg) >= 90:
        raise ValueError('require a downstream-descending shock and positive upstream x velocity')
    beta = np.deg2rad(shock_angle_deg)
    mn = mach*np.sin(beta)
    if mn <= 1:
        raise ValueError('shock-normal upstream Mach must exceed one')
    reference = Air5PostShockReference(chemistry, upstream_mach=mn,
        upstream_temperature=temperature, upstream_pressure=pressure,
        upstream_tv=tv, mass_fraction=mass_fraction)
    points = (reference.upstream, reference.postshock)
    for point in points:
        if not chemistry.temperature_bounds[0] <= point.temperature <= chemistry.temperature_bounds[1]:
            raise ValueError('jump temperature outside mechanism domain')
        if not chemistry.pressure_bounds[0] <= point.pressure <= chemistry.pressure_bounds[1]:
            raise ValueError('jump pressure outside mechanism domain')
    alpha = np.deg2rad(inflow_angle_deg)
    front = beta-alpha
    normal = np.array([np.sin(front), np.cos(front), 0.])
    tangent = np.array([np.cos(front), -np.sin(front), 0.])
    upstream_speed = reference.upstream.speed/np.sin(beta)
    tangent_speed = upstream_speed*np.cos(beta)
    velocities = (upstream_speed*np.array([np.cos(alpha), np.sin(alpha), 0.]),
                  reference.postshock.speed*normal+tangent_speed*tangent)
    states = []
    for point, velocity in zip(points, velocities):
        q = reference.conservative_state(point)
        q[1:4] = point.density*velocity
        # Restore tangential kinetic energy absent from the normal reference.
        q[4] += .5*point.density*(float(velocity@velocity)-point.speed**2)
        states.append(q)
    y = points[0].mass_fraction
    gas = float(y@chemistry.gas_constant)
    gamma = 1+gas/float(y@chemistry.cv_tr)
    return ObliqueJump(*states, normal, tangent, shock_angle_deg, inflow_angle_deg,
                       tuple(p.pressure for p in points),
                       tuple(p.temperature for p in points), tv, gamma,
                       tuple(float(np.sqrt(gamma*gas*p.temperature)) for p in points))


def top_state(jump: ObliqueJump, x: float, top_x: float) -> np.ndarray:
    if not np.isfinite((x, top_x)).all():
        raise ValueError('top boundary coordinates must be finite')
    # Assign the jump point itself to the downstream state on every MPI rank.
    return (jump.upstream_q if x < top_x else jump.downstream_q).copy()


def jump_metadata(jump: ObliqueJump, *, top_x: float, top_y: float) -> dict:
    if not np.isfinite((top_x, top_y)).all() or top_x < 0 or top_y <= 0:
        raise ValueError('require finite top_x >= 0 and top_y > 0')
    q1, q2 = jump.upstream_q, jump.downstream_q
    velocity2 = q2[1:4]/q2[0]
    f1 = normal_flux(q1, jump.pressure[0], jump.normal)
    f2 = normal_flux(q2, jump.pressure[1], jump.normal)
    scaled = abs(f1-f2)/np.maximum(np.maximum(abs(f1), abs(f2)), 1.)
    return dict(schema='air5_oblique_shock_states_v1', status='boundary-data-only-not-physical-pass',
        top_x=top_x, top_y=top_y, shock_angle_deg=jump.shock_angle_deg,
        inflow_angle_deg=jump.inflow_angle_deg,
        geometric_wall_intersection_x=top_x+top_y/np.tan(np.deg2rad(jump.shock_angle_deg-jump.inflow_angle_deg)),
        normal=jump.normal.tolist(), gamma_tr=jump.gamma_tr,
        deflection_deg=float(jump.inflow_angle_deg-np.rad2deg(np.arctan2(velocity2[1], velocity2[0]))),
        downstream_mach=float(np.linalg.norm(velocity2)/jump.sound_speed[1]),
        downstream_mach_x=float(velocity2[0]/jump.sound_speed[1]),
        top_inward_normal_mach=float(-velocity2[1]/jump.sound_speed[1]),
        max_scaled_normal_flux_residual=float(scaled.max()),
        normal_flux_residual_by_component=scaled.tolist(),
        pressure=list(jump.pressure), temperature=list(jump.temperature), tv=jump.tv,
        q_order=['rho', 'rho_u', 'rho_v', 'rho_w', 'rho_E',
                 'rho_N2', 'rho_O2', 'rho_N', 'rho_O', 'rho_NO', 'rho_Ev'],
        upstream_q=q1.tolist(), downstream_q=q2.tolist(),
        scope='Prescribed complete states, not NSCBC. Geometric intersection is not a viscous impingement prediction.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mechanism', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    for name in ('mach', 'temperature', 'tv', 'pressure', 'shock-angle-deg', 'top-x', 'top-y'):
        parser.add_argument('--'+name, type=float, required=True)
    parser.add_argument('--mass-fractions', help='N2,O2,N,O,NO; defaults to the existing normal-shock gate composition')
    parser.add_argument('--inflow-angle-deg', type=float, default=0.,
                        help='Upstream velocity angle above +x; shock angle is relative to that velocity')
    args = parser.parse_args()
    try:
        fractions = None if args.mass_fractions is None else np.array([float(v) for v in args.mass_fractions.split(',')])
        jump = frozen_oblique_jump(Air5RadauReference(args.mechanism), mach=args.mach,
            temperature=args.temperature, tv=args.tv, pressure=args.pressure,
            shock_angle_deg=args.shock_angle_deg, inflow_angle_deg=args.inflow_angle_deg,
            mass_fraction=fractions)
        metadata = jump_metadata(jump, top_x=args.top_x, top_y=args.top_y)
    except ValueError as error:
        parser.error(str(error))
    if metadata['max_scaled_normal_flux_residual'] > 2e-12:
        parser.error('frozen jump violates the normal conservative flux gate')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: metadata[k] for k in ('status', 'deflection_deg',
        'downstream_mach', 'max_scaled_normal_flux_residual')}))


if __name__ == '__main__':
    main()
