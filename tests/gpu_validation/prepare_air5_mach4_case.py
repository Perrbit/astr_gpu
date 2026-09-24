#!/usr/bin/env python3
"""Prepare an ASTR-only Mach-4 flat-plate precursor, without incident forcing.

The quartic inlet is a seed, not a compressible reacting similarity solution.
An incident jump is recorded for planning but is deliberately not activated.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from air5_radau_reference import Air5RadauReference
from air5_transport_reference import Air5TransportReference
from generate_air5_oblique_shock_states import frozen_oblique_jump, jump_metadata
from prepare_air5_c4_case import prepare_case, replace_after_marker, replace_boundary_types

ROOT = Path(__file__).resolve().parents[2]


def rescale_inlet_profile(path, scale):
    """Change only the inlet thickness; preserve wall/edge states and domain top."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('inlet thickness scale must be finite and positive')
    rows = np.loadtxt(path)
    if rows.ndim != 2 or rows.shape[1] != 13 or rows.shape[0] < 3:
        raise ValueError('unexpected inlet profile layout')
    rows[:-1,0] *= scale
    if not np.isfinite(rows).all() or np.any(np.diff(rows[:,0])<=0):
        raise ValueError('scaled profile must remain below the domain top')
    headers = [line for line in path.read_text().splitlines() if line.startswith('#')]
    headers.append(f'# diagnostic_inlet_thickness_scale={scale:.17e}')
    path.write_text('\n'.join(headers)+'\n'+
        '\n'.join(' '.join(f'{v:.17e}' for v in row) for row in rows)+'\n')


def seed_profile():
    chemistry = Air5RadauReference(ROOT / "chemMech/air5_kimjo12.json")
    transport = Air5TransportReference(ROOT / "chemMech/air5_kimjo12.json")
    ys = np.array([.767, .233, 0., 0., 0.])
    gas = float(ys @ chemistry.gas_constant)
    gamma = 1. + gas / float(ys @ chemistry.cv_tr)
    temperature, wall_temperature, pressure, mach = 1500., 3000., 20000., 4.
    speed = mach * np.sqrt(gamma * gas * temperature)
    rho = pressure / (gas * temperature)
    mu = transport.transport_properties(temperature, temperature, pressure, ys)[0]
    # This Reynolds number sets an engineering seed length, not a DNS resolution gate.
    delta_star = 1000. * mu / (rho * speed)
    eta = np.linspace(0., 1., 4097)
    f = 2.*eta - 2.*eta**3 + eta**4
    temp = wall_temperature + (temperature-wall_temperature)*f
    shape_integral = np.trapezoid(1.-temperature/temp*f, eta)
    thickness = delta_star / shape_integral
    y = np.unique(np.concatenate((eta*thickness, [12.*delta_star])))
    s = np.minimum(y/thickness, 1.)
    f = 2.*s - 2.*s**3 + s**4
    temp = wall_temperature + (temperature-wall_temperature)*f
    rows = np.column_stack((y, pressure/(gas*temp), speed*f,
        np.zeros_like(y), np.zeros_like(y), np.full_like(y, pressure), temp, temp,
        np.broadcast_to(ys, (len(y), 5))))
    jump = frozen_oblique_jump(chemistry, mach=mach, temperature=temperature,
        tv=temperature, pressure=pressure, shock_angle_deg=25., mass_fraction=ys)
    meta = jump_metadata(jump, top_x=20.*delta_star, top_y=12.*delta_star)
    meta.update(case_id="air5_mach4_low_pressure_v1", mach_definition="frozen translational",
        ref_len=delta_star, seed_re_delta_star=1000., seed_thickness=thickness,
        edge_viscosity=mu, edge_density=rho, edge_velocity=float(speed),
        wall_temperature=wall_temperature, domain=[80.*delta_star,12.*delta_star,2.*delta_star],
        flow_through_time=80.*delta_star/speed,
        initial_field="uniform-x analytic quartic seed; not a developed boundary layer",
        x_origin=20.*delta_star, lfilter=False,
        minimum_transport_temperature=1000.,
        validation_scope="ASTR precursor engineering gate, not paper reproduction or physical acceptance")
    return rows, meta


def prepare(destination, grid="63,63,7", maxstep=19, deltat="2.d-9", use_gpu="t"):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing evidence: {destination}")
    dimensions = tuple(int(v) for v in grid.split(','))
    if len(dimensions) != 3 or min(dimensions) < 7:
        raise ValueError("three grid upper bounds >= 7 are required")
    dt = float(deltat.lower().replace('d','e'))
    if maxstep < 0 or not np.isfinite(dt) or dt <= 0 or use_gpu not in ('t', 'f'):
        raise ValueError("invalid time integration configuration")
    rows, meta = seed_profile()
    input_file = prepare_case(ROOT / "examples/Taylor_Green_Vortex_SI/datin",
        destination, grid, maxstep, deltat, "t", "f", use_gpu,
        "high-enthalpy-boundary-layer", 1, "uniform")
    lines = input_file.read_text().splitlines()
    replace_after_marker(lines, "ref_tem,ref_vel,ref_len,ref_den",
        ','.join(f'{v:.17e}' for v in (1500.,meta['edge_velocity'],meta['ref_len'],meta['edge_density'])))
    replace_after_marker(lines, "recon_schem, lchardecomp,bfacmpld,shkcrt", "3,f,0.3d0,0.05d0")
    replace_boundary_types(lines, ("11,free",50,"41,3000.d0",51,1,1))
    input_file.write_text('\n'.join(lines)+'\n')
    datin = destination/'datin'
    (datin/'air5_hbl_profile.dat').write_text(
        '# Analytic startup seed; ASTR must perform predevelopment.\n'
        f'# x_origin={meta["x_origin"]:.17e}\n'
        '# y rho u v w p T Tv Y_N2 Y_O2 Y_N Y_O Y_NO\n' +
        '\n'.join(' '.join(f'{v:.17e}' for v in row) for row in rows)+'\n')
    (datin/'air5_hbl_domain.dat').write_text('air5_hbl_domain_v1\n'+
        ' '.join(f'{v:.17e}' for v in meta['domain'])+'\n')
    control = datin/'controller'
    lines = control.read_text().splitlines()
    replace_after_marker(lines, "maxstep,feqchkpt,feqwsequ,feqslice,feqlist,feqavg",
        f'{maxstep},{max(maxstep,1)},1000000,1000000,1,1000000')
    control.write_text('\n'.join(lines)+'\n')
    meta.update(grid=grid, maxstep=maxstep, deltat=dt, flowtype='air5hbl',
        status='prepared-precursor-only', incident_shock_enabled=False)
    (destination/'mach4_case_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    (destination/'validation').mkdir()
    return input_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--grid',default='63,63,7')
    parser.add_argument('--maxstep',type=int,default=19)
    parser.add_argument('--deltat',default='2.d-9')
    parser.add_argument('--use-gpu',choices=('t','f'),default='t')
    args = parser.parse_args()
    print(prepare(args.destination,args.grid,args.maxstep,args.deltat,args.use_gpu))
