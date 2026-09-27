#!/usr/bin/env python3
"""Attach the approved Mach4 incident boundary to a compensated HBL checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from air5_radau_reference import Air5RadauReference
from check_air5_compensation_checkpoint import read
from generate_air5_oblique_shock_states import frozen_oblique_jump, jump_metadata
from run_air5_sbli_preflight import set_value


def prepare(baseline, destination):
    baseline, destination = Path(baseline), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    checkpoint = baseline/'outdat/flowfield.h5'
    q, _, phase, closure = read(checkpoint)
    profile = np.loadtxt(baseline/'datin/air5_hbl_profile.dat')
    edge = profile[-1]
    rho, u, v, w, pressure, temp, tv = edge[1:8]
    ys = edge[8:13]
    if w != 0 or temp < 1000 or np.any(ys < 0):
        raise ValueError('unsupported edge state')
    chemistry = Air5RadauReference(Path(__file__).resolve().parents[2]/'chemMech/air5_kimjo12.json')
    gas = float(ys@chemistry.gas_constant)
    gamma = 1+gas/float(ys@chemistry.cv_tr)
    mach = float(np.hypot(u, v)/np.sqrt(gamma*gas*temp))
    if not np.allclose([mach, temp, tv, pressure], [4., 1500., 1500., 20000.],
                       rtol=2e-11, atol=0.):
        raise ValueError('edge does not match the approved Mach4 condition')
    jump = frozen_oblique_jump(chemistry, mach=mach, temperature=temp, tv=tv,
        pressure=pressure, shock_angle_deg=25.,
        inflow_angle_deg=float(np.degrees(np.arctan2(v, u))), mass_fraction=ys)
    # Match the saved physical edge, without clipping or normalizing its state.
    if not np.allclose(q[:, -1, :, :], jump.upstream_q, atol=1e-20, rtol=2e-11):
        raise ValueError('saved top edge differs from the prescribed upstream state')
    lines = (baseline/'datin/air5_hbl_domain.dat').read_text().splitlines()
    if lines[0] != 'air5_hbl_domain_v1':
        raise ValueError('unknown HBL geometry schema')
    domain = list(map(float, lines[1].split()))
    meta = jump_metadata(jump, top_x=domain[0]/4, top_y=domain[1])
    if not 0 < meta['geometric_wall_intersection_x'] < domain[0]:
        raise ValueError('geometric shock foot outside domain')
    if meta['max_scaled_normal_flux_residual'] > 2e-12:
        raise ValueError('frozen jump flux mismatch')
    grid = ','.join(str(n-1) for n in reversed(q.shape[:3]))
    meta.update(flowtype='air5sbli', grid=grid, domain=domain,
        wall_temperature=float(profile[0, 6]), lfilter=False,
        baseline=str(baseline.resolve()), checkpoint_phase=phase,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        initial_closure=closure, upstream_mach=mach,
        validation_scope='incident startup from an unsteady precursor; not developed SBLI validation')
    shutil.copytree(baseline/'datin', destination/'datin')
    (destination/'outdat').mkdir()
    for name in ('flowfield.h5', 'auxiliary.txt'):
        shutil.copy2(baseline/'outdat'/name, destination/'outdat'/name)
    set_value(destination/'datin/input.air5_c4', 'flowtype', 'air5sbli')
    rows = [[meta['top_x'], meta['top_y'], *jump.normal], jump.upstream_q, jump.downstream_q]
    (destination/'datin/air5_incident_shock.dat').write_text('air5_incident_shock_v1\n'+
        '\n'.join(' '.join(f'{v:.17e}' for v in row) for row in rows)+'\n')
    (destination/'incident_shock_metadata.json').write_text(json.dumps(meta, indent=2)+'\n')
    return meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.baseline, args.destination), indent=2))
