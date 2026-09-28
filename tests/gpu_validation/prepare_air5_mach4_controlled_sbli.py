#!/usr/bin/env python3
"""Prepare, but never launch, the controlled Mach-4 reacting SBLI campaign."""
import argparse
import hashlib
import json
from pathlib import Path

from air5_radau_reference import Air5RadauReference
from generate_air5_oblique_shock_states import frozen_oblique_jump
from prepare_air5_mach4_case import ROOT, prepare, seed_profile
from run_air5_sbli_preflight import set_value


def prepare_campaign(destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    rows, physics = seed_profile()
    physics['scope'] = ('Frozen jump supplies characteristic boundary targets; '
                        'geometric intersection is not a viscous impingement prediction.')
    model = Air5RadauReference(ROOT/'chemMech/air5_kimjo12.json')
    jump = frozen_oblique_jump(model, mach=4., temperature=1500., tv=1500.,
        pressure=20000., shock_angle_deg=25., mass_fraction=rows[-1, 8:13])
    if physics['max_scaled_normal_flux_residual'] > 2e-12:
        raise ValueError('incident jump fails normal flux conservation')
    tau = physics['domain'][1] / (physics['edge_velocity']/4.)
    common = dict(ASTR_FORCE_MPI_TOPOLOGY='2,1,1', ASTR_GPU_SYNC_MODE='explicit',
        ASTR_GPU_PRECISION_MODE='fp64', ASTR_AIR5_COMPENSATION='on',
        ASTR_AIR5_CONVECTION_LIMITER='symmetric_species',
        ASTR_AIR5_DIFFUSION_LIMITER='layered', ASTR_AIR5_C4_CONSERVATION='f',
        ASTR_AIR5_TOP_MODE='characteristic', ASTR_AIR5_TOP_TAU=f'{tau:.17e}')
    cases = []
    for name, incident, source in (
        ('precursor', False, 'coupled'),
        ('flat_plate_coupled', False, 'coupled'),
        ('sbli_vt_only', True, 'vt'),
        ('sbli_coupled', True, 'coupled'),
    ):
        case = destination/name
        input_file = prepare(case, grid='63,127,15', maxstep=1, deltat='2.5d-10')
        set_value(input_file, 'lrestar', 'f')
        if incident:
            set_value(input_file, 'flowtype', 'air5sbli')
            shock_rows = [[physics['top_x'], physics['top_y'], *jump.normal],
                          jump.upstream_q, jump.downstream_q]
            (case/'datin/air5_incident_shock.dat').write_text(
                'air5_incident_shock_v1\n' + '\n'.join(
                    ' '.join(f'{v:.17e}' for v in row) for row in shock_rows)+'\n')
        metadata = json.loads((case/'mach4_case_metadata.json').read_text())
        metadata.update(flowtype='air5sbli' if incident else 'air5hbl',
            incident_shock_enabled=incident, status='prepared-template-not-admitted',
            source_mode=source)
        (case/'mach4_case_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
        env = dict(common, ASTR_AIR5_SOURCE_MODE=source)
        (case/'environment.json').write_text(json.dumps(env, indent=2)+'\n')
        cases.append(dict(name=name, incident=incident, source_mode=source,
            admission='short-gate-review' if name=='precursor' else 'await-developed-precursor',
            input_sha256=hashlib.sha256(input_file.read_bytes()).hexdigest()))
    manifest = dict(schema='air5_mach4_controlled_sbli_v1', prepared_only=True,
        physical_acceptance=False, physics=physics, cases=cases,
        grid_upper_bounds=[63,127,15], grid_points=[64,128,16], np=2,
        deltat=2.5e-10, maxstep=1, tau=tau,
        inlet_status='quartic startup seed, not a developed inlet',
        prerequisite='ASTR precursor development and inlet extraction; no existing checkpoint approved',
        guard='GPU characteristic validation opt-in deliberately absent; no launch script supplied',
        comparison='same inlet and boundaries; inspect pre-interaction profiles separately')
    (destination/'campaign.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_campaign(args.destination), indent=2))
