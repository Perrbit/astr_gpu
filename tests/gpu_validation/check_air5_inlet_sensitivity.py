#!/usr/bin/env python3
"""Compare same-checkpoint inlet perturbations; no steady or physical pass."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from air5_radau_reference import Air5RadauReference
from air5_transport_reference import Air5TransportReference
from check_air5_c5_normal_shock import primitive_metrics
from check_air5_numq11_shock_tube import q_array
from check_air5_precursor_evolution import wall_response
from compare_q_validation_snapshots import read_q_snapshot
from run_air5_sbli_long import FIELDS, state_metrics


def compare(output):
    matrix = json.loads((output/'matrix.json').read_text())
    if matrix['status'] != 'completed-pending-comparison':
        raise ValueError('inlet matrix is incomplete')
    meta = json.loads((output/'control/gpu/mach4_case_metadata.json').read_text())
    thermo = Air5RadauReference(output/'control/mechanism.json')
    transport = Air5TransportReference(output/'control/mechanism.json')
    source_hash = matrix['source_checkpoint_sha256']
    uref, pref = meta['edge_velocity'], meta['pressure'][0]
    results, reference, common_time = [], None, None
    for entry in matrix['cases']:
        root = output/entry['case']
        contract = json.loads((root/'contract.json').read_text())
        if contract['files']['gpu/outdat/flowfield.h5'] != source_hash:
            raise ValueError('initial interior checkpoints differ')
        with h5py.File(root/'checkpoint_step001500.h5') as stream:
            fields = {key:stream[key][()] for key in FIELDS}
            time = float(stream['time'][()].item())
        if reference is None:
            reference, common_time = fields, time
        if time != common_time:
            raise ValueError('comparison phases differ')
        metrics = state_metrics(fields,thermo,uref,pref)
        stage_min, stage_max, closure, count = float('inf'), 0., 0., 0
        ev_bounds = [np.array([thermo.species_vibrational_energy(s,t) for s in range(5)])
                     for t in thermo.temperature_bounds]
        for label in ('post_chemistry','pre_rhs','post_update','post_transport'):
            for path in sorted((root/'gpu/validation').glob(f'air5.{label}.*.bin')):
                snapshot = read_q_snapshot(path)
                im,jm,km,hm,_ = snapshot.header
                q = q_array(snapshot)[hm:hm+im+1,hm:hm+jm+1,hm:hm+km+1]
                rho, species, temp, pressure, _ = primitive_metrics(q,thermo)
                if (not np.isfinite(q).all() or rho.min()<=0 or species.min()<0 or
                    temp.min()<meta['minimum_transport_temperature'] or
                    temp.max()>thermo.temperature_bounds[1] or
                    pressure.min()<thermo.pressure_bounds[0] or
                    pressure.max()>thermo.pressure_bounds[1] or
                    np.any(q[...,10]<species@ev_bounds[0]) or
                    np.any(q[...,10]>species@ev_bounds[1])):
                    raise ValueError(f'final-stage state gate failed: {path}')
                closure = max(closure,float(abs(species.sum(axis=-1)-rho).max()/rho.min()))
                stage_min,stage_max = min(stage_min,float(temp.min())),max(stage_max,float(temp.max()))
                count += 1
        if count != 18 or closure>2e-12:
            raise ValueError('incomplete stage snapshots or mass closure failure')
        stations = [1,2,4,8,16,32,47,62]
        walls = {str(n):wall_response(fields,meta['domain'],transport,meta['edge_density'],
                 uref,stations,stencil_points=n) for n in (3,7)}
        profile = np.loadtxt(root/'gpu/datin/air5_hbl_profile.dat')
        y = np.linspace(0.,meta['domain'][1],fields['ro'].shape[1])
        inlet_u = np.interp(y,profile[:,0],profile[:,2])
        inlet_t = np.interp(y,profile[:,0],profile[:,6])
        ys = profile[0,8:]
        if not np.all(profile[:,8:]==ys):
            raise ValueError('this inlet audit requires constant inlet composition')
        gas,cv = float(ys@thermo.gas_constant),float(ys@thermo.cv_tr)
        mach = inlet_u/np.sqrt((1.+gas/cv)*gas*inlet_t)
        inlet = dict(first_point_frozen_normal_mach=float(mach[1]),
            subsonic_interior_indices=np.flatnonzero((mach<1.)&(y>0.)).tolist(),
            delta99=float(np.interp(.99,inlet_u/uref,y)),
            sonic_height=float(np.interp(1.,mach,y)))
        differences = []
        for ix in stations:
            values = dict(ix=ix,x=ix*meta['domain'][0]/(fields['ro'].shape[2]-1))
            for key,scale in [('u1',uref),('u2',uref),('p',pref),('t',1500.),('tv',1500.)]:
                values[key] = float(abs(fields[key][:-1,:,ix]-reference[key][:-1,:,ix]).max()/scale)
            differences.append(values)
        results.append(dict(case=entry['case'],scale=entry['scale'],time=time,
            max_cfl=entry['status']['max_cfl'],metrics=metrics,wall=walls,inlet=inlet,
            station_max_normalized_difference=differences,
            final_stage=dict(snapshots=count,min_temperature=stage_min,
                             max_temperature=stage_max,max_mass_closure=closure)))
    result = dict(status='bounded-inlet-step-response-not-steady-validation',
        initial_checkpoint_sha256=source_hash,normalizations=dict(u1=uref,u2=uref,p=pref,t=1500.,tv=1500.),
        cases=results)
    (output/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    result=compare(parser.parse_args().output.resolve())
    print(json.dumps([{k:v for k,v in case.items() if k not in ('wall','station_max_normalized_difference')}
                      for case in result['cases']],indent=2))
