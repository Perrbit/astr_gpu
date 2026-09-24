#!/usr/bin/env python3
"""ASTR-only matched-time wall-normal mesh and timestep diagnostic matrix."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from air5_transport_reference import Air5TransportReference
from check_air5_precursor_evolution import review_final_states, summarize, wall_response
from run_air5_sbli_long import FIELDS, digest, run


def matched_comparison(left, right):
    """Evaluate on the coarser grid; interpolation is diagnostic, not a truth oracle."""
    for name in ('astr','mechanism.json','gpu/datin/air5_hbl_profile.dat','gpu/datin/air5_hbl_domain.dat'):
        if digest(left/name)!=digest(right/name):
            raise ValueError(f'resolution comparison changed a fixed input: {name}')
    a = summarize(left)['history'][-1]
    b = summarize(right)['history'][-1]
    if not np.isclose(a['time'],b['time'],rtol=1e-10,atol=1e-18):
        raise ValueError('physical times differ')
    def load(root,step):
        with h5py.File(root/f'checkpoint_step{step:06d}.h5') as f:
            return {key:f[key][()] for key in FIELDS}
    fa,fb = load(left,a['step']),load(right,b['step'])
    ma = json.loads((left/'gpu/mach4_case_metadata.json').read_text())
    mb = json.loads((right/'gpu/mach4_case_metadata.json').read_text())
    if ma['domain']!=mb['domain'] or fa['ro'].shape[::2]!=fb['ro'].shape[::2]:
        raise ValueError('comparison requires identical domain and x/z mesh')
    ya,yb = [np.linspace(0.,ma['domain'][1],f['ro'].shape[1]) for f in (fa,fb)]
    if len(ya)>len(yb):
        raise ValueError('left must be the coarser or same grid')
    stations = [1,2,4,8,16,32,47,62]
    transport = Air5TransportReference(right/'mechanism.json')
    walls,wall_units = {},{}
    for n in (3,7):
        all_walls = [wall_response(f,ma['domain'],transport,ma['edge_density'],
                    ma['edge_velocity'],range(1,f['ro'].shape[2]-1),n) for f in (fa,fb)]
        wall_units[str(n)] = {name:dict(min_station_mean=min(x['yplus_first'] for x in w),
            max_point=max(x['yplus_first_max'] for x in w))
            for name,w in zip(('left','right'),all_walls)}
        wa,wb = [[x for x in w if x['ix'] in stations] for w in all_walls]
        walls[str(n)] = [dict(ix=x['ix'],left=x,right=y,
            relative_to_right={k:abs(x[k]-y[k])/max(abs(y[k]),1e-300)
                for k in ('skin_friction','heat_flux_tr','heat_flux_v','heat_flux_total')})
            for x,y in zip(wa,wb)]
    profile_errors=[]
    for ix in stations:
        errors={}
        for key in FIELDS:
            av,bv = fa[key][:-1,:,ix].mean(axis=0),fb[key][:-1,:,ix].mean(axis=0)
            errors[key]=float(abs(av-np.interp(ya,yb,bv)).max())
        profile_errors.append(dict(ix=ix,max_abs=errors))
    global_difference = ({key:float(abs(fa[key]-fb[key]).max()) for key in FIELDS}
                         if fa['ro'].shape==fb['ro'].shape else None)
    return dict(left=str(left),right=str(right),time=a['time'],wall=walls,
        wall_unit_ranges=wall_units,same_grid_global_max_abs=global_difference,
        profile_max_abs_on_left_grid=profile_errors,
        thickness_and_pressure=[dict(ix=x['ix'],relative_to_right={k:abs(x[k]-y[k])/abs(y[k])
            for k in ('displacement_thickness','wall_pressure')})
            for x,y in zip(a['stations'],b['stations'])])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--pilot',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--review-only',action='store_true',help='Review a completed matrix without running ASTR')
    args=parser.parse_args()
    base,pilot,out = (p.resolve() for p in (args.baseline,args.pilot,args.output))
    if args.review_only:
        result=json.loads((out/'matrix.json').read_text())
        if result['status']!='completed-diagnostics-not-physical-pass':
            raise ValueError('cannot review an incomplete matrix')
        comparisons=[matched_comparison(Path(c['left']),Path(c['right'])) for c in result['comparisons']]
        (out/'extended_comparison.json').write_text(json.dumps(dict(
            status='diagnostics-not-physical-pass',comparisons=comparisons),indent=2,allow_nan=False)+'\n')
        return
    out.mkdir(parents=True,exist_ok=False)
    executable=base/'astr'
    if digest(executable)!=digest(pilot/'astr'):
        raise ValueError('pilot and baseline executables differ')
    review_final_states(pilot)
    review_final_states(base)
    result=dict(status='running-diagnostic-matrix',cases=[],comparisons=[])
    def save():
        (out/'matrix.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    save()
    for name,ny,dt,updates in [('y256_dt2ns',256,2e-9,1001),
                              ('y512_dt2ns',512,2e-9,1001),
                              ('y512_dt1ns',512,1e-9,2001)]:
        destination=out/name
        try:
            run(SimpleNamespace(output=destination,case='mach4-precursor',grid=f'63,{ny-1},7',
                baseline=None,executable=executable,updates=updates,dt=dt,
                checkpoint=(updates-1)//4,timeout=2400.,inlet_thickness_scale=1.))
            fields=review_final_states(destination)
            evolution=summarize(destination)
            if not np.isclose(evolution['history'][-1]['time'],2e-6,rtol=1e-10,atol=1e-18):
                raise ValueError('target checkpoint time was not reached')
            result['cases'].append(dict(name=name,status=json.loads((destination/'status.json').read_text()),
                                        final_stage=fields))
            previous=base if ny==256 else out/('y256_dt2ns' if dt==2e-9 else 'y512_dt2ns')
            result['comparisons'].append(matched_comparison(previous,destination))
            save()
        except BaseException as error:
            result.update(status='stopped',failed_case=name,reason=str(error))
            save()
            raise
    result['status']='completed-diagnostics-not-physical-pass'
    save()


if __name__=='__main__':
    main()
