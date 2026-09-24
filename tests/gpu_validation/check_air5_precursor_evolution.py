#!/usr/bin/env python3
"""Evaluate ASTR precursor checkpoints; no flow integration or automatic steady pass."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from air5_hbl_diagnostics import finite_difference_weights
from air5_radau_reference import Air5RadauReference
from air5_transport_reference import Air5TransportReference
from run_air5_sbli_long import FIELDS, precursor_profiles, state_metrics


def review_final_states(output):
    """Check the archived last update separately from the complete-RK checkpoint."""
    from check_air5_c5_normal_shock import primitive_metrics
    from check_air5_numq11_shock_tube import q_array
    from compare_q_validation_snapshots import read_q_snapshot

    contract = json.loads((output/'contract.json').read_text())
    meta = json.loads((output/'gpu/mach4_case_metadata.json').read_text())
    thermo = Air5RadauReference(output/'mechanism.json')
    step = contract['start_step']+contract['updates']-1
    ranks = int(np.prod(contract['topology']))
    ev = [np.array([thermo.species_vibrational_energy(s,t) for s in range(5)])
          for t in thermo.temperature_bounds]
    result = dict(status='final-stage-state-pass-not-physical-pass',snapshots=0,
        min_temperature=float('inf'),max_temperature=0.,min_pressure=float('inf'),
        max_pressure=0.,max_mass_closure=0.)
    for label,stages in [('post_chemistry',2),('pre_rhs',3),('post_update',3),('post_transport',1)]:
        for stage in range(1,stages+1):
            for rank in range(ranks):
                path = output/'gpu/validation'/f'air5.{label}.step{step:08d}.rk{stage:02d}.rank{rank:08d}.bin'
                snapshot = read_q_snapshot(path)
                im,jm,km,hm,_ = snapshot.header
                q = q_array(snapshot)[hm:hm+im+1,hm:hm+jm+1,hm:hm+km+1]
                rho,ys,t,p,_ = primitive_metrics(q,thermo)
                if (not np.isfinite(q).all() or rho.min()<=0 or ys.min()<0 or
                    t.min()<meta['minimum_transport_temperature'] or
                    t.max()>thermo.temperature_bounds[1] or
                    p.min()<thermo.pressure_bounds[0] or p.max()>thermo.pressure_bounds[1] or
                    np.any(q[...,10]<ys@ev[0]) or np.any(q[...,10]>ys@ev[1])):
                    raise ValueError(f'final-stage model/positivity gate failed: {path}')
                closure = float(np.max(abs(ys.sum(axis=-1)-rho)/rho))
                if closure>2e-12:
                    raise ValueError(f'final-stage mass closure failed: {path}')
                result['snapshots'] += 1
                for key,value in [('temperature',t),('pressure',p)]:
                    result['min_'+key] = min(result['min_'+key],float(value.min()))
                    result['max_'+key] = max(result['max_'+key],float(value.max()))
                result['max_mass_closure'] = max(result['max_mass_closure'],closure)
    (output/'field_review.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def wall_response(fields, domain, transport, rho_inf, u_inf, stations, stencil_points=7):
    """Wall derivative diagnostic, positive heat flux into the fluid.

    This samples continuous wall observables, not the solver's discrete RHS.
    The noncatalytic wall imposes zero normal species enthalpy flux.
    """
    nz, ny, nx = fields['ro'].shape
    if stencil_points not in (3,5,7) or ny < stencil_points or nz < 2 or nx < 2:
        raise ValueError('wall response requires a 3/5/7-point stencil and a periodic z endpoint')
    if max(np.abs(fields[key][:,0,:]).max() for key in ('u1','u2','u3')) > 2e-12*u_inf:
        raise ValueError('wall response requires no slip')
    y = np.linspace(0.,domain[1],ny)
    weights = finite_difference_weights(y[:stencil_points],y[0])
    gradients = {key:np.einsum('j,kji->ki',weights,fields[key][:-1,:stencil_points,:])
                 for key in ('u1','t','tv')}
    result = []
    for ix in stations:
        values = []
        wall_units = []
        for iz in range(nz-1):
            ys = np.array([fields[f'sp{s:03d}'][iz,0,ix] for s in range(1,6)])
            mu, kt, kv, _ = transport.transport_properties(fields['t'][iz,0,ix],
                fields['tv'][iz,0,ix],fields['p'][iz,0,ix],ys)
            rho_w = fields['ro'][iz,0,ix]
            tau_w = mu*gradients['u1'][iz,ix]
            u_tau = np.sqrt(abs(tau_w)/rho_w)
            wall_units.append((tau_w,u_tau,(y[1]-y[0])*rho_w*u_tau/mu,
                               mu/rho_w))
            values.append((2*mu*gradients['u1'][iz,ix]/(rho_inf*u_inf**2),
                -kt*gradients['t'][iz,ix],-kv*gradients['tv'][iz,ix]))
        cf, heat_tr, heat_v = np.mean(values,axis=0)
        units = np.asarray(wall_units)
        result.append(dict(ix=ix, skin_friction=float(cf), heat_flux_tr=float(heat_tr),
            heat_flux_v=float(heat_v), heat_flux_total=float(heat_tr+heat_v),
            tau_wall=float(units[:,0].mean()),u_tau=float(units[:,1].mean()),
            first_node_distance=float(y[1]-y[0]),yplus_first=float(units[:,2].mean()),
            yplus_first_max=float(units[:,2].max()),nu_wall=float(units[:,3].mean())))
    return result


def summarize(output):
    meta = json.loads((output/'gpu/mach4_case_metadata.json').read_text())
    thermo = Air5RadauReference(output/'mechanism.json')
    transport = Air5TransportReference(output/'mechanism.json')
    history = []
    for path in sorted(output.glob('checkpoint_step*.h5')):
        with h5py.File(path) as stream:
            fields = {key:stream[key][()] for key in FIELDS}
            step,time = int(stream['nstep'][()].item()),float(stream['time'][()].item())
        metrics = state_metrics(fields,thermo,meta['edge_velocity'],meta['pressure'][0])
        if metrics['min_temperature'] < meta['minimum_transport_temperature']:
            raise ValueError('checkpoint left the selected transport-fit temperature range')
        profiles = precursor_profiles(fields,meta['domain'])
        stations = [s['ix'] for s in profiles['stations']]
        walls = wall_response(fields,meta['domain'],transport,meta['edge_density'],
                              meta['edge_velocity'],stations)
        three_point = wall_response(fields,meta['domain'],transport,meta['edge_density'],
                                    meta['edge_velocity'],stations,stencil_points=3)
        samples = []
        for profile,wall in zip(profiles['stations'],walls):
            samples.append({**{k:v for k,v in profile.items() if k!='fields'},**wall})
        history.append(dict(step=step,time=time,metrics=metrics,stations=samples,
                            three_point_wall_diagnostic=three_point))
    if len(history) < 2:
        raise ValueError('at least two archived checkpoints are required')
    if any(b['time']<=a['time'] for a,b in zip(history,history[1:])):
        raise ValueError('checkpoint time must increase')
    recent = []
    for a,b in zip(history[-2]['stations'],history[-1]['stations']):
        recent.append(dict(ix=b['ix'], **{key:abs(b[key]-a[key])/max(abs(a[key]),abs(b[key]),1e-300)
            for key in ('displacement_thickness','wall_pressure','skin_friction','heat_flux_total')}))
    result = dict(status='diagnostics-only-no-steady-or-physical-pass',
        heat_flux_convention='W/m^2 positive into fluid; independent seven-point wall derivative',
        skin_friction_definition='2*mu_w*du_dy_w/(rho_inf*U_inf^2); no-slip plane wall',
        history=history, recent_time_window=history[-1]['time']-history[-2]['time'],
        recent_relative_changes=recent,
        flow_through_times=history[-1]['time']/meta['flow_through_time'])
    (output/'precursor_evolution.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    result = summarize(parser.parse_args().output.resolve())
    print(json.dumps({k:v for k,v in result.items() if k!='history'},indent=2))
