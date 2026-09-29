#!/usr/bin/env python3
"""Read-only precursor diagnostics; never grant automatic physical admission."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from air5_radau_reference import Air5RadauReference
from air5_transport_reference import Air5TransportReference
from check_air5_precursor_evolution import wall_response
from run_air5_sbli_long import FIELDS, digest, state_metrics


def thickness(y, rho_u, rho_u2, rho_edge, u_edge, velocity):
    """Compressible integral thicknesses with the top node as edge reference."""
    arrays = [np.asarray(v, dtype=float) for v in (y, rho_u, rho_u2, velocity)]
    y, rho_u, rho_u2, velocity = arrays
    if (y.ndim != 1 or len(y) < 3 or any(v.shape != y.shape for v in arrays)
            or not np.isfinite(arrays).all() or np.any(np.diff(y) <= 0)
            or not np.isfinite([rho_edge, u_edge]).all() or min(rho_edge, u_edge) <= 0):
        raise ValueError('invalid boundary-layer profile')
    flux = rho_u/(rho_edge*u_edge)
    ds = float(np.trapezoid(1-flux, y))
    theta = float(np.trapezoid(flux-rho_u2/(rho_edge*u_edge**2), y))
    # Use the outermost crossing so a local overshoot cannot give a false edge.
    below = np.flatnonzero(velocity < .99*u_edge)
    if not len(below) or below[-1] == len(y)-1:
        raise ValueError('no resolved outer 99-percent velocity crossing')
    j = below[-1]
    d99 = y[j]+(.99*u_edge-velocity[j])*(y[j+1]-y[j])/(velocity[j+1]-velocity[j])
    return dict(delta99=float(d99), displacement_thickness=ds,
                momentum_thickness=theta, shape_factor=ds/theta if theta > 0 else None)


def monitor_windows(profiles, walls, edges_us, uref):
    if not np.isfinite(profiles).all() or not np.isfinite(walls).all():
        raise ValueError('nonfinite monitor')
    result = []
    for ix in np.unique(profiles[:, 3]).astype(int):
        samples = profiles[profiles[:, 3] == ix]
        times = np.unique(samples[:, 1])
        frames = [samples[samples[:, 1] == t] for t in times]
        frames = [f[np.argsort(f[:, 4])] for f in frames]
        ny = len(frames[0])
        if any(len(f) != ny or not np.array_equal(f[:, 4], np.arange(ny)) for f in frames):
            raise ValueError('incomplete or duplicate profile monitor')
        cube = np.stack(frames)
        if not np.all(cube[:, :, 6] == cube[0, :, 6]):
            raise ValueError('monitor coordinates changed')
        wm = walls[walls[:, 2] == ix]
        wm = wm[np.argsort(wm[:, 1])]
        if not np.array_equal(wm[:, 1], times):
            raise ValueError('wall/profile sample times differ')
        if not np.allclose(np.diff(times), np.diff(times)[0], rtol=1e-8, atol=1e-18):
            raise ValueError('sample averages require uniform cadence')
        windows = []
        for lo, hi in zip(edges_us, edges_us[1:]):
            mask = (times >= lo*1e-6) & (times < hi*1e-6)
            expected = (hi-lo)*1e-6/np.diff(times)[0]
            if abs(mask.sum()-expected) > 1.01:
                raise ValueError('requested window is not fully sampled')
            p = cube[mask].mean(axis=0)
            w = wm[mask].mean(axis=0)
            windows.append(dict(interval_us=[lo, hi], samples=int(mask.sum()),
                mean_profile=p.tolist(), mean_tau=float(w[6]), mean_heat=float(w[7]),
                thickness=thickness(p[:, 6], (cube[mask, :, 8]*cube[mask, :, 9]).mean(axis=0),
                    (cube[mask, :, 8]*cube[mask, :, 9]**2).mean(axis=0),
                    p[-1, 8], p[-1, 9], p[:, 9])))
        changes = []
        for a, b in zip(windows, windows[1:]):
            pa, pb = np.asarray(a['mean_profile']), np.asarray(b['mean_profile'])
            changes.append(dict(intervals_us=[a['interval_us'], b['interval_us']],
                max_du_over_uinf=float(abs(pa[:, 9]-pb[:, 9]).max()/uref),
                max_dT=float(abs(pa[:, 12]-pb[:, 12]).max()),
                max_dTv=float(abs(pa[:, 13]-pb[:, 13]).max()),
                relative_to_later={k:abs(b[k]-a[k])/abs(b[k]) if b[k] != 0 else None
                                   for k in ('mean_tau', 'mean_heat')}))
        result.append(dict(ix=int(ix), windows=windows, changes=changes))
    return result


def analyze(case, mechanism):
    domain = np.loadtxt(case/'datin/air5_hbl_domain.dat', skiprows=1)
    inlet = np.loadtxt(case/'datin/air5_hbl_profile.dat')
    rhoinf, uinf, pinf = inlet[-1, 1], inlet[-1, 2], inlet[-1, 5]
    path = case/'outdat/flowfield.h5'
    with h5py.File(path) as h:
        fields = {k:h[k][()] for k in FIELDS}
        time = float(h['time'][()].item())
        step = int(h['nstep'][()].item())
    thermo = Air5RadauReference(mechanism)
    state = state_metrics(fields, thermo, uinf, pinf)
    y = np.linspace(0, domain[1], fields['ro'].shape[1])
    stations = []
    for ix in range(1, fields['ro'].shape[2]-1):
        f = {k:v[:-1, :, ix] for k,v in fields.items()}
        p = {k:v.mean(axis=0) for k,v in f.items()}
        stations.append(dict(ix=ix, x=domain[0]*ix/(fields['ro'].shape[2]-1),
            **thickness(y, (f['ro']*f['u1']).mean(axis=0),
                (f['ro']*f['u1']**2).mean(axis=0), p['ro'][-1], p['u1'][-1], p['u1']),
            outer_20_percent_range={k:float(np.ptp(v[int(.8*len(y)):])) for k,v in p.items()}))
    transport = Air5TransportReference(mechanism)
    walls = {str(n):wall_response(fields, domain, transport, rhoinf, uinf,
                range(1, fields['ro'].shape[2]-1), n) for n in (3,5,7)}
    monitors = monitor_windows(np.loadtxt(case/'monitor/air5_profiles.dat'),
        np.loadtxt(case/'monitor/air5_wall.dat'), [12.,16.,20.,24.,28.], uinf)
    return dict(status='diagnostics-complete-physical-admission-pending',
        case=str(case), checkpoint_sha256=digest(path), mechanism_sha256=digest(mechanism),
        checkpoint_step=step, checkpoint_time=time,
        flow_through_time=float(domain[0]/uinf), state=state,
        conventions=dict(edge='local top node; outer plateau still requires physical assessment',
            wall_heat='positive into gas; translation plus vibration; independent derivative',
            checkpoint='archived complete-RK state; retain legacy stored time label',
            monitor='fixed-z plane, uniform-time samples; not spanwise/turbulent statistics'),
        spanwise_max_range={k:float(np.ptp(v, axis=0).max()) for k,v in fields.items()},
        y=y.tolist(), profiles={k:v[:-1].mean(axis=0).tolist() for k,v in fields.items()},
        stations=stations, wall_by_stencil=walls, monitor_windows=monitors,
        unclosed=['stationarity', 'matched-time timestep sensitivity',
                  'wall-normal grid sensitivity', 'filter sensitivity', 'independent physical reference'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--mechanism', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.case.resolve(), args.mechanism.resolve())
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(result['status'])
