#!/usr/bin/env python3
"""Equal-duration block statistics of fixed probes and planar wall diagnostics."""
import argparse
import json
from pathlib import Path

import numpy as np

VARIABLES = ('rho', 'u', 'v', 'w', 'T', 'Tv', 'p', 'Y_N2', 'Y_O2', 'Y_N', 'Y_O', 'Y_NO')


def load_series(path, key_columns, value_column):
    rows = np.loadtxt(path, ndmin=2)
    if not np.isfinite(rows).all():
        raise ValueError('nonfinite monitor')
    times = np.unique(rows[:, 1])
    if len(times) < 4:
        raise ValueError('need at least four samples for two blocks')
    spacing = np.diff(times)
    if not np.allclose(spacing, spacing[0], rtol=1e-8, atol=0):
        raise ValueError('nonuniform sampling requires time-weighted statistics')
    frames = [rows[rows[:, 1] == t] for t in times]
    keys = frames[0][:, key_columns]
    if len(np.unique(keys, axis=0)) != len(keys):
        raise ValueError('duplicate spatial samples')
    if any(f.shape != frames[0].shape or not np.array_equal(f[:, key_columns], keys) for f in frames):
        raise ValueError('changing probe layout')
    coordinates = frames[0][:, value_column-3:value_column]
    if any(not np.array_equal(f[:, value_column-3:value_column], coordinates) for f in frames):
        raise ValueError('moving probes require a separate interpretation')
    return times, keys, np.stack([f[:, value_column:] for f in frames])


def block_statistics(values):
    if values.shape[0] < 2 or values.shape[-1] != 12 or np.any(values[..., 0] <= 0):
        raise ValueError('invalid primitive samples')
    mean = values.mean(axis=0)
    rms = np.sqrt(np.mean((values-mean)**2, axis=0))
    rho = values[..., :1]
    velocity = values[..., 1:4]
    favre = np.sum(rho*velocity, axis=0)/np.sum(rho, axis=0)
    favre_rms = np.sqrt(np.sum(rho*(velocity-favre)**2, axis=0)/np.sum(rho, axis=0))
    return dict(mean=mean.tolist(), rms=rms.tolist(), favre_velocity=favre.tolist(),
                favre_velocity_rms=favre_rms.tolist())


def separated_intervals(x, shear):
    """Return all negative-shear intervals; None denotes a truncated boundary."""
    if np.any(np.diff(x) <= 0):
        raise ValueError('wall x must increase')
    intervals = []
    indices = np.flatnonzero(shear < 0)
    if not len(indices):
        return intervals
    def crossing(i, j):
        return float(x[i]-shear[i]*(x[j]-x[i])/(shear[j]-shear[i]))
    for group in np.split(indices, np.flatnonzero(np.diff(indices)>1)+1):
        a, b = int(group[0]), int(group[-1])
        start = crossing(a-1, a) if a else None
        end = crossing(b, b+1) if b+1 < len(x) else None
        intervals.append(dict(start=start, end=end,
            length=end-start if start is not None and end is not None else None))
    return intervals


def analyze_probes(path, window_samples):
    times, keys, data = load_series(path, [2], 6)
    if window_samples < 2 or len(times) < 2*window_samples:
        raise ValueError('two complete probe windows required')
    a = block_statistics(data[-2*window_samples:-window_samples])
    b = block_statistics(data[-window_samples:])
    return dict(scope='fixed probes, not a statistical-stationarity admission',
        variables=VARIABLES, keys=keys.tolist(), window_samples=window_samples,
        previous=a, recent=b,
        previous_time_range=times[-2*window_samples:-window_samples][[0,-1]].tolist(),
        recent_time_range=times[-window_samples:][[0,-1]].tolist(),
        block_mean_change=(np.array(b['mean'])-np.array(a['mean'])).tolist())


def analyze(directory, window_samples):
    probes = directory/'air5_probes.dat'
    if not (directory/'air5_profiles.dat').exists():
        return analyze_probes(probes, window_samples)
    times, keys, data = load_series(directory/'air5_profiles.dat', [2, 3, 4], 8)
    wt, wk, wall = load_series(directory/'air5_wall.dat', [2], 6)
    if not np.array_equal(times, wt) or len(times) < 2*window_samples or window_samples < 2:
        raise ValueError('matching wall/profile series and two complete windows required')
    recent = data[-window_samples:]
    previous = data[-2*window_samples:-window_samples]
    a, b = block_statistics(previous), block_statistics(recent)
    wall_rows = np.loadtxt(directory/'air5_wall.dat', ndmin=2)
    x = wall_rows[wall_rows[:,1] == wt[0], 3]
    profile_rows = np.loadtxt(directory/'air5_profiles.dat', ndmin=2)
    initial = profile_rows[profile_rows[:,1] == times[0]]
    massflow = []
    for station in np.unique(keys[:,0]):
        mask = keys[:,0] == station
        y = initial[mask,6]
        if np.any(np.diff(y) <= 0):
            raise ValueError('profile y must increase')
        flux = np.trapezoid(data[:,mask,0]*data[:,mask,1], y, axis=1)
        massflow.append(dict(station=int(station), time=times.tolist(),
                             mass_flux_per_unit_span=flux.tolist()))
    separation = [dict(time=float(t), intervals=separated_intervals(x, f[:,0]))
                  for t, f in zip(wt, wall)]
    result = dict(scope='fixed-plane diagnostics, not turbulent or statistical-stationarity admission',
        variables=VARIABLES, keys=keys.tolist(), window_samples=window_samples,
        window_duration=float(window_samples*(times[1]-times[0])),
        previous=a, recent=b, block_mean_change=(np.array(b['mean'])-np.array(a['mean'])).tolist(),
        wall_variables=['tau_xy', 'q_into_gas', 'p'], wall_indices=wk.tolist(),
        wall_previous_mean=wall[-2*window_samples:-window_samples].mean(axis=0).tolist(),
        wall_recent_mean=wall[-window_samples:].mean(axis=0).tolist(),
        wall_recent_rms=wall[-window_samples:].std(axis=0).tolist(), separation=separation,
        massflow=massflow,
        warning='No confidence interval without temporal-correlation assessment; no automatic pass threshold.')
    if probes.exists():
        result['configured_probes'] = analyze_probes(probes, window_samples)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--window-samples', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.write_text(json.dumps(analyze(args.directory, args.window_samples), indent=2, allow_nan=False)+'\n')
