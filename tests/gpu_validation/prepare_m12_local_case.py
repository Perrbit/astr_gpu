#!/usr/bin/env python3
"""Subset the read-only M12 resources for the approved local interface gate."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import h5py
import numpy as np


def replace_value(lines, section, value):
    candidates = [i for i, line in enumerate(lines)
                  if line.strip().startswith('# ' + section)]
    if len(candidates) != 1:
        raise ValueError(f'expected one input section: {section}')
    index = candidates[0] + 1
    while index < len(lines) and (not lines[index].strip() or
                                  lines[index].lstrip().startswith('#')):
        index += 1
    if index == len(lines):
        raise ValueError(f'missing value: {section}')
    lines[index] = value


def selected_nodes(grid, intervals=(64, 32, 24)):
    x = grid['x'][0, :]
    if x.shape != (2251,) or not np.all(np.diff(x) > 0):
        raise ValueError('unexpected source x coordinates')
    if intervals == (450, 130, 96):
        ix = np.arange(0, 2251, 5)
        iy = np.arange(0, 261, 2)
    elif intervals == (64, 32, 24):
        targets = np.linspace(x[0], x[-1], 65)
        ix = np.abs(x[:, None] - targets[None, :]).argmin(axis=0)
        iy = np.rint(np.linspace(0, 260, 33)).astype(int)
    else:
        raise ValueError('M12 supports the local interface or approved medium grid')
    if np.unique(ix).size != intervals[0]+1 or np.unique(iy).size != intervals[1]+1:
        raise ValueError('subset must contain distinct original nodes')
    blow_x = x[ix]
    if not (np.any((blow_x > 20) & (blow_x < 40)) and
            np.any((blow_x > 40) & (blow_x < 60))):
        raise ValueError('subset misses a nonzero wall-blowing segment')
    return ix, iy, blow_x


def prepare(source, destination, backend, intervals=(64, 32, 24)):
    source = source.resolve(strict=True)
    destination = destination.resolve()
    if destination.exists() or destination == source or source in destination.parents:
        raise ValueError('destination must be a new directory outside the read-only case')
    datin = source / 'datin'
    with h5py.File(datin / 'grid.2d', 'r') as grid:
        ix, iy, xline = selected_nodes(grid, intervals)
        if any(grid[key].shape != (261, 2251) for key in ('x', 'y', 'z')):
            raise ValueError('unexpected source grid shape')
    profile = (datin / 'inlet.prof').read_text(encoding='ascii').splitlines()
    if len(profile) != 265:
        raise ValueError('profile must have four headers and 261 node records')
    input_lines = (datin / 'input.3d').read_text(encoding='ascii').splitlines()
    replace_value(input_lines, 'im,jm,km', ','.join(map(str, intervals)))
    replace_value(input_lines, 'nondimen,diffterm',
                  't,t,f,f,f,f,f,f,' + ('t' if backend == 'gpu' else 'f'))
    controller = (datin / 'controller').read_text(encoding='ascii').splitlines()
    replace_value(controller, 'lwsequ,lwslic', 'f,f,f,f')
    # ASTR's inclusive nstep loop advances from 0 through maxstep.
    replace_value(controller, 'maxstep,feqchkpt', '9,1,9999,9999,1,9999')
    destination.mkdir(parents=True)
    for name in ('datin', 'outdat', 'checkpoint', 'testout', 'diagnostics'):
        (destination / name).mkdir()
    sources = {}
    for filename, keys in (('grid.2d', ('x', 'y', 'z')),
                           ('flowini2d.h5', ('ro', 'u1', 'u2', 'u3', 'p', 't'))):
        with h5py.File(datin / filename, 'r') as original, \
                h5py.File(destination / 'datin' / filename, 'x') as subset:
            for key in keys:
                if original[key].shape != (261, 2251):
                    raise ValueError(f'unexpected source field shape: {filename}/{key}')
                values = original[key][...][np.ix_(iy, ix)]
                if not np.isfinite(values).all():
                    raise ValueError(f'nonfinite source field: {filename}/{key}')
                subset.create_dataset(key, data=values)
        sources[filename] = hashlib.sha256((datin / filename).read_bytes()).hexdigest()
    (destination / 'datin' / 'inlet.prof').write_text(
        '\n'.join(profile[:4] + [profile[4 + j] for j in iy]) + '\n', encoding='ascii')
    for filename in ('wallbs.dat', 'slice.dat'):
        shutil.copyfile(datin / filename, destination / 'datin' / filename)
    (destination / 'datin' / 'input.dat').write_text(
        '\n'.join(input_lines) + '\n', encoding='ascii')
    (destination / 'datin' / 'controller').write_text(
        '\n'.join(controller) + '\n', encoding='ascii')
    (destination / 'datin' / 'input.output').write_text(
        "&output\n directory='outdat/new',\n"
        " host_budget_bytes=67108864,device_budget_bytes=67108864,\n"
        " buffer_bytes=1048576,device_reserve_bytes=1073741824\n/\n"
        "&checkpoint\n enabled=f\n/\n&volume\n enabled=f\n/\n"
        "&slices\n enabled=f\n/\n", encoding='ascii')
    provenance = {
        'purpose': 'local interface validation, not production DNS',
        'intervals': list(intervals), 'completed_steps': 10, 'deltat': 0.02,
        'x_selection': ('every fifth original node' if intervals[0] == 450 else
                        'nearest original nodes to 65 uniformly spaced physical targets'),
        'y_selection': ('every second original node' if intervals[1] == 130 else
                        '33 rounded uniformly spaced original node indices'),
        'x_indices': ix.tolist(), 'y_indices': iy.tolist(),
        'nonzero_blowing_x': xline[(xline > 20) & (xline < 60)].tolist(),
        'source_sha256': sources,
        'changes': ['subset dimensions', 'runtime usegpu', 'LF input text',
                    'ten-step local limit and per-step reporting',
                    'crash repair disabled for first-failure gate',
                    'field and checkpoint output disabled'],
        'unchanged': ['743e MP-LD', '643e diffusion', 'RK3', 'physical parameters',
                      'boundary types', 'wall blowing parameters', 'sponge extents',
                      'selected source grid and initial field values', 'profile headers'],
    }
    (destination / 'fixture.json').write_text(
        json.dumps(provenance, indent=2) + '\n', encoding='ascii')
    return provenance


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--backend', choices=('cpu', 'gpu'), required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.destination, args.backend), indent=2))
