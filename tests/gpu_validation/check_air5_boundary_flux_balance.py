#!/usr/bin/env python3
"""Check the final convective flux divergence, not total reacting-flow conservation."""
import argparse
import json
from pathlib import Path

import numpy as np

from check_air5_c5_frozen_transport import read_rhs_snapshot


def balance(rhs, faces, topology, periodic):
    if len(rhs) != int(np.prod(topology)) or len(faces) != len(rhs):
        raise ValueError('incomplete rank set')
    volume = np.zeros(11)
    boundary = np.zeros(11)
    scale = np.zeros(11)
    pairs = 0
    for rank, field in enumerate(rhs):
        coords = [rank % topology[0], (rank//topology[0]) % topology[1], rank//(topology[0]*topology[1])]
        dims = np.array(field.shape[:3])-1
        if field.shape[-1] != 11 or not np.isfinite(field).all():
            raise ValueError('invalid convective RHS')
        # Shared end nodes belong to the next rank. Physical end nodes do not advance.
        owned = [slice(int(not periodic[a] and coords[a] == 0), int(dims[a])) for a in range(3)]
        active = field[tuple(owned)]
        volume += active.sum(axis=(0, 1, 2))
        scale += abs(active).sum(axis=(0, 1, 2))
        for axis in range(3):
            plane = faces[rank][axis]
            transverse = [a for a in range(3) if a != axis]
            expected = tuple(int(dims[a]+1) for a in transverse)+(12, 4)
            if plane.shape != expected or not np.isfinite(plane).all():
                raise ValueError('invalid face payload')
            selection = tuple(owned[a] for a in transverse)+(slice(0, 11),)
            if not periodic[axis]:
                if coords[axis] == 0:
                    term = plane[selection+(0,)]
                    boundary += term.sum(axis=(0, 1))
                    scale += abs(term).sum(axis=(0, 1))
                if coords[axis] == topology[axis]-1:
                    term = plane[selection+(1,)]
                    boundary -= term.sum(axis=(0, 1))
                    scale += abs(term).sum(axis=(0, 1))
            for shift, sent, received in ((-1, 0, 2), (1, 1, 3)):
                neighbor = coords.copy()
                neighbor[axis] += shift
                if not 0 <= neighbor[axis] < topology[axis]:
                    if not periodic[axis]:
                        continue
                    neighbor[axis] %= topology[axis]
                other = neighbor[0]+topology[0]*(neighbor[1]+topology[1]*neighbor[2])
                if not np.array_equal(plane[..., sent].copy().view(np.uint64),
                                      faces[other][axis][..., received].copy().view(np.uint64)):
                    raise ValueError('shared final face differs bitwise')
                pairs += 1
    residual = abs(volume-boundary)/np.maximum(scale, 1.)
    if np.max(residual) > 2e-11:
        raise ValueError(f'boundary flux divergence mismatch: {residual.tolist()}')
    return dict(max_scaled_residual=float(residual.max()), scaled_residual=residual.tolist(),
                rhs_sum=volume.tolist(), net_boundary_flux=boundary.tolist(), plane_pairs=pairs)


def run(root):
    contract = json.loads((root/'contract.json').read_text())
    result = json.loads((root/'result.json').read_text())
    if result['status'] != 'bounded-replay-completed-not-physical-pass':
        raise ValueError('replay did not complete')
    case = root/contract['backend']
    lines = (case/'datin/input.air5_c4').read_text().splitlines()
    marker = next(i for i, line in enumerate(lines) if 'lihomo,ljhomo,lkhomo' in line)
    tokens = [v.strip().lower() for v in lines[marker+1].split(',')]
    if len(tokens) != 3 or any(v not in ('t', 'f') for v in tokens):
        raise ValueError('invalid periodicity flags')
    periodic = tuple(v == 't' for v in tokens)
    topology = tuple(map(int, contract['topology'].split(',')))
    entries = []
    for step in sorted({contract['snapshot_step'], contract.get('snapshot_step_secondary')} - {None}):
        for stage in range(1, 4):
            rhs, faces = [], []
            for rank in range(int(np.prod(topology))):
                suffix = f'.step{step:08d}.rk{stage:02d}.rank{rank:08d}.bin'
                snap = read_rhs_snapshot(case/'validation'/('air5.conv'+suffix))
                im, jm, km, nq = snap.header
                rhs.append(snap.values.reshape(im+1, jm+1, km+1, nq, order='F'))
                rank_faces = []
                for axis in range(1, 4):
                    with (case/'validation'/(f'air5.shared_faces.axis{axis}'+suffix)).open('rb') as stream:
                        header = np.fromfile(stream, np.int32, 7)
                        payload = np.fromfile(stream, np.float64)
                    if len(header) != 7 or tuple(header[:4]) != (step, stage, rank, axis) or header[6] != 12:
                        raise ValueError('invalid face header')
                    rank_faces.append(payload.reshape(*header[4:], 4, order='F'))
                faces.append(rank_faces)
            entries.append(dict(step=step, stage=stage, **balance(rhs, faces, topology, periodic)))
    if not entries:
        raise ValueError('no snapshot stages')
    report = dict(status='convective-boundary-flux-balance-pass', stages=entries,
                  scope='final convection only; excludes source, diffusion and boundary-state updates')
    (root/'boundary_flux_balance.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'], stages=len(entries),
                         max_scaled_residual=max(e['max_scaled_residual'] for e in entries),
                         plane_pairs=sum(e['plane_pairs'] for e in entries))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root)
