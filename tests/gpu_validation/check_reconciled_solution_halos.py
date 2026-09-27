#!/usr/bin/env python3
"""Require saved RHS-input face halos to equal the final donor owned state."""
import argparse
import json
from pathlib import Path

import numpy as np

from check_air5_c5_hbl import _load_parallel_layout, _phase_files
from compare_q_validation_snapshots import read_q_snapshot


def check(case, step, stage, periodic_axes=(2,), prefix='air5'):
    layouts = _load_parallel_layout(case/'datin/parallel.info')
    files = _phase_files(case/'validation'/prefix, 'pre_rhs', stage, step)
    if files.keys() != layouts.keys():
        raise ValueError('missing rank snapshots')
    arrays, dims, origins, widths = {}, {}, {}, {}
    for rank, layout in layouts.items():
        snapshot = read_q_snapshot(files[rank])
        im, jm, km, hm, nq = snapshot.header
        dims[rank] = np.array([im, jm, km])
        origins[rank] = np.array([layout.i0, layout.j0, layout.k0])
        widths[rank] = hm
        arrays[rank] = snapshot.values.reshape(tuple(dims[rank]+2*hm+1)+(nq,), order='F')
    extent = np.max([origins[r]+dims[r] for r in layouts], axis=0)
    rows = []
    for rank in layouts:
        for axis in range(3):
            for side in (-1, 1):
                location = origins[rank].copy()
                location[axis] += side*dims[rank][axis]
                if axis in periodic_axes:
                    location[axis] %= extent[axis]
                donors = [r for r in layouts if np.array_equal(origins[r], location)]
                if not donors:
                    if (side < 0 and origins[rank][axis] == 0 or
                        side > 0 and origins[rank][axis]+dims[rank][axis] == extent[axis]):
                        continue
                    raise ValueError('unsupported unequal decomposition or missing donor')
                donor, = donors
                hm, hd = widths[rank], widths[donor]
                target = [slice(hm, hm+d+1) for d in dims[rank]]
                source = [slice(hd, hd+d+1) for d in dims[donor]]
                target[axis] = slice(0, hm) if side < 0 else slice(hm+dims[rank][axis]+1, 2*hm+dims[rank][axis]+1)
                source[axis] = slice(hd+dims[donor][axis]-hm, hd+dims[donor][axis]) if side < 0 else slice(hd+1, hd+hm+1)
                a, b = arrays[rank][tuple(target)], arrays[donor][tuple(source)]
                if a.shape != b.shape:
                    raise ValueError('incompatible halo slab shapes')
                rows.append(dict(rank=rank, donor=donor, axis=axis+1, side=side,
                                 equal=bool(np.array_equal(a, b)),
                                 finite=bool(np.isfinite(a).all() and np.isfinite(b).all()),
                                 max_abs=float(np.max(abs(a-b)))))
    return dict(step=step, stage=stage, faces=rows,
                passed=bool(rows) and all(r['equal'] and r['finite'] for r in rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', type=Path)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--stage', type=int, default=3)
    parser.add_argument('--periodic-axes', default='z')
    parser.add_argument('--prefix', default='air5')
    args = parser.parse_args()
    result = check(args.case, args.step, args.stage,
                   tuple('xyz'.index(a) for a in args.periodic_axes), args.prefix)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
