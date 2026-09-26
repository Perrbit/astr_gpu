#!/usr/bin/env python3
"""Check same-phase AIR5 conservative/carry checkpoints, not physical accuracy."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def read(path):
    with h5py.File(path, 'r') as data:
        if int(data['air5_compensation_version'][()].item()) != 1:
            raise ValueError('unsupported compensation schema')
        q = np.stack([data[f'acq{i:02}'][()] for i in range(1, 12)], axis=-1)
        carry = np.stack([data[f'acc{i:02}'][()] for i in range(1, 12)], axis=-1)
        phase = (int(data['nstep'][()].item()), float(data['time'][()].item()))
    if not np.isfinite(q).all() or not np.isfinite(carry).all():
        raise ValueError('nonfinite checkpoint state or carry')
    if np.any(q[..., 0] <= 0) or np.any(q[..., 5:10] < 0):
        raise ValueError('nonpositive density or negative species')
    ysum = np.zeros(q.shape[:-1])
    for i in range(5, 10):
        ysum += q[..., i] / q[..., 0]
    residual = float(np.max(np.abs(ysum - 1)))
    if residual > 128 * np.finfo(float).eps:
        raise ValueError(f'sequential mass fraction closure exceeds 128 epsilon: {residual:.17e}')
    return q, carry, phase, residual


def compare(reference, candidate, exact=False):
    q0, c0, phase0, r0 = read(reference)
    q1, c1, phase1, r1 = read(candidate)
    if phase0 != phase1 or q0.shape != q1.shape:
        raise ValueError('checkpoint phase or shape differs')
    maxima = np.max(np.abs(q1 - q0), axis=tuple(range(q0.ndim - 1)))
    if exact:
        passed = (np.array_equal(q0.view(np.uint64), q1.view(np.uint64)) and
                  np.array_equal(c0.view(np.uint64), c1.view(np.uint64)))
    else:
        passed = bool(np.all(np.abs(q1-q0) <= 1e-9 + 1e-10*np.abs(q0)))
    result = dict(passed=passed, exact=exact, phase=phase0,
                  conservative_max_abs=maxima.tolist(),
                  carry_max_abs=float(np.max(np.abs(c1-c0))),
                  closure_max=[r0, r1], reference=str(reference), candidate=str(candidate))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--exact', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = compare(args.reference, args.candidate, args.exact)
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(text)
    print(text, end='')
    if not result['passed']:
        raise SystemExit('compensation checkpoint comparison failed')
