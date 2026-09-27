#!/usr/bin/env python3
"""Check the fixed 16^3 free-stream CFL regression matrix, including locations."""
import argparse
import json
from pathlib import Path

import numpy as np


def records(path):
    text = path.read_text()
    if 'The job is done!' not in text:
        raise ValueError(f'incomplete run: {path}')
    selected = [line for line in text.splitlines() if line.startswith(
        ('ASTR_CFL directional', 'ASTR_CFL maximum', 'ASTR_CFL_Y'))]
    values = np.array([list(map(float, line.split('=')[1].split()))
                       for line in selected if line.startswith('ASTR_CFL directional')])
    locations = [line for line in selected if line.startswith('ASTR_CFL maximum')]
    if values.shape != (2, 5) or len(locations) != 8:
        raise ValueError(f'missing two-step records: {path}')
    return selected, values, locations


def check(root):
    groups = [('curve', ['curve_np1', 'curve_x', 'curve_y']),
              ('cartesian', [f'cart_zero_{sign}_{name}' for sign in (1, -1)
                             for name in ('np1', 'x', 'y')])]
    result = dict(passed=True, groups={})
    for group, names in groups:
        baseline = None
        count = 0
        for name in names:
            for backend in ('cpu', 'gpu'):
                case = root/name
                for filename in ('flowfield_compare.txt', 'cpu_freestream_drift.txt',
                                 'gpu_freestream_drift.txt'):
                    if 'status: pass' not in (case/filename).read_text():
                        raise ValueError(f'flow gate failed: {case}/{filename}')
                selected, values, locations = records(case/backend/(backend+'.log'))
                if baseline is None:
                    baseline = selected
                if selected != baseline:
                    raise ValueError(f'CFL values, locations or profiles differ: {case}/{backend}')
                if group == 'cartesian':
                    directional = (10+np.abs([0.7, -0.2, 0.1]))*16/(2*np.pi)*1e-4
                    expected = np.r_[directional, directional.sum(), directional.sum()]
                    np.testing.assert_allclose(values, np.broadcast_to(expected, values.shape),
                                               rtol=5e-13, atol=1e-14)
                count += 1
        result['groups'][group] = dict(runs=count, exact_log_agreement=True,
                                      last_maxima=values[-1].tolist(), locations=locations[-4:])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    text = json.dumps(check(args.root), indent=2)+'\n'
    if args.output:
        args.output.write_text(text)
    print(text, end='')
