#!/usr/bin/env python3
"""Measure mass-invariant drift in saved ASTR stages; never integrate a state."""
import argparse
import json
from pathlib import Path

import numpy as np

from check_air5_c5_hbl import _active_array


def metrics(q, node):
    if q.ndim != 4 or q.shape[-1] != 11 or not np.isfinite(q).all():
        raise ValueError('expected finite physical-node q with 11 components')
    if len(node) != 3 or any(i < 0 or i >= n for i, n in zip(node, q.shape[:3])):
        raise ValueError('diagnostic node outside physical-node extent')
    if np.any(q[..., 0] <= 0):
        raise ValueError('nonpositive density')
    y = q[..., 5:10] / q[..., :1]
    sequential = np.zeros(q.shape[:3])
    for s in range(5):
        sequential += y[..., s]
    error = sequential - 1.0
    q_extended = q.astype(np.longdouble)
    residual = q_extended[..., 5:10].sum(axis=-1) - q_extended[..., 0]
    relative = residual / q_extended[..., 0]
    ijk = tuple(int(v) for v in np.unravel_index(np.argmax(abs(error)), error.shape))
    point = tuple(node)
    return dict(min_species_density=float(q[..., 5:10].min()),
                transport_sum_tolerance=128 * np.finfo(float).eps,
                transport_sum_violations=int(np.count_nonzero(abs(error) > 128*np.finfo(float).eps)),
                max_sum_error=float(error[ijk]), max_sum_error_ijk=ijk,
                max_extended_relative_residual=float(abs(relative).max()),
                node_q=q[point].tolist(), node_sum_error=float(error[point]),
                node_extended_mass_residual=float(residual[point]),
                node_extended_relative_residual=float(relative[point]))


def run(case, rank, node, steps):
    if rank < 0 or not steps or any(step < 0 for step in steps):
        raise ValueError('require nonnegative rank and snapshot steps')
    entries = []
    phases = [('pre_chemistry', 1), ('post_chemistry', 1)]
    phases += [(label, stage) for stage in range(1, 4)
               for label in ('pre_rhs', 'post_update')]
    phases += [('post_transport', 1), ('post_chemistry', 2)]
    for step in steps:
        previous = None
        for label, stage in phases:
            path = case/'validation'/f'air5.{label}.step{step:08d}.rk{stage:02d}.rank{rank:08d}.bin'
            if not path.exists():
                entries.append(dict(step=step, label=label, stage=stage, saved=False))
                continue
            row = metrics(_active_array(path), node)
            current = row['node_extended_mass_residual']
            row.update(step=step, label=label, stage=stage, saved=True,
                       node_residual_change=None if previous is None else current-previous)
            entries.append(row)
            previous = current
    if not any(row['saved'] for row in entries):
        raise ValueError('no selected snapshots')
    return dict(case=str(case.resolve()), rank=rank, node=list(node),
                status='diagnostic-not-acceptance',
                extended_precision_bits=int(np.finfo(np.longdouble).nmant), entries=entries)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--node', type=int, nargs=3, required=True)
    parser.add_argument('--steps', type=int, nargs='+', required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report = run(args.case, args.rank, args.node, args.steps)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    for row in report['entries']:
        if row['saved']:
            print(row['step'], row['label'], row['stage'],
                  row['node_extended_relative_residual'], row['node_residual_change'],
                  row['transport_sum_violations'])
