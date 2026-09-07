#!/usr/bin/env python3
"""Validate complete per-rank RK records and report the slowest rank per step."""

import argparse
import math
from pathlib import Path


def slowest_rank_samples(text, ranks, steps):
    if ranks < 1 or steps < 1:
        raise ValueError('ranks and steps must be positive')
    records = {}
    for line in text.splitlines():
        if not line.startswith('ASTR_GPU_RANK_RK_TIMING '):
            continue
        parts = line.split()
        if len(parts) != 6:
            raise ValueError('malformed rank RK record')
        rank, step = map(int, parts[1:3])
        prepare, integration, total = map(float, parts[3:])
        if not 0 <= rank < ranks or not 0 <= step < steps:
            raise ValueError('rank or step out of range')
        if any(not math.isfinite(x) or x < 0 for x in (prepare, integration, total)):
            raise ValueError('invalid RK duration')
        if not math.isclose(prepare + integration, total, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('inconsistent complete RK duration')
        if (rank, step) in records:
            raise ValueError('duplicate rank RK record')
        records[rank, step] = total
    if len(records) != ranks * steps:
        raise ValueError('missing rank RK records')
    return [max(records[rank, step] for rank in range(ranks)) for step in range(steps)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', required=True, type=Path)
    parser.add_argument('--ranks', required=True, type=int)
    parser.add_argument('--steps', required=True, type=int,
                        help='Number of RK records per rank, including step zero')
    parser.add_argument('--discard', type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.discard < args.steps:
        parser.error('discard must retain at least one step')
    samples = slowest_rank_samples(args.log.read_text(), args.ranks, args.steps)
    print('step\tmax_rank_rk_seconds')
    for step in range(args.discard, args.steps):
        print(f'{step}\t{samples[step]:.16e}')


if __name__ == '__main__':
    main()
