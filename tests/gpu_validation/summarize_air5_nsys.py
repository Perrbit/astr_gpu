#!/usr/bin/env python3
"""Summarize two steady reacting cycles from an Nsight Systems SQLite export."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3


def merged(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def duration(intervals):
    return sum(b-a for a, b in merged(intervals))


def analyze(filename):
    db = sqlite3.connect(f'file:{filename.resolve()}?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    names = dict(db.execute('SELECT id,value FROM StringIds'))
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    kernels = [dict(r) for r in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL')]
    ranks = list(db.execute('SELECT * FROM MPI_RANKS'))
    if not kernels or not ranks:
        raise ValueError('missing GPU or MPI trace')
    result = dict(source=str(filename), window='third to seventh chemistry launch start per rank',
                  cycles=2, scope='two steady chemistry-to-chemistry cycles; not unprofiled timing', ranks={})
    for rank_row in ranks:
        tid, rank = rank_row['globalTid'], rank_row['rank']
        # Nsight encodes process identity above the low 24 thread-id bits.
        pid = (tid >> 24) << 24
        own = [r for r in kernels if r['globalPid'] == pid]
        chemical = sorted(r['start'] for r in own
                          if 'air5_chemistry_half_step_kernel' in names[r['shortName']])
        if len(chemical) != 8:
            raise ValueError('expected four full steps / eight chemistry launches per rank')
        start, end = chemical[2], chemical[6]
        inside = [r for r in own if start <= r['start'] < end]
        if any(r['end'] > end for r in inside):
            raise ValueError('kernel crosses the cycle boundary')
        by_kernel = defaultdict(lambda: dict(calls=0, seconds=0., registers=0, local_bytes=0))
        for r in inside:
            row = by_kernel[names[r['shortName']]]
            row['calls'] += 1
            row['seconds'] += (r['end']-r['start'])*1e-9
            row['registers'] = max(row['registers'], r['registersPerThread'])
            row['local_bytes'] = max(row['local_bytes'], r['localMemoryPerThread'])
        kernel_intervals = [(r['start'], r['end']) for r in inside]
        memory = defaultdict(lambda: dict(calls=0, bytes=0, seconds=0.))
        busy = list(kernel_intervals)
        copy_names = dict(db.execute('SELECT id,label FROM ENUM_CUDA_MEMCPY_OPER'))
        for r in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE globalPid=? AND start>=? AND start<?',
                            (pid, start, end)):
            row = memory[copy_names[r['copyKind']]]
            row['calls'] += 1
            row['bytes'] += r['bytes']
            row['seconds'] += (min(end, r['end'])-r['start'])*1e-9
            busy.append((r['start'], min(end, r['end'])))
        if 'CUPTI_ACTIVITY_KIND_MEMSET' in tables:
            busy.extend((r['start'], min(end, r['end'])) for r in db.execute(
                'SELECT * FROM CUPTI_ACTIVITY_KIND_MEMSET WHERE globalPid=? AND start>=? AND start<?',
                (pid, start, end)))
        api, mpi, mpi_intervals = defaultdict(lambda: [0, 0.]), defaultdict(lambda: [0, 0.]), []
        for table, key, target in [('CUPTI_ACTIVITY_KIND_RUNTIME', 'nameId', api)]+[
                (t, 'textId', mpi) for t in sorted(tables) if t.startswith('MPI_') and t.endswith('_EVENTS')]:
            for r in db.execute(f'SELECT * FROM {table} WHERE globalTid=? AND start<? AND end>?',
                                (tid, end, start)):
                a, b = max(start, r['start']), min(end, r['end'])
                row = target[names[r[key]]]
                row[0] += 1
                row[1] += (b-a)*1e-9
                if target is mpi:
                    mpi_intervals.append((a, b))
        busy_ns = duration(busy)
        mpi_exposed_ns = duration(busy+mpi_intervals)-busy_ns
        result['ranks'][str(rank)] = dict(window_seconds=(end-start)*1e-9,
            kernel_seconds=duration(kernel_intervals)*1e-9, device_busy_seconds=busy_ns*1e-9,
            exposed_mpi_seconds=mpi_exposed_ns*1e-9,
            other_device_idle_seconds=(end-start-busy_ns-mpi_exposed_ns)*1e-9,
            kernels=dict(sorted(by_kernel.items(), key=lambda x: -x[1]['seconds'])),
            copies=dict(memory), cuda_api=dict(api), mpi=dict(mpi))
    db.close()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sqlite', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.sqlite)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    for rank, row in report['ranks'].items():
        print(rank, {k: v for k, v in row.items() if k.endswith('_seconds')})
