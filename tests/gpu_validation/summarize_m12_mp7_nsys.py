#!/usr/bin/env python3
"""Six matched MP-stage cycles in an eight-step M12 Nsight Systems trace."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3

from summarize_air5_nsys import duration


def analyze(filename):
    db = sqlite3.connect(f'file:{filename.resolve()}?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    names = dict(db.execute('SELECT id,value FROM StringIds'))
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    kernels = [dict(r) for r in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL')]
    ranks = list(db.execute('SELECT * FROM MPI_RANKS'))
    result = dict(source=str(filename), cycles=6,
                  scope='step1 first MP face to step7 first MP face; aligned cycles, not formal RK timing',
                  ranks={})
    for row in ranks:
        tid, rank = row['globalTid'], row['rank']
        pid = (tid >> 24) << 24
        own = [r for r in kernels if r['globalPid'] == pid]
        faces = sorted((r for r in own if 'mp_faces' in names[r['shortName']]), key=lambda r:r['start'])
        if len(faces) != 72:
            raise ValueError('expected 8 steps x 3 stages x 3 MP face launches')
        start, end = faces[9]['start'], faces[63]['start']
        inside = [r for r in own if start <= r['start'] < end]
        if any(r['end'] > end for r in inside):
            raise ValueError('kernel crosses matched cycle boundary')
        grouped = defaultdict(lambda: dict(calls=0, seconds=0., registers=0, local_bytes=0))
        busy = [(r['start'],r['end']) for r in inside]
        kernel_ns = duration(busy)
        for r in inside:
            entry = grouped[names[r['shortName']]]
            entry['calls'] += 1
            entry['seconds'] += (r['end']-r['start'])*1e-9
            entry['registers'] = max(entry['registers'],r['registersPerThread'])
            entry['local_bytes'] = max(entry['local_bytes'],r['localMemoryPerThread'])
        copies = defaultdict(lambda: dict(calls=0, bytes=0, seconds=0.))
        copy_names = dict(db.execute('SELECT id,label FROM ENUM_CUDA_MEMCPY_OPER'))
        for r in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE globalPid=? AND start<? AND end>?',
                            (pid,end,start)):
            a,b = max(start,r['start']),min(end,r['end'])
            entry = copies[copy_names[r['copyKind']]]
            entry['calls'] += 1
            entry['bytes'] += r['bytes']
            entry['seconds'] += (b-a)*1e-9
            busy.append((a,b))
        if 'CUPTI_ACTIVITY_KIND_MEMSET' in tables:
            busy.extend((max(start,r['start']),min(end,r['end'])) for r in db.execute(
                'SELECT * FROM CUPTI_ACTIVITY_KIND_MEMSET WHERE globalPid=? AND start<? AND end>?',
                (pid,end,start)))
        cuda,mpi = defaultdict(lambda:[0,0.]),defaultdict(lambda:[0,0.])
        sync_intervals,mpi_intervals = [],[]
        for table in ('CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER'):
            if table not in tables:
                continue
            for r in db.execute(f'SELECT * FROM {table} WHERE start<? AND end>?', (end,start)):
                if (r['globalTid'] >> 24) << 24 != pid:
                    continue
                a,b = max(start,r['start']),min(end,r['end'])
                name = names[r['nameId']]
                cuda[name][0] += 1
                cuda[name][1] += (b-a)*1e-9
                if 'Synchronize' in name:
                    sync_intervals.append((a,b))
        messages = defaultdict(lambda:dict(calls=0,metadata_bytes=0))
        for table in sorted(tables):
            if not (table.startswith('MPI_') and table.endswith('_EVENTS')):
                continue
            for r in db.execute(f'SELECT * FROM {table} WHERE globalTid=? AND start<? AND end>?',
                                (tid,end,start)):
                a,b = max(start,r['start']),min(end,r['end'])
                name = names[r['textId']]
                mpi[name][0] += 1
                mpi[name][1] += (b-a)*1e-9
                mpi_intervals.append((a,b))
                if table == 'MPI_P2P_EVENTS' and r['remoteRank'] is not None and r['remoteRank'] >= 0:
                    entry = messages[f"{name}/tag{r['tag']}"]
                    entry['calls'] += 1
                    entry['metadata_bytes'] += r['size']
        busy_ns = duration(busy)
        exposed_mpi = duration(busy+mpi_intervals)-busy_ns
        exposed_sync = duration(busy+sync_intervals)-busy_ns
        result['ranks'][str(rank)] = dict(window_seconds=(end-start)*1e-9,
            kernel_seconds=kernel_ns*1e-9,device_busy_seconds=busy_ns*1e-9,
            exposed_mpi_seconds=exposed_mpi*1e-9,
            sync_without_device_work_seconds=exposed_sync*1e-9,
            sync_without_device_work_fraction=exposed_sync/(end-start),
            other_device_idle_seconds=(end-start-busy_ns-exposed_mpi)*1e-9,
            kernels=dict(sorted(grouped.items(),key=lambda p:-p[1]['seconds'])),
            copies=dict(copies),cuda_api=dict(cuda),mpi=dict(mpi),p2p_metadata=dict(messages),
            accounting='exposed sync can overlap exposed MPI; never add the two; metadata is not wire traffic')
    db.close()
    if not result['ranks']:
        raise ValueError('no matched MPI/GPU rank')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sqlite', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.sqlite)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for rank,row in report['ranks'].items():
        print(rank,{k:v for k,v in row.items() if k.endswith(('_seconds','_fraction'))})
