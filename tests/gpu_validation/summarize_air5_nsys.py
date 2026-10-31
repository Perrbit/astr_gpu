#!/usr/bin/env python3
"""Summarize two steady reacting cycles from an Nsight Systems SQLite export."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3


def is_chemistry_kernel(name):
    return any(marker in name for marker in ('air5_chemistry_half_step_kernel',
        'air5_chemistry_report_kernel','air5_chemistry_quiet_kernel'))


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


def overlap_duration(left,right):
    return duration([(max(a,c),min(b,d)) for a,b in left for c,d in right])


def posted_request_scopes(posts,waits):
    scopes=[]
    missing=0
    for post in posts:
        following=[row for row in waits if row['requestHandle']==post['requestHandle']
                   and row['start']>=post['end']]
        if not following:
            missing+=1
            continue
        scopes.append((post['end'],min(following,key=lambda row:row['start'])['end']))
    return scopes,missing


def chemistry_cycle_window(entries, dynamic_top=False):
    entries = sorted(entries, key=lambda row: row['start'])
    if dynamic_top and len(entries) == 16:
        # Current bounded fixture launches the interior followed by the smaller top face.
        for interior, top in zip(entries[::2], entries[1::2]):
            if interior['gridX'] <= top['gridX'] or interior['end'] > top['start']:
                raise ValueError('ambiguous interior/top chemistry launches')
        entries = entries[::2]
    if len(entries) != 8:
        raise ValueError('expected four full steps / eight interior chemistry launches per rank')
    return entries[2]['start'], entries[6]['start']


def analyze(filename, dynamic_top=False):
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
        chemical = [r for r in own if is_chemistry_kernel(names[r['shortName']])]
        start, end = chemistry_cycle_window(chemical, dynamic_top)
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
        progress_intervals=[]
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
                    if 'testall' in names[r[key]].lower():
                        progress_intervals.append((a,b))
        busy_ns = duration(busy)
        mpi_exposed_ns = duration(busy+mpi_intervals)-busy_ns
        packs=[(r['start'],r['end']) for r in inside
               if 'chemistry_flux_exchange_gpu_pack_flux_' in names[r['shortName']]]
        posts=[dict(r) for r in db.execute('SELECT * FROM MPI_P2P_EVENTS WHERE globalTid=? AND start>=? AND end<=? '
                'AND tag BETWEEN 22101 AND 22106',(tid,start,end))
                if names[r['textId']] in ('MPI_Isend','MPI_Irecv')] if 'MPI_P2P_EVENTS' in tables else []
        waits=[dict(r) for r in db.execute('SELECT * FROM MPI_START_WAIT_EVENTS WHERE globalTid=? AND start>=? AND end<=?',
                (tid,start,end))] if 'MPI_START_WAIT_EVENTS' in tables else []
        scopes,missing=posted_request_scopes(posts,waits)
        result['ranks'][str(rank)] = dict(window_seconds=(end-start)*1e-9,
            kernel_seconds=duration(kernel_intervals)*1e-9, device_busy_seconds=busy_ns*1e-9,
            exposed_mpi_seconds=mpi_exposed_ns*1e-9,
            other_device_idle_seconds=(end-start-busy_ns-mpi_exposed_ns)*1e-9,
            kernels=dict(sorted(by_kernel.items(), key=lambda x: -x[1]['seconds'])),
            copies=dict(memory), cuda_api=dict(api), mpi=dict(mpi),
            flux_pack_calls=len(packs),flux_pack_seconds=duration(packs)*1e-9,
            flux_pack_mpi_progress_trace_available=bool(progress_intervals),
            flux_pack_mpi_progress_overlap_calls=(sum(overlap_duration([p],progress_intervals)>0 for p in packs)
                if progress_intervals else None),
            flux_pack_mpi_progress_overlap_seconds=(overlap_duration(packs,progress_intervals)*1e-9
                if progress_intervals else None),
            flux_pack_posted_request_trace_available=bool(scopes),
            flux_pack_posted_request_overlap_calls=(sum(overlap_duration([p],scopes)>0 for p in packs)
                if scopes else None),
            flux_pack_posted_request_overlap_seconds=overlap_duration(packs,scopes)*1e-9 if scopes else None,
            flux_requests_without_wait_trace=missing,
            overlap_scope='posted request API return to associated MPI wait completion; not network payload overlap or speedup')
    db.close()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sqlite', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dynamic-top', action='store_true', help='Allow the bounded interior/top launch pair')
    args = parser.parse_args()
    report = analyze(args.sqlite, args.dynamic_top)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    for rank, row in report['ranks'].items():
        print(rank, {k: v for k, v in row.items() if k.endswith('_seconds')})
