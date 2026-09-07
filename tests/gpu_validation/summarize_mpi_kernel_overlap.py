"""Report MPI API/kernel time intersection, not communication-engine utilization."""

import argparse
import bisect
import json
from pathlib import Path
import sqlite3


def summarize_completion(db):
    """Count observed Testall flags during kernels; do not infer byte progress."""
    tables = {row[0] for row in db.execute('SELECT name FROM sqlite_master WHERE type="table"')}
    if not {'CUPTI_ACTIVITY_KIND_KERNEL', 'StringIds'} <= tables:
        raise ValueError('Nsight export must include CUDA kernels and StringIds')
    if 'NVTX_EVENTS' not in tables:
        return {}
    kernels = {}
    for start, end, process, name in db.execute(
            'SELECT k.start,k.end,k.globalPid >> 24,s.value '
            'FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id '
            'ORDER BY k.start'):
        kernels.setdefault(process, []).append((start, end, name))
    # Sweep each process independently; a marker can coincide with several streams.
    positions = {}
    active = {}
    result = {}
    for time, process, name in db.execute(
            'SELECT n.start,n.globalTid >> 24,COALESCE(n.text,s.value) '
            'FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id '
            "WHERE COALESCE(n.text,s.value) IN ('ASTR_MPI_TESTALL_PENDING',"
            "'ASTR_MPI_TESTALL_COMPLETE') ORDER BY n.start"):
        sequence = kernels.get(process, [])
        index = positions.get(process, 0)
        running = [item for item in active.get(process, []) if item[1] > time]
        while index < len(sequence) and sequence[index][0] <= time:
            if sequence[index][1] > time:
                running.append(sequence[index])
            index += 1
        positions[process] = index
        active[process] = running
        record = result.setdefault(name, dict(markers=0, markers_during_kernel=0,
                                              kernel_names={}, processes={}))
        record['markers'] += 1
        record['markers_during_kernel'] += bool(running)
        process_record = record['processes'].setdefault(str(process),
                                                       dict(markers=0, markers_during_kernel=0))
        process_record['markers'] += 1
        process_record['markers_during_kernel'] += bool(running)
        for kernel in {item[2] for item in running}:
            record['kernel_names'][kernel] = record['kernel_names'].get(kernel, 0) + 1
    return result


def summarize(db):
    tables = {row[0] for row in db.execute('SELECT name FROM sqlite_master WHERE type="table"')}
    if not {'CUPTI_ACTIVITY_KIND_KERNEL', 'StringIds'} <= tables:
        raise ValueError('Nsight export must include CUDA kernels and StringIds')
    # Union kernel intervals per process to avoid double counting concurrent streams.
    kernels = {}
    for start, end, process in db.execute(
            'SELECT start,end,globalPid >> 24 FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start'):
        intervals = kernels.setdefault(process, [])
        if intervals and start <= intervals[-1][1]:
            intervals[-1] = (intervals[-1][0], max(end, intervals[-1][1]))
        else:
            intervals.append((start, end))
    ends = {process: [end for _, end in intervals] for process, intervals in kernels.items()}
    events = set()
    start_wait_events = set()
    for table in ('MPI_START_WAIT_EVENTS', 'MPI_OTHER_EVENTS', 'MPI_P2P_EVENTS'):
        if table in tables:
            rows = set(db.execute(
                f'SELECT e.start,e.end,e.globalTid,s.value FROM {table} e '
                'JOIN StringIds s ON e.textId=s.id WHERE s.value IN '
                "('MPI_Testall','MPI_Waitall','MPI_Sendrecv','MPI_Irecv','MPI_Isend')"))
            events.update(rows)
            if table == 'MPI_START_WAIT_EVENTS':
                start_wait_events.update(rows)
    result = {}
    for start, end, thread, name in sorted(events):
        record = result.setdefault(name, dict(unique_calls=0, calls_overlapping_kernel=0,
                                              api_interval_sum_ns=0, overlap_interval_sum_ns=0,
                                              start_wait_calls=0, start_wait_calls_overlapping_kernel=0,
                                              start_wait_overlap_interval_sum_ns=0))
        process = thread >> 24
        intervals = kernels.get(process, [])
        overlap = 0
        index = bisect.bisect_right(ends.get(process, []), start)
        while index < len(intervals) and intervals[index][0] < end:
            left, right = intervals[index]
            overlap += max(0, min(end, right) - max(start, left))
            index += 1
        record['unique_calls'] += 1
        record['calls_overlapping_kernel'] += int(overlap > 0)
        record['api_interval_sum_ns'] += end - start
        record['overlap_interval_sum_ns'] += overlap
        if (start, end, thread, name) in start_wait_events:
            record['start_wait_calls'] += 1
            record['start_wait_calls_overlapping_kernel'] += int(overlap > 0)
            record['start_wait_overlap_interval_sum_ns'] += overlap
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sqlite', type=Path)
    parser.add_argument('--completion', action='store_true',
                        help='Summarize profiling-interposer NVTX completion flags instead of MPI API times')
    args = parser.parse_args()
    with sqlite3.connect(args.sqlite.resolve().as_uri() + '?mode=ro', uri=True) as connection:
        print(json.dumps((summarize_completion if args.completion else summarize)(connection),
                         indent=2, sort_keys=True))
