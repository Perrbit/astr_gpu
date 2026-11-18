"""Exercise global two-level PCR and compare actual transport results."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np


def records(directory, precision='fp64'):
    result = {}
    for path in sorted(directory.glob('rank*.bin')):
        with path.open('rb') as stream:
            while True:
                header = np.fromfile(stream, dtype=np.int32, count=9)
                if not header.size:
                    break
                if header.size != 9:
                    raise AssertionError(f'{path}: incomplete compact record')
                axis, kind, periodic, intervals, nrhs, first, owned, plane, debug = map(int, header)
                nodes = owned + int(periodic or first + owned != intervals + 1)
                count = (owned + nodes)*nrhs
                values = np.fromfile(stream, dtype=np.float32 if precision=='fp32' else np.float64, count=count)
                key = tuple(map(int, header))
                if values.size != count or not np.isfinite(values).all() or key in result:
                    raise AssertionError(f'{path}: invalid or duplicate compact payload {key}')
                result[key] = values
    return result


def summarize(directory, ranks, transport, precision='fp64'):
    logs = (directory/'run.log').read_text()
    if sum(line.startswith('CPP_DISTRIBUTED_PASS ') for line in logs.splitlines()) != ranks:
        raise AssertionError(f'{directory}: missing rank completion')
    rows = []
    for rank in range(ranks):
        with (directory/f'rank{rank:04d}.csv').open() as stream:
            local = list(csv.DictReader(stream))
        if len(local) != 216:
            raise AssertionError(f'{directory}: expected 216 cases per rank')
        rows.extend(local)
    maximum = {name: max(float(row[name]) for row in rows)
               for name in ['scaled_error', 'residual', 'restore_error']}
    if precision=='fp64' and (maximum['residual'] > 1e-12 or max(maximum['scaled_error'], maximum['restore_error']) > 2e-10):
        raise AssertionError(f'{directory}: numerical gate {maximum}')
    wire=re.findall(r'^CPP_COMPACT_WIRE\s+(\d+)\s+(\d+)\s+((?:\d+\s*){5})$',logs,re.M)
    width=4 if precision=='fp32' else 8
    if len(wire)!=ranks or any(int(w)!=width or int(values.split()[-1])!=0 for _,w,values in wire):
        raise AssertionError('compact packet MPI type/build precision evidence')
    if transport == 'device-aware' and any(int(row['pinned_bytes']) or int(row['host_stage_bytes']) for row in rows):
        raise AssertionError(f'{directory}: device MPI staged interface data')
    if transport == 'pinned' and any(int(row['host_stage_bytes']) <= 0 for row in rows):
        raise AssertionError(f'{directory}: no pinned transfer recorded')
    return maximum | dict(cases=len(rows), max_device_bytes=max(int(row['device_bytes']) for row in rows),
        element_bytes=width,wire=[dict(rank=int(rank),counters=list(map(int,values.split()))) for rank,_,values in wire])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--mpiexec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--topologies', nargs='+', default=['1,1,1', '2,1,1', '1,2,1', '1,1,2'])
    parser.add_argument('--transport', choices=['pinned', 'device-aware'], required=True)
    parser.add_argument('--compare', type=Path, help='same topology records from another transport')
    parser.add_argument('--precision',choices=['fp64','fp32'],default='fp64')
    parser.add_argument('--sanitizer',choices=['memcheck','initcheck'])
    options = parser.parse_args()
    options.output.mkdir(parents=True, exist_ok=False)
    # MPI/UCX backend configuration is supplied explicitly by the caller.
    report = {}
    for value in options.topologies:
        dims = tuple(map(int, value.split(',')))
        if len(dims) != 3 or min(dims) < 1:
            raise ValueError('topology must contain three positive dimensions')
        ranks = int(np.prod(dims))
        name = 'np%d_%dx%dx%d' % (ranks, *dims)
        directory = options.output/name
        directory.mkdir()
        command = [str(options.mpiexec), '--bind-to', 'none', '-n', str(ranks)]
        if options.sanitizer:
            command += ['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer',
                '--tool',options.sanitizer,'--error-exitcode','99','--log-file',str(directory.resolve()/'sanitizer.%p.log')]
            if options.sanitizer=='memcheck':command+=['--leak-check','full']
        command += [str(options.binary.resolve()), str(directory.resolve()), *map(str, dims), options.transport]
        with (directory/'run.log').open('w') as log:
            subprocess.run(command, env=os.environ, stdin=subprocess.DEVNULL, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        summary = summarize(directory, ranks, options.transport,options.precision)
        actual = records(directory,options.precision)
        if len(actual) != 216*ranks:
            raise AssertionError(f'{name}: incomplete binary observations')
        if options.compare:
            expected = records(options.compare/name,options.precision)
            if actual.keys() != expected.keys() or any(
                value.dtype!=expected[key].dtype or value.tobytes()!=expected[key].tobytes()
                for key, value in actual.items()
            ):
                raise AssertionError(f'{name}: transport changed solved/shared nodes')
            summary['transport_bitwise_equal'] = True
        if options.sanitizer:
            paths=list(directory.glob('sanitizer.*.log'))
            if len(paths)!=ranks or any('ERROR SUMMARY: 0 errors' not in p.read_text() for p in paths):
                raise AssertionError('compact per-rank sanitizer evidence')
            if options.sanitizer=='memcheck' and any('LEAK SUMMARY: 0 bytes leaked' not in p.read_text() for p in paths):
                raise AssertionError('compact allocation leak')
        report[name] = dict(command=command, transport=options.transport,precision=options.precision,
            binary_sha256=hashlib.sha256(options.binary.read_bytes()).hexdigest(),sanitizer=options.sanitizer, **summary)
        print(name, summary, flush=True)
        (options.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
