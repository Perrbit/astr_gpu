#!/usr/bin/env python3
"""CP4 algebraic gates; whole-line reconstruction is test-only, after MPI exits."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np


def reconstruct(directory):
    fields = {}
    owners = {}
    for filename in sorted(directory.glob('rank*.bin')):
        with filename.open('rb') as stream:
            while header := np.fromfile(stream, dtype=np.int32, count=8).tolist():
                if len(header) != 8:
                    raise AssertionError(f'truncated header: {filename}')
                axis, kind, periodic, intervals, columns, first, owned, plane = header
                assert plane == 0
                count = owned * columns
                data = np.fromfile(stream, dtype=np.float64, count=count)
                if data.size != count:
                    raise AssertionError(f'truncated field: {filename}')
                key = (axis, kind, periodic, intervals, columns)
                n = intervals if periodic else intervals + 1
                if key not in fields:
                    fields[key] = np.empty((n, columns))
                    owners[key] = np.zeros(n, dtype=np.int32)
                fields[key][first:first + owned] = data.reshape((owned, columns), order='F')
                owners[key][first:first + owned] += 1
    for key, field in fields.items():
        if not np.all(owners[key] == 1) or not np.isfinite(field).all():
            raise AssertionError(f'invalid ownership or field: {directory}, {key}')
    return fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--scope', choices=['full', 'smoke', 'tiny'], default='full')
    parser.add_argument('--capacity', type=int, choices=[1, 5, 11], default=11)
    parser.add_argument('--sanitizer', choices=['memcheck', 'racecheck', 'initcheck', 'synccheck'])
    parser.add_argument('--suppressions', type=Path)
    parser.add_argument('--ucx-tls', default='self,sm,cuda_copy,cuda_ipc',
                        help='explicit UCX transport list for device-aware tests; never a fallback')
    parser.add_argument('--backends', nargs='+', choices=['pinned', 'device-aware'],
                        default=['pinned', 'device-aware'])
    parser.add_argument('--topologies', nargs='+', default=[
        '1,1,1', '2,1,1', '1,2,1', '1,1,2', '3,1,1', '1,3,1', '1,1,3',
        '4,1,1', '2,2,1', '2,1,2', '1,2,2', '3,2,1', '5,1,1', '2,2,2', '8,1,1'])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    exe = args.exe.resolve()
    summary = {'executable': str(exe), 'sha256': hashlib.sha256(exe.read_bytes()).hexdigest(),
               'scope': args.scope, 'capacity': args.capacity, 'sanitizer': args.sanitizer,
               'ucx_tls': args.ucx_tls, 'runs': []}
    baseline = None
    for backend in args.backends:
        for topology in args.topologies:
            dims = tuple(int(value) for value in topology.split(','))
            if len(dims) != 3 or min(dims) < 1:
                raise ValueError(f'invalid topology: {topology}')
            directory = (args.out / f'{backend}_{topology.replace(",", "x")}').resolve()
            directory.mkdir()
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=env.get('CUDA_VISIBLE_DEVICES', '0,1'),
                       ASTR_GPU_COMPACT_TRANSPORT=backend,
                       OMPI_MCA_coll='^hcoll,ucc,cuda', OMPI_MCA_coll_hcoll_enable='0')
            if backend == 'pinned':
                env.update(OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,vader,tcp',
                           OMPI_MCA_osc='pt2pt', OMPI_MCA_opal_cuda_support='0')
            else:
                env.update(OMPI_MCA_pml='ucx', UCX_MEMTYPE_CACHE='n',
                           UCX_CUDA_COPY_ENABLE_FABRIC='no', UCX_CUDA_COPY_DMABUF='no',
                           UCX_CUDA_IPC_ENABLE_MNNVL='no', UCX_TLS=args.ucx_tls)
                env.pop('OMPI_MCA_opal_cuda_support', None)
            probe = [str(exe), str(directory), *(str(n) for n in dims), args.scope, str(args.capacity)]
            if args.sanitizer:
                instrument = ['compute-sanitizer', '--tool', args.sanitizer, '--error-exitcode', '99',
                              '--log-file', str(directory / 'sanitizer.%p.log')]
                if args.sanitizer == 'memcheck':
                    instrument += ['--leak-check', 'full']
                if args.suppressions:
                    instrument += ['--suppressions', str(args.suppressions.resolve())]
                probe = instrument + probe
            command = ['mpirun', '--oversubscribe', '--bind-to', 'none', '-np', str(np.prod(dims)), *probe]
            started = time.monotonic()
            with (directory / 'run.log').open('w') as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            rows = []
            for filename in sorted(directory.glob('rank*.csv')):
                with filename.open() as stream:
                    rows.extend(csv.DictReader(stream))
            batches = sum(count <= args.capacity for count in [1, 5, 11])
            expected = 3 * {'full': 9, 'smoke': 2, 'tiny': 3}[args.scope] * 4 * batches * np.prod(dims)
            if len(rows) != expected:
                raise AssertionError(f'missing probe rows: {len(rows)} != {expected}')
            if {row['device_aware'] for row in rows} != ({'T'} if backend == 'device-aware' else {'F'}):
                raise AssertionError('selected transport was not used')
            fields = reconstruct(directory)
            expected_fields = expected // int(np.prod(dims))
            if len(fields) != expected_fields:
                raise AssertionError(f'missing global reference-line records: {directory}')
            if baseline is None:
                baseline = fields
            if fields.keys() != baseline.keys():
                raise AssertionError('different global case sets across partitions')
            abs_partition = 0.0
            scaled_partition = 0.0
            for key, field in fields.items():
                delta = np.max(np.abs(field - baseline[key]), axis=0)
                abs_partition = max(abs_partition, float(np.max(delta)))
                scale = np.maximum(1.0, np.max(np.abs(baseline[key]), axis=0))
                scaled_partition = max(scaled_partition, float(np.max(delta / scale)))
            metrics = {name: max(float(row[name]) for row in rows) for name in
                       ['abs_error', 'normalized_error', 'residual', 'restore_error']}
            if 'cpu_normalized_error' in rows[0]:
                metrics.update({name: max(float(row[name]) for row in rows) for name in
                                ['cpu_abs_error', 'cpu_normalized_error']})
            if metrics['normalized_error'] > 2e-10 or metrics['residual'] > 1e-12 or scaled_partition > 2e-10:
                raise AssertionError(f'CP4 numerical gate failed: {directory}')
            record = dict(backend=backend, topology=dims, rows=len(rows),
                          seconds=time.monotonic() - started, partition_absolute=abs_partition,
                          partition_normalized=scaled_partition, **metrics)
            summary['runs'].append(record)
            (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
            print(json.dumps(record), flush=True)
    summary['status'] = 'pass'
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
