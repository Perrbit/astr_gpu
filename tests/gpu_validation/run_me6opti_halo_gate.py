"""Exact ME6 primitive halo checks on uneven periodic partitions."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--mpiexec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--transport', choices=['cpu', 'pinned', 'device-aware'], required=True)
    parser.add_argument('--memcheck', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(OMPI_MCA_coll='^cuda,hcoll,ucc', OMPI_MCA_osc='pt2pt')
    if args.transport == 'device-aware':
        env.update(OMPI_MCA_pml='ucx', OMPI_MCA_opal_cuda_support='true',
                   UCX_TLS='self,sm,tcp,cuda_copy,cuda_ipc', UCX_MEMTYPE_CACHE='n')
    else:
        env.update(OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,vader,tcp', OMPI_MCA_opal_cuda_support='false')
    if args.transport != 'cpu':
        env['ASTR_GPU_HALO_TRANSPORT'] = args.transport
    matrix = [(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2),
              (2, 2, 1), (2, 1, 2), (1, 2, 2), (2, 2, 2)]
    if args.memcheck:
        matrix = [(2, 2, 2)]
    records = []
    for topology in matrix:
        command = [str(args.mpiexec.resolve()), '--oversubscribe', '-np', str(math.prod(topology))]
        if args.memcheck:
            command += ['compute-sanitizer', '--tool', 'memcheck', '--error-exitcode', '99']
        command += [str(args.executable.resolve()), *map(str, topology)]
        name = 'x'.join(map(str, topology))
        with (args.output / f'{name}.log').open('w') as stream:
            result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        records.append(dict(topology=topology, command=command, returncode=result.returncode))
        (args.output / 'summary.json').write_text(json.dumps(dict(transport=args.transport, runs=records), indent=2))
        result.check_returncode()
        if 'ME6_EXTENDED_HALO_PASS' not in (args.output / f'{name}.log').read_text():
            raise AssertionError(f'halo comparison result missing: {name}')
        print(args.transport, name, 'exact node/face/edge/corner comparison passed', flush=True)


if __name__ == '__main__':
    main()
