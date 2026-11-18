"""Recheck frozen NR1 point inputs with rebuilt production projection probes."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from run_air5_flux_components import check_flux_constraints, compare, read, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('capture', 'components', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out/'replay.py')
    probes = {'cpu': args.components/'cpu/probe', 'old_gpu': args.components/'old_gpu/probe',
              'cpp': args.components/'candidate'}
    receipt = dict(probes={k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in probes.items()},
                   input_sha256={}, commands=[], cases=[])
    for stem, op in [('projection_endpoints', 1), ('face_neighbors', 4)]:
        path = out/f'{stem}.bin'
        shutil.copy2(args.capture/path.name, path)
        receipt['input_sha256'][stem] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open('rb') as stream:
            n = int(np.fromfile(stream, '<i4', 1)[0])
            x = np.fromfile(stream, '<f8').reshape(n, 80)
        assert np.all(x[:,79] == op)
        labels = [f'{stem}:{i}' for i in range(n)]
        for cache in (0, 1):
            results = {}
            for backend, binary in probes.items():
                tag = f'{stem}_{backend}_{cache}'
                result = out/f'{tag}.bin'
                command = [binary.resolve(), path, result, cache]
                receipt['commands'].append(list(map(str, command)))
                run(command, out, f'{tag}.log')
                results[backend] = read(result)
            statuses = {b: {int(s): int(np.count_nonzero(v[0] == s)) for s in np.unique(v[0])}
                        for b, v in results.items()}
            row = dict(input=stem, cache=cache, statuses=statuses)
            receipt['cases'].append(row)
            (out/'report.json').write_text(json.dumps(receipt, indent=2)+'\n')
            print(row, flush=True)
            assert all(np.all(v[0] == 0) for v in results.values()), statuses
            cpu_status, cpu = results['cpu']
            for backend, (status, actual) in results.items():
                row[backend] = dict(comparison=compare(actual, cpu, cpu_status, x, labels, operations=[op]),
                                    constraints=check_flux_constraints(x, status, actual, labels, [None]*n))
            (out/'report.json').write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    main()
