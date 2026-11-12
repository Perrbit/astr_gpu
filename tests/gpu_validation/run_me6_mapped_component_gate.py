"""Check G2 components on immutable, solver-generated geometry resources."""
import argparse
import json
import math
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu', type=Path, required=True)
    parser.add_argument('--gpu', type=Path, required=True)
    parser.add_argument('--curve', type=Path, nargs=3, required=True)
    parser.add_argument('--stretched', type=Path, nargs=3, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'comparisons': [], 'orders': {}}
    destination = args.output / 'report.json'
    for mapping in ('curve', 'stretched'):
        errors = []
        sizes = []
        for level, geometry in enumerate(getattr(args, mapping)):
            record = {'mapping': mapping, 'geometry': str(geometry.resolve())}
            report['comparisons'].append(record)
            for backend in ('cpu', 'gpu'):
                log = args.output / f'{mapping}_{level}_{backend}.log'
                with log.open('w') as stream:
                    result = subprocess.run([str(getattr(args, backend).resolve()), str(geometry.resolve())],
                                            stdout=stream, stderr=subprocess.STDOUT)
                record[f'{backend}_exit'] = result.returncode
                destination.write_text(json.dumps(report, indent=2) + '\n')
                if result.returncode:
                    raise RuntimeError(f'G2 {backend} component failed: {log}')
                lines = log.read_text().splitlines()
                line, = [line for line in lines if line.startswith('G2 actual geometry intervals=')]
                sizes_text, values_text = line.split('momentum/heat/work errors=')
                shape = list(map(int, sizes_text.split('=')[1].split()))
                values = list(map(float, values_text.split()))
                if len(values) != 5 or not all(math.isfinite(v) for v in values):
                    raise AssertionError(f'invalid G2 manufactured errors: {log}')
                record[f'{backend}_errors'] = values
                record['intervals'] = shape
                if backend == 'cpu':
                    errors.append(values)
                    sizes.append(shape)
                else:
                    line, = [line for line in lines if line.startswith('G2 CPU/GPU scaled difference=')]
                    difference = float(line.split('=')[1])
                    record['cpu_gpu_scaled'] = difference
                    if not math.isfinite(difference) or difference > 2e-10:
                        raise AssertionError(f'G2 CPU/GPU difference: {log}')
            destination.write_text(json.dumps(report, indent=2) + '\n')
        if sizes != [[32]*3, [64]*3, [128]*3]:
            raise AssertionError(f'expected 32/64/128 actual geometries, got {sizes}')
        orders = [[math.log(a/b, 2) for a, b in zip(errors[level], errors[level+1])]
                  for level in range(2)]
        report['orders'][mapping] = orders
        destination.write_text(json.dumps(report, indent=2) + '\n')
        if min(orders[-1]) < 5.5:
            raise AssertionError(f'G2 {mapping} lacks asymptotic sixth order: {orders[-1]}')
    report['status'] = 'passed'
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
