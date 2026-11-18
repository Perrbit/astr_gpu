"""Package existing FP64 ASTR snapshots for exact-input sensor replay; no advancement."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from run_cpp_multi_tgv_gate import raw_field


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--physical-axes', default='', help='physical axes, e.g. x; other axes are periodic')
    parser.add_argument('--step', type=int, default=0)
    parser.add_argument('--threshold', type=float, default=.001)
    args = parser.parse_args()
    if any(axis not in 'xyz' for axis in args.physical_axes):
        parser.error('physical axes must use x/y/z')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    ends = [int(axis in args.physical_axes) for axis in 'xyz']
    report = dict(reference=str(args.reference.resolve()), step=args.step,
                  physical_low=ends, physical_high=ends, threshold=args.threshold, files={})
    for label, name in (('cpu', 'cpu'), ('gpu', 'old_gpu_fp64')):
        for stage in (1, 2, 3):
            prefix = args.reference/name/'outdat/rhs'
            suffix = f'.step{args.step:08d}.rk{stage:02d}.rank00000000.bin'
            source = [Path(str(prefix)+'.'+kind+suffix)
                      for kind in ('pre_rhs_primitives', 'gradients', 'sensor')]
            h, primitive = raw_field(source[0], primitive=True)
            upper = np.asarray(h[:3]); nodes = int(np.prod(upper+1))
            with source[1].open('rb') as stream:
                gh = np.fromfile(stream, np.int32, 5)
                gradient = np.fromfile(stream, np.float64, nodes*9)
            with source[2].open('rb') as stream:
                sh = np.fromfile(stream, np.int32, 3)
                raw = np.fromfile(stream, np.float64, nodes)
                mask = np.fromfile(stream, np.int8, nodes)
                extra = stream.read()
            if (not np.array_equal(gh[:3], upper) or not np.array_equal(sh, upper)
                    or gradient.size != nodes*9 or raw.size != nodes or mask.size != nodes or extra):
                raise AssertionError('snapshot header/payload mismatch')
            if not np.isfinite(gradient).all() or not np.isfinite(raw).all() or np.any((mask != 0) & (mask != 1)):
                raise AssertionError('invalid physical sensor input/output')
            path = root/f'{label}.rk{stage:02d}.bin'
            with path.open('xb') as stream:
                np.asarray([*h[:4], *ends, *ends], np.int32).tofile(stream)
                np.asarray([args.threshold], np.float64).tofile(stream)
                primitive[..., 4].ravel(order='F').tofile(stream)
                gradient.tofile(stream); raw.tofile(stream); mask.tofile(stream)
            report['files'][path.name] = dict(
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                sources={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in source})
    (root/'reference.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(output=str(root), packets=len(report['files']), physical_axes=args.physical_axes)))


if __name__ == '__main__':
    main()
