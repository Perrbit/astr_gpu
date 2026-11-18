"""Pack existing immutable wall trajectories, without rerunning either solver."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from run_cpp_multi_tgv_gate import raw_field


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {}
    for backend in ('cpu', 'gpu'):
        directory = args.reference/(backend+'_y')
        scalars = np.loadtxt(directory/'flowstate.dat', skiprows=1, ndmin=2)
        with h5py.File(directory/'outdat/new/resources/geometry.h5') as geometry:
            coords = np.stack([geometry[f'q{n:04}'][...].transpose(2, 1, 0) for n in (1, 2, 3)], axis=-1)
        for row in scalars:
            step = int(row[0])
            suffix = f'.step{step:08}.rk01.rank00000000.bin'
            header, q = raw_field(directory/('outdat/rhs.pre_rhs'+suffix))
            ph, p = raw_field(directory/('outdat/rhs.pre_rhs_primitives'+suffix), primitive=True)
            if tuple(ph) != tuple(header[:4]):
                raise AssertionError('channel primitive layout')
            upper = header[:3]
            nodes = int(np.prod(np.asarray(upper)+1))
            with (directory/('outdat/rhs.gradients'+suffix)).open('rb') as source:
                gh = np.fromfile(source, np.int32, 5)
                gradient = np.fromfile(source, np.float64, nodes*9)
            if tuple(gh[:3]) != tuple(upper) or gradient.size != nodes*9:
                raise AssertionError('channel gradient layout')
            with (directory/'geometry_input.bin').open('rb') as source:
                layout = np.fromfile(source, np.int32, 4)
                if tuple(layout) != tuple(header[:4]):
                    raise AssertionError('channel geometry layout')
                source.seek(16+q[...,0].size*9*8)
                jac = np.fromfile(source, np.float64)
            if jac.size != q[...,0].size:
                raise AssertionError('channel Jacobian layout')
            with (args.output/f'{backend}.step{step:08}.bin').open('xb') as target:
                np.array(header[:4], np.int32).tofile(target)
                q.ravel(order='F').tofile(target)
                p.ravel(order='F').tofile(target)
                coords.ravel(order='F').tofile(target)
                gradient.tofile(target)
                jac.tofile(target)
                row[[2,3,5]].astype(np.float64).tofile(target)
        report[backend] = dict(source=str(directory.resolve()), steps=scalars[:,0].astype(int).tolist(),
                               fields=['massflux', 'fbcx', 'wrms'])
    (args.output/'reference.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
