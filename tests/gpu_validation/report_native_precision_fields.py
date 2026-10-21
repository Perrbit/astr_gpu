"""Offline matched-stage field/profile reports from immutable ASTR short runs."""

import argparse
import json
from pathlib import Path
import re

import h5py
import numpy as np

from compare_flowstate import read_flowstate
from run_native_precision_tgv_gate import component_report, physical_state, primitive_state


PRIMITIVES = ('rho', 'u', 'v', 'w', 'p', 'T')
PHASE = 'step00000009.rk03'


def ownership(row, intervals, physical):
    cells, origin = np.asarray(row[4:7]), np.asarray(row[7:10])
    counts = cells+((origin+cells == intervals) & physical)
    local = tuple(slice(0, int(n)) for n in counts)
    target = tuple(slice(int(o), int(o+n)) for o, n in zip(origin, counts))
    return local, target


def assemble(case, precision, physical):
    partitions = np.loadtxt(case/'datin/parallel.info', skiprows=3, dtype=int, ndmin=2)
    intervals = np.max(partitions[:, 4:7]+partitions[:, 7:10], axis=0)
    shape = tuple(intervals+np.asarray(physical))
    q = np.empty((*shape, 5))
    primitive = np.empty((*shape, 6))
    owners = np.zeros(shape, dtype=np.int32)
    prefix = case/'diagnostics/state' if any(physical) else case/'state'
    for row in partitions:
        rank = int(row[0])
        suffix = PHASE+f'.rank{rank:08d}.bin'
        a = physical_state(Path(str(prefix)+'.pre_rhs.'+suffix), precision)
        sides = tuple((not physical[d] or row[7+d]>0,
                       not physical[d] or row[7+d]+row[4+d]<intervals[d]) for d in range(3))
        b = primitive_state(Path(str(prefix)+'.pre_rhs_primitives.'+suffix), precision, halo_sides=sides)
        local, target = ownership(row, intervals, physical)
        q[target] = a[local]
        primitive[target] = b[local]
        owners[target] += 1
    if not np.all(owners == 1):
        raise ValueError('Offline field ownership is not unique and complete')
    if not np.isfinite(q).all() or not np.isfinite(primitive).all() or np.any(primitive[..., [0, 4, 5]] <= 0):
        raise ValueError('Invalid physical state in native-precision source')
    return q, primitive, intervals


def compare_statistics(a, b):
    names, x = read_flowstate(a)
    other_names, y = read_flowstate(b)
    if names != other_names or x.shape != y.shape or not np.array_equal(x[:, 0], y[:, 0]):
        raise ValueError('Statistics sample identity differs')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite source statistics')
    return dict(zip(names[1:], component_report(x[:, 1:], y[:, 1:])))


def profiles(destination, reference, candidate, a, b):
    with h5py.File(reference/'datin/grid.h5') as grid:
        x, y = grid['x'][0, 0, :], grid['y'][0, :, :]
    with h5py.File(candidate/'datin/grid.h5') as grid:
        cx, cy = grid['x'][0, 0, :], grid['y'][0, :, :]
    report = []
    # z is periodic; these are instantaneous spanwise means, not time statistics.
    mean_a, mean_b = a.mean(axis=2), b.mean(axis=2)
    for fraction in (.25, .5, .75):
        i = int(np.argmin(abs(x-(x[0]+fraction*(x[-1]-x[0])))))
        table = np.column_stack((y[:, i], cy[:, i], mean_a[i], mean_b[i]))
        name = f'profile_x{i:04d}.csv'
        np.savetxt(destination/name, table, delimiter=',', comments='',
            header=','.join(['y_fp64', 'y_fp32']+[field+'_fp64' for field in PRIMITIVES]+
                            [field+'_fp32' for field in PRIMITIVES]))
        report.append(dict(file=name, x_index=i, requested_x_fraction=fraction,
            x_fp64=float(x[i]), x_fp32=float(cx[i]),
            y_coordinate_max_abs_difference=float(abs(y[:, i]-cy[:, i]).max()),
            fields=dict(zip(PRIMITIVES, component_report(mean_a[i], mean_b[i])))))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fp32-root', type=Path, required=True)
    parser.add_argument('--fp64-root', type=Path, required=True)
    parser.add_argument('--case', choices=('tgv', 'plate'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source32, source64 = args.fp32_root.resolve(), args.fp64_root.resolve()
    tags = [{p.name for p in root.iterdir() if p.is_dir() and re.fullmatch(
        r'\dx\dx\d_filter_[01]', p.name)} for root in (source32, source64)]
    if not tags[0] or tags[0] != tags[1]:
        raise ValueError('Paired source matrices have different configurations')
    physical = (True, True, False) if args.case == 'plate' else (False, False, False)
    output = args.out.resolve(); output.mkdir(exist_ok=False)
    report = dict(status='report-only-no-cross-precision-threshold', case=args.case,
        fp64_source=str(source64), fp32_source=str(source32),
        phase='step9/RK3 pre_rhs, not a completed-step output field',
        normalization='max absolute difference / max(1, max absolute FP64 component); no local-value divisor',
        analysis_precision='offline FP64 only, never fed back to the solver', cases={})
    if args.case == 'plate':
        report['wall_diagnostics'] = dict(
            fbcx='existing bottom-wall area average of mu(T) du/dy',
            wallheatflux='existing bottom-wall area average of k(T) dT/dy; legacy signed quantity, not relabeled heat-into-gas',
            scope='short transient statistics, not a converged turbulent mean or external physical acceptance')
    for tag in sorted(tags[0]):
        left, right = source64/tag/'gpu', source32/tag/'gpu'
        q64, p64, intervals = assemble(left, 'fp64', physical)
        q32, p32, other_intervals = assemble(right, 'fp32', physical)
        if not np.array_equal(intervals, other_intervals):
            raise ValueError('Logical source grids differ')
        record = dict(grid_intervals=intervals.tolist(),
            q=component_report(q64, q32), primitives=dict(zip(PRIMITIVES, component_report(p64, p32))),
            statistics=compare_statistics(left, right))
        if args.case == 'plate':
            directory = output/tag; directory.mkdir()
            record['profiles'] = profiles(directory, left, right, p64, p32)
        report['cases'][tag] = record
        print(tag+': paired immutable fields reported', flush=True)
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
