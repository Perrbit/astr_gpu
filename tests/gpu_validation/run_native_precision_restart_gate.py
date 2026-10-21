"""Completed-step native restart checks using only the ASTR time integrator."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from run_output_restart_validation import compare_fields, run_case


ROOT = Path(__file__).resolve().parents[2]
FINAL = Path('outdat/new/checkpoints/step000000000012')


def check_state(path, precision):
    width = 4 if precision == 'fp32' else 8
    with h5py.File(path) as state:
        if state['identity'][1] != 3 or state['numerical_kind_bytes'][:].tolist() != [width]:
            raise AssertionError('checkpoint precision identity differs')
        fields = []
        components = int(state['identity'][6])
        if components not in (5, 11):
            raise AssertionError('unsupported nonreacting checkpoint layout')
        for component in range(1, components + 1):
            field = state[f'q{component:04d}']
            if field.dtype != np.dtype(f'<f{width}') or not np.isfinite(field[:]).all():
                raise AssertionError(f'invalid native payload: {field.name}')
            if component <= 5:
                fields.append(field[:].astype(np.float64))
        density = fields[0]
        if not (density > 0).all():
            raise AssertionError('checkpoint physical density is not positive')
        internal = fields[4] - sum(v * v for v in fields[1:4]) / (2 * density)
        if not (internal > 0).all():
            raise AssertionError('checkpoint physical state is not positive')
        return dict(dtype=str(state['q0001'].dtype), density_min=float(density.min()),
                    internal_energy_min=float(internal.min()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--precision', choices=['fp32', 'fp64'], required=True)
    parser.add_argument('--cpu', type=Path, required=True)
    parser.add_argument('--gpu', type=Path, required=True)
    parser.add_argument('--mpiexec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['steps', 'time'], default='steps')
    parser.add_argument('--backends', nargs='+', choices=['cpu', 'gpu'], default=['cpu', 'gpu'])
    parser.add_argument('--schemes', nargs='+', choices=['643e', '643c'], default=['643e', '643c'])
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(exist_ok=False)
    os.environ.update(OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,vader,tcp',
                      OMPI_MCA_osc='pt2pt', OMPI_MCA_coll='^cuda,hcoll,ucc',
                      OMPI_MCA_opal_cuda_support='0')
    report = dict(precision=args.precision, steps=12, restart_step=5, checkpoint_schedule_mode=args.mode, cases={})
    for scheme in args.schemes:
        for backend in args.backends:
            for ranks in ([1, 2] if scheme == '643e' else [2]):
                label = f'{backend}_{scheme}_np{ranks}'
                local = SimpleNamespace(
                    output=args.output, executable=getattr(args, backend).resolve(),
                    mpiexec=args.mpiexec.resolve(), case='tgv', mode=args.mode, restart_step=5,
                    initial_dimension=0, legacy_statistics=False, statistics=scheme == '643e',
                    initial_restart=False, filter_workspace='scalar', force='feedback',
                    axis='x', no_samples=True, wall_samples=False)
                options = dict(scheme=scheme, gpu_precision_mode='')
                continuous, _ = run_case(local, ROOT, backend, ranks, scheme + '_continuous', 12, **options)
                seed, _ = run_case(local, ROOT, backend, ranks, scheme + '_seed', 5, **options)
                resumed, _ = run_case(local, ROOT, backend, ranks, scheme + '_resumed', 12,
                    restore=seed / 'outdat/new/checkpoints/step000000000005', **options)
                result = dict(state=check_state(resumed / FINAL / 'state.h5', args.precision))
                names = ['state.h5'] + (['statistics.h5'] if local.statistics else [])
                result['bit_identical_hdf5_datasets'] = {
                    name: compare_fields(continuous / FINAL / name, resumed / FINAL / name)
                    for name in names}
                for name in ('control.bin', 'archives.bin', 'insitu_control.bin'):
                    if (continuous / FINAL / name).read_bytes() != (resumed / FINAL / name).read_bytes():
                        raise AssertionError(f'{label}: {name} differs')
                result['control_files_bit_identical'] = True
                report['cases'][label] = result
                (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
                print(label + ': exact state/control' + ('/statistics' if local.statistics else ''), flush=True)
    report['status'] = 'passed'
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
