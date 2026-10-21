"""Verify cross-kind restore before RK, then advance with the real ASTR solver."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import h5py
import numpy as np

from run_output_restart_validation import run_case
from run_native_precision_restart_gate import check_state


ROOT = Path(__file__).resolve().parents[2]


def check_conversion(source, resumed, target):
    dtype = np.dtype('<f4' if target == 'fp32' else '<f8')
    result = dict(contract='source cast to target kind, then original x/y/z periodic endpoint averages; before RK',
                  periodic_rebuild_max_abs={})
    with h5py.File(source / 'state.h5') as original, h5py.File(resumed / 'outdat/restore_probe.h5') as restored:
        for component in range(1, 6):
            name = f'q{component:04d}'
            converted = original[name][:].astype(dtype)
            expected = converted.copy()
            # Non-exact restart deliberately invokes the existing qswap. File
            # endpoints can differ after RK/filtering, so this is not a raw copy.
            for axis in (2, 1, 0):
                lower = [slice(None)] * 3
                upper = lower.copy()
                lower[axis], upper[axis] = 0, -1
                mean = dtype.type(0.5) * (expected[tuple(lower)] + expected[tuple(upper)])
                expected[tuple(lower)] = mean
                expected[tuple(upper)] = mean
            actual = restored[name][:]
            if actual.dtype != dtype or actual.tobytes() != expected.tobytes():
                raise AssertionError(f'actual restored {name} differs from conversion plus original periodic rebuild')
            result['periodic_rebuild_max_abs'][name] = float(np.max(np.abs(expected - converted)))
        if int(restored['identity'][8]) != 5:
            raise AssertionError('restore probe is not at the completed source step')
    result['all_five_fields_bit_identical'] = True
    return result


def check_merged_statistics(directory, metadata):
    shape = tuple(metadata['shape_kji'])
    selected = tuple(slice(0, n) for n in shape)
    sources = []
    for entry in metadata['sources']:
        with h5py.File(entry['path']) as source:
            sources.append(np.stack([source[f'q{m:04d}'][selected].astype(np.float64)
                                     for m in range(8, 34)], axis=-1))
    a, b = sources
    report = {}
    with h5py.File(directory / 'statistics.h5') as result:
        np.testing.assert_array_equal(result['point/duration'][:], a[..., 0] + b[..., 0])
        np.testing.assert_array_equal(result['point/mass_duration'][:], a[..., 1] + b[..., 1])
        for kind, slot, mean_start, central_start in (('reynolds', 0, 2, 8), ('favre', 1, 5, 17)):
            wa, wb = a[..., slot], b[..., slot]
            total = wa + wb
            ua, ub = a[..., mean_start:mean_start + 3], b[..., mean_start:mean_start + 3]
            mean = (wa[..., None] * ua + wb[..., None] * ub) / total[..., None]
            central = result[f'point/central_{kind}'][:]
            report[f'mean_{kind}_max_abs'] = float(np.max(np.abs(mean - result[f'point/mean_{kind}'][:])))
            error = 0.0
            for j in range(3):
                for i in range(3):
                    component = central_start + i + 3*j
                    reference = (a[..., component] + wa * (ua[..., i] - mean[..., i]) * (ua[..., j] - mean[..., j]) +
                                 b[..., component] + wb * (ub[..., i] - mean[..., i]) * (ub[..., j] - mean[..., j]))
                    error = max(error, float(np.max(np.abs(reference - central[..., i, j]))))
            report[f'central_{kind}_max_abs'] = error
            np.testing.assert_array_equal(central, central.swapaxes(-1, -2))
            covariance = result[f'point/covariance_{kind}'][:]
            np.testing.assert_array_equal(covariance, central / total[..., None, None])
            np.testing.assert_array_equal(result[f'point/rms_{kind}'][:],
                                          np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1)))
        np.testing.assert_array_equal(result['point/density_stress'][:],
            result['point/central_favre'][:] / result['point/duration'][:][..., None, None])
        if not all(np.isfinite(result['point'][key][:]).all() for key in result['point']):
            raise AssertionError('nonfinite merged point statistics')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for backend in ('cpu', 'gpu'):
        for precision in ('fp32', 'fp64'):
            parser.add_argument(f'--{backend}-{precision}', type=Path, required=True)
    parser.add_argument('--mpiexec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['steps', 'time'], default='steps')
    parser.add_argument('--merge-statistics', action='store_true')
    parser.add_argument('--backends', nargs='+', choices=['cpu', 'gpu'], default=['cpu', 'gpu'])
    parser.add_argument('--schemes', nargs='+', choices=['643e', '643c'], default=['643e', '643c'])
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(exist_ok=False)
    os.environ.update(OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,vader,tcp',
                      OMPI_MCA_osc='pt2pt', OMPI_MCA_coll='^cuda,hcoll,ucc',
                      OMPI_MCA_opal_cuda_support='0')
    report = dict(source_steps=5, target_steps=7, checkpoint_schedule_mode=args.mode, cases={})
    for scheme in args.schemes:
        for backend in args.backends:
            for ranks in ([1, 2] if scheme == '643e' else [2]):
                for source_precision, target_precision in (('fp32', 'fp64'), ('fp64', 'fp32')):
                    label = f'{backend}_{scheme}_np{ranks}_{source_precision}_to_{target_precision}'
                    local = SimpleNamespace(
                        output=args.output, executable=getattr(args, f'{backend}_{source_precision}').resolve(),
                        mpiexec=args.mpiexec.resolve(), case='tgv', mode=args.mode, restart_step=5,
                        initial_dimension=0, legacy_statistics=False, statistics=scheme == '643e',
                        initial_restart=False, filter_workspace='scalar', force='feedback',
                        axis='x', no_samples=True, wall_samples=False)
                    options = dict(scheme=scheme, gpu_precision_mode='')
                    seed, _ = run_case(local, ROOT, backend, ranks,
                        f'{scheme}_{source_precision}_seed_for_{target_precision}', 5, **options)
                    source = seed / 'outdat/new/checkpoints/step000000000005'
                    before = {p: (p.stat().st_size, p.stat().st_mtime_ns)
                              for p in seed.rglob('*') if p.is_file()}
                    local.executable = getattr(args, f'{backend}_{target_precision}').resolve()
                    resumed, _ = run_case(local, ROOT, backend, ranks,
                        f'{scheme}_{target_precision}_resumed_from_{source_precision}', 12,
                        restore=source, exact_restart=False, restore_probe=True, **options)
                    result = dict(restored_copy=check_conversion(source, resumed, target_precision))
                    if 'ASTR_OUTPUT_PRECISION_SWITCH' not in (resumed / 'run.log').read_text():
                        raise AssertionError('precision transition was not recorded')
                    final = resumed / 'outdat/new/checkpoints/step000000000012'
                    result['final_state'] = check_state(final / 'state.h5', target_precision)
                    if local.statistics:
                        with h5py.File(resumed / 'outdat/restore_statistics_probe.h5') as statistics:
                            if np.any(statistics['q0008'][:] != 0) or np.any(statistics['q0009'][:] != 0):
                                raise AssertionError('source statistics were silently inherited across precision')
                        with h5py.File(final / 'statistics.h5') as statistics:
                            weights = statistics['q0008'][:]
                            # This periodic fixture samples unique nodes, not the
                            # repeated terminal planes padded by the GPU provider.
                            unique = weights[:-1, :-1, :-1]
                            previous = statistics['q0034'][:-1, :-1, :-1]
                            if not np.isfinite(weights).all() or not (unique > 0).all() or not (previous == 1).all():
                                raise AssertionError('new statistics segment has invalid duration')
                            result['new_segment_duration_range'] = [float(unique.min()), float(unique.max())]
                        if args.merge_statistics:
                            merged = args.output / (label + '_merged')
                            subprocess.run([sys.executable, str(ROOT / 'scripts/output/merge_statistics_segments.py'),
                                str(source / 'statistics.h5'), str(final / 'statistics.h5'),
                                '--output', str(merged), '--buffer-bytes', '65536'], check=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                            result['offline_merge'] = json.loads((merged / 'COMPLETE.json').read_text())
                            result['offline_merge_algebra'] = check_merged_statistics(merged, result['offline_merge'])
                    if any((p.stat().st_size, p.stat().st_mtime_ns) != value for p, value in before.items()):
                        raise AssertionError('source checkpoint run was modified')
                    result['source_files_unchanged'] = True
                    report['cases'][label] = result
                    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
                    print(label + ': passed', flush=True)
    report['status'] = 'passed'
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
