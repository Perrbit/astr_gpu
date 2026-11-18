"""Actual ASTR time statistics, observation isolation and exact continuation."""
import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tests/gpu_validation'))
from test_index_slice_runtime import arguments, product, segments, frames
from run_output_restart_validation import run_case
from run_cpp_insitu_gate import compare_hdf
from run_cpp_tgv_gate import logged_steps


def configuration(enabled):
    unlimited = np.iinfo(np.int64).max // 4096
    return f"""&insitu_run
 enabled={'.true.' if enabled else '.false.'},statistics={'.true.' if enabled else '.false.'},render=f,
 statistics_window=.0005,.0115,output_directory='outdat/render',
 schedule_mode='steps',step_interval=4,initial_frame=f,final_frame=t,
 host_budget_bytes={unlimited},device_budget_bytes={unlimited},device_reserve_bytes=0
/
"""


def compare_statistics(actual, reference, grid, precision='fp64'):
    errors = {}
    dtype = np.dtype('float32' if precision == 'fp32' else 'float64')
    word_type = np.int32 if precision == 'fp32' else np.int64
    limit = None if precision == 'fp32' else 2e-10
    with h5py.File(actual) as a, h5py.File(reference) as b:
        if any(f['numerical_kind_bytes'][...].tolist() != [dtype.itemsize] for f in (a, b)):
            raise AssertionError('statistics numerical kind differs')
        if not np.array_equal(a['identity'][...][:-1], b['identity'][...][:-1]):
            raise AssertionError('statistics flow identity differs')
        if int(a['identity'][...][-1]) != 4 or int(b['identity'][...][-1]) not in (3, 4):
            raise AssertionError('statistics storage role')
        first, second = a['metadata'][...], b['metadata'][...]
        if not np.array_equal(first[[0, 2, 3]], second[[0, 2, 3]]):
            raise AssertionError('statistics metadata phase differs')
        if not np.array_equal(a['partitions'][...], b['partitions'][...]):
            raise AssertionError('statistics partition differs')
        x, y = [words[4:].astype(word_type).view(dtype).astype(np.float64) for words in (first, second)]
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise AssertionError('nonfinite regional statistics')
        errors['regional_state'] = float(np.max(abs(x-y))/max(1., float(np.max(abs(y)))))
        for c in range(1, 35):
            key = f'q{c:04d}'
            if a[key].dtype != dtype or b[key].dtype != dtype:
                raise AssertionError((key, 'statistics payload kind differs'))
            x, y = a[key][:grid, :grid, :grid], b[key][:grid, :grid, :grid]
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise AssertionError((key, 'nonfinite point state'))
            errors[key] = float(np.max(abs(x-y))/max(1., float(np.max(abs(y)))))
        extras = a['rank_extras'][...]
        if not np.isfinite(extras).all() or np.count_nonzero(extras):
            raise AssertionError('native statistics duplicate-node padding is not zero')
    if not all(np.isfinite(value) and (limit is None or value <= limit) for value in errors.values()):
        raise AssertionError(errors)
    return errors


def final_output(case, rank):
    path = case/f'outdat/render/sample.statistics.step00000012.rank{rank:08d}.bin'
    with path.open('rb') as stream:
        if stream.read(8) != b'ASTRST01':
            raise AssertionError('final statistics magic')
        header = np.fromfile(stream, '<i4', 9)
        metadata = np.fromfile(stream, '<f8', 10)
        values = np.fromfile(stream, '<f8')
    if header[0] != 1 or header[1] != 12 or header[2] != rank or not np.isfinite(values).all():
        raise AssertionError('final statistics phase/layout')
    return metadata, values.reshape((*header[3:6], 41), order='F')


def compare_output(actual, reference, ranks):
    error = raw_rms = 0.
    for rank in range(ranks):
        _, x = final_output(actual, rank)
        _, y = final_output(reference, rank)
        if x.shape != y.shape:
            raise AssertionError('statistics result dimensions differ')
        # Unique periodic nodes; CPU may retain different seam copies.
        x, y = x[:-1, :-1, :-1].copy(), y[:-1, :-1, :-1].copy()
        raw_rms = max(raw_rms, float(np.max(abs(x[..., 8:14]-y[..., 8:14]))))
        x[..., 8:14] **= 2
        y[..., 8:14] **= 2
        for c in range(41):
            value = float(np.max(abs(x[..., c]-y[..., c]))/max(1., float(np.max(abs(y[..., c])))))
            error = max(error, value)
        if error > 2e-10:
            raise AssertionError(('derived statistics', error))
    return dict(result_scaled=error, raw_rms_maxabs=raw_rms)


def mean_products(grid):
    fields = ['velocity_x','velocity_y','velocity_z',
              'velocity_gradient_xx','velocity_gradient_yx','velocity_gradient_zx',
              'velocity_gradient_xy','velocity_gradient_yy','velocity_gradient_zy',
              'velocity_gradient_xz','velocity_gradient_yz','velocity_gradient_zz',
              'Q_rs','velocity_divergence','vorticity_x','vorticity_y','vorticity_z']
    definitions = [product('mean-reynolds', ['density',*fields,'density_gradient_magnitude'], 'reynolds'),
                   product('mean-favre', fields, 'favre'),
                   product('unavailable-mean-pressure', ['pressure'], 'reynolds')]
    for definition in definitions:
        definition['statistics_source'] = 'velocity-time'
        definition['sampling'] = dict(mode='grid_index',axis='k',index=grid//4,
                                      index_ranges=dict(i=[0,grid],j=[0,grid]))
    return definitions


def compare_mean_products(actual, reference, exact=False):
    a = {p.parent.parent.name: p for p in segments(actual)}
    b = {p.parent.parent.name: p for p in segments(reference)}
    if set(a) != {'mean-reynolds','mean-favre'} or set(a) != set(b):
        raise AssertionError(('mean product identities', set(a), set(b)))
    status = json.loads((actual/'outdat/new/products.status.json').read_text())
    for item in status['products']:
        expected = 'invalid-configuration' if item['id']=='unavailable-mean-pressure' else 'active'
        if item['state'] != expected:
            raise AssertionError(item)
    errors = {}
    for name in a:
        if exact:
            # A restarted archive creates a new segment at its first output.
            # Geometry values stay exact; its creation clock belongs to that segment.
            with h5py.File(a[name]/'geometry.h5') as left, h5py.File(b[name]/'geometry.h5') as right:
                if set(left) != set(right):
                    raise AssertionError('geometry dataset identities differ')
                for key in set(left)-{'clock','step'}:
                    if left[key][...].tobytes() != right[key][...].tobytes():
                        raise AssertionError(('restored geometry bits',key))
            for segment in (a[name],b[name]):
                with h5py.File(segment/'geometry.h5') as geometry, h5py.File(frames(segment)[0]) as first:
                    for key in ('clock','step'):
                        if not np.array_equal(geometry[key][...],first[key][...]):
                            raise AssertionError(('segment creation identity',key))
            error = 0.
        else:
            error = compare_hdf(a[name]/'geometry.h5', b[name]/'geometry.h5')
        expected_frames = ['step000000000009.h5'] if exact else [f'step{step:012d}.h5' for step in (1,5,9)]
        current_frames = frames(a[name])
        if [p.name for p in current_frames] != expected_frames:
            raise AssertionError(('mean product sampling clock', name, current_frames))
        for path in current_frames:
            error = max(error, compare_hdf(path,b[name]/path.name,exact=exact))
        errors[name] = error
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--cpu-reference', type=Path, required=True)
    parser.add_argument('--gpu-reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ranks', type=int, default=1)
    parser.add_argument('--axis', choices=['x', 'y', 'z'], default='x')
    parser.add_argument('--mapping', choices=['periodic'])
    parser.add_argument('--automatic', action='store_true')
    parser.add_argument('--means', action='store_true')
    parser.add_argument('--memcheck', action='store_true')
    options = parser.parse_args()
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    args = arguments(output)
    args.executable = options.binary.resolve();args.axis = options.axis
    args.unrestricted = True;args.insitu_write_eps = '0'
    grid = 32 if options.mapping else 16
    common = dict(grid=','.join([str(grid)]*3), tgv_mapping=options.mapping,
                  numerics=dict(filter_scheme='explicit10'), checkpoint_interval=5)
    definitions = mean_products(grid) if options.means else []
    def control(steps):
        timestep = ("mode='auto',cfl_target=.5,initial_dt=.001,max_dt=.002,growth_factor=1.1" if options.automatic
                    else "mode='fixed',fixed_dt=.001")
        return f"""&run_control
 max_steps={steps},reload_steps=1
/
&timestep
 {timestep}
/
&logging
 list_steps=1
/
&repair
 crashfix=f
/
&statistics
 enabled=f
/
"""
    baseline, _ = run_case(args, ROOT, 'gpu', options.ranks, 'cpp-unobserved', 12,
        insitu_config=configuration(False), run_config_text=control(12), product_catalog_records=[], **common)
    actual, _ = run_case(args, ROOT, 'gpu', options.ranks, 'cpp-statistics', 12,
        insitu_config=configuration(True), run_config_text=control(12), product_catalog_records=definitions,
        memcheck=options.memcheck, **common)
    if 'backend=cuda' not in (actual/'run.log').read_text():
        raise AssertionError('new C++ backend was not selected')
    final = 'outdat/new/checkpoints/step000000000012'
    compare_hdf(actual/final/'state.h5', baseline/final/'state.h5', exact=True)
    if not np.array_equal(logged_steps(actual, 12), logged_steps(baseline, 12)):
        raise AssertionError('statistics changed step control')
    report = dict(ranks=options.ranks, axis=options.axis, mapping=options.mapping, automatic=options.automatic, means=options.means,
                  memcheck=options.memcheck,authoritative_on_off_bits=True, references={})
    for name, binary, backend in (('cpu', options.cpu_reference, 'cpu'), ('old_gpu', options.gpu_reference, 'gpu')):
        args.executable = binary.resolve()
        reference, _ = run_case(args, ROOT, backend, options.ranks, name, 12,
            insitu_config=configuration(True), run_config_text=control(12), product_catalog_records=definitions, **common)
        report['references'][name] = dict(
            terminal_scaled=compare_hdf(actual/final/'state.h5', reference/final/'state.h5'),
            state=compare_statistics(actual/final/'statistics.h5', reference/final/'statistics.h5', grid),
            output=compare_output(actual, reference, options.ranks))
        if options.means:
            report['references'][name]['means'] = compare_mean_products(actual,reference)
    args.executable = options.binary.resolve()
    prefix, _ = run_case(args, ROOT, 'gpu', options.ranks, 'cpp-prefix', 5,
        insitu_config=configuration(True), run_config_text=control(5), product_catalog_records=definitions, **common)
    resumed, _ = run_case(args, ROOT, 'gpu', options.ranks, 'cpp-resumed', 12,
        restore=prefix/'outdat/new/checkpoints/step000000000005',
        insitu_config=configuration(True), run_config_text=control(12), product_catalog_records=definitions, **common)
    for name in ('state.h5', 'statistics.h5'):
        compare_hdf(actual/final/name, resumed/final/name, exact=True)
    if not np.array_equal(logged_steps(actual, 12)[5:], logged_steps(resumed, 7, first_step=6)):
        raise AssertionError('restored statistics changed step control')
    for rank in range(options.ranks):
        first, second = final_output(actual, rank), final_output(resumed, rank)
        if any(a.tobytes()!=b.tobytes() for a, b in zip(first, second)):
            raise AssertionError('same-backend final statistics output bits differ')
    report['same_backend_restart'] = dict(state_bits=True, statistics_bits=True, output_bits=True, timestep_history=True)
    if options.means:
        report['same_backend_restart']['means'] = compare_mean_products(resumed,actual,exact=True)
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
