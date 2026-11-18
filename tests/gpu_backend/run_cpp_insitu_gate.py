"""Completed-step products from the actual C++ ASTR state; no Python solver."""
import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests/gpu_validation'))
from test_index_slice_runtime import arguments, product, segments, frames
from run_output_restart_validation import run_case
from run_cpp_tgv_gate import logged_steps, cfl_log


def compare_hdf(first, second, exact=False, limit=2e-10):
    maximum = 0.
    with h5py.File(first) as a, h5py.File(second) as b:
        names = []
        a.visititems(lambda name, item: names.append(name) if isinstance(item, h5py.Dataset) else None)
        other = []
        b.visititems(lambda name, item: other.append(name) if isinstance(item, h5py.Dataset) else None)
        if names != other:
            raise AssertionError((first, second, 'dataset identities differ'))
        for name in names:
            x, y = a[name][...], b[name][...]
            if x.shape != y.shape or x.dtype != y.dtype:
                raise AssertionError((name, 'layout differs'))
            if exact:
                if x.tobytes() != y.tobytes():
                    raise AssertionError((name, 'authoritative bits changed by observation'))
            elif x.dtype.kind == 'f':
                if not np.isfinite(x).all() or not np.isfinite(y).all():
                    raise AssertionError((name, 'nonfinite comparison'))
                error = float(np.max(abs(x-y), initial=0.) / max(1., float(np.max(abs(y), initial=0.))))
                maximum = max(maximum, error)
                if limit is not None and error > limit:
                    raise AssertionError((name, error))
            elif not np.array_equal(x, y):
                raise AssertionError((name, 'nonfloating metadata differs'))
    return maximum


def run_control(options, steps):
    if not options.automatic:
        return None
    return f"""&run_control
 max_steps={steps},reload_steps=1,stop_time=-1
/
&timestep
 mode='auto',cfl_target=0.5,initial_dt=0.001,max_dt=0.002,growth_factor=1.1
/
&logging
 list_steps=1
/
&repair
 crashfix=.false.
/
&statistics
 enabled=.false.,sample_clock='steps',sample_steps=1,write_steps=0
/
"""


def check_clock(options, actual, baseline, reference):
    rows = logged_steps(actual)
    if not np.array_equal(rows, logged_steps(baseline)):
        raise AssertionError('observation changed step control')
    if not np.allclose(rows, logged_steps(reference), rtol=0., atol=2e-15):
        raise AssertionError('new and frozen step clocks differ')
    cfl = cfl_log(actual)
    if not np.isfinite(cfl).all() or (options.automatic and np.max(cfl[:, -1]) > .5):
        raise AssertionError('nonfinite or excessive acoustic CFL')
    if options.automatic:
        if options.precision == 'fp32':
            expected = np.empty(4, dtype=np.float32)
            expected[0] = .001
            for index in range(1, 4):
                expected[index] = expected[index-1] * np.float32(1.1)
            same = np.array_equal(rows[:, 3], expected)
        else:
            same = np.allclose(rows[:, 3], .001*1.1**np.arange(4), rtol=0., atol=2e-18)
        if not same:
            raise AssertionError('fixture did not exercise adaptive step growth')
    return dict(steps=rows.tolist(), maximum_cfl_upper_bound=float(np.max(cfl[:, -1])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ranks', type=int, default=1)
    parser.add_argument('--axis', choices=['x', 'y', 'z'], default='x')
    parser.add_argument('--images', action='store_true')
    parser.add_argument('--restart', action='store_true')
    parser.add_argument('--pipeline', choices=['standard-device', 'direct-device'], default='standard-device')
    parser.add_argument('--transport', choices=['pinned', 'device-aware'], default='pinned')
    parser.add_argument('--mapping', choices=['periodic', 'y-wavy'])
    parser.add_argument('--automatic', action='store_true')
    parser.add_argument('--precision', choices=['fp64', 'fp32'], default='fp64')
    options = parser.parse_args()
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    args = arguments(output)
    args.executable = options.binary.resolve()
    args.axis = options.axis
    args.unrestricted = True
    args.insitu_write_eps = '0'
    args.native_precision = options.precision
    args.wall_samples = False
    if options.mapping == 'y-wavy' or (options.mapping and options.precision == 'fp32'):
        args.scale_timestep = 2e-5
        args.maximum_cfl = .5
    if options.images:
        render_gate(options, args, output)
        return
    grid = 32 if options.mapping else 16
    common = dict(grid=','.join([str(grid)]*3), tgv_mapping=options.mapping, solver_halo_transport='pinned',
                  numerics=dict(filter_scheme='explicit10'),
                  gpu_precision_mode='' if options.precision == 'fp32' else 'fp64')
    fields = ['density', 'velocity_x', 'velocity_y', 'velocity_z', 'pressure', 'temperature',
              'velocity_gradient_xx', 'velocity_gradient_yx', 'velocity_gradient_zx',
              'velocity_gradient_xy', 'velocity_gradient_yy', 'velocity_gradient_zy',
              'velocity_gradient_xz', 'velocity_gradient_yz', 'velocity_gradient_zz',
              'Q_rs', 'velocity_divergence', 'vorticity_x', 'vorticity_y', 'vorticity_z',
              'density_gradient_magnitude', 'temperature_gradient_x', 'temperature_gradient_y',
              'temperature_gradient_z', 'lambda2', 'lambda_ci']
    if options.mapping:
        # The mature shared CURVE source does not admit thermal/vortex additions.
        fields = fields[:21]
    definitions = []
    for axis in 'ijk':
        definition = product(name='plane-'+axis, fields=fields)
        definition['sampling'] = dict(mode='grid_index', axis=axis, index=4,
            index_ranges={other: [0, grid] for other in 'ijk' if other != axis})
        definitions.append(definition)
    limit = None if options.precision == 'fp32' else 2e-10
    report = dict(ranks=options.ranks, axis=options.axis, mapping=options.mapping, solver_transport='pinned',
                  precision=options.precision, difference_limit=limit,
                  numerical_strip_transport='explicit-host', requested_fields=fields, cases={})
    checkpoint_interval = 2 if options.restart else 4
    baseline, _ = run_case(args, ROOT, 'gpu', options.ranks, 'cpp-unobserved', 4,
        checkpoint_interval=checkpoint_interval, product_catalog_records=[], run_config_text=run_control(options, 4), **common)
    actual, _ = run_case(args, ROOT, 'gpu', options.ranks, 'cpp-products', 4,
        checkpoint_interval=checkpoint_interval, product_catalog_records=definitions, run_config_text=run_control(options, 4), **common)
    if 'backend=cuda' not in (actual/'run.log').read_text():
        raise AssertionError('new C++ main was not selected')
    final = 'outdat/new/checkpoints/step000000000004/state.h5'
    compare_hdf(actual/final, baseline/final, exact=True)
    args.executable = options.reference.resolve()
    reference, _ = run_case(args, ROOT, 'gpu', options.ranks, 'legacy-products', 4,
        checkpoint_interval=checkpoint_interval, product_catalog_records=definitions, run_config_text=run_control(options, 4), **common)
    report['clock'] = check_clock(options, actual, baseline, reference)
    report['terminal_scaled_error'] = compare_hdf(actual/final, reference/final, limit=limit)
    actual_segments, reference_segments = segments(actual), segments(reference)
    if len(actual_segments) != 3 or len(reference_segments) != 3:
        raise AssertionError('three numerical slice products must be active')
    status = json.loads((actual/'outdat/new/products.status.json').read_text())
    if any(p['state'] != 'active' or p['last_output_step'] != 4 for p in status['products']):
        raise AssertionError(status)
    for a, b in zip(actual_segments, reference_segments):
        left, right = frames(a), frames(b)
        if [p.name for p in left] != ['step000000000000.h5', 'step000000000004.h5'] or [p.name for p in left] != [p.name for p in right]:
            raise AssertionError('completed-step identities differ')
        report['cases'][a.parent.parent.name] = dict(geometry=compare_hdf(a/'geometry.h5', b/'geometry.h5', limit=limit),
            fields=max(compare_hdf(x, y, limit=limit) for x, y in zip(left, right)))
    if options.restart:
        args.executable = options.binary.resolve()
        prefix, _ = run_case(args, ROOT, 'gpu', options.ranks, 'cpp-prefix', 2,
            checkpoint_interval=2, product_catalog_records=definitions, run_config_text=run_control(options, 2), **common)
        changed = [product(name='changed-on-disk', fields=['pressure'])]
        resumed, _ = run_case(args, ROOT, 'gpu', options.ranks, 'cpp-resumed', 4,
            restore=prefix/'outdat/new/checkpoints/step000000000002',
            checkpoint_interval=checkpoint_interval, product_catalog_records=changed,
            run_config_text=run_control(options, 4), **common)
        compare_hdf(resumed/final, actual/final, exact=True)
        resumed_segments = segments(resumed)
        if len(resumed_segments) != 3:
            raise AssertionError('saved catalog was not restored')
        for a, b in zip(resumed_segments, actual_segments):
            output_frames = frames(a)
            if [p.name for p in output_frames] != ['step000000000004.h5']:
                raise AssertionError('saved numerical product schedule changed')
            compare_hdf(output_frames[0], b/output_frames[0].name, exact=True)
        report['same_backend_restart'] = dict(authoritative_bits=True, slice_bits=True, saved_catalog_and_schedule=True)
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


def render_gate(options, args, output):
    from test_runtime_catalog_images import image_product, q_product, line_product, image_segments
    from test_insitu_device_products import resident_configuration
    from PIL import Image
    config = resident_configuration(options.pipeline, options.transport, statistics=False, interval=1,
                                    profile='curve_demo' if options.mapping else 'velocity_slice')
    # Positive representation of insitu_run_config's existing unlimited_budget_bytes.
    # This legacy provider envelope does not accept the -1 input spelling.
    unlimited = np.iinfo(np.int64).max // 4096
    config = config.replace('host_budget_bytes=4294967296', f'host_budget_bytes={unlimited}')
    config = config.replace('device_budget_bytes=2147483648', f'device_budget_bytes={unlimited}')
    config = config.replace('device_reserve_bytes=1073741824', 'device_reserve_bytes=0')
    definitions = [image_product('slice'), q_product('vortices'), line_product('lines')]
    if options.automatic:
        for definition in definitions:
            definition['schedule'] = dict(mode='time', interval=.002)
    common = dict(grid='32,32,32', tgv_mapping=options.mapping, numerics=dict(filter_scheme='explicit10'),
                  gpu_precision_mode='' if options.precision == 'fp32' else 'fp64')
    limit = None if options.precision == 'fp32' else 2e-10
    checkpoint_interval = 2 if options.restart else 4
    args.executable = options.reference.resolve()
    reference, _ = run_case(args, ROOT, 'gpu', options.ranks, 'legacy-images', 4,
        checkpoint_interval=checkpoint_interval, product_catalog_records=definitions, insitu_config=config,
        postprocess_transport=options.transport, run_config_text=run_control(options, 4), **common)
    args.executable = options.binary.resolve()
    baseline, _ = run_case(args, ROOT, 'gpu', options.ranks, 'cpp-no-render', 4,
        checkpoint_interval=checkpoint_interval, product_catalog_records=[], run_config_text=run_control(options, 4), **common)
    actual, _ = run_case(args, ROOT, 'gpu', options.ranks, 'cpp-images', 4,
        checkpoint_interval=checkpoint_interval, product_catalog_records=definitions, insitu_config=config,
        postprocess_transport=options.transport, run_config_text=run_control(options, 4), **common)
    final = 'outdat/new/checkpoints/step000000000004/state.h5'
    compare_hdf(actual/final, baseline/final, exact=True)
    if 'backend=cuda' not in (actual/'run.log').read_text():
        raise AssertionError('new C++ main was not selected')
    report = dict(ranks=options.ranks, axis=options.axis, mapping=options.mapping,pipeline=options.pipeline, transport=options.transport,
                  precision=options.precision, difference_limit=limit,
                  automatic=options.automatic, product_clock='time' if options.automatic else 'steps',
                  terminal_scaled_error=compare_hdf(actual/final, reference/final, limit=limit), products={})
    with h5py.File(actual/final) as checkpoint:
        width = 4 if options.precision == 'fp32' else 8
        if any(checkpoint[name].dtype.itemsize != width for name in checkpoint if name.startswith('q')):
            raise AssertionError('rendering did not use the requested native state type')
        rho = checkpoint['q0001'][...].astype(np.float64)
        momenta = [checkpoint[f'q{index:04d}'][...].astype(np.float64) for index in (2, 3, 4)]
        internal = checkpoint['q0005'][...].astype(np.float64) - .5*sum(m*m for m in momenta)/rho
        if np.any(rho <= 0) or not np.isfinite(internal).all() or np.any(internal <= 0):
            raise AssertionError('rendered main has nonpositive density/internal energy')
        if any(np.any(checkpoint[f'q{index:04d}'][...] <= 0) for index in (10, 11)):
            raise AssertionError('rendered main has nonpositive pressure/temperature')
        report['native_state_bytes'] = width
    report['clock'] = check_clock(options, actual, baseline, reference)
    event_times = {0: 0., **{int(row[0]): row[2] for row in report['clock']['steps']}}
    status = json.loads((actual/'outdat/new/products.status.json').read_text())
    if len(status['products']) != 3 or any(p['state'] != 'active' or p['last_output_step'] != 4 for p in status['products']):
        raise AssertionError(status)
    actual_segments, reference_segments = image_segments(actual), image_segments(reference)
    if len(actual_segments) != 3 or len(reference_segments) != 3:
        raise AssertionError('slice/Q/streamline image products must all be present')
    for a, b in zip(actual_segments, reference_segments):
        left, right = sorted(a.glob('*.jpeg')), sorted(b.glob('*.jpeg'))
        if [p.name for p in left] != [p.name for p in right] or len(left) != 3:
            raise AssertionError('expected step 0/2/4 images in both backends')
        differences = []
        for x, y in zip(left, right):
            image, old = np.asarray(Image.open(x).convert('RGB')), np.asarray(Image.open(y).convert('RGB'))
            if image.shape != old.shape or float(image.std()) < 1.:
                raise AssertionError('blank or dimensionally inconsistent rendered product')
            differences.append(float(np.mean(abs(image.astype(float)-old.astype(float)))))
        receipts = [json.loads(path.read_text()) for path in a.glob('frame.*.rank*.json')]
        if len(receipts) != 3*options.ranks or any(r['field_d2h_bytes'] != 0 or r['geometry_host_bytes'] != 0 for r in receipts):
            raise AssertionError('device images used field/geometry readback')
        for step in (0, 2, 4):
            at_step = [r for r in receipts if r['step'] == step]
            if len(at_step) != options.ranks or any(r['time'] != event_times[step] for r in at_step):
                raise AssertionError('product receipt is not at the completed-step time')
            if sum(r['product']['local_cells'] for r in receipts if r['step'] == step) <= 0:
                raise AssertionError('rendered product geometry is globally empty')
        if list(a.glob('*.eps')):
            raise AssertionError('EPS was explicitly disabled')
        report['products'][a.parent.parent.name] = dict(images=len(left), receipts=len(receipts),
            old_new_pixel_mean_absolute_difference=differences)
    if options.restart:
        args.executable = options.binary.resolve()
        prefix, _ = run_case(args, ROOT, 'gpu', options.ranks, 'cpp-images-prefix', 2,
            checkpoint_interval=checkpoint_interval, product_catalog_records=definitions, insitu_config=config,
            postprocess_transport=options.transport, run_config_text=run_control(options, 2), **common)
        resumed, _ = run_case(args, ROOT, 'gpu', options.ranks, 'cpp-images-resumed', 4,
            restore=prefix/'outdat/new/checkpoints/step000000000002', checkpoint_interval=checkpoint_interval,
            product_catalog_records=[], insitu_config=config, postprocess_transport=options.transport,
            run_config_text=run_control(options, 4), **common)
        compare_hdf(resumed/final, actual/final, exact=True)
        for name in ('control.bin', 'insitu_control.bin'):
            if (resumed/final).with_name(name).read_bytes() != (actual/final).with_name(name).read_bytes():
                raise AssertionError('same-backend rendering control differs: '+name)
        if not np.array_equal(logged_steps(resumed, count=2, first_step=3), logged_steps(actual)[2:]):
            raise AssertionError('restart changed adaptive step history')
        resumed_segments = image_segments(resumed)
        if len(resumed_segments) != 3:
            raise AssertionError('image catalog was not restored')
        differences = {}
        for a, b in zip(resumed_segments, actual_segments):
            files = sorted(a.glob('*.jpeg'))
            if len(files) != 1 or files[0].name != 'frame.step000000000004.jpeg':
                raise AssertionError('image schedule repeated/skipped a saved event')
            receipts = [json.loads(path.read_text()) for path in a.glob('frame.*.rank*.json')]
            if len(receipts) != options.ranks or any(r['step'] != 4 or r['time'] != event_times[4] for r in receipts):
                raise AssertionError('restored product time differs')
            x, y = np.asarray(Image.open(files[0]).convert('RGB')), np.asarray(Image.open(b/files[0].name).convert('RGB'))
            differences[a.parent.parent.name] = float(np.mean(abs(x.astype(float)-y.astype(float))))
        if any(value != 0. for value in differences.values()):
            raise AssertionError('same-backend continuation changed image pixels')
        status = json.loads((resumed/'outdat/new/products.status.json').read_text())
        if any(p['state'] != 'active' or p['last_output_step'] != 4 for p in status['products']):
            raise AssertionError('restored image products are inactive or use the wrong phase')
        report['same_backend_restart'] = dict(authoritative_bits=True, saved_image_catalog_and_schedule=True,
                                              pixel_mean_absolute_difference=differences)
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
