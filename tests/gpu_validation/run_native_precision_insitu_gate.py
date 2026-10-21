"""Actual native-kind in-situ products and exact continuation, driven by ASTR."""

import argparse
import json
import os
from pathlib import Path
import re
import sqlite3
from types import SimpleNamespace

import h5py
import numpy as np
from PIL import Image

from run_native_precision_restart_gate import check_state
from run_output_restart_validation import archive_schedule_payload, compare_fields, run_case
from test_insitu_device_products import resident_configuration
from test_output_insitu_restart import configuration as host_configuration
from run_native_precision_adaptive_gate import CONFIG as ADAPTIVE_CONFIG, GROUPS as ADAPTIVE_GROUPS, check_log
from run_output_archive_validation import frames, check_series
from test_insitu_curve_derivatives import index_coordinates
from test_insitu_wall_scalar_statistics import read_statistics as read_wall_statistics


ROOT = Path(__file__).resolve().parents[2]
TGV_PRODUCTS = {'velocity_slice', 'q_surface', 'instantaneous_streamlines', 'crossing_streamlines',
                'mean_reynolds_streamlines', 'mean_favre_streamlines'}


def configuration(library, pipeline, profile='all', statistics=True, interval=2):
    value = resident_configuration(pipeline, 'pinned', statistics=statistics,
                                   interval=interval, profile=profile)
    value = re.sub(r"implementation_path='[^']+'", f"implementation_path='{library}'", value)
    return value.replace('final_frame=t', 'final_frame=f')


def images(case, ranks, steps, pipeline, products, width, wall_statistics=False):
    output = case / 'outdat/render'
    report = {}
    for step in steps:
        totals = dict.fromkeys(products, 0)
        for rank in range(ranks):
            receipt = json.loads((output / f'mesh_step{step:08d}_rank{rank}.json').read_text())
            if receipt['rendering_pipeline'] != pipeline or receipt['geometry_host_bytes'] != 0 or \
                    set(receipt['products']) != products:
                raise AssertionError('device product identity/readback differs')
            for name, product in receipt['products'].items():
                totals[name] += product['local_cells']
        if any(count <= 0 for count in totals.values()):
            raise AssertionError('device product has no cells globally')
        for name in products:
            with Image.open(output / f'{name}.step{step:08d}.jpeg') as image:
                pixels = np.asarray(image.convert('RGB'))
            if pixels.shape != (600, 800, 3) or np.count_nonzero(np.any(pixels < 220, axis=-1)) < 500:
                raise AssertionError('missing/blank native product image')
        report[str(step)] = totals
    samples = list(output.glob('sample.*.bin'))
    if list(output.glob('*.eps')) or list(output.glob('*.vtp')) or any(not wall_statistics or not re.fullmatch(
            r'sample\.wall_statistics\.step\d{8}\.rank\d{8}\.bin', path.name) for path in samples):
        raise AssertionError('JPEG-only strict rendering emitted an extra geometry product')
    if wall_statistics:
        report['wall_statistics_exports'] = []
        for path in samples:
            header, clock, count, xyz, values, _ = read_wall_statistics(path)
            if header[0] != (2 if width == 4 else 1) or values.dtype.itemsize != width:
                raise AssertionError('wall statistics export precision differs')
            report['wall_statistics_exports'].append(dict(file=path.name, numerical_bytes=width, samples=count))
    text = (case / 'run.log').read_text()
    for field, display in re.findall(r'field_maxabs=(\S+) display_maxabs=(\S+)', text):
        if not np.isfinite(float(field)) or float(display) != 0 or (width == 8 and float(field) > 2e-10):
            raise AssertionError('native numerical/display audit failed')
    report['numerical_audits'] = re.findall(r'ASTR_INSITU_RESIDENT_AUDIT[^\n]+', text)
    return report


def state_pair(reference, candidate, precision):
    report = check_state(candidate, precision)
    with h5py.File(reference) as a, h5py.File(candidate) as b:
        errors = {}
        for name in a:
            if not name.startswith('q'):
                continue
            aa, bb = a[name][:].astype(np.float64), b[name][:].astype(np.float64)
            difference = np.abs(aa-bb)
            error = float(difference.max())
            if not np.isfinite(error) or (precision == 'fp64' and error > 2e-10):
                raise AssertionError(f'CPU/GPU native field regression: {name}')
            errors[name] = dict(max_abs=error, index_zyx=list(map(int, np.unravel_index(
                difference.argmax(), difference.shape))))
    report['cpu_gpu_field_differences'] = errors
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--precision', choices=('fp32', 'fp64'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--groups', nargs='+', choices=('derivatives', 'tgv', 'curve', 'joint', 'planes', 'wallmeans', 'safety', 'observe', 'preview'),
                        default=['derivatives', 'tgv', 'curve', 'joint'])
    parser.add_argument('--mpiexec', type=Path, default=Path(
        '/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpiexec'))
    opts = parser.parse_args()
    root = opts.out.resolve()
    root.mkdir(exist_ok=False)
    os.environ.update(OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,vader,tcp', OMPI_MCA_osc='pt2pt',
                      OMPI_MCA_coll='^cuda,hcoll,ucc', OMPI_MCA_opal_cuda_support='0')
    local = SimpleNamespace(output=root, executable=opts.exe.resolve(), mpiexec=opts.mpiexec.absolute(),
        case='tgv', mode='steps', restart_step=3, initial_dimension=0, legacy_statistics=False,
        statistics=True, initial_restart=False, filter_workspace='scalar', force='feedback', axis='x',
        no_samples=True, wall_samples=False, native_precision=opts.precision, insitu_write_eps='0',
        directory_budget_bytes=256*1024**2, runtime_timeout_seconds=600)
    width = 4 if opts.precision == 'fp32' else 8
    result = dict(precision=opts.precision, numerical_bytes=width, cases={})
    library = opts.library.resolve(strict=True)
    def save(name, record):
        result['cases'][name] = record
        (root / 'report.json').write_text(json.dumps(result, indent=2)+'\n')
        print(name + ': passed', flush=True)
    if 'preview' in opts.groups:
        import subprocess
        from test_insitu_preview import prepare_native_preview, source_snapshot, candidate_views
        config = configuration(library, 'standard-device', interval=4)
        config = config.replace('statistics_window=0.0005,0.0115', 'statistics_window=0.003,0.004')
        source, _ = run_case(local, ROOT, 'gpu', 2, 'timing_pinned', 4, grid='32,32,32',
            checkpoint_interval=4, insitu_config=config, gpu_precision_mode='')
        os.environ.update(ASTR_INSITU_PREVIEW_SOURCE_ROOT=str(root),
            ASTR_OUTPUT_INSITU_EXE=str(opts.exe.resolve()), ASTR_OUTPUT_INSITU_BACKEND=str(library))
        directory = root/'preview'; directory.mkdir()
        source_root, bundle, exe, launcher, env, output = prepare_native_preview(
            directory, 2, True, candidate_views())
        env.update(ASTR_INSITU_WRITE_EPS='0', OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,vader,tcp',
                   OMPI_MCA_coll='^cuda,hcoll,ucc', OMPI_MCA_osc='pt2pt', OMPI_MCA_opal_cuda_support='0')
        before = source_snapshot(source_root)
        with (directory/'run.log').open('w') as log:
            subprocess.run([str(launcher), '-np', '2', str(exe), 'insitu-preview', 'datin/input.tgv'],
                cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
        if source_snapshot(source_root) != before:
            raise AssertionError('Native preview modified its source')
        text = (directory/'run.log').read_text()
        if 'advances=0' not in text or text.count('flow_mismatches=0 statistics_mismatches=0') != 2:
            raise AssertionError('Native preview changed flow/statistics')
        cached = re.findall(r'ASTR_INSITU_PREVIEW_CACHE rank=(\d+) step=4 candidate=(\w+) '
                            r'velocity=(\w+) seeds=(\S+) integrated=(\d+)', text)
        for rank in range(2):
            counts = {name: int(integrated) for r, name, _, _, integrated in cached if int(r) == rank}
            if counts != dict(front=1, oblique=0, speed=0, stratified=1, reynolds=1, favre=1):
                raise AssertionError('Native preview trajectory reuse differs')
        pictures = list(output.glob('preview.*.jpeg'))
        if len(pictures) != 6 or list(output.glob('*.eps')) or list(output.glob('*.vtp')):
            raise AssertionError('Native preview product set differs')
        records = [json.loads(path.read_text()) for path in output.glob('preview.*.rank*.json')]
        if len(records) != 12 or any(record['geometry_host_bytes'] != 0 for record in records):
            raise AssertionError('Native preview geometry was read back')
        for path in pictures:
            with Image.open(path) as image:
                pixels = np.asarray(image.convert('RGB'))
            if pixels.shape != (600, 800, 3) or np.count_nonzero(np.any(pixels < 220, axis=-1)) < 500:
                raise AssertionError('Native preview image is missing/blank')
        save('preview', dict(source_unchanged=True, advances=0, trajectory_cache_checked=True,
                             pictures=6, numerical_bytes=width))
    if 'derivatives' in opts.groups:
        captures = {}
        for flow, derivative in (('gpu', 'cpu'), ('gpu', 'gpu')):
            config = host_configuration(statistics=False, interval=2)
            config = re.sub(r"implementation_path='[^']+'", f"implementation_path='{library}'", config)
            config = config.replace('scripts/insitu/tgv_pipeline.py',
                                    'tests/gpu_validation/insitu_diagnostic_capture.py')
            config = config.replace('render=.true.,', f"render=.true.,derivative_backend='{derivative}',")
            case, _ = run_case(local, ROOT, flow, 2, 'derivatives_'+derivative, 4, grid='32,32,32',
                checkpoint_interval=2, insitu_config=config, gpu_precision_mode='')
            captures[flow, derivative] = case
            for step in (2, 4):
                with h5py.File(case / f'outdat/new/checkpoints/step{step:012d}/state.h5') as state:
                    for rank in range(2):
                        indices = index_coordinates(2, 'x', rank)
                        # The sampled copy canonicalizes periodic upper endpoints to node zero.
                        selection = tuple(indices[d] % 32 for d in (2, 1, 0))
                        with np.load(case / f'outdat/render/fields.step{step:08d}.rank{rank:08d}.npz') as fields:
                            for name in fields.files:
                                if fields[name].dtype != np.dtype(f'<f{width}'):
                                    raise AssertionError('non-native host bridge payload: '+name)
                            for component in range(1, 6):
                                if not np.array_equal(fields[f'q{component}'], state[f'q{component:04d}'][:][selection]):
                                    raise AssertionError('host bridge differs from completed authority')
        record = {}
        reference = captures['gpu', 'cpu']
        for key, candidate in captures.items():
            errors = {}
            for step in (2, 4):
                for rank in range(2):
                    filename = f'outdat/render/fields.step{step:08d}.rank{rank:08d}.npz'
                    with np.load(reference / filename) as expected, np.load(candidate / filename) as actual:
                        if set(expected.files) != set(actual.files):
                            raise AssertionError('native derivative field inventory differs')
                        for name in expected.files:
                            error = float(np.max(np.abs(expected[name].astype(np.float64)-actual[name])))
                            if not np.isfinite(error) or (width == 8 and error > 2e-10):
                                raise AssertionError('native derivative field regression: '+name)
                            errors[name] = max(errors.get(name, 0.), error)
            record['_'.join(key)] = errors
        save('native_host_bridge_and_derivatives', record)
    if 'tgv' in opts.groups:
        for ranks in (1, 2):
            off = host_configuration(render=False, interval=2).replace(
                'final_frame=t', 'final_frame=f')
            cpu, _ = run_case(local, ROOT, 'cpu', ranks, 'cpu_reference', 4, grid='32,32,32',
                checkpoint_interval=1, insitu_config=off, gpu_precision_mode='')
            plain, _ = run_case(local, ROOT, 'gpu', ranks, 'gpu_reference', 4, grid='32,32,32',
                checkpoint_interval=1, insitu_config=off, gpu_precision_mode='')
            final = 'outdat/new/checkpoints/step000000000004'
            save(f'tgv_reference_np{ranks}', state_pair(cpu / final / 'state.h5', plain / final / 'state.h5', opts.precision))
            for pipeline in ('standard-device', 'direct-device'):
                name = f'tgv_{pipeline}_np{ranks}'
                active, _ = run_case(local, ROOT, 'gpu', ranks, name, 4, grid='32,32,32',
                    checkpoint_interval=1, insitu_config=configuration(library, pipeline),
                    gpu_precision_mode='', resident_audit=True)
                record = images(active, ranks, (2, 4), pipeline, TGV_PRODUCTS, width)
                for filename in ('state.h5', 'statistics.h5'):
                    compare_fields(plain / final / filename, active / final / filename)
                save(name, record)
    if 'curve' in opts.groups:
        curve_local = SimpleNamespace(**vars(local))
        curve_local.statistics = False
        curve_local.scale_timestep = 2e-5
        curve_local.maximum_cfl = .5
        for mapping in ('periodic', 'y-wavy'):
            plain, _ = run_case(curve_local, ROOT, 'gpu', 2, mapping+'_off', 4, grid='32,32,32',
                tgv_mapping=mapping, checkpoint_interval=1, insitu_config='&insitu_run enabled=f /\n',
                gpu_precision_mode='')
            for pipeline in ('standard-device', 'direct-device'):
                for profile in (('q_surface', 'streamlines', 'channel_walls') if mapping == 'y-wavy'
                                else ('q_surface', 'streamlines')):
                    name = f'{mapping}_{profile}_{pipeline}'
                    means = profile == 'streamlines'
                    config = configuration(library, pipeline, profile, means).replace(
                        'statistics_window=0.0005,0.0115', 'statistics_window=0.00001,0.0002')
                    if means:
                        config = config.replace('statistics=.true.',
                                                'statistics=.true.,mean_streamline_render=t')
                    active, _ = run_case(curve_local, ROOT, 'gpu', 2, name, 4, grid='32,32,32',
                        tgv_mapping=mapping, checkpoint_interval=1, insitu_config=config,
                        gpu_precision_mode='', resident_audit=True)
                    products = (TGV_PRODUCTS-{'velocity_slice', 'q_surface'} if means else
                        ({'q_surface'} if profile == 'q_surface' else
                         {'wall_pressure', 'wall_shear_x', 'wall_heat_into_gas'}))
                    record = images(active, 2, (2, 4), pipeline, products, width)
                    final = 'outdat/new/checkpoints/step000000000004/state.h5'
                    compare_fields(plain / final, active / final)
                    record['state'] = check_state(active / final, opts.precision)
                    save(name, record)
    if 'planes' in opts.groups:
        for mapping in (None, 'periodic'):
            plane_local = SimpleNamespace(**vars(local))
            plane_local.statistics = False
            if mapping:
                plane_local.scale_timestep = 2e-5
                plane_local.maximum_cfl = .5
            plain, _ = run_case(plane_local, ROOT, 'gpu', 2, f'plane_{mapping}_off', 4,
                grid='32,32,32', tgv_mapping=mapping, checkpoint_interval=1,
                insitu_config='&insitu_run enabled=f /\n', gpu_precision_mode='')
            for pipeline in ('standard-device', 'direct-device'):
                config = configuration(library, pipeline, 'velocity_slice', False).replace(
                    "products='velocity_slice',", "products='velocity_slice',slice_definition='plane',"
                    "slice_origin=3.,0.,0.,slice_normal=1.,.25,-.125,")
                name = f'plane_{mapping}_{pipeline}'
                active, _ = run_case(plane_local, ROOT, 'gpu', 2, name, 4, grid='32,32,32',
                    tgv_mapping=mapping, checkpoint_interval=1, insitu_config=config, gpu_precision_mode='')
                record = images(active, 2, (2, 4), pipeline, {'velocity_slice'}, width)
                final = 'outdat/new/checkpoints/step000000000004/state.h5'
                compare_fields(plain/final, active/final)
                save(name, record)
    if 'wallmeans' in opts.groups:
        wall_local = SimpleNamespace(**vars(local))
        wall_local.statistics = False
        wall_local.scale_timestep = 2e-5
        wall_local.maximum_cfl = .5
        for pipeline in ('standard-device', 'direct-device'):
            config = configuration(library, pipeline, 'channel_walls', True).replace(
                'statistics_window=0.0005,0.0115', 'statistics_window=0.00001,0.0002').replace(
                'statistics=.true.', 'statistics=.true.,wall_mean_render=t')
            options = dict(grid='32,32,32', tgv_mapping='y-wavy', checkpoint_interval=1,
                           insitu_config=config, gpu_precision_mode='')
            active, _ = run_case(wall_local, ROOT, 'gpu', 2, 'wallmeans_'+pipeline, 4, **options)
            source = active / 'outdat/new/checkpoints/step000000000003'
            resumed, _ = run_case(wall_local, ROOT, 'gpu', 2, 'wallmeans_resume_'+pipeline, 4,
                                  restore=source, **options)
            products = {'wall_pressure', 'wall_shear_x', 'wall_heat_into_gas'}
            products |= {'mean_'+name for name in products}
            record = images(active, 2, (2, 4), pipeline, products, width, wall_statistics=True)
            images(resumed, 2, (4,), pipeline, products, width, wall_statistics=True)
            final = 'outdat/new/checkpoints/step000000000004'
            for name in ('state.h5', 'statistics.h5'):
                compare_fields(active/final/name, resumed/final/name)
            for path in (resumed/'outdat/render').glob('*.jpeg'):
                if path.read_bytes() != (active/'outdat/render'/path.name).read_bytes():
                    raise AssertionError('wall-mean continuation image differs')
            save('wallmeans_'+pipeline, record)
    if 'safety' in opts.groups:
        case, _ = run_case(local, ROOT, 'gpu', 2, 'resident_memcheck', 2, grid='32,32,32',
            checkpoint_interval=1, insitu_config=configuration(library, 'standard-device'),
            gpu_precision_mode='', memcheck=True)
        save('resident_memcheck', images(case, 2, (2,), 'standard-device', TGV_PRODUCTS, width))
    if 'observe' in opts.groups:
        off, _ = run_case(local, ROOT, 'gpu', 2, 'resource_off', 2, grid='32,32,32',
            enabled=False, checkpoint_enabled=False, gpu_precision_mode='',
            insitu_config=host_configuration(render=False, interval=2), monitor_resources=True)
        baseline = json.loads((off/'resources.sampled.json').read_text())
        for pipeline in ('standard-device', 'direct-device'):
            config=configuration(library,pipeline)
            observed, _ = run_case(local, ROOT, 'gpu', 2, 'resource_'+pipeline, 2, grid='32,32,32',
                enabled=False, checkpoint_enabled=False, gpu_precision_mode='', insitu_config=config,
                monitor_resources=True, resource_baseline=baseline)
            trace, _ = run_case(local, ROOT, 'gpu', 2, 'trace_'+pipeline, 2, grid='32,32,32',
                enabled=False, checkpoint_enabled=False, gpu_precision_mode='', insitu_config=config,
                nsys_trace=True, insitu_timing=True, pixel_audit=True)
            rows=[]
            for rank in range(2):
                database=trace/f'trace.rank{rank}.sqlite'
                with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as db:
                    if db.execute('select count(*) from CUPTI_ACTIVITY_KIND_KERNEL').fetchone()[0]==0:
                        raise AssertionError('native in-situ capture has no kernels')
                    names=('ASTR_IS8_DEVICE_GEOMETRY_EXTRACTION','ASTR_IS8_DEVICE_INSTANT_LINES',
                           'ASTR_IS8_DEVICE_CROSSING_LINES','ASTR_IS8_DEVICE_REYNOLDS_LINES',
                           'ASTR_IS8_DEVICE_FAVRE_LINES','ASTR_IS8_RESIDENT_RENDER')
                    ledger={}
                    for name in names:
                        copies=db.execute('select e.name,m.bytes from CUPTI_ACTIVITY_KIND_MEMCPY m '
                            'join ENUM_CUDA_MEMCPY_OPER e on e.id=m.copyKind where exists '
                            '(select 1 from NVTX_EVENTS n where n.text=? and m.start>=n.start and m.end<=n.end)',
                            (name,)).fetchall()
                        if any(kind=='CUDA_MEMCPY_KIND_UVM_DTOH' or
                               (kind=='CUDA_MEMCPY_KIND_DTOH' and count>2048) for kind,count in copies):
                            raise AssertionError('native product copied a field/geometry payload to the host')
                        ledger[name]=copies
                    rows.append(dict(rank=rank,copies=ledger))
            save('observed_'+pipeline,dict(
                resources=json.loads((observed/'resources.sampled.json').read_text()), transfer=rows,
                scope='CUDA product ranges; pinned halo staging is separate; OpenGL pixel reads are logged independently'))
    if 'joint' in opts.groups:
        for pipeline in ('standard-device', 'direct-device'):
            config = configuration(library, pipeline, interval=4)
            options = dict(grid='32,32,32', checkpoint_interval=5,
                insitu_config=config, gpu_precision_mode='', archive_groups=ADAPTIVE_GROUPS,
                adaptive_config=ADAPTIVE_CONFIG, tgv_reynolds=1)
            continuous, _ = run_case(local, ROOT, 'gpu', 2, pipeline+'_continuous', 12, **options)
            seed, _ = run_case(local, ROOT, 'gpu', 2, pipeline+'_seed', 5, **options)
            source = seed / 'outdat/new/checkpoints/step000000000005'
            before = {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in source.iterdir() if p.is_file()}
            resumed, _ = run_case(local, ROOT, 'gpu', 2, pipeline+'_resumed', 12, restore=source, **options)
            final = 'outdat/new/checkpoints/step000000000012'
            for name in ('state.h5', 'statistics.h5'):
                compare_fields(continuous / final / name, resumed / final / name)
            for name in ('control.bin', 'insitu_control.bin'):
                if (continuous / final / name).read_bytes() != (resumed / final / name).read_bytes():
                    raise AssertionError('exact in-situ control differs: '+name)
            if archive_schedule_payload(continuous / final / 'archives.bin') != archive_schedule_payload(
                    resumed / final / 'archives.bin'):
                raise AssertionError('exact archive clock differs')
            record = images(resumed, 2, (8, 12), pipeline, TGV_PRODUCTS, width)
            record['adaptive'] = check_log(resumed, 5, 12)
            record['joint_reynolds'] = 1
            record['archives'] = {}
            for product in ('fields', 'slices'):
                check_series(resumed, product)
                expected_frames = frames(continuous, product)
                actual_frames = frames(resumed, product)
                record['archives'][product] = sorted(actual_frames)
                for step, path in actual_frames.items():
                    with h5py.File(path/'data.h5') as actual, h5py.File(expected_frames[step]/'data.h5') as expected:
                        def same_payload(name, array):
                            if not isinstance(array, h5py.Dataset) or array.dtype.kind != 'f':
                                return
                            a, b = array[:], expected[name][:]
                            if array.dtype.itemsize != width or a.dtype != b.dtype or a.shape != b.shape or \
                                    a.tobytes() != b.tobytes():
                                raise AssertionError('joint archive continuation payload differs: '+name)
                        actual.visititems(same_payload)
            for path in (resumed / 'outdat/render').glob('*.jpeg'):
                if path.read_bytes() != (continuous / 'outdat/render' / path.name).read_bytes():
                    raise AssertionError('same-kind continuation image differs: '+path.name)
            if before != {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in source.iterdir() if p.is_file()}:
                raise AssertionError('exact source checkpoint changed')
            save('joint_'+pipeline, record)
    result['status'] = 'passed'
    (root / 'report.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
