#!/usr/bin/env python3
"""Four-GPU original-grid M12 product coverage and paired timing, not a DNS campaign."""
import argparse
from collections import defaultdict
import hashlib
import io
import json
import os
from pathlib import Path
import re
from statistics import median
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
from PIL import Image

from prepare_m12_local_case import FULL_INTERVALS, prepare, replace_value
from insitu_cfl_gate import CflGate
from insitu_resource_monitor import run_monitored

ROOT = Path(__file__).resolve().parents[2]
GIB = 1024**3
sys.path.insert(0, str(ROOT/'scripts/insitu'))
from image_publication import publish_pair, eps_output_enabled
from catalog_examples import m12_short_products,tgv32_products,write_catalog
from catalog_image_inspection import inspect_images


def save(path, record):
    path.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')


def publication_preflight(directory):
    """Exercise the selected image formats on the destination filesystem."""
    write_eps = eps_output_enabled()
    work = Path(tempfile.mkdtemp(prefix='image-publication-', dir=directory))
    image = Image.new('RGB', (8, 8), (170, 20, 40))
    jpeg = io.BytesIO()
    eps = io.BytesIO() if write_eps else None
    image.save(jpeg, 'JPEG', quality=95, dpi=(300, 300))
    if write_eps:
        image.save(eps, 'EPS')
    path = work/'probe.jpeg'
    eps_bytes = eps.getvalue() if write_eps else None
    failure = publish_pair(path, jpeg.getvalue(), eps_bytes)
    if failure:
        raise OSError(f'Image filesystem preflight failed: {failure}; {work}')
    with Image.open(path) as decoded:
        decoded.load()
        if decoded.size != (8, 8):
            raise ValueError('Published JPEG cannot be decoded')
    if write_eps and path.with_suffix('.eps').read_bytes() != eps_bytes:
        raise ValueError('Incomplete published EPS')
    if list(work.glob('*.partial')):
        raise ValueError('Incomplete image publication')
    try:
        publish_pair(path, jpeg.getvalue(), eps_bytes)
    except FileExistsError:
        pass
    else:
        raise ValueError('Image publisher overwrote an existing frame')
    result = dict(status='passed', directory=str(work),
        image_formats=['jpeg', 'eps'] if write_eps else ['jpeg'],
        checks=['JPEG decode']+(['EPS bytes'] if write_eps else [])+[
        'fsync and hard-link publication', 'no partial files', 'existing destination protected'])
    save(work/'report.json', result)
    return result


def render_configuration(args, images=True):
    return f"""&insitu_run
 enabled=t, statistics=f, render=t,
 products='m12_diagnostics',streamline_seeds='bl-layered256',
 processing_backend='device',derivative_backend='gpu',rendering_pipeline='standard-device',
 postprocess_transport='pinned',output_directory='outdat/render',
 host_budget_bytes={args.host_budget},device_budget_bytes={args.device_budget},device_reserve_bytes={args.reserve},
 implementation_path='{args.library}',pipeline_file='{ROOT}/scripts/insitu/device_render_pipeline.py'
/
"""


def render_catalog(args,shape,images=True):
    if tuple(shape)==(64,32,24):
        records=m12_short_products(interval=5)
    else:
        path=getattr(args,'product_catalog',None)
        if path is None:
            raise ValueError('Full-grid M12 requires an explicitly reviewed --product-catalog; the short-fixture catalog is not rescaled')
        rows=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        if not rows or rows[0]!={'catalog_schema':'astr-insitu-products-v1'}:
            raise ValueError('Product catalog header differs')
        records=rows[1:]
    if getattr(args,'m12_products','all')=='slices-wall':
        records=[row for row in records if row['kind'] not in ('q_surface','streamlines')]
    # The paired control retains explicitly selected numerical monitors, never hidden profile diagnostics.
    return records if images else [row for row in records if row['output']['channel']!='image']


def prepare_case(args, case, shape, steps, images, checkpoint=False):
    prepare(args.source, case, 'gpu', shape)
    boundary = getattr(args, 'm12_boundary', 'original')
    if boundary == 'nscbc':
        path = case/'datin/input.dat'
        lines = path.read_text().splitlines()
        first = next(i for i, line in enumerate(lines) if line.startswith('# bctype'))+1
        lines[first+1] = '22,1.d0'
        lines[first+3] = '52'
        path.write_text('\n'.join(lines)+'\n')
    controller = case/'datin/controller'
    lines = controller.read_text().splitlines()
    replace_value(lines, 'maxstep,feqchkpt', f'{steps-1},1,999999,999999,1,999999')
    controller.write_text('\n'.join(lines)+'\n')
    fixture = json.loads((case/'fixture.json').read_text())
    fixture.update(completed_steps=steps, topology=[4,1,1], images=images, checkpoint=checkpoint,
        boundary=boundary, products=getattr(args, 'm12_products', 'all'))
    if boundary == 'nscbc':
        fixture['unchanged'] = [item for item in fixture.get('unchanged', []) if item != 'boundary types']
        fixture['changes'] = fixture.get('changes', [])+['x+ bc22 and y+ bc52, validated compatibility mode']
    save(case/'fixture.json', fixture)
    (case/'outdat/render').mkdir()
    (case/'datin/input.insitu').write_text(render_configuration(args, images))
    write_catalog(case/'input.products.jsonl',render_catalog(args,shape,images))
    if checkpoint:
        # The checkpoint writer owns exclusive children, not its caller's root.
        (case/'outdat/new').mkdir(mode=0o700)
        path = case/'datin/input.output'
        path.write_text(path.read_text().replace('&checkpoint\n enabled=f',
            f'&checkpoint\n enabled=t,mode=\'steps\',interval_steps={steps},keep=1,initial_frame=f'))


def runtime_environment(solver_halo, boundary='original'):
    env = {key: value for key, value in os.environ.items() if not key.startswith('ASTR_')}
    env.update(ASTR_FORCE_MPI_TOPOLOGY='4,1,1', ASTR_WALL_BLOWING_MODE='legacy_random',
        ASTR_GPU_HALO_TRANSPORT=solver_halo, ASTR_GPU_SYNC_MODE='explicit', ASTR_GPU_PRECISION_MODE='fp64',
        ASTR_GPU_MP_POSITIVITY='off', ASTR_GPU_UPWIND_FLUX_MODE='split',
        ASTR_GPU_RK_TIMING='1', ASTR_GPU_RANK_RK_TIMING='1', ASTR_COMPLETE_STEP_TIMING='1',
        ASTR_INSITU_TIMING='1', ASTR_INSITU_CONFIG='datin/input.insitu',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    if boundary == 'nscbc':
        env['ASTR_NSCBC_FARFIELD_MODE'] = 'compatibility'
    for key in ('DISPLAY', 'VTK_EGL_DEVICE_INDEX', 'CATALYST_IMPLEMENTATION_PREFER_ENV'):
        env.pop(key, None)
    return env


def timing_summary(log, steps, warmup=5):
    stages = defaultdict(lambda: defaultdict(float))
    iterations, rk = defaultdict(dict), defaultdict(dict)
    pipeline = defaultdict(float)
    for line in log.splitlines():
        parts = line.split()
        if parts and parts[0] == 'ASTR_INSITU_STAGE_TIMING' and len(parts) == 5:
            stage, step, rank, value = parts[1], int(parts[2]), int(parts[3]), float(parts[4])
            if not np.isfinite(value) or value < 0:
                raise ValueError('Invalid stage timing')
            stages[stage][rank] += value
            if stage == 'completed_step_inclusive' and warmup < step <= steps:
                if step in iterations[rank]:
                    raise ValueError('Duplicate complete-iteration timing')
                iterations[rank][step] = value
        elif parts and parts[0] == 'ASTR_GPU_RANK_RK_TIMING' and len(parts) == 6:
            rank, step = int(parts[1]), int(parts[2])+1
            values = list(map(float, parts[3:]))
            if not np.isfinite(values).all() or min(values) < 0:
                raise ValueError('Invalid RK timing')
            if warmup < step <= steps:
                if step in rk[rank]:
                    raise ValueError('Duplicate RK timing')
                rk[rank][step] = values
        elif line.startswith('ASTR_INSITU_PIPELINE_TIMING '):
            record = json.loads(line.split(' ', 1)[1])
            # Image encoding/publication occur on rank zero. Rank-local render
            # and capture times are reported separately, never summed over MPI.
            for name, value in record['seconds'].items():
                if not np.isfinite(value) or value < 0:
                    raise ValueError('Invalid image pipeline timing')
                pipeline[f'rank{record["rank"]}/{name}'] += value
    expected = set(range(warmup+1, steps+1))
    if set(iterations) != set(range(4)) or set(rk) != set(range(4)) or any(
            set(data) != expected for table in (iterations, rk) for data in table.values()):
        raise ValueError('Incomplete four-rank post-warmup timing window')
    return dict(warmup_steps=warmup, measured_steps=[warmup+1, steps],
        complete_iteration_seconds=max(sum(data.values()) for data in iterations.values()),
        complete_iteration_median_seconds=median(max(data[s] for data in iterations.values()) for s in sorted(expected)),
        rk_integration_seconds=max(sum(value[1] for value in data.values()) for data in rk.values()),
        rk_prepare_plus_integration_seconds=max(sum(value[2] for value in data.values()) for data in rk.values()),
        stage_rank_seconds_whole_run={name: dict(values) for name, values in stages.items()},
        image_pipeline_rank_seconds_whole_run=dict(pipeline),
        definition='max over rank-local window sums; complete iteration includes diagnostics/render/I-O; nested stages are not additive')


def compact_diagnostics(case, shape, steps):
    files=sorted((case/'outdat/new').glob('monitor_products/*/version*/segment*/*.csv'))
    files+=sorted((case/'outdat/new').glob('wall_products/*/version*/segment*/*.csv'))
    rows=0
    for path in files:
        text='\n'.join(line for line in path.read_text().splitlines() if not line.startswith('#'))
        values=np.atleast_1d(np.genfromtxt(io.StringIO(text),delimiter=',',names=True))
        if not np.isfinite(np.column_stack([values[key] for key in values.dtype.names])).all():
            raise ValueError('Nonfinite explicitly requested monitor: '+str(path))
        rows+=len(values)
    return dict(csv_files=len(files),rows=rows,scope='only explicit catalog monitors; no implicit 72-probe or wall profile')


def inspect_products(case, steps, profile='all'):
    return inspect_images(case,4,steps)


def run_m12(args, label, shape, steps, images, baseline=None, checkpoint=False):
    case = args.output/label
    prepare_case(args, case, shape, steps, images, checkpoint)
    executable = args.diagnostic_executable if checkpoint else args.executable
    env = runtime_environment(args.solver_halo, getattr(args, 'm12_boundary', 'original'))
    if args.component_only:
        # Shared allocation accounting uses MPI_Win_allocate_shared locally.
        env['OMPI_MCA_osc'] = 'sm'
    command = [str(args.mpiexec), '--bind-to', 'none', '-np', '4', str(executable), 'run', 'datin/input.dat']
    receipt = dict(status='running', intervals=shape, topology=[4,1,1], steps=steps, dt=.02,
        images=images, checkpoint=checkpoint, command=command,
        boundary=getattr(args, 'm12_boundary', 'original'), product_profile=getattr(args, 'm12_products', 'all'),
        executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
        runtime={k:v for k,v in env.items() if k.startswith(('ASTR_', 'OMPI_MCA_', 'UCX_'))})
    save(case/'receipt.json', receipt)
    gate = CflGate(case/'run.log', 1., .02, 0, steps)
    last_disk_check = [0]
    def progress():
        gate.check()
        # Directory scans once per second, not at every 20 ms memory sample.
        import time
        now = time.monotonic()
        if now-last_disk_check[0] >= 1.:
            last_disk_check[0] = now
            if sum(p.stat().st_size for p in case.rglob('*') if p.is_file()) > args.directory_budget:
                raise ValueError('Run-directory disk budget exceeded')
    try:
        with (case/'run.log').open('wb') as log:
            resources = run_monitored(command, case, env, log, case/'resources.sampled.json', baseline=baseline,
                device_extra_budget_bytes=args.device_budget, host_extra_budget_bytes=args.host_budget,
                device_reserve_bytes=args.reserve, timeout_seconds=args.timeout, check_progress=progress)
        gate.check(final=True)
        log = (case/'run.log').read_text(errors='replace')
        if 'The job is done!' not in log:
            raise ValueError('Missing normal solver completion')
        downloads = re.findall(r'ASTR_INSITU_GPU_FLOW rank=(\d+) frame_downloads=(\d+)', log)
        if len(downloads) != 4 or {int(rank) for rank, _ in downloads} != set(range(4)) or any(
                int(count) != 0 for _, count in downloads):
            raise ValueError('Missing or nonzero full-frame download counters')
        if not checkpoint and (list((case/'outdat/new/checkpoints').rglob('*.h5')) or
                               list((case/'outdat/new/volume').rglob('*.h5'))):
            raise ValueError('Full-size coverage unexpectedly wrote field/checkpoint HDF5')
        receipt.update(status='completed', max_cfl=gate.maximum, resources=resources,
            compact=compact_diagnostics(case, shape, steps), timing=timing_summary(log, steps),
            postprocess_frame_downloads=0,
            caveat='input initialization and the small exact-state checkpoint are separate I/O; legacy crash repair, if invoked, is not a postprocessing transfer')
        if images:
            receipt['products'] = inspect_products(case, steps, getattr(args, 'm12_products', 'all'))
            if not receipt['products']['publication_complete']:
                receipt['status'] = 'completed-with-missing-products'
        elif list((case/'outdat/new/render_products').rglob('*.jpeg')):
            raise ValueError('Images-off control produced pictures')
    except BaseException as error:
        receipt.update(status='failed', error=str(error), max_cfl=gate.maximum)
        raise
    finally:
        save(case/'receipt.json', receipt)
    return case, receipt


def compare_pair(off, on, shape, steps, checkpoint=False):
    if checkpoint:
        from run_output_restart_validation import compare_fields
        relative = f'outdat/new/checkpoints/step{steps:012d}/state.h5'
        compare_fields(off/relative, on/relative)
    for subdir in ('monitor_products','wall_products','slice_products'):
        a=off/'outdat/new'/subdir;b=on/'outdat/new'/subdir
        files={path.relative_to(a) for path in a.rglob('*') if path.suffix in ('.csv','.h5')}
        if files!={path.relative_to(b) for path in b.rglob('*') if path.suffix in ('.csv','.h5')}:
            raise ValueError('Paired catalog monitor inventory differs: '+subdir)
        for name in files:
            if name.suffix=='.h5':
                from run_output_restart_validation import compare_fields
                compare_fields(a/name,b/name)
            elif (a/name).read_bytes()!=(b/name).read_bytes():
                raise ValueError('Images changed paired compact diagnostics: '+str(name))
    return dict(authoritative_q='bitwise equal' if checkpoint else 'not downloaded at full scale',
        compact_files='byte-identical', intervals=shape, steps=steps)


def tgv_means(args):
    from run_output_restart_validation import run_case
    local = SimpleNamespace(output=args.output, executable=args.executable, mpiexec=args.mpiexec,
        case='tgv', mode='steps', restart_step=5, initial_dimension=0, legacy_statistics=False,
        statistics=True, initial_restart=False, filter_workspace='scalar', force='feedback',
        axis='x', no_samples=True, maximum_cfl=1., runtime_timeout_seconds=args.timeout,
        directory_budget_bytes=args.directory_budget, resource_host_extra_budget_bytes=args.host_budget,
        resource_device_extra_budget_bytes=args.device_budget, resource_device_reserve_bytes=args.reserve)
    config = f"""&insitu_run
 enabled=t, statistics=t, render=t, initial_frame=f, final_frame=f,
 statistics_window=0.0005,0.0115, products='all',
 processing_backend='device',derivative_backend='gpu',rendering_pipeline='standard-device',
 postprocess_transport='device-aware',output_directory='outdat/render',
 host_budget_bytes={args.host_budget},device_budget_bytes={args.device_budget},device_reserve_bytes={args.reserve},
 implementation_path='{args.library}',pipeline_file='{ROOT}/scripts/insitu/device_render_pipeline.py'
/
"""
    case, _ = run_case(local, ROOT, 'gpu', 2, 'tgv_means', 10, enabled=False, grid='32,32,32',
        insitu_config=config, checkpoint_enabled=False, monitor_resources=True,
        product_catalog_records=tgv32_products(5,statistics=True),
        postprocess_transport='device-aware', solver_halo_transport='device-aware', insitu_timing=True)
    result=inspect_images(case,2,10)
    result.update(status='completed' if result['publication_complete'] else 'completed-with-missing-products',
        scope='existing short-window means and explicit rendering, not statistical stationarity')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--executable', type=Path)
    parser.add_argument('--diagnostic-executable', type=Path)
    parser.add_argument('--probe-directory', type=Path)
    parser.add_argument('--mpiexec', type=Path)
    parser.add_argument('--library', type=Path)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--component-only', action='store_true', help='local four-rank correctness; not four-GPU qualification')
    parser.add_argument('--solver-halo', choices=('device-aware','pinned'), default='device-aware')
    parser.add_argument('--m12-products', choices=('all','slices-wall'), default='all',
        help='Filter the explicit catalog: slices-wall omits Q and streamlines; it adds no products')
    parser.add_argument('--product-catalog',type=Path,
        help='Reviewed native JSONL for the full grid; no short-fixture index rescaling or implicit products')
    parser.add_argument('--m12-boundary', choices=('original','nscbc'), default='original')
    parser.add_argument('--timeout', type=float, default=7*3600)
    parser.add_argument('--host-budget', type=int)
    parser.add_argument('--device-budget', type=int)
    parser.add_argument('--reserve', type=int)
    parser.add_argument('--directory-budget', type=int, default=32*GIB)
    args = parser.parse_args()
    for key, local, platform in (('host_budget',16,32),('device_budget',12,32),('reserve',2,8)):
        if getattr(args,key) is None:
            setattr(args,key,(local if args.component_only else platform)*GIB)
    if any(getattr(args, key) <= 0 for key in ('timeout','host_budget','device_budget','directory_budget')):
        parser.error('Resource budgets must be positive')
    if args.reserve < 0:
        parser.error('Device free-memory reserve must be nonnegative')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', base_commit=os.environ.get('ASTR_PLATFORM_SOURCE_COMMIT', 'unknown'),
        source='tracked snapshot plus any explicitly recorded deployment patch',
        boundary=args.m12_boundary, products=args.m12_products,
        scope='platform capability and short-window timing, not production physics',
        resource_limits=dict(host_extra_budget_bytes=args.host_budget,
            device_extra_budget_bytes=args.device_budget, device_reserve_bytes=args.reserve), tests={})
    try:
        report['tests']['image_publication'] = publication_preflight(args.output)
        if not args.preflight_only:
            for name in ('source','executable','diagnostic_executable','mpiexec','library'):
                if getattr(args, name) is None:
                    parser.error('--'+name.replace('_','-')+' is required for flow tests')
                path = getattr(args, name)
                if name == 'mpiexec':
                    # HPC-X dispatches its wrapper by argv[0], not the link target.
                    path = path.absolute()
                    if not path.is_file() or not os.access(path, os.X_OK):
                        raise ValueError(f'MPI launcher is not executable: {path}')
                else:
                    path = path.resolve(strict=True)
                setattr(args, name, path)
            if not args.component_only:
                if args.product_catalog is None:
                    parser.error('--product-catalog is required for full-grid images/monitors')
                if args.probe_directory is None:
                    parser.error('--probe-directory is required for four-GPU qualification')
                args.probe_directory = args.probe_directory.resolve(strict=True)
                from run_platform_insitu_acceptance import dependency_gate
                report['tests']['dependencies'] = dependency_gate(SimpleNamespace(output=args.output,
                    probe_directory=args.probe_directory, mpiexec=args.mpiexec, m12_topologies=['4,1,1'],
                    probe_timeout_seconds=1800))
                if args.m12_products == 'all':
                    report['tests']['tgv_means'] = tgv_means(args)
            shape = (64,32,24)
            off, a = run_m12(args, 'small_off', shape, 10, False, checkpoint=True)
            on, b = run_m12(args, 'small_on', shape, 10, True, baseline=a['resources'], checkpoint=True)
            report['tests']['small_pair'] = compare_pair(off, on, shape, 10, checkpoint=True)
            report['tests']['small_products'] = b.get('products')
            save(args.output/'report.json', report)
            if not args.component_only:
                off, a = run_m12(args, 'full_off', FULL_INTERVALS, 50, False)
                on, b = run_m12(args, 'full_on', FULL_INTERVALS, 50, True, baseline=a['resources'])
                report['tests']['full_pair'] = compare_pair(off, on, FULL_INTERVALS, 50)
                report['tests']['full_products'] = b.get('products')
                report['tests']['paired_timing'] = dict(off=a['timing'], on=b['timing'],
                    additional_complete_iteration_seconds=b['timing']['complete_iteration_seconds']-a['timing']['complete_iteration_seconds'],
                    note='single paired measurement, not a statistical speedup claim')
        incomplete = any(isinstance(item, dict) and (item.get('publication_complete') is False or
            item.get('status') == 'completed-with-missing-products') for item in report['tests'].values())
        report['status'] = 'completed-with-missing-products' if incomplete else 'completed'
    except BaseException as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        save(args.output/'report.json', report)
        print(f'PLATFORM_FULLSIZE_STATUS={report["status"]} report={args.output / "report.json"}', flush=True)


if __name__ == '__main__':
    main()
