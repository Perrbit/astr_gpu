#!/usr/bin/env python3
"""Allocated-device smoke gate; not a production or scaling benchmark."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

from compare_flowstate import read_flowstate
from insitu_cfl_gate import CflGate
from run_m12_insitu_acceptance import configuration as m12_configuration, check_images, m12_short_products
from run_m12_local_compare import run as run_m12, check_pair
from run_output_restart_validation import run_case, compare_fields
from test_insitu_device_products import field_difference,check_device_render
from catalog_image_inspection import inspect_images

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/insitu'))
from catalog_examples import tgv32_products
FINAL = 'outdat/new/checkpoints/step000000000010/state.h5'
M12_TOPOLOGIES = ('2,1,1', '4,1,1')


def runtime_options(args):
    defaults = dict(directory_budget_bytes=32*1024**3, runtime_timeout_seconds=3600,
        resource_host_extra_budget_bytes=32*1024**3,
        resource_device_extra_budget_bytes=8*1024**3, resource_device_reserve_bytes=2*1024**3)
    return {name: getattr(args, name, default) for name, default in defaults.items()}


def render_configuration(factory, args):
    options = runtime_options(args)
    return factory(args.library, host_budget_bytes=options['resource_host_extra_budget_bytes'],
        device_budget_bytes=options['resource_device_extra_budget_bytes'],
        device_reserve_bytes=options['resource_device_reserve_bytes'])


def tgv_configuration(library, host_budget_bytes=4*1024**3, device_budget_bytes=2*1024**3,
                      device_reserve_bytes=1024**3):
    return f"""&insitu_run
 enabled=t, statistics=f, render=t,
 processing_backend='device', derivative_backend='gpu', products='all',
 rendering_pipeline='standard-device', postprocess_transport='device-aware',
 output_directory='outdat/render',
 host_budget_bytes={host_budget_bytes}, device_budget_bytes={device_budget_bytes}, device_reserve_bytes={device_reserve_bytes},
 implementation_path='{library}', pipeline_file='{ROOT / 'scripts/insitu/device_render_pipeline.py'}'
/
"""


def execute(command, output, timeout=600):
    command = list(map(str, command))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
        output.write_text(stdout + stderr)
        raise
    output.write_text(stdout + stderr)
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command, stdout, stderr)
    return stdout


def dependency_gate(args):
    probes = args.probe_directory
    timeout = getattr(args, 'probe_timeout_seconds', 600)
    rank_counts = {1, 2}
    for topology in getattr(args, 'm12_topologies', M12_TOPOLOGIES):
        parts = [int(value) for value in topology.split(',')]
        if len(parts) != 3 or min(parts) <= 0:
            raise ValueError('M12 topology must contain three positive MPI dimensions')
        rank_counts.add(int(np.prod(parts)))
    required = max(rank_counts)
    expected = json.loads(execute([sys.executable, ROOT / 'tests/gpu_validation/insitu_device_identity.py'],
                                  args.output / 'cuda_egl.json', timeout=timeout))
    if len(expected) < required or len({d['uuid'] for d in expected}) != len(expected):
        raise ValueError(f'Requested topologies need {required} distinct allocated CUDA/EGL devices; '
                         f'found {len(expected)}')
    mappings = []
    for ranks in sorted(rank_counts):
        log = execute([args.mpiexec, '--mca', 'coll_hcoll_enable', '0', '-np', ranks,
                       probes / 'insitu_device_map_probe'], args.output / f'mapping_np{ranks}.log', timeout=timeout)
        actual = re.findall(r'MAPPING rank=(\d+) egl=(\d+) uuid=([0-9a-f-]{36})', log)
        if len(actual) != ranks or {int(row[0]) for row in actual} != set(range(ranks)):
            raise ValueError('Incomplete native CUDA/EGL mapping')
        for rank, index, uuid in actual:
            want = expected[int(rank)]
            if int(index) != want['egl_index'] or uuid != want['uuid']:
                raise ValueError('Solver and rendering GPU identity differ')
        mappings.append(dict(ranks=ranks, actual=actual))
    graphics = []
    for ordinal in range(required):
        log = execute([args.mpiexec, '-np', '1', probes / 'insitu_standard_device_probe', ordinal, '--conduit-clipped'],
                      args.output / f'graphics_device{ordinal}.log', timeout=timeout)
        rows = re.findall(r'ASTR_INSITU_STANDARD_DEVICE_PROBE (\{[^\n]+\})', log)
        if len(rows) != 1:
            raise ValueError('Missing CUDA/VTK/EGL interop probe receipt')
        receipt = json.loads(rows[0])
        if receipt['cuda_device'] != ordinal or receipt['gl_cuda_device'] != ordinal or \
                receipt['geometry_host_bytes'] != 0 or receipt['host_access_refused'] is not True or \
                receipt['pixels_passed'] is not True or receipt['conduit'] is not True or receipt.get('clipping_planes') != 6 or \
                receipt['uuid'].removeprefix('GPU-') != expected[ordinal]['uuid']:
            raise ValueError('Device buffer rendering probe failed')
        graphics.append(receipt)
    log = execute([args.mpiexec, '-np', '2', probes / 'halo_cuda_aware_probe', '33', '33', 'early-bind'],
                  args.output / 'aware_halo_payload.log', timeout=timeout)
    if log.count('supported=1') != 2 or 'CUDA-aware device-buffer exact tests passed' not in log:
        raise ValueError('CUDA-aware MPI exact payload gate failed; no fallback permitted')
    return dict(devices=expected, used_devices=required, mappings=mappings, graphics=graphics,
                cuda_aware_payload='passed')


def tgv_gate(args):
    directory = args.output / 'tgv'
    directory.mkdir()
    local = SimpleNamespace(output=directory, executable=args.executable, mpiexec=args.mpiexec,
        case='tgv', mode='steps', restart_step=5, initial_dimension=0, legacy_statistics=False,
        statistics=False, initial_restart=False, filter_workspace='full', force='feedback',
        axis='z', no_samples=True, maximum_cfl=1., **runtime_options(args))
    receipts = []
    for ranks in (1, 2):
        cases = {}
        baseline = None
        for backend, label in (('cpu', 'reference'), ('gpu', 'off'), ('gpu', 'on')):
            render = label == 'on'
            case, _ = run_case(local, ROOT, backend, ranks, label, 10, grid='32,32,32',
                checkpoint_interval=10, checkpoint_enabled=True,
                insitu_config=render_configuration(tgv_configuration, args) if render else None,
                product_catalog_records=tgv32_products(5,statistics=False) if render else [],
                postprocess_transport='device-aware' if render else None,
                solver_halo_transport='device-aware' if backend == 'gpu' and ranks == 2 else 'pinned',
                resident_audit=render, monitor_resources=backend == 'gpu', resource_baseline=baseline,
                insitu_timing=render)
            cfl = CflGate(case / 'run.log', 1., .001, 0, 10)
            cfl.check(final=True)
            cases[label] = case
            if label == 'off':
                baseline = json.loads((case / 'resources.sampled.json').read_text())
            if render:
                products=inspect_images(case,ranks,10)
                if not products['publication_complete']: raise ValueError(products['warnings'])
        compare_fields(cases['off'] / FINAL, cases['on'] / FINAL)
        maximum = field_difference(cases['reference'] / FINAL, cases['on'] / FINAL, 2e-10)
        headers, values = zip(*(read_flowstate(cases[label]) for label in ('reference', 'off', 'on')))
        if not all(header == headers[0] for header in headers) or any(v.shape != values[0].shape for v in values):
            raise ValueError('TGV statistics inventory mismatch')
        statistics_error = float(np.max(np.abs(values[0]-values[2])))
        if not all(np.isfinite(v).all() for v in values) or statistics_error > 2e-10 or \
                values[1].tobytes() != values[2].tobytes():
            raise ValueError('TGV statistics equivalence/isolation failed')
        receipts.append(dict(ranks=ranks, topology=[1, 1, ranks], field_maxabs=maximum,
            statistics_maxabs=statistics_error, render_observer='exact state/statistics unchanged',
            solver_halo='device-aware' if ranks == 2 else 'pinned', postprocess_halo='device-aware'))
        print(f'PASS platform TGV NP={ranks}', flush=True)
    return receipts


def m12_gate(args):
    local = SimpleNamespace(source=args.source, output=args.output / 'm12', executable=args.executable,
        mpiexec=args.mpiexec, memcheck=False, diagnostics=True, halo_transport='device-aware',
        **runtime_options(args))
    local.output.mkdir()
    receipts = []
    for topology in getattr(args, 'm12_topologies', M12_TOPOLOGIES):
        ranks = int(np.prod([int(v) for v in topology.split(',')]))
        cpu = run_m12(local, 'cpu', topology, case_label='cpu_reference')
        off = run_m12(local, 'gpu', topology, case_label='gpu_off')
        baseline = json.loads((off / 'resources.sampled.json').read_text())
        on = run_m12(local, 'gpu', topology, case_label='gpu_on',
                     insitu_config=render_configuration(m12_configuration, args), resource_baseline=baseline,
                     product_catalog_records=m12_short_products())
        receipt = check_pair(cpu, on, ranks)
        isolation = check_pair(off, on, ranks, reference_backend='gpu')
        if isolation['q']['max_abs'] != 0. or any(isolation['statistics_max_abs'].values()):
            raise ValueError('M12 render observer changes solver state or statistics')
        receipt.update(topology=topology, solver_halo='device-aware', postprocess_halo='pinned',
                       graphics=check_images(on, ranks), observer=isolation)
        receipts.append(receipt)
        print('PASS platform M12 topology='+topology, flush=True)
    return receipts


def native_gate(args):
    directory = args.output / 'native'
    directory.mkdir()
    local = SimpleNamespace(output=directory, executable=args.native_executable, mpiexec=args.mpiexec,
        case='tgv', mode='steps', restart_step=5, initial_dimension=0, legacy_statistics=False,
        statistics=False, initial_restart=False, filter_workspace='full', force='feedback',
        axis='z', no_samples=True, maximum_cfl=1., **runtime_options(args))
    cases = {}
    baseline = None
    for label in ('off', 'on'):
        render = label == 'on'
        case, _ = run_case(local, ROOT, 'gpu', 2, label, 10, grid='32,32,32',
            checkpoint_interval=10, checkpoint_enabled=True,
            insitu_config=render_configuration(tgv_configuration, args) if render else None,
            product_catalog_records=tgv32_products(5,statistics=False) if render else [],
            postprocess_transport='device-aware' if render else None,
            solver_halo_transport='device-aware', monitor_resources=True,
            resource_baseline=baseline, insitu_timing=render)
        cases[label] = case
        if render:
            products=inspect_images(case,2,10)
            if not products['publication_complete']: raise ValueError(products['warnings'])
        else:
            baseline = json.loads((case / 'resources.sampled.json').read_text())
    compare_fields(cases['off'] / FINAL, cases['on'] / FINAL)
    compare_fields(cases['on'] / FINAL, args.output / 'tgv/gpu_np2_on' / FINAL)
    local = SimpleNamespace(source=args.source, output=directory, executable=args.native_executable,
        mpiexec=args.mpiexec, memcheck=False, diagnostics=False, halo_transport='device-aware',
        **runtime_options(args))
    receipts = []
    for topology in getattr(args, 'm12_topologies', M12_TOPOLOGIES):
        ranks = int(np.prod([int(v) for v in topology.split(',')]))
        off = run_m12(local, 'gpu', topology, case_label='m12_off', observation='plain',
                     checkpoint_interval=10)
        baseline = json.loads((off / 'resources.sampled.json').read_text())
        on = run_m12(local, 'gpu', topology, case_label='m12_on', observation='images',
                    checkpoint_interval=10, insitu_config=render_configuration(m12_configuration, args),
                    resource_baseline=baseline,product_catalog_records=m12_short_products())
        compare_fields(off / FINAL, on / FINAL)
        receipts.append(dict(topology=topology, ranks=ranks, state_isolation='exact',
                             graphics=check_images(on, ranks, audit=False)))
        print(f'PASS native production-build observer isolation NP={ranks}', flush=True)
    return dict(tgv_topology=[1, 1, 2], tgv_state_isolation='exact', m12=receipts,
                scope='BUILD_TESTING=OFF; no validation geometry/volume oracle downloads; bounded final state checkpoints remain')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'output', 'executable', 'native-executable', 'mpiexec', 'library', 'probe-directory'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--m12-topologies', nargs='+', default=M12_TOPOLOGIES,
                        help='MPI dimensions as x,y,z; one allocated GPU per rank')
    parser.add_argument('--timeout-seconds', type=int, default=3600,
                        help='Maximum solver time per short run, excluding validation')
    parser.add_argument('--probe-timeout-seconds', type=int, default=600)
    parser.add_argument('--host-budget-gib', type=int, default=32)
    parser.add_argument('--device-budget-gib', type=int, default=8)
    parser.add_argument('--device-reserve-gib', type=int, default=2)
    parser.add_argument('--directory-budget-gib', type=int, default=32)
    args = parser.parse_args()
    for option in ('host_budget', 'device_budget', 'device_reserve', 'directory_budget'):
        value = getattr(args, option+'_gib')
        if value <= 0:
            parser.error('--'+option.replace('_', '-')+'-gib must be positive')
        target = {'host_budget': 'resource_host_extra_budget_bytes',
                  'device_budget': 'resource_device_extra_budget_bytes',
                  'device_reserve': 'resource_device_reserve_bytes',
                  'directory_budget': 'directory_budget_bytes'}[option]
        setattr(args, target, value*1024**3)
    if min(args.timeout_seconds, args.probe_timeout_seconds) <= 0:
        parser.error('solver and probe timeouts must be positive')
    args.runtime_timeout_seconds = args.timeout_seconds
    for name in ('source', 'executable', 'native_executable', 'library', 'probe_directory'):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    args.mpiexec = args.mpiexec.absolute()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', steps=10, frame_steps=[5, 10], grids=dict(tgv=[32]*3, m12=[64, 32, 24]),
        scope='Bounded platform port smoke; diagnostic snapshots/final checkpoints are not production output',
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), acceptance_absolute_tolerance=2e-10,
        m12_topologies=args.m12_topologies, resource_options=runtime_options(args))
    try:
        report['dependencies'] = dependency_gate(args)
        report['tgv'] = tgv_gate(args)
        report['m12'] = m12_gate(args)
        report['native'] = native_gate(args)
        report['status'] = 'passed-bounded-platform-not-production'
    except BaseException as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        (args.output / 'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
