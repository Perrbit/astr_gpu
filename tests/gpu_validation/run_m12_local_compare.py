#!/usr/bin/env python3
"""Run the approved M12 short matrix; stop at the first failed gate."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import h5py
import numpy as np

from compare_flowstate import read_flowstate
from compare_q_validation_snapshots import compare_snapshot_sets, comparison_values, read_q_snapshot
from insitu_cfl_gate import CflGate
from insitu_resource_monitor import run_monitored
from prepare_m12_local_case import prepare,replace_value
from analyze_m12_sensor_diagnostics import analyze_pair

TOPOLOGIES = ('1,1,1', '2,1,1', '1,2,1', '1,1,2',
              '4,1,1', '2,2,1', '2,1,2', '1,2,2')


def positive_state(path):
    snapshot = read_q_snapshot(path)
    im, jm, km, _, components = snapshot.header
    if components != 5:
        raise ValueError('M12 requires exactly five conservative components')
    q = comparison_values(snapshot, True).reshape((im+1, jm+1, km+1, 5), order='F')
    rho = q[..., 0]
    if not np.all(rho > 0):
        raise ValueError(f'nonpositive density: {path}')
    internal_energy = q[..., 4] - .5 * np.sum(q[..., 1:4]**2, axis=-1) / rho
    if not np.all(np.isfinite(internal_energy) & (internal_energy > 0)):
        raise ValueError(f'nonpositive/nonfinite internal energy (temperature): {path}')
    return {'rho_min': float(rho.min()), 'internal_energy_min': float(internal_energy.min())}


def check_pair(cpu, gpu, ranks, reference_backend='cpu'):
    fields = compare_snapshot_sets(cpu/'diagnostics/state', gpu/'diagnostics/state',
        ('initialized', 'pre_rhs', 'post_update'), 2e-10, 0., active_only=True)
    if not fields.passed or fields.file_count != 61*ranks:
        raise ValueError(f'authority-state gate failed: {fields.max_abs}, count={fields.file_count}')
    header, reference = read_flowstate(cpu)
    gpu_header, candidate = read_flowstate(gpu)
    if header != gpu_header or reference.shape != candidate.shape:
        raise ValueError('statistics identity differs')
    difference = np.abs(candidate-reference)
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate)) or np.any(difference > 2e-10):
        raise ValueError(f'statistics gate failed: {difference.max()}')
    with h5py.File(cpu/'datin/grid.h5', 'r') as a, h5py.File(gpu/'datin/grid.h5', 'r') as b:
        for key in ('x', 'y', 'z'):
            if not np.array_equal(a[key][...], b[key][...]):
                raise ValueError(f'generated coordinate identity differs: {key}')
    with h5py.File(cpu/'diagnostics/geometry.h5', 'r') as a, \
            h5py.File(gpu/'diagnostics/geometry.h5', 'r') as b:
        if set(a) != set(b) or len(a) != 10:
            raise ValueError('geometry diagnostic inventory differs')
        for key in a:
            values = a[key][...]
            if not np.isfinite(values).all() or not np.array_equal(values, b[key][...]):
                raise ValueError(f'geometry metric identity/finite gate failed: {key}')
            if key == 'jacob' and not np.all(values > 0):
                raise ValueError('nonpositive grid Jacobian')
    return {'q': asdict(fields), 'rhs_sensor': check_rhs_sensor(cpu, gpu, ranks, reference_backend),
            'statistics_max_abs': dict(zip(header, difference.max(axis=0).tolist())),
            'generated_grid': 'bitwise-identical CPU/GPU'}


def check_rhs_sensor(cpu, gpu, ranks, reference_backend='cpu'):
    if reference_backend not in ('cpu', 'gpu'):
        raise ValueError('unknown reference solver backend')
    maxima = {}
    for label in ('conv_after_x', 'conv_after_y', 'conv_after_z', 'conv', 'full', 'sensor'):
        reference = sorted((cpu/'diagnostics').glob(f'state.{label}.*.bin'))
        candidate = sorted((gpu/'diagnostics').glob(f'state.{label}.*.bin'))
        if len(reference) != 30*ranks or [p.name for p in reference] != [p.name for p in candidate]:
            raise ValueError(f'missing three-stage/ten-step {label} diagnostics')
        maximum = 0.
        for a, b in zip(reference, candidate):
            values = []
            headers = []
            masks = []
            for path in (a, b):
                with path.open('rb') as stream:
                    shape = np.fromfile(stream, dtype=np.int32, count=3 if label == 'sensor' else 4)
                    count = int(np.prod(shape[:3]+1)) * (1 if label == 'sensor' else shape[3])
                    headers.append(shape)
                    values.append(np.fromfile(stream, dtype=np.float64, count=count))
                    if label == 'sensor':
                        masks.append(np.fromfile(stream, dtype=np.int8, count=count))
                    if values[-1].size != count or stream.read(1):
                        raise ValueError(f'invalid {label} diagnostic payload')
            if not np.array_equal(headers[0], headers[1]) or any(not np.isfinite(v).all() for v in values):
                raise ValueError(f'{label} identity/finite gate failed')
            if label == 'sensor' and (masks[0].size != count or masks[1].size != count or
                                      not np.array_equal(masks[0], masks[1])):
                raise ValueError('CPU/GPU characteristic activation masks differ')
            # CPU per-axis snapshots precede qrhs=-qrhs in rhscal; GPU already stores RHS.
            sign = -1 if reference_backend == 'cpu' and label.startswith('conv_after_') else 1
            difference = float(np.max(np.abs(sign*values[0]-values[1])))
            if label != 'sensor' and difference > 2e-10:
                raise ValueError(f'{label} same-definition gate failed: {a.name}: {difference}')
            maximum = max(maximum, difference)
        maxima[label] = maximum
    replay_receipt = analyze_pair(cpu, gpu, expected_snapshots=30*ranks)
    return {'max_abs': maxima, 'file_count_per_label': 30*ranks,
            'sensor_captured_input_gate': replay_receipt,
            'characteristic_activation_masks': 'bitwise-identical',
            'reference_backend': reference_backend,
            'per_axis_reference_sign': -1 if reference_backend == 'cpu' else 1,
            'completed_convection_full_rhs_reference_sign': 1}


def check_private_diagnostics(cpu, gpu, ranks):
    reference = sorted((cpu/'diagnostics').glob('private_step*.bin'))
    candidate = sorted((gpu/'diagnostics').glob('private_step*.bin'))
    if len(reference) != 2*ranks or [p.name for p in reference] != [p.name for p in candidate]:
        raise ValueError('missing first/last completed-step private diagnostics')
    maximum = 0.
    for a, b in zip(reference, candidate):
        with a.open('rb') as stream:
            shape_a = np.fromfile(stream, dtype=np.int32, count=4)
            values_a = np.fromfile(stream, dtype=np.float64)
        with b.open('rb') as stream:
            shape_b = np.fromfile(stream, dtype=np.int32, count=4)
            values_b = np.fromfile(stream, dtype=np.float64)
        expected = int(np.prod(shape_a[:3]+1))*14
        if not np.array_equal(shape_a, shape_b) or shape_a[-1] != 14 or \
                values_a.size != expected or values_b.size != expected:
            raise ValueError('private diagnostic shape/identity differs')
        if not np.isfinite(values_a).all() or not np.isfinite(values_b).all():
            raise ValueError('nonfinite private diagnostic field')
        difference = float(np.max(np.abs(values_a-values_b)))
        if difference > 2e-10:
            raise ValueError(f'CPU/GPU private field gate failed: {a.name}: {difference}')
        maximum = max(maximum, difference)
    for case in (cpu, gpu):
        lines = [line for line in (case/'run.log').read_text().splitlines()
                 if line.startswith('ASTR_M12_PRIVATE_DIAGNOSTIC step=')]
        if len(lines) != 2 or any('q_bits_rng_unchanged=1' not in line for line in lines):
            raise ValueError('private diagnostic analytic/isolation proof missing')
    return {'max_abs': maximum, 'file_count': len(reference), 'phase': 'completed steps 1 and 10',
            'analytic_affine_and_rng_q_bit_isolation': 'passed inside both solver backends'}


def run(args, backend, topology, insitu_config=None, case_label=None, resource_baseline=None,
        observation=None, steps=10, restore=None, checkpoint_interval=None,product_oracles=False):
    if steps not in (5,10) or (restore is not None and (steps!=10 or checkpoint_interval is None)):
        raise ValueError('M12 restart gate is limited to ten steps or five-plus-five')
    if observation not in (None, 'images', 'geometry', 'trace'):
        raise ValueError('unknown M12 observation mode')
    if observation and (backend != 'gpu' or insitu_config is None or args.memcheck):
        raise ValueError('observation requires an uninstrumented GPU render run without memcheck')
    if product_oracles and (backend!='gpu' or insitu_config is None or observation is not None):
        raise ValueError('product continuation oracles require a separate instrumented GPU render gate')
    ranks = int(np.prod([int(value) for value in topology.split(',')]))
    profiler_allowance = (observation == 'trace' and ranks == 4 and
                          getattr(args, 'np4_profiler_host_budget_8gib', False))
    external_host_budget = (8 if profiler_allowance else 4)*1024**3
    case = args.output / ((case_label or backend) + '_np' + str(ranks) + '_' + topology.replace(',', 'x'))
    prepare(args.source, case, backend)
    first_step=5 if restore is not None else 0
    if steps!=10:
        path=case/'datin/controller'
        lines=path.read_text().splitlines()
        replace_value(lines,'maxstep,feqchkpt',f'{steps-1},1,9999,9999,1,9999')
        path.write_text('\n'.join(lines)+'\n')
    if checkpoint_interval is not None:
        (case/'outdat/new').mkdir()
        (case/'datin/input.output').write_text(f"""&output
 directory='outdat/new',restore_directory='{restore or ''}',restart_output='saved',
 host_budget_bytes=67108864,device_budget_bytes=67108864,buffer_bytes=1048576,
 device_reserve_bytes=1073741824
/
&checkpoint
 enabled=t,mode='steps',interval_steps={checkpoint_interval},keep=2,initial_frame=f
/
&volume
 enabled=f
/
&slices
 enabled=f
/
""")
    if restore is not None:
        # Initialization and the UDF must use only the checkpoint's frozen resources.
        for name in ('grid.2d','flowini2d.h5','inlet.prof','wallbs.dat'):
            (case/'datin'/name).unlink()
    env = os.environ.copy()
    if observation:
        for key in list(env):
            if key.startswith(('ASTR_VALIDATION_', 'ASTR_M12_', 'ASTR_INSITU_TEST_')):
                env.pop(key)
        env.pop('ASTR_VTK_PIXEL_AUDIT', None)
    for key in ('ASTR_INSITU_CONFIG', 'ASTR_OUTPUT_CONFIG', 'ASTR_OUTPUT_ADAPTIVE_CONFIG',
                'ASTR_VALIDATION_RHS_STEP_SECONDARY', 'ASTR_VALIDATION_RK_SNAPSHOT',
                'ASTR_M12_DIAGNOSTIC_CHECK', 'ASTR_M12_RENDER_ISOLATION',
                'ASTR_INSITU_SAMPLE_PREFIX', 'ASTR_INSITU_RESIDENT_AUDIT',
                'ASTR_INSITU_TEST_PLANE_PREFIX', 'ASTR_INSITU_TEST_CURVE_Q_PREFIX',
                'ASTR_INSITU_TEST_CURVE_TRACE_PREFIX', 'ASTR_INSITU_TEST_CURVE_STEP_SCALE'):
        env.pop(key, None)
    env.update(ASTR_FORCE_MPI_TOPOLOGY=topology, ASTR_WALL_BLOWING_MODE='legacy_random',
               ASTR_GPU_SYNC_MODE='explicit', ASTR_GPU_HALO_TRANSPORT='pageable',
               ASTR_GEOMETRY_DUMP='diagnostics/geometry.h5',
               ASTR_VALIDATION_RHS_PREFIX='diagnostics/state',
               ASTR_M12_SENSOR_DETAIL='1',
               ASTR_VALIDATION_RHS_STEP='0', ASTR_VALIDATION_RHS_STEP_END='9')
    if observation:
        for key in ('ASTR_GEOMETRY_DUMP', 'ASTR_VALIDATION_RHS_PREFIX', 'ASTR_M12_SENSOR_DETAIL',
                    'ASTR_VALIDATION_RHS_STEP', 'ASTR_VALIDATION_RHS_STEP_END'):
            env.pop(key, None)
    if args.diagnostics:
        env['ASTR_M12_DIAGNOSTIC_CHECK'] = '1'
    if insitu_config is not None:
        if backend != 'gpu':
            raise ValueError('approved M12 render preset requires the GPU solver')
        (case/'outdat/render').mkdir()
        (case/'datin/input.insitu').write_text(insitu_config)
        env['ASTR_INSITU_CONFIG'] = 'datin/input.insitu'
        env['ASTR_INSITU_TIMING'] = '1'
        env['ASTR_INSITU_RESIDENT_AUDIT'] = '1'
        env['ASTR_M12_RENDER_ISOLATION'] = '1'
        if observation:
            env.pop('ASTR_INSITU_RESIDENT_AUDIT')
            env.pop('ASTR_M12_RENDER_ISOLATION')
            env['ASTR_INSITU_TEST_ORACLE_IO'] = '0'
        if observation == 'geometry' or product_oracles:
            env.update(ASTR_INSITU_TEST_PLANE_PREFIX='diagnostics/plane',
                       ASTR_INSITU_TEST_CURVE_Q_PREFIX='diagnostics/surface',
                       ASTR_INSITU_TEST_CURVE_TRACE_PREFIX='diagnostics/trace')
        if observation == 'trace':
            env['ASTR_VTK_PIXEL_AUDIT'] = '1'
        for key in ('DISPLAY', 'PYTHONPATH', 'CATALYST_IMPLEMENTATION_PREFER_ENV', 'VTK_EGL_DEVICE_INDEX'):
            env.pop(key, None)
    command = [str(args.mpiexec), '--mca', 'pml', 'ob1', '--mca', 'btl', 'self,vader,tcp',
               '--mca', 'coll_hcoll_enable', '0', '--oversubscribe', '-np', str(ranks)]
    if args.memcheck and backend == 'gpu':
        env.update(OMPI_MCA_opal_cuda_support='false',
                   OMPI_MCA_osc='sm' if ranks==4 and insitu_config is not None else 'pt2pt',
                   OMPI_MCA_coll_ucc_enable='0')
        command += [str(args.sanitizer), '--tool', 'memcheck', '--error-exitcode', '99',
                    '--log-file', str(case/'memcheck.%p.log')]
    if observation == 'trace':
        command += [str(args.nsys), 'profile', '--trace=cuda,nvtx,mpi', '--mpi-impl=openmpi',
                    '--sample=none', '--cpuctxsw=none', '--cuda-memory-usage=true',
                    '--cuda-um-cpu-page-faults=true', '--cuda-um-gpu-page-faults=true',
                    '--export=sqlite', '--output='+str(case/'trace.rank%q{OMPI_COMM_WORLD_RANK}')]
    command += [str(args.executable), 'run', 'datin/input.dat']
    cfl = CflGate(case/'run.log', 1., .02, first_step, steps)
    checked = {}

    def progress(final=False):
        cfl.check(final=final)
        for path in sorted((case/'diagnostics').glob('state.*.bin')):
            if path in checked or not ('.post_update.' in path.name or '.initialized.' in path.name):
                continue
            if path.stat().st_size < 20:
                continue
            with path.open('rb') as stream:
                im, jm, km, hm, nq = np.fromfile(stream, dtype=np.int32, count=5)
            expected = 20 + 8*(im+2*hm+1)*(jm+2*hm+1)*(km+2*hm+1)*nq
            if path.stat().st_size < expected:
                continue
            checked[path] = positive_state(path)
        if sum(p.stat().st_size for p in case.rglob('*') if p.is_file()) > 4*1024**3:
            raise ValueError('approved 4 GiB directory budget exceeded')

    with (case/'run.log').open('wb') as log:
        if backend == 'gpu':
            resources = run_monitored(command, case, env, log, case/'resources.sampled.json',
                                      baseline=resource_baseline,
                                      host_extra_budget_bytes=external_host_budget,
                                      timeout_seconds=180, check_progress=progress)
            if resource_baseline is None and (resources['host_rss_peak_bytes'] > 4*1024**3 or any(
                    entry['peak_bytes'] > 2*1024**3 for entry in resources['devices'].values())):
                raise ValueError('total sampled job memory exceeds even the additional-memory allowance')
        else:
            process = subprocess.Popen(command, cwd=case, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                deadline = time.monotonic()+180
                while process.poll() is None:
                    progress()
                    if time.monotonic() > deadline:
                        raise TimeoutError('M12 short gate exceeded 180 seconds')
                    time.sleep(.02)
                if process.returncode:
                    raise subprocess.CalledProcessError(process.returncode, command)
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                raise
    progress(final=True)
    if (not observation and len(checked) != (1+3*(steps-first_step))*ranks) or 'The job is done!' not in (case/'run.log').read_text():
        raise ValueError('missing successful ten-step/three-stage state checks')
    if args.memcheck and backend == 'gpu':
        logs = list(case.glob('memcheck.*.log'))
        if len(logs) != ranks or any('ERROR SUMMARY: 0 errors' not in p.read_text() for p in logs):
            raise ValueError('Compute Sanitizer memory gate failed')
    (case/'gate.json').write_text(json.dumps({'cfl_max': cfl.maximum, 'state_count': len(checked),
        'rho_min': min((s['rho_min'] for s in checked.values()), default=None),
        'internal_energy_min': min((s['internal_energy_min'] for s in checked.values()), default=None),
        'observation': observation,
        'external_host_budget_bytes': external_host_budget,
        'native_postprocessing_host_budget_bytes': 4*1024**3 if insitu_config is not None else None,
        'strict_absolute_tolerance': 2e-10, 'statistics_phase': 'legacy first-RK-stage listing',
        'final_q_phase': 'third RK update, after sequential x/y sponge',
        'memcheck': bool(args.memcheck and backend == 'gpu')}, indent=2))
    return case


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--mpiexec', type=Path, required=True)
    parser.add_argument('--topologies', nargs='+', choices=TOPOLOGIES, default=TOPOLOGIES)
    parser.add_argument('--memcheck', action='store_true')
    parser.add_argument('--diagnostics', action='store_true')
    parser.add_argument('--sanitizer', type=Path)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.executable = args.executable.resolve(strict=True)
    if args.memcheck and args.sanitizer is None:
        parser.error('--memcheck requires --sanitizer')
    args.output.mkdir(parents=True, exist_ok=True)
    for topology in args.topologies:
        cpu = run(args, 'cpu', topology)
        gpu = run(args, 'gpu', topology)
        ranks = int(np.prod([int(value) for value in topology.split(',')]))
        receipt = check_pair(cpu, gpu, ranks)
        if args.diagnostics:
            receipt['private_diagnostics'] = check_private_diagnostics(cpu, gpu, ranks)
        (args.output/('compare_'+topology.replace(',', 'x')+'.json')).write_text(json.dumps(receipt, indent=2))
        print(f"PASS topology={topology} q_max={receipt['q']['max_abs']:.8e}", flush=True)
