"""Ordinary compact TGV against immutable CPU and CUDA Fortran executables."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

from run_cpp_tgv_gate import prepare_case, checkpoint, compare, logged_steps, namelist
from run_cpp_restart_gate import datasets
from run_cpp_multi_tgv_gate import layouts, raw_field, physical, global_region


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--mpiexec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--form', choices=['explicit_explicit', 'compact_explicit', 'explicit_compact', 'compact_compact'], required=True)
    parser.add_argument('--convection-form', choices=['standard', 'keep'], default='standard')
    parser.add_argument('--viscous-form', choices=['standard', 'me6_opti'], default='standard')
    parser.add_argument('--reference-only', action='store_true')
    parser.add_argument('--grid', type=int, default=16)
    parser.add_argument('--mapping', choices=['periodic'])
    parser.add_argument('--checkpoint', action='store_true')
    parser.add_argument('--analyze-only', action='store_true')
    parser.add_argument('--topologies', nargs='+', default=['1,1,1', '2,1,1', '1,2,1', '1,1,2'])
    parser.add_argument('--transport', choices=['pinned', 'device-aware'], default='pinned')
    parser.add_argument('--halo', choices=['pinned', 'device'], default='pinned')
    parser.add_argument('--workspace', choices=['scalar', 'full'], default='scalar')
    parser.add_argument('--sync', choices=['explicit', 'dependency'], default='explicit')
    parser.add_argument('--memcheck', action='store_true')
    parser.add_argument('--references', type=Path, help='reuse matching immutable-run observations')
    options = parser.parse_args()
    options.output = options.output.resolve()
    if not options.analyze_only:
        options.output.mkdir(parents=True, exist_ok=False)
    frozen = options.reference.resolve()
    original = frozen/'evidence/standard_tgv_explicit10/cpu_np1_keep_x/datin'
    derivative, filtering = options.form.split('_')
    scheme = '643c' if derivative == 'compact' else '643e'
    filter_scheme = 'compact10' if filtering == 'compact' else 'explicit10'
    base = os.environ | dict(ASTR_GPU_FILTER_WORKSPACE=options.workspace, ASTR_GPU_SYNC_MODE='dependency',
                             ASTR_GPU_COMPACT_TRANSPORT=options.transport, ASTR_GPU_COMPACT_SYNC=options.sync,
                             ASTR_GPU_HALO_TRANSPORT=options.halo, ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',
                             ASTR_VALIDATION_RHS_STEP='0', ASTR_VALIDATION_RHS_STEP_SECONDARY='4')
    report = dict(reference=str(frozen), form=options.form, transport=options.transport, halo=options.halo,
                  workspace=options.workspace, sync=options.sync,
                  convection_form=options.convection_form, viscous_form=options.viscous_form,
                  grid=options.grid, mapping=options.mapping,
                  batch_rhs=int(base.get('ASTR_GPU_COMPACT_BATCH_RHS', '65535')), cases={})

    def run(name, binary, gpu, cpp, dims):
        directory = options.output/name
        if options.analyze_only:
            return directory
        prepare_case(directory, original, gpu, cpp, False, grid=options.grid,
                     numerics=dict(conschm=scheme, difschm=scheme, filter_scheme=filter_scheme,
                                   convection_form=options.convection_form, viscous_form=options.viscous_form),
                     mesh=dict(lreadgrid=True, gridfile='datin/grid.tgv.h5') if options.mapping else None)
        if options.mapping:
            generator = Path(__file__).resolve().parents[1]/'gpu_validation/generate_curvilinear_tgv_grid.py'
            subprocess.run([sys.executable, str(generator), '--grid', ','.join([str(options.grid)]*3),
                            '--amplitude', '0.15', '--mapping', options.mapping,
                            '--output', str(directory/'datin/grid.tgv.h5'),
                            '--report', str(directory/'grid_report.txt')], check=True)
        if cpp and options.checkpoint:
            (directory/'datin/input.output').write_text(namelist(dict(
                output=dict(directory='outdat/new', exact_restart=True, host_budget_bytes=-1,
                            device_budget_bytes=-1, device_reserve_bytes=0),
                checkpoint=dict(enabled=True, mode='steps', interval_steps=2, keep=2),
                volume=dict(enabled=False), slices=dict(enabled=False))))
        env = base | {'ASTR_FORCE_MPI_TOPOLOGY': ','.join(map(str, dims))}
        if not cpp:
            env |= dict(ASTR_GPU_SYNC_MODE='explicit', ASTR_GPU_COMPACT_SYNC='explicit', ASTR_GPU_COMPACT_TRANSPORT='pinned')
        application = [str(binary.resolve()), 'run']
        if cpp and options.memcheck:
            application = ['compute-sanitizer', '--tool', 'memcheck', '--leak-check', 'full',
                           '--error-exitcode', '77', '--log-file', str(directory/'memcheck_rank%q{OMPI_COMM_WORLD_RANK}.log'),
                           *application]
        command = [str(options.mpiexec), '--bind-to', 'none', '-n', str(int(np.prod(dims))), *application]
        with (directory/'run.log').open('w') as log:
            subprocess.run(command, cwd=directory, env=env, stdin=subprocess.DEVNULL,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        if cpp and options.memcheck:
            for rank in range(int(np.prod(dims))):
                text = (directory/f'memcheck_rank{rank}.log').read_text()
                if 'ERROR SUMMARY: 0 errors' not in text or 'LEAK SUMMARY: 0 bytes leaked' not in text:
                    raise AssertionError('compact ordinary ASTR memory/leak check')
        return directory

    references = {}
    for backend, gpu, binary in [('cpu', False, frozen/'bin/astr_cpu'), ('old_gpu', True, frozen/'bin/astr_cuda_fortran')]:
        directory = options.references/backend if options.references else run(backend, binary, gpu, False, (1, 1, 1))
        references[backend] = directory
    final = Path('outdat/new/checkpoints/step000000000004/state.h5')
    reference_q = {name: checkpoint(directory/final) for name, directory in references.items()}
    report['reference_crosscheck'] = compare(reference_q['cpu'], reference_q['old_gpu'])
    if options.reference_only:
        differences = dict(statistics=compare(np.loadtxt(references['cpu']/'flowstate.dat', skiprows=1),
                                              np.loadtxt(references['old_gpu']/'flowstate.dat', skiprows=1)))
        for stage in range(1, 4):
            for primitive in (False, True):
                label = 'pre_rhs_primitives' if primitive else 'pre_rhs'
                values = []
                for backend in ['cpu', 'old_gpu']:
                    header, data = raw_field(references[backend]/f'outdat/rhs.{label}.step00000000.rk{stage:02}.rank00000000.bin', primitive)
                    values.append(physical(data, dict(upper=header[:3], halo=(header[3],))))
                differences[f'{label}/rk{stage}'] = compare(*values)
            for label in ['conv', 'full']:
                values = []
                for backend in ['cpu', 'old_gpu']:
                    with (references[backend]/f'outdat/rhs.{label}.step00000000.rk{stage:02}.rank00000000.bin').open('rb') as stream:
                        shape = tuple(np.fromfile(stream, np.int32, 4))
                        values.append(np.fromfile(stream, np.float64).reshape(tuple(n+1 for n in shape[:3])+(shape[3],), order='F'))
                differences[f'{label}/rk{stage}'] = compare(*values)
        report['reference_operators'] = differences
        (options.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print('Matched frozen references:', report['reference_crosscheck'], flush=True)
        return
    for value in options.topologies:
        dims = tuple(map(int, value.split(',')))
        name = 'cpp_np%d_%dx%dx%d' % (np.prod(dims), *dims)
        directory = run(name, options.binary, True, True, dims)
        layout = layouts(directory)
        result = dict(terminal={}, stages={}, operators={}, statistics={}, steps={})
        if options.checkpoint:
            actual_geometry = datasets(directory/'outdat/new/resources/geometry.h5')
            geometry_errors = {}
            for backend, refdir in references.items():
                expected_geometry = datasets(refdir/'outdat/new/resources/geometry.h5')
                if actual_geometry.keys() != expected_geometry.keys():
                    raise AssertionError('checkpoint geometry dataset identities differ')
                # Rank-local redundant geometry is topology-specific; compare it
                # bitwise in same-topology restart, not against an NP=1 layout.
                physical_keys = [f'q{n:04d}' for n in range(1, 14)]
                geometry_errors[backend] = {key: compare(actual_geometry[key].reshape(-1, 1), expected_geometry[key].reshape(-1, 1))
                    for key in physical_keys}
            result['geometry'] = geometry_errors
            final_q = checkpoint(directory/final)
            for rank, own in layout.items():
                _, raw = raw_field(directory/f'outdat/rhs.cpp_final.step00000004.rk01.rank{rank:08}.bin')
                if not np.array_equal(physical(raw, own).view(np.uint64), global_region(final_q, own).view(np.uint64)):
                    raise AssertionError('published checkpoint q differs from complete-step device observation')
        if options.viscous_form == 'me6_opti':
            log = (directory/'run.log').read_text()
            allocations = {int(rank): tuple(map(int, values)) for rank, *values in re.findall(
                r'ASTR_CPP_ME6 rank=(\d+) mode=(\d+) device_bytes=(\d+) pinned_bytes=(\d+)', log)}
            traffic = {int(rank): tuple(map(int, values)) for rank, *values in re.findall(
                r'ASTR_CPP_ME6_TRAFFIC rank=(\d+) sent_bytes=(\d+) host_stage_bytes=(\d+)', log)}
            if allocations.keys() != layout.keys() or traffic.keys() != layout.keys():
                raise AssertionError('missing rank-local ME6 transport evidence')
            for rank, own in layout.items():
                upper = own['upper'];h, = own['halo'];parts = own['topology']
                packets = []
                if parts[1] > 1:
                    packets.append(4*h*(upper[0]+2*h+1)*(upper[2]+1))
                if parts[2] > 1:
                    packets.append(4*h*(upper[0]+2*h+1)*(upper[1]+2*h+1))
                capacity = max(packets, default=0)
                device_bytes = 2*int(np.prod(np.array(upper)+2*h+1))*8+162*8+4*capacity*8
                pinned_bytes = 4*capacity*8 if options.halo == 'pinned' else 0
                mode = int(options.halo == 'device')
                sent = 12*2*8*sum(packets)
                staging = 2*sent if options.halo == 'pinned' else 0
                if allocations[rank] != (mode, device_bytes, pinned_bytes) or traffic[rank] != (sent, staging):
                    raise AssertionError(f'ME6 rank {rank}: unexpected allocation/transport traffic')
            result['me6_allocations'] = allocations
            result['me6_traffic'] = traffic
        for backend, refdir in references.items():
            errors = []
            for rank, own in layout.items():
                _, data = raw_field(directory/f'outdat/rhs.cpp_final.step00000004.rk01.rank{rank:08}.bin')
                errors.append(compare(physical(data, own), global_region(reference_q[backend], own)))
            result['terminal'][backend] = np.max(errors, axis=0).tolist()
            result['statistics'][backend] = compare(np.loadtxt(directory/'flowstate.dat', skiprows=1),
                                                    np.loadtxt(refdir/'flowstate.dat', skiprows=1))
            result['steps'][backend] = compare(logged_steps(directory), logged_steps(refdir))
            maximum = np.zeros(6)
            for stage in range(1, 4):
                for primitive in (False, True):
                    label = 'pre_rhs_primitives' if primitive else 'pre_rhs'
                    header, ref = raw_field(refdir/f'outdat/rhs.{label}.step00000000.rk{stage:02}.rank00000000.bin', primitive)
                    ref = physical(ref, dict(upper=header[:3], halo=(header[3],)))
                    for rank, own in layout.items():
                        _, data = raw_field(directory/f'outdat/rhs.{label}.step00000000.rk{stage:02}.rank{rank:08}.bin', primitive)
                        differences = compare(physical(data, own), global_region(ref, own))
                        maximum[:len(differences)] = np.maximum(maximum[:len(differences)], differences)
            result['stages'][backend] = maximum.tolist()
            maximum = np.zeros(5)
            for stage in range(1, 4):
                for label in ['conv', 'full']:
                    path = f'outdat/rhs.{label}.step00000000.rk{stage:02}.rank00000000.bin'
                    with (refdir/path).open('rb') as stream:
                        shape = tuple(np.fromfile(stream, np.int32, 4))
                        ref = np.fromfile(stream, np.float64).reshape(tuple(n+1 for n in shape[:3])+(shape[3],), order='F')
                    for rank, own in layout.items():
                        path = f'outdat/rhs.{label}.step00000000.rk{stage:02}.rank{rank:08}.bin'
                        with (directory/path).open('rb') as stream:
                            shape = tuple(np.fromfile(stream, np.int32, 4))
                            actual = np.fromfile(stream, np.float64).reshape(tuple(n+1 for n in shape[:3])+(shape[3],), order='F')
                        maximum = np.maximum(maximum, compare(actual, global_region(ref, own)))
            result['operators'][backend] = maximum.tolist()
        report['cases'][name] = result
        print(name, result, flush=True)
        (options.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
