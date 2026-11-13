"""Real multi-rank TGV: frozen fields and every exchanged halo phase.

The Python code prepares inputs and reads ASTR observations; it never advances
the flow or supplies halos to the solver.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np

from run_cpp_tgv_gate import prepare_case, checkpoint, compare, logged_steps, cfl_log

PACKETS = [(0, 0, 0), (13, 1, 0), (0, 2, 0),
           (0, 0, 1), (0, 1, 1), (0, 2, 1),
           (0, 0, 0), (0, 1, 0), (0, 2, 0),
           (11, 0, 0), (12, 0, 0), (11, 1, 0), (12, 1, 0), (11, 2, 0), (12, 2, 0)]
COMPONENTS = {0: 5, 13: 5, 11: 6, 12: 3}


def validation_cases(quad=False):
    result = [('np1_trace', 1, (1, 1, 1), True, 'dependency')]
    for axis, topology in [('x', (2, 1, 1)), ('y', (1, 2, 1)), ('z', (1, 1, 2))]:
        result.extend([(f'np2_{axis}_trace', 2, topology, True, 'dependency'),
                       (f'np2_{axis}_plain', 2, topology, False, 'dependency')])
    result.append(('np2_x_explicit', 2, (2, 1, 1), False, 'explicit'))
    if quad:
        result.extend([('np4_xy_trace', 4, (2, 2, 1), True, 'dependency'),
                       ('np4_xy_plain', 4, (2, 2, 1), False, 'dependency')])
    return result


def layouts(directory):
    result = {}
    for line in (directory/'run.log').read_text().splitlines():
        if not line.startswith('ASTR_GPU_CPP_LAYOUT '):
            continue
        fields = {name: tuple(map(int, value.split())) for name, value in
                  re.findall(r'(\w+)=\s*((?:-?\d+\s*)+)', line)}
        rank, = fields.pop('rank')
        if rank in result:
            raise AssertionError('duplicate layout record')
        if set(fields) != {'global', 'origin', 'upper', 'halo', 'topology', 'lower', 'higher'}:
            raise AssertionError('incomplete solver-owned layout record')
        result[rank] = fields
    if not result or sorted(result) != list(range(int(np.prod(result[0]['topology'])))):
        raise AssertionError('missing MPI rank layout')
    return result


def raw_field(path, primitive=False, packet=False):
    with path.open('rb') as stream:
        header = tuple(map(int, np.fromfile(stream, np.int32, 9 if packet else (4 if primitive else 5))))
        values = np.fromfile(stream, np.float64)
    im, jm, km, hm = header[:4]
    components = 6 if primitive else header[4]
    shape = (im+2*hm+1, jm+2*hm+1, km+2*hm+1, components)
    if values.size != np.prod(shape):
        raise AssertionError(f'{path}: field payload size')
    return header, values.reshape(shape, order='F')


def physical(values, layout):
    h, = layout['halo']
    return values[tuple(slice(h, h+n+1) for n in layout['upper'])+(slice(None),)]


def global_region(values, layout):
    return values[tuple(slice(o, o+n+1) for o, n in zip(layout['origin'], layout['upper']))+(slice(None),)]


def bits_equal(actual, expected, label):
    if actual.shape != expected.shape or not np.array_equal(actual.view(np.uint64), expected.view(np.uint64)):
        difference = float(np.max(abs(actual-expected))) if actual.shape == expected.shape else None
        raise AssertionError(f'{label}: bits differ, max_abs={difference}')


def axis_region(values, layout, axis):
    h, = layout['halo']
    selection = [slice(h, h+n+1) for n in layout['upper']]
    selection[axis] = slice(None)
    return values[tuple(selection)+(slice(None),)]


def check_halos(fields, layout, axis, reconcile, label, exact=True):
    for rank, values in fields.items():
        own = layout[rank]
        h, = own['halo']
        n = own['upper'][axis]
        region = axis_region(values, own, axis)
        for side in (0, 1):
            peer = (rank if own['topology'][axis] == 1 else own['lower' if side == 0 else 'higher'][axis])
            other = layout[peer]
            oh, = other['halo']
            for a in range(3):
                if a != axis and (own['origin'][a], own['upper'][a]) != (other['origin'][a], other['upper'][a]):
                    raise AssertionError('transverse neighbor coordinates differ')
            neighbor = axis_region(fields[peer], other, axis)
            offsets = np.arange(1, h+1)
            destination = h-offsets if side == 0 else h+n+offsets
            source = oh+other['upper'][axis]-offsets if side == 0 else oh+offsets
            actual = np.take(region, destination, axis=axis)
            expected = np.take(neighbor, source, axis=axis)
            name = f'{label}/rank{rank}/axis{axis}/side{side}'
            if not np.isfinite(actual).all() or not np.isfinite(expected).all():
                raise AssertionError(name+': nonfinite consumed halo')
            if exact:
                bits_equal(actual, expected, name)
            else:
                compare(actual, expected)
            if reconcile:
                endpoint = h if side == 0 else h+n
                other_endpoint = oh+other['upper'][axis] if side == 0 else oh
                bits_equal(np.take(region, endpoint, axis=axis), np.take(neighbor, other_endpoint, axis=axis),
                           name+'/shared endpoint')


def analyze(directory, frozen):
    layout = layouts(directory)
    reference = frozen/'evidence/standard_tgv_explicit10'
    terminal = {backend: checkpoint(reference/f'{backend}_np1_keep_x/outdat/new/checkpoints/step000000000004/state.h5')
                for backend in ('cpu', 'gpu')}
    compare(terminal['cpu'], terminal['gpu'])
    report = {'topology': layout[0]['topology'], 'terminal': {}, 'stages': {}, 'halo_phase_checks': 0}
    for rank, meta in layout.items():
        header, q = raw_field(directory/f'outdat/rhs.cpp_final.step00000004.rk01.rank{rank:08d}.bin')
        if header != (*meta['upper'], *meta['halo'], 5):
            raise AssertionError('terminal layout differs from solver metadata')
        q = physical(q, meta)
        pressure = .4*(q[..., 4]-.5*np.sum(q[..., 1:4]**2, axis=-1)/q[..., 0])
        if not (q[..., 0] > 0).all() or not (pressure > 0).all():
            raise AssertionError('nonpositive physical state')
        report['terminal'][rank] = {b: compare(q, global_region(v, meta)) for b, v in terminal.items()}
    for backend in ('cpu', 'gpu'):
        ref = reference/f'{backend}_np1_keep_x'
        report[backend+'_scalars'] = compare(np.loadtxt(directory/'flowstate.dat', skiprows=1),
                                            np.loadtxt(ref/'flowstate.dat', skiprows=1))
        report[backend+'_steps'] = compare(logged_steps(directory), logged_steps(ref))
        report[backend+'_cfl'] = compare(cfl_log(directory), cfl_log(ref))
    traced = list((directory/'outdat').glob('rhs.cpp_halo.*.bin'))
    if not traced:
        return report
    if len(traced) != len(PACKETS)*3*len(layout):
        raise AssertionError('incomplete exchange phase observations')
    for stage in (1, 2, 3):
        stage_references = {}
        for backend in ('cpu', 'gpu'):
            ref = reference/f'{backend}_np1_keep_x/outdat'
            suffix = f'.step00000000.rk{stage:02d}.rank00000000.bin'
            rh, rq = raw_field(ref/('rhs.pre_rhs'+suffix))
            ph, rp = raw_field(ref/('rhs.pre_rhs_primitives'+suffix), primitive=True)
            ref_layout = {'upper': rh[:3], 'halo': (rh[3],)}
            stage_references[backend] = (physical(rq, ref_layout), physical(rp, ref_layout))
        compare(stage_references['cpu'][0], stage_references['gpu'][0])
        compare(stage_references['cpu'][1], stage_references['gpu'][1])
        qfields, primitives = {}, {}
        for rank, meta in layout.items():
            suffix = f'.step00000000.rk{stage:02d}.rank{rank:08d}.bin'
            qheader, qfields[rank] = raw_field(directory/'outdat'/('rhs.pre_rhs'+suffix))
            pheader, primitives[rank] = raw_field(directory/'outdat'/('rhs.pre_rhs_primitives'+suffix), primitive=True)
            if qheader[:4] != (*meta['upper'], *meta['halo']) or pheader != qheader[:4]:
                raise AssertionError('prepared-state layout')
            values = physical(qfields[rank], meta)
            rho = values[..., :1]
            velocity = values[..., 1:4]/rho
            pressure = .4*(values[..., 4:5]-.5*rho*np.sum(velocity**2, axis=-1, keepdims=True))
            expected = np.concatenate([rho, velocity, pressure, .014*pressure/rho], axis=-1)
            result = {'cache_coherence': compare(physical(primitives[rank], meta), expected)}
            for backend, (rq, rp) in stage_references.items():
                result[backend+'_q'] = compare(values, global_region(rq, meta))
                result[backend+'_primitives'] = compare(physical(primitives[rank], meta),
                                                        global_region(rp, meta))
            report['stages'][f'rank{rank}/rk{stage}'] = result
        for axis in range(3):
            check_halos(qfields, layout, axis, True, f'prepared/rk{stage}')
            check_halos(primitives, layout, axis, False, f'primitive/rk{stage}', exact=False)
        for sequence, (field, axis, reconcile) in enumerate(PACKETS, 1):
            fields = {}
            for rank, meta in layout.items():
                filename = f'rhs.cpp_halo.step00000000.rk{stage:02d}.exchange{sequence:02d}.rank{rank:08d}.bin'
                header, fields[rank] = raw_field(directory/'outdat'/filename, packet=True)
                if header != (*meta['upper'], *meta['halo'], COMPONENTS[field], field, axis, reconcile, sequence):
                    raise AssertionError('field/axis/reconciliation identity mismatch')
                coordinates = np.indices(fields[rank].shape[:3])-meta['halo'][0]
                keep = np.ones(coordinates.shape[1:], bool)
                for a in range(3):
                    if a != axis:
                        keep &= (coordinates[a] >= 0) & (coordinates[a] <= meta['upper'][a])
                if np.any(fields[rank][~keep].view(np.uint64) != 0):
                    raise AssertionError('observation padding is not positive zero')
            check_halos(fields, layout, axis, bool(reconcile), f'field{field}/rk{stage}/exchange{sequence}')
            if sequence == 9:
                for rank, values in fields.items():
                    bits_equal(axis_region(values, layout[rank], axis), axis_region(qfields[rank], layout[rank], axis),
                               'observation versus independent full-q download')
            report['halo_phase_checks'] += len(layout)
    return report


def main():
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--binary', type=Path)
    action.add_argument('--analyze', type=Path)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--mpiexec', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--sanitizer', choices=['memcheck', 'initcheck'])
    parser.add_argument('--include-quad', action='store_true')
    options = parser.parse_args()
    if options.analyze:
        result = analyze(options.analyze, options.reference)
        (options.analyze/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2))
        return
    if options.output is None or options.mpiexec is None:
        parser.error('--binary requires --output and --mpiexec')
    options.output = options.output.resolve()
    options.output.mkdir(parents=True, exist_ok=False)
    frozen = options.reference.resolve()
    source = frozen/'evidence/standard_tgv_explicit10/cpu_np1_keep_x/datin'
    environment = os.environ | dict(OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,vader,tcp', OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc', OMPI_MCA_opal_cuda_support='0', ASTR_GPU_FILTER_WORKSPACE='full',
        ASTR_GPU_HALO_TRANSPORT='pinned', ASTR_CFL_PROFILE_Y='1', ASTR_VALIDATION_RHS_PREFIX='outdat/rhs')
    report = {}
    matrix = validation_cases(options.include_quad)
    if options.sanitizer:
        matrix = [('np2_x_trace', 2, (2, 1, 1), True, 'dependency')]
    for name, ranks, topology, trace, sync in matrix:
        directory = options.output/name
        prepare_case(directory, source, True, True, False)
        env = environment | dict(ASTR_FORCE_MPI_TOPOLOGY=','.join(map(str, topology)), ASTR_GPU_SYNC_MODE=sync,
                                 ASTR_VALIDATION_RHS_STEP='0' if trace else '4', ASTR_VALIDATION_RHS_STEP_SECONDARY='4')
        application = [str(options.binary.resolve()), 'run']
        if options.sanitizer:
            application = ['compute-sanitizer', '--tool', options.sanitizer, '--error-exitcode', '1',
                           '--log-file', str(directory/'sanitizer_rank%q{OMPI_COMM_WORLD_RANK}.log'),
                           *(['--leak-check', 'full'] if options.sanitizer == 'memcheck' else []), *application]
        command = [str(options.mpiexec), '--bind-to', 'none', '-n', str(ranks), *application]
        with (directory/'run.log').open('w') as log:
            subprocess.run(command, cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        report[name] = analyze(directory, frozen)
        if not trace:
            target = options.output/('np2_x_plain' if sync == 'explicit' else name.replace('_plain', '_trace'))
            for rank, meta in layouts(directory).items():
                suffix = f'outdat/rhs.cpp_final.step00000004.rk01.rank{rank:08d}.bin'
                _, actual = raw_field(directory/suffix)
                _, expected = raw_field(target/suffix)
                bits_equal(physical(actual, meta), physical(expected, meta), name+'/trace-or-sync isolation')
            report[name]['trace_or_sync_physical_bits_equal'] = True
        (options.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(name, 'terminal/stages/halos passed', flush=True)


if __name__ == '__main__':
    main()
