#!/usr/bin/env python3
"""G2 same-phase physical-node error check, not an independent CFD reference."""
import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path

import h5py
import numpy as np

from air5_radau_reference import Air5RadauReference
from check_air5_actual_error import error_values, NAMES
from run_air5_performance_diagnosis import digest

FLOW_NAMES = (*NAMES, 'rho_kg_per_m3', 'u_m_per_s', 'v_m_per_s', 'w_m_per_s', 'p_Pa', 'q5_J_per_m3')


def flow_scales(initial_ev, freestream):
    values = np.array([initial_ev, *(freestream[name] for name in ('rho', 'u', 'p', 'q5'))])
    if not np.isfinite(values).all() or initial_ev < 0 or freestream['rho'] <= 0 or freestream['p'] <= 0:
        raise ValueError('invalid frozen flow error scales')
    velocity_scale = 1e-10 * max(abs(freestream['u']), 1.)
    absolute = np.array([1e-12]*5 + [1e-10*max(abs(initial_ev), 1.), 1e-5, 1e-5,
        1e-12*freestream['rho'], *([velocity_scale]*3), 1e-12*freestream['p'],
        1e-12*max(abs(freestream['q5']), 1.)])
    relative = np.array([1e-6]*5 + [1e-7]*9)
    return absolute, relative


def observations(model, q):
    q = np.asarray(q, dtype=float)
    if q.shape != (11,) or not np.isfinite(q).all() or q[0] <= 0 or np.any(q[5:11] < 0):
        raise ValueError('physical q requires positive density and finite nonnegative species/ev')
    fractions = q[5:10] / q[0]
    if abs(math.fsum(map(float, fractions)) - 1.) > 128*np.finfo(float).eps:
        raise ValueError('physical q violates species closure')
    temperature = model.temperature_from_q5(q[0], q[1:4], q[5:10], q[10], q[4])
    tv = model.tv_from_ev(q[5:10], q[10])
    pressure = model.pressure(q[5:10], temperature)
    return np.r_[fractions, q[10]/q[0], temperature, tv, q[0], q[1:4]/q[0], pressure, q[4]]


def compare_node(model, actual, actual_carry, target, target_carry, initial_ev, freestream):
    actual, actual_carry, target, target_carry = [np.asarray(row, dtype=float)
        for row in (actual, actual_carry, target, target_carry)]
    if any(row.shape != (11,) or not np.isfinite(row).all()
           for row in (actual, actual_carry, target, target_carry)):
        raise ValueError('flow comparison requires eleven finite q/carry components')
    absolute, relative = flow_scales(initial_ev, freestream)
    result = {}
    for name, carry, ref_carry in (('q', np.zeros(11), np.zeros(11)),
                                    ('q-minus-carry', actual_carry, target_carry)):
        represented, reference = actual-carry, target-ref_carry
        observed, expected = observations(model, represented), observations(model, reference)
        row = error_values(observed, expected, absolute, relative)
        # Resolve differences before rounding q-carry or dividing by density.
        delta = np.array([math.fsum((float(h), -float(c), -float(t), float(r)))
                          for h, c, t, r in zip(actual, carry, target, ref_carry)])
        difference = np.asarray(row['absolute_error'])
        difference[8], difference[13] = abs(delta[0]), abs(delta[4])
        for column, component in (*zip(range(5), range(5, 10)), (5, 10), (9, 1), (10, 2), (11, 3)):
            difference[column] = abs(math.fsum((float(delta[component]),
                -float(expected[column])*float(delta[0])))) / represented[0]
        row.update(absolute_error=difference.tolist(),
            relative_error=[float(d/abs(t)) if t != 0 else None for d, t in zip(difference, expected)],
            normalized_error=(difference / (absolute+relative*abs(expected))).tolist())
        result[name] = row
    return result


def empty_maxima():
    return {name: dict(max_absolute_error=[0.]*14, max_normalized_error=[0.]*14,
        max_relative_error=[None]*14, undefined_relative_nodes=[0]*14,
        max_error_node_zyx=[None]*14) for name in ('q', 'q-minus-carry')}


def accumulate(maxima, result, node):
    for name, row in result.items():
        output = maxima[name]
        for component, value in enumerate(row['normalized_error']):
            output['max_absolute_error'][component] = max(output['max_absolute_error'][component],
                                                         row['absolute_error'][component])
            if output['max_error_node_zyx'][component] is None or value > output['max_normalized_error'][component]:
                output['max_normalized_error'][component] = value
                output['max_error_node_zyx'][component] = list(node)
            relative = row['relative_error'][component]
            if relative is None:
                output['undefined_relative_nodes'][component] += 1
            else:
                previous = output['max_relative_error'][component]
                output['max_relative_error'][component] = relative if previous is None else max(previous, relative)


def compare_checkpoints(model, initial, reference, preceding, candidate, freestream):
    paths = dict(initial=initial, reference=reference, preceding=preceding, candidate=candidate)
    with ExitStack() as stack:
        files = {name: stack.enter_context(h5py.File(path)) for name, path in paths.items()}
        headers = {name: file['identity'][:] for name, file in files.items()}
        for name, header in headers.items():
            clock = header[9:12].view(np.float64)
            if (header.shape != (13,) or header[0] != int('4153545251533031', 16) or
                    header[1] not in (2, 3) or header[2] != 1 or header[12] != 7 or header[6] not in (34, 35) or
                    not np.isfinite(clock).all() or np.any(clock < 0) or clock[2] <= 0 or
                    (header[8] > 0 and clock[1] <= 0) or
                    files[name]['metadata'][:].tolist() != [1, 1]):
                raise ValueError(f'{name}: require a completed-step compensated AIR5 state')
            if header[1] == 3 and files[name]['numerical_kind_bytes'][:].tolist() != [8]:
                raise ValueError(f'{name}: G2 reference comparison requires FP64 state')
        target_header = headers['reference']
        same_phase = [0, 2, 3, 4, 5, 8, 9, 10, 11, 12]
        for name in ('preceding', 'candidate'):
            if not np.array_equal(headers[name][same_phase], target_header[same_phase]):
                raise ValueError(f'{name}: full-flow phase/grid/dt mismatch')
        if (not np.array_equal(headers['initial'][3:6], target_header[3:6]) or
                headers['initial'][8] >= target_header[8] or
                headers['initial'][9:10].view(np.float64)[0] >= target_header[9:10].view(np.float64)[0]):
            raise ValueError('initial: require an earlier same-grid state for frozen ev scales')
        shape = tuple(map(int, target_header[3:6][::-1]))
        for name, file in files.items():
            for component in range(1, 23):
                if file[f'q{component:04}'].shape != shape:
                    raise ValueError(f'{name}: physical dataset shape differs from state grid')
        report = dict(scope='same-phase physical q/q-carry; not cache equivalence, exact restart or independent CFD',
            components=list(FLOW_NAMES), grid_intervals=(target_header[3:6]-1).tolist(),
            completed_step=int(target_header[8]), time=float(target_header[9:10].view(np.float64)[0]),
            physical_nodes=math.prod(shape), freestream=freestream,
            reference_pair=empty_maxima(), candidate=empty_maxima())
        # Read global physical planes, never rank_extras/ghosts or whole 3D fields.
        for z in range(shape[0]):
            slab = {name: np.stack([file[f'q{m:04}'][z, :, :] for m in range(1, 23)], axis=-1)
                    for name, file in files.items()}
            for y, x in np.ndindex(shape[1:]):
                nodes = {name: values[y, x] for name, values in slab.items()}
                origin = nodes['initial'][:11]
                observations(model, origin)
                initial_ev = origin[10] / origin[0]
                ref = nodes['reference']
                for source, label in (('preceding', 'reference_pair'), ('candidate', 'candidate')):
                    point = nodes[source]
                    result = compare_node(model, point[:11], point[11:22], ref[:11], ref[11:22],
                                          initial_ev, freestream)
                    accumulate(report[label], result, (z, y, x))
        maximum = lambda rows: max(max(row['max_normalized_error']) for row in rows.values())
        report['within_reference_margin'] = maximum(report['reference_pair']) <= .01
        report['within_candidate_error'] = maximum(report['candidate']) <= 1.
        return report


def require_flow_error(report):
    if not report['within_reference_margin']:
        raise ValueError('full-flow reference margin exceeds 1% of the approved budget')
    if not report['within_candidate_error']:
        raise ValueError('full-flow candidate error exceeds the approved budget')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('initial', 'reference', 'preceding', 'candidate', 'freestream', 'report'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    mechanism = Path(__file__).resolve().parents[2] / 'chemMech/air5_kimjo12.json'
    model = Air5RadauReference(mechanism)
    report = dict(status='running', checker_sha256=digest(Path(__file__)),
        mechanism_sha256=digest(mechanism),
        thermodynamics_code_sha256=digest(Path(__file__).with_name('air5_radau_reference.py')),
        error_norm_code_sha256=digest(Path(__file__).with_name('check_air5_actual_error.py')),
        files={name: dict(path=str(getattr(args, name).resolve()), sha256=digest(getattr(args, name)))
               for name in ('initial', 'reference', 'preceding', 'candidate', 'freestream')})
    try:
        report.update(compare_checkpoints(model, args.initial, args.reference, args.preceding, args.candidate,
                                          json.loads(args.freestream.read_text())))
        require_flow_error(report)
        report['status'] = 'flow-error-budget-passed'
    except BaseException as exc:
        report.update(status='failed', error=str(exc))
        raise
    finally:
        with args.report.open('x') as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
