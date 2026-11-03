#!/usr/bin/env python3
"""Compare saved same-topology CPU/GPU station evidence; never advances CFD."""
import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

TOPOLOGIES = ((1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2), (2, 2, 1), (2, 1, 2), (1, 2, 2))


def read(path):
    lines = [line for line in path.read_text().splitlines() if not line.startswith('#')]
    return np.genfromtxt(lines, delimiter=',', names=True)


def thickness_metadata(path):
    result = {}
    for line in path.read_text().splitlines():
        if not line.startswith('# thickness_') or '=' not in line:
            continue
        key, value = line[12:].split('=', 1)
        try:
            result[key] = float(value)
        except ValueError:
            result[key] = value
    return result


def restart_logs(records):
    logs = []
    for path in records:
        for case in ET.parse(path).iter('testcase'):
            if case.find('failure') is not None or case.find('error') is not None:
                raise ValueError(f'failed verification record: {path}')
            if case.find('skipped') is not None:
                continue
            properties = {p.attrib['name']: p.attrib['value'] for p in case.iter('property')}
            if 'case_directory' not in properties:
                continue
            backend = re.search(r'-(cpu|gpu)\]$', case.attrib['name'])
            if backend is None:
                raise ValueError(f'missing parameterized backend identity: {case.attrib["name"]}')
            logs.append((Path(properties['case_directory'])/'run.log', backend[1]))
    return logs


def analyse(root, family='station', restart_records=()):
    cases, visited = {}, set()
    patterns = (('test_completed_wall_means*/*/run.log','test_real_wall_references*/*/run.log') if family == 'wall' else
                ('test_real_station_temporal_pro*/*/run.log', 'test_flatplate_station_profile*/*/run.log',
                 'test_real_wall_unit_profiles*/*/run.log', 'test_read_grid_wall_unit_profi*/*/run.log',
                 'test_real_thickness_profiles*/*/run.log'))
    product_directory = 'wall_profile_products' if family == 'wall' else 'profile_products'
    logs = restart_logs(restart_records) if restart_records else [(log,None) for pattern in patterns for log in root.glob(pattern)]
    for log, declared_backend in sorted(logs):
        case = log.parent.resolve()
        if case in visited:
            continue
        visited.add(case)
        if not (case / 'outdat/new' / product_directory).is_dir():
            continue
        backend = declared_backend or case.name.split('_')[0]
        text = log.read_text()
        topology = re.search(r'forced mpi topology=\s*(\d+)\s+(\d+)\s+(\d+)', text)
        if not topology or 'The job is done!' not in text:
            raise ValueError(f'missing completed solver identity: {case}')
        identity = backend, tuple(map(int, topology.groups()))
        if identity in cases:
            raise ValueError(f'duplicate identity: {identity}')
        cases[identity] = case
    required = {(b, t) for b in ('cpu', 'gpu') for t in TOPOLOGIES}
    if set(cases) != required:
        raise ValueError(f'case matrix mismatch: {set(cases)^required}')
    comparisons = []
    for topology in TOPOLOGIES:
        cpu, gpu = cases['cpu', topology], cases['gpu', topology]
        a = {p.parent.parent.parent.name+'/'+p.name: p for p in
             (cpu / 'outdat/new' / product_directory).glob('*/*/segment*/frame.step*.csv')}
        b = {p.parent.parent.parent.name+'/'+p.name: p for p in
             (gpu / 'outdat/new' / product_directory).glob('*/*/segment*/frame.step*.csv')}
        if not a or a.keys() != b.keys():
            raise ValueError(f'frame mismatch: {topology}')
        worst = absolute = rms_raw = 0.
        for key, path in a.items():
            x, y = read(path), read(b[key])
            meta_x, meta_y = thickness_metadata(path), thickness_metadata(b[key])
            if meta_x.keys() != meta_y.keys():
                raise ValueError(f'thickness availability/definition mismatch: {key}')
            for name, value in meta_x.items():
                other = meta_y[name]
                if isinstance(value, str):
                    if value != other:
                        raise ValueError(f'thickness definition mismatch: {key}/{name}')
                else:
                    if not np.isfinite([value, other]).all():
                        raise ValueError(f'nonfinite thickness diagnostic: {key}/{name}')
                    absolute = max(absolute, abs(value-other))
                    worst = max(worst, abs(value-other)/max(1., abs(value), abs(other)))
            if x.shape != y.shape or x.dtype.names != y.dtype.names:
                raise ValueError(f'column/shape mismatch: {key}')
            for field in x.dtype.names:
                u, v = x[field], y[field]
                if field in ('y_first_plus', 'dx_minus_plus', 'dx_plus_plus', 'dz_min_plus', 'dz_max_plus') or \
                        (field.endswith('_plus') and 'wall_scale_available' in x.dtype.names):
                    np.testing.assert_array_equal(np.isnan(u), np.isnan(v), err_msg=key+'/'+field)
                    missing_scale = x['wall_scale_available'] == 0
                    if not np.isnan(u[missing_scale]).all() or not np.isnan(v[missing_scale]).all():
                        raise ValueError(f'grid units require an available wall scale: {key}/{field}')
                    selected = ~np.isnan(u)
                    u, v = u[selected], v[selected]
                    if u.size == 0:
                        continue
                availability = {'u_tau': 'wall_scale_available', 'nu_wall': 'wall_scale_available',
                                'viscous_length': 'wall_scale_available', 'Cf': 'cf_available',
                                'Cq_inf': 'cq_available', 'St_stat': 'stanton_available',
                                'cumulative_displacement': 'thickness_integrals_available',
                                'cumulative_momentum': 'thickness_integrals_available'}.get(field)
                if availability and availability in x.dtype.names:
                    np.testing.assert_array_equal(x[availability], y[availability], err_msg=key+'/'+availability)
                    selected = x[availability] == 1
                    if not np.isnan(u[~selected]).all() or not np.isnan(v[~selected]).all():
                        raise ValueError(f'unavailable reference must have an empty CSV cell: {key}/{field}')
                    u, v = u[selected], v[selected]
                    if u.size == 0:
                        continue
                if not np.isfinite(u).all() or not np.isfinite(v).all():
                    raise ValueError(f'nonfinite values: {key}/{field}')
                if field.startswith('rms_velocity_'):
                    rms_raw = max(rms_raw, float(np.max(np.abs(u-v))))
                    u, v = u*u, v*v
                error = np.abs(u-v)
                absolute = max(absolute, float(error.max()))
                worst = max(worst, float(np.max(error/np.maximum(1., np.maximum(np.abs(u), np.abs(v))))))
        if worst > 2e-10:
            raise ValueError(f'same-topology profile gate failed: {topology}: {worst}')
        entry = dict(topology=topology, cpu=str(cpu), gpu=str(gpu), frames=len(a),
                     max_scaled_error=worst, max_absolute_error=absolute)
        if family == 'station':
            entry['rms_raw_maxabs'] = rms_raw
        comparisons.append(entry)
    return dict(scope='same-topology-'+family+'-profile-trajectories', comparisons=comparisons)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--family', choices=('station', 'wall'), default='station')
    parser.add_argument('--restart-records', type=Path, nargs='+', default=[],
                        help='passed JUnit records identifying the continuous reference cases in restart tests')
    args = parser.parse_args()
    result = analyse(args.root, args.family, args.restart_records)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    keys = ['max_scaled_error', 'max_absolute_error']
    if args.family == 'station':
        keys.append('rms_raw_maxabs')
    print(json.dumps({key: max(c[key] for c in result['comparisons']) for key in keys}))
