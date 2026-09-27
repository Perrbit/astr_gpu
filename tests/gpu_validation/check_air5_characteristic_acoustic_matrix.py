#!/usr/bin/env python3
"""Recompute acoustic reports and compare controls without running the solver."""
import argparse
import json
import math
from pathlib import Path

from run_air5_characteristic_acoustic import analyze

CASES = ('baseline', 'half_tau', 'double_tau', 'extended', 'coarse', 'half_dt', 'prescribed')


def compare(entries, *, partial=False):
    names = CASES[:len(entries)] if partial else CASES
    if not entries or set(entries) != set(names):
        raise ValueError('the complete seven-case matrix is required')
    baseline, result = entries['baseline']
    differences = {}
    for name in names:
        meta, report = entries[name]
        if meta.get('use_gpu', False) != baseline.get('use_gpu', False):
            raise ValueError(f'{name}: mixed backends require a separate equivalence bridge')
        for key in ('executable_sha256', 'rho', 'pressure', 'sound_speed', 'gamma',
                    'epsilon', 'probe_xyz', 'sampling_phase', 'incident_center',
                    'reflected_center', 'window_halfwidth'):
            if meta[key] != baseline[key]:
                raise ValueError(f'{name}: mismatched {key}')
        expected_ny = 128 if name == 'coarse' else 256
        expected_dt = 2.5e-9 if name == 'half_dt' else 5e-9
        factor = {'half_tau': .5, 'double_tau': 2.}.get(name, 1.)
        mode = 'prescribed' if name == 'prescribed' else 'characteristic'
        extended = name == 'extended'
        expected_grid = [16, expected_ny*(2 if extended else 1), 8]
        expected_domain = [.08, .02 if extended else .01, .002]
        if (meta['ny'] != expected_ny or meta['dt'] != expected_dt or
            meta['updates'] != round(7.5e-6/expected_dt) or
            meta['tau_factor'] != factor or meta['mode'] != mode or
            meta['extended'] != extended or meta['grid_intervals'] != expected_grid or
            meta['domain'] != expected_domain):
            raise ValueError(f'{name}: outside approved matrix')
        if not math.isclose(meta['tau'], factor*.01/meta['sound_speed'], rel_tol=1e-14):
            raise ValueError(f'{name}: wrong explicit relaxation time')
        env = meta['environment']
        if (env['ASTR_AIR5_SOURCE_MODE'] != 'frozen' or env['ASTR_AIR5_COMPENSATION'] != 'on' or
            env['ASTR_AIR5_TOP_MODE'] != mode or float(env['ASTR_AIR5_TOP_TAU']) != meta['tau']):
            raise ValueError(f'{name}: source or boundary control mismatch')
        for key in ('R_peak', 'R_L2', 'incident_peak', 'incident_L2'):
            if not math.isfinite(report[key]) or report[key] < 0:
                raise ValueError(f'{name}: invalid {key}')
        if report['incident_peak'] == 0 or report['incident_L2'] == 0:
            raise ValueError(f'{name}: missing incident wave')
        closure = report['maximum_sequential_closure']
        if not math.isfinite(closure) or closure < 0 or closure > 128*math.ulp(1.):
            raise ValueError(f'{name}: composition closure')
        if name not in ('extended', 'prescribed'):
            threshold = .10 if name in ('half_tau', 'double_tau') else .05
            if max(report['R_peak'], report['R_L2']) > threshold:
                raise ValueError(f'{name}: reflection threshold')
        difference = abs(report['R_L2']-result['R_L2'])
        bound = {'half_tau': .03, 'double_tau': .03, 'coarse': .02, 'half_dt': .005}.get(name)
        if bound is not None:
            differences[name] = difference
            if difference > bound:
                raise ValueError(f'{name}: R_L2 difference {difference} exceeds {bound}')
    incident_errors = {}
    if 'extended' in entries:
        reference = entries['extended'][1]
        incident_errors = {key: abs(result[key]/reference[key]-1)
                           for key in ('incident_peak', 'incident_L2')}
    if incident_errors and max(incident_errors.values()) > .05:
        raise ValueError(f'extended-domain incident mismatch: {incident_errors}')
    return dict(passed=True, complete=len(entries) == len(CASES), checked_cases=list(names),
                R_L2_differences=differences, incident_relative_errors=incident_errors,
                characteristic_R_L2=result['R_L2'],
                prescribed_R_L2=entries['prescribed'][1]['R_L2'] if 'prescribed' in entries else None,
                scope='frozen acoustic matrix only; reacting, oblique, GPU, MPI and restart gates remain separate')


def load_case(path):
    log = (path/'run.log').read_text()
    if 'The job is done!' not in log or 'ieee_invalid' in log.lower():
        raise ValueError(f'case has not completed cleanly: {path}')
    # Re-read raw states/monitors rather than trusting an old result.json pass flag.
    return json.loads((path/'gate.json').read_text()), analyze(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in CASES:
        parser.add_argument('--'+name.replace('_', '-'), type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare({name: load_case(getattr(args, name)) for name in CASES}), indent=2))
