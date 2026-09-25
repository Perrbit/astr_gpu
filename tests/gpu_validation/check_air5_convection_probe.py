#!/usr/bin/env python3
"""Validate read-only convection operands against ASTR's recorded RK increments."""
import argparse
import json
from pathlib import Path

import numpy as np

from check_air5_inlet_velocity_budget import UPDATE_WEIGHTS, step_budget
from check_air5_mach4_error_replay import cartesian_jacobian


PREFIX = 'AIR5_CONVECTION_PROBE '
SPECIES = ('N2', 'O2', 'N', 'O', 'NO')


def read_probe(log):
    records, current = [], None
    for line in log.read_text().splitlines():
        if not line.startswith(PREFIX):
            continue
        key, value = line[len(PREFIX):].split('=', 1)
        if key == 'step/stage/rank/i/j/k':
            indices = list(map(int, value.split()))
            if len(indices) != 6:
                raise ValueError('invalid probe location')
            current = dict(zip(('step', 'stage', 'rank', 'i', 'j', 'k'), indices))
            current.update(faces=[], shared=[])
            records.append(current)
        elif current is None:
            raise ValueError('probe payload precedes location')
        elif key == 'face axis/side':
            axis, side = map(int, value.split())
            current['faces'].append(dict(axis=axis, side=side))
        elif key == 'correction_SI':
            if not current['faces'] or 'correction' in current['faces'][-1]:
                raise ValueError('duplicate or unpaired face correction')
            current['faces'][-1]['correction'] = list(map(float, value.split()))
        elif key == 'shared step/stage/axis/theta':
            fields = value.split()
            if len(fields) != 5 or list(map(int, fields[:2])) != [current['step'], current['stage']]:
                raise ValueError('shared face phase mismatch')
            current['shared'].append(dict(axis=int(fields[2]), theta=list(map(float, fields[3:]))))
        elif key in ('base_SI', 'negative_budget_SI', 'species_ratio',
                     'roundoff_scale_SI', 'face_other_and_actual_ratio'):
            if key in current:
                raise ValueError('duplicate probe payload')
            current[key] = list(map(float, value.split()))
        else:
            raise ValueError(f'unknown probe record: {key}')
    if not records:
        raise ValueError('no convection probe records')
    phases = [(r['step'], r['stage']) for r in records]
    if len(phases) != len(set(phases)):
        raise ValueError('duplicate probe phase')
    for record in records:
        validate_record(record)
    return records


def validate_record(record):
    if len(record['faces']) != 6 or {(f['axis'], f['side']) for f in record['faces']} != {
            (a, s) for a in range(1, 4) for s in (1, 2)}:
        raise ValueError('probe requires six distinct faces')
    if len(record['shared']) != 3 or {f['axis'] for f in record['shared']} != {1, 2, 3}:
        raise ValueError('probe requires three shared face pairs')
    lengths = dict(base_SI=11, negative_budget_SI=6, species_ratio=5,
                   roundoff_scale_SI=5, face_other_and_actual_ratio=2)
    for key, length in lengths.items():
        data = np.asarray(record[key])
        if data.shape != (length,) or not np.isfinite(data).all():
            raise ValueError(f'invalid {key}')
    corrections = np.asarray([f['correction'] for f in record['faces']])
    if corrections.shape != (6, 11) or not np.isfinite(corrections).all():
        raise ValueError('invalid face corrections')
    negative = np.minimum(corrections[:, 5:10], 0).sum(axis=0)
    if not np.allclose(negative, record['negative_budget_SI'][:5], rtol=1e-12, atol=1e-280):
        raise ValueError('inconsistent species negative budget')
    ratios = np.array(record['species_ratio']+record['face_other_and_actual_ratio'])
    if np.any((ratios < 0) | (ratios > 1)) or not np.isclose(
            ratios[-1], min(ratios[:-1]), rtol=1e-12, atol=1e-280):
        raise ValueError('actual ratio does not match recorded constraints')
    for face in record['shared']:
        theta = np.asarray(face['theta'])
        if theta.shape != (2,) or not np.isfinite(theta).all() or np.any(
                (theta < 0) | (theta > ratios[-1])):
            raise ValueError('invalid shared coefficient')


def correction_by_axis(record):
    """Applied limiter correction relative to raw high-order flux, not total RHS."""
    theta = {f['axis']: f['theta'] for f in record['shared']}
    values = np.zeros((3, 11))
    for face in record['faces']:
        values[face['axis']-1] += (theta[face['axis']][face['side']-1]-1)*np.array(face['correction'])
    return values


def run(directory):
    contract = json.loads((directory/'contract.json').read_text())
    result = json.loads((directory/'result.json').read_text())
    if result['status'] != 'bounded-replay-completed-not-physical-pass' or contract['backend'] != 'cpu':
        raise ValueError('probe requires a completed CPU replay')
    if contract['convection_limiter'] != 'species_budget':
        raise ValueError('this attribution requires the species-budget candidate')
    case = directory/'cpu'
    records = read_probe(case/'run.log')
    steps = range(contract['start_step'], contract['start_step']+contract['updates'])
    if [(r['step'], r['stage']) for r in records] != [(s, k) for s in steps for k in (1, 2, 3)]:
        raise ValueError('incomplete or reordered probe phases')
    node = tuple(records[0][key] for key in ('i', 'j', 'k'))
    rank = records[0]['rank']
    jacobian = cartesian_jacobian(case)
    for step in steps:
        selected = [r for r in records if r['step'] == step]
        if any((r['rank'], r['i'], r['j'], r['k']) != (rank, *node) for r in selected):
            raise ValueError('probe node changed')
        actual = sum(w*correction_by_axis(r).sum(axis=0)[:2]
                     for w, r in zip(UPDATE_WEIGHTS, selected))
        _, _, terms, _ = step_budget(case, step, contract['dt'], rank, node, jacobian)
        expected = terms['convection_limiter']
        scale = sum(w*abs(correction_by_axis(r)).sum(axis=0)[:2]
                    for w, r in zip(UPDATE_WEIGHTS, selected))
        if np.any(abs(actual-expected) > 1e-10*scale+1e-15):
            raise ValueError('probe limiter increments disagree with recorded RHS')
    for record in records:
        base = np.array(record['base_SI'])
        corrections = np.array([f['correction'] for f in record['faces']])
        record['raw_trial_species_SI'] = (base+corrections.sum(axis=0))[5:10].tolist()
        record['limiter_rho_momentum_increment_by_axis'] = correction_by_axis(record)[:, :2].tolist()
        limiting = int(np.argmin(record['species_ratio']))
        record['minimum_species_constraint'] = (
            SPECIES[limiting] if record['species_ratio'][limiting] < 1 else None)
    return dict(status='operand-and-RHS-identity-pass-not-physical-pass',
                contract=contract, records=records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report = run(args.case)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    for record in report['records']:
        print(json.dumps({key: record[key] for key in ('step', 'stage', 'species_ratio',
            'minimum_species_constraint', 'face_other_and_actual_ratio')}))
