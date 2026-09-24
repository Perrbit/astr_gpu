#!/usr/bin/env python3
"""Read ASTR single-node diffusion evidence; never advance or repair a field."""
import argparse
import json
from pathlib import Path

import numpy as np


PREFIX = 'AIR5_DIFFUSION_PROBE '


def read_probe(log):
    records = []
    current = None
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
        elif key == 'face axis/side/mask/species/ratio/thermal':
            fields = value.split()
            if len(fields) != 6:
                raise ValueError('invalid face record')
            current['faces'].append(dict(zip(('axis', 'side', 'mask', 'species_mask'),
                                             map(int, fields[:4])),
                                         ratio=float(fields[4]), thermal=float(fields[5])))
        elif key == 'correction_SI':
            current['faces'][-1]['correction'] = list(map(float, value.split()))
        elif key == 'shared step/stage/axis/theta':
            fields = value.split()
            if len(fields) != 5 or list(map(int, fields[:2])) != [current['step'], current['stage']]:
                raise ValueError('shared face phase mismatch')
            current['shared'].append(dict(axis=int(fields[2]), theta=list(map(float, fields[3:]))))
        elif key == 'raw_trial_admissible':
            if value.strip() not in ('T', 'F'):
                raise ValueError('invalid admissibility flag')
            current[key] = value.strip() == 'T'
        elif key in ('base_SI', 'negative_budget_SI', 'raw_trial_SI', 'ratio budget/actual/relaxed'):
            current[key] = list(map(float, value.split()))
        else:
            raise ValueError(f'unknown probe record: {key}')
    if not records:
        raise ValueError('no diffusion probe records')
    for record in records:
        validate_record(record)
    return records


def validate_record(record):
    if {(f['axis'], f['side']) for f in record['faces']} != {
            (axis, side) for axis in range(1, 4) for side in range(1, 3)} or len(record['faces']) != 6:
        raise ValueError('probe requires six distinct faces')
    if len(record['shared']) != 3 or {f['axis'] for f in record['shared']} != {1, 2, 3}:
        raise ValueError('probe requires three shared face pairs')
    base = np.asarray(record['base_SI'])
    corrections = np.asarray([f['correction'] for f in record['faces']])
    trial = np.asarray(record['raw_trial_SI'])
    negative = np.asarray(record['negative_budget_SI'])
    ratios = np.asarray(record['ratio budget/actual/relaxed'])
    if base.shape != (11,) or trial.shape != (11,) or corrections.shape != (6, 11):
        raise ValueError('invalid conservative vector length')
    if negative.shape != (6,) or ratios.shape != (3,):
        raise ValueError('invalid budget length')
    for array in (base, trial, corrections, negative, ratios):
        if not np.isfinite(array).all():
            raise ValueError('nonfinite probe payload')
    # Opposing face stresses can cancel almost completely; scale rounding by operands.
    rounding_bound = 64*np.finfo(float).eps*(abs(base)+abs(corrections).sum(axis=0))
    if np.any(abs(trial-(base+corrections.sum(axis=0))) > rounding_bound):
        raise ValueError('raw trial does not match recorded face increments')
    if not np.allclose(negative[:5], np.minimum(corrections[:, 5:10], 0).sum(axis=0),
                       rtol=1e-10, atol=1e-280):
        raise ValueError('species negative budget is inconsistent')
    if not np.isclose(ratios[1], min(ratios[0], *(f['ratio'] for f in record['faces'])),
                      rtol=1e-12, atol=1e-280):
        raise ValueError('actual ratio does not match recorded constraints')
    if not np.isclose(ratios[2], min(ratios[0], *(f['thermal'] for f in record['faces'])),
                      rtol=1e-12, atol=1e-280):
        raise ValueError('relaxed diagnostic ratio is inconsistent')
    if 'raw_trial_admissible' not in record:
        raise ValueError('missing admissibility decision')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    records = read_probe(args.log)
    args.report.write_text(json.dumps(dict(status='diagnostic-not-physical-pass',
        log=str(args.log), records=records), indent=2, allow_nan=False)+'\n')
    for record in records:
        print(json.dumps({key: record[key] for key in ('step', 'stage', 'rank', 'i', 'j', 'k',
            'ratio budget/actual/relaxed', 'raw_trial_admissible')}))
