"""Nearest coefficient projections in finite density-error chambers, not CFD."""
import argparse
from fractions import Fraction
from itertools import product
import json
from pathlib import Path

import numpy as np

from review_high_order_current_local_filter import (
    density_accuracy_training, parameters7, project_density_accuracy, review)
from review_high_order_p6_filter_compatibility import load_native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, nargs='+', required=True)
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--kind', type=int, choices=(41, 42, 421), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    matrices = {}
    for path in args.native:
        matrices.update(load_native(path))
    design = json.loads(args.design.read_text())
    start = np.array(design['floating_parameters'][:7])
    training = density_accuracy_training(matrices, args.kind)
    projections = []
    # Both fields and stencil are reflection symmetric. Include all seven
    # non-endpoint lower-strip nodes, including unchanged deep rows 5..7.
    for row0, sign0, row1, sign1 in product(range(1, 8), (-1., 1.), range(1, 8), (-1., 1.)):
        result, record = project_density_accuracy(training, start, [(row0, sign0), (row1, sign1)])
        record.update(success=bool(result.success), message=str(result.message),
                      parameters=parameters7(result.x).tolist(),
                      distance=float(np.linalg.norm(result.x-start)))
        projections.append(record)
    trials = []
    winner = None
    for record in sorted(projections, key=lambda item: item['distance']):
        if not record['success'] or record['maximum_scaled_constraint_violation'] > 1e-10:
            continue
        values = np.array(record['parameters'][:7])
        samples, failure = review(matrices, args.kind, values)
        trials.append(dict(projection=record, samples_visited=len(samples), first_failure=failure))
        if failure is None:
            rational = [Fraction(float(x)).limit_denominator(4096) for x in values]
            rounded_samples, rounded_failure = review(matrices, args.kind, np.array([float(x) for x in rational]))
            record['rational_parameters'] = [str(x) for x in parameters7(rational)]
            record['rational_samples_visited'] = len(rounded_samples)
            record['rational_failure'] = rounded_failure
            if rounded_failure is None:
                winner = record
                break
        if len(trials)%20 == 0:
            print('reviewed nearest projections', len(trials), flush=True)
    report = dict(status='finite nearest-projection study only; no production admission',
                  design=str(args.design), bctype=args.kind, projections=projections, trials=trials,
                  winner=winner,
                  caveat='One nearest point per chamber is not an impossibility proof for the full coefficient family')
    with (args.output/'report.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print('projections', len(projections), 'reviewed', len(trials), 'winner', winner is not None, flush=True)


if __name__ == '__main__':
    main()
