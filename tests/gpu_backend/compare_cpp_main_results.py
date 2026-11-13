"""Compare downloaded ordinary-main DCU results with native frozen references."""
import argparse
import json
from pathlib import Path
import re

import numpy as np

from run_cpp_tgv_gate import compare, checkpoint, snapshot, logged_steps, cfl_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--observed', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--steps', type=int, required=True)
    options = parser.parse_args()
    report = {'observed': str(options.observed.resolve()), 'reference': str(options.reference.resolve()), 'cases': {}}
    for mode, cpu, gpu in [('fixed', 'cpu_fixed_reference', 'gpu_fixed_reference'),
                            ('automatic', 'cpu_automatic', 'old_gpu_automatic')]:
        directory = options.observed/mode
        if 'ASTR_GPU_CPP rank=0 backend=gpufusion' not in (directory/'run.log').read_text():
            raise AssertionError('missing ordinary GPUfusion main selection')
        q = snapshot(directory/f'rhs.cpp_final.step{options.steps:08d}.rk01.rank00000000.bin')
        rho = q[..., 0]
        pressure = .4*(q[..., 4]-.5*np.sum(q[..., 1:4]**2, axis=-1)/rho)
        if not np.isfinite(q).all() or not (rho > 0).all() or not (pressure > 0).all():
            raise AssertionError('invalid physical terminal state')
        result = {}
        for backend, name in [('cpu', cpu), ('old_gpu', gpu)]:
            reference = options.reference/name
            statistics = np.loadtxt(directory/'flowstate.dat', skiprows=1)
            if statistics.shape != (options.steps-1, 5) or not np.array_equal(statistics[:, 0], np.arange(1, options.steps)):
                raise AssertionError('TGV scalar sampling sequence changed')
            result[backend] = dict(q=compare(q, checkpoint(reference/f'outdat/new/checkpoints/step{options.steps:012d}/state.h5')),
                                   scalars=compare(statistics, np.loadtxt(reference/'flowstate.dat', skiprows=1)),
                                   steps=compare(logged_steps(directory, options.steps), logged_steps(reference, options.steps)),
                                   cfl=compare(cfl_log(directory, options.steps), cfl_log(reference, options.steps)))
        report['cases'][mode] = result
        print(mode, result, flush=True)
    diagnostic = list(options.observed.glob('main128-diagnostics-*.log'))
    if len(diagnostic) != 1:
        raise AssertionError('one diagnostic component log required')
    text = diagnostic[0].read_text()
    rows = re.findall(r'^FLUID rank=0 debug=[01]\s+.*?\s+([0-9.+\-Ee]+)$', text, re.MULTILINE)
    if len(rows) != 80 or any(not 0 <= float(value) <= 2e-10 for value in rows):
        raise AssertionError('missing or failed diagnostic component comparisons')
    report['diagnostic_max_scaled'] = max(map(float, rows))
    (options.observed/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
