#!/usr/bin/env python3
"""Bridge the completed CPU acoustic baseline, then run the GPU-only matrix."""
import argparse
import json
from pathlib import Path

import numpy as np

from check_air5_characteristic_acoustic_matrix import load_case
from run_air5_characteristic_acoustic import prepare, run, load_monitor
from run_air5_characteristic_backend_gate import fields, compare, TOLERANCE
from run_air5_characteristic_acoustic_sequence import sequence


def advance(output, executable, cpu):
    output.mkdir(parents=True, exist_ok=False)
    old_meta, old_report = load_case(cpu)
    baseline = output/'baseline'
    meta = prepare(baseline, executable, use_gpu=True)
    run(baseline, meta)
    comparisons = compare(fields(cpu, (0, 1499)), fields(baseline, (0, 1499)))
    maximum = 0.
    for index in (1, 2, 3):
        a = load_monitor(cpu, old_meta, index)[:, :6]
        b = load_monitor(baseline, meta, index)[:, :6]
        maximum = max(maximum, float((abs(a-b)/np.maximum(1.,abs(a))).max()))
    if maximum > TOLERANCE:
        raise ValueError(f'full-window CPU/GPU probe mismatch: {maximum}')
    report = dict(passed=True, comparison=comparisons, monitor_max_scaled=maximum,
                  cpu_executable_sha256=old_meta['executable_sha256'],
                  gpu_executable_sha256=meta['executable_sha256'],
                  cpu_R_L2=old_report['R_L2'])
    (output/'bridge.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'CPU/GPU bridge passed, monitor max scaled={maximum}', flush=True)
    half = output/'half_tau'
    run(half, prepare(half, executable, tau_factor=.5, use_gpu=True))
    return sequence(baseline, half, output/'controls')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--cpu-reference', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(advance(args.output.resolve(), args.executable.resolve(),
                             args.cpu_reference.resolve()), indent=2))
