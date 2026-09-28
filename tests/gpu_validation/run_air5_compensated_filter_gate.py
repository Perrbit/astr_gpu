#!/usr/bin/env python3
"""Bounded characteristic-top, compensated-filter backend matrix."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from run_air5_characteristic_backend_gate import execute, compare
from check_air5_compensation_checkpoint import read


def run(executable, output):
    executable = executable.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', production_ready=False,
                  executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
                  cases={}, represented_checkpoints={})
    def save():
        (output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    save()
    try:
        reference = None
        for topology in ('1,1,1', '2,1,1', '1,2,1', '1,1,2'):
            for workspace in ('full', 'scalar'):
                for gpu in (False, True):
                    name = topology.replace(',', 'x')+'_'+workspace+('_gpu' if gpu else '_cpu')
                    print(name, flush=True)
                    data = execute(output/name, executable, gpu, topology,
                                   source_mode='coupled', viscous=True, dt=1e-10,
                                   diffusion_limiter='layered', convection_limiter='symmetric_species',
                                   filtering=True, compensation=True, filter_workspace=workspace)
                    high, carry, phase, closure = read(output/name/'outdat/flowfield.h5')
                    represented = high.astype(np.longdouble)-carry.astype(np.longdouble)
                    residual = np.max(abs(represented[...,5:10].sum(axis=-1)-represented[...,0]) /
                                      represented[...,0])
                    if (represented[...,0].min() <= 0 or represented[...,5:10].min() < 0 or
                            residual > 128*np.finfo(float).eps or not np.count_nonzero(carry)):
                        raise ValueError('invalid or unexercised compensated checkpoint: '+name)
                    report['represented_checkpoints'][name] = dict(
                        phase=phase, high_closure=closure, represented_closure=float(residual),
                        nonzero_carry=int(np.count_nonzero(carry)))
                    if reference is None:
                        reference = data
                    report['cases'][name] = compare(reference, data)
                    save()
    except BaseException as error:
        report.update(status='failed', error=str(error))
        save()
        raise
    report['status'] = 'short-matrix-pass-not-production-admission'
    save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.executable, args.output)
