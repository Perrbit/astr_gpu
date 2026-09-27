#!/usr/bin/env python3
"""Short frozen pulse at the top: matched RK fields and MPI decomposition gate."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from air5_radau_reference import Air5RadauReference
from check_air5_c5_hbl import _assemble_global, _load_parallel_layout
from check_air5_mach4_error_replay import state_gate
from check_air5_mass_closure_stages import metrics
from run_air5_characteristic_acoustic import prepare, load_monitor
from run_air5_sbli_preflight import ROOT, environment, set_value

TOLERANCE = 2e-10
PHASES = [('post_chemistry', 1), ('pre_rhs', 1), ('post_update', 1),
          ('pre_rhs', 2), ('post_update', 2), ('pre_rhs', 3), ('post_update', 3),
          ('post_transport', 1), ('post_chemistry', 2)]


def fields(case, steps=(0, 2)):
    layouts = _load_parallel_layout(case/'datin/parallel.info')
    return {f'{step}:{label}:{stage}': _assemble_global(
        case/'validation/air5', label, stage, layouts, step)
        for step in steps for label, stage in PHASES}


def compare(reference, candidate):
    if not reference or reference.keys() != candidate.keys():
        raise ValueError('phase set mismatch')
    result = {}
    for name in reference:
        a, b = reference[name], candidate[name]
        if a.shape != b.shape:
            raise ValueError(f'{name}: shape mismatch')
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            raise ValueError(f'{name}: nonfinite state')
        scaled = abs(a-b)/np.maximum(1., abs(a))
        result[name] = dict(max_scaled=float(scaled.max()),
                           component_max_abs=abs(a-b).max(axis=(0, 1, 2)).tolist(),
                           top_max_scaled=float(scaled[:, -1].max()))
        if scaled.max() > TOLERANCE:
            raise ValueError(f'{name}: phase mismatch {result[name]}')
    return result


def execute(case, executable, gpu, topology, sanitizer=False):
    meta = prepare(case, executable, ny=128, nz=16, pulse_y=.0096, short_updates=3, use_gpu=gpu)
    set_value(case/'datin/input.air5_c4',
              'nondimen,diffterm,lfilter,lreadgrid,lfftz,limmbou,ltimrpt,lcomb',
              'f,f,f,f,f,f,t,t,'+('t' if gpu else 'f'))
    env = environment(topology)
    env.update(meta['environment'], ASTR_AIR5_TOP_GPU_VALIDATION='on')
    command = [shutil.which('mpirun'), '--oversubscribe', '-np',
               str(int(np.prod([int(x) for x in topology.split(',')])))]
    if sanitizer:
        command += [shutil.which('compute-sanitizer'), '--tool', 'memcheck',
                    '--error-exitcode', '99', '--log-file', 'validation/memcheck.%p.log']
    command += [str(executable), 'run', 'datin/input.air5_c4']
    meta.update(topology=topology, use_gpu=gpu, gate='short backend equivalence, not acoustic acceptance',
                tolerance=TOLERANCE, environment={k:v for k,v in env.items() if k.startswith('ASTR_')})
    (case/'gate.json').write_text(json.dumps(meta, indent=2)+'\n')
    with (case/'run.log').open('w') as log:
        subprocess.run(['timeout', '--kill-after=10s', '900s', *command], cwd=case, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    log = (case/'run.log').read_text()
    if 'The job is done!' not in log or 'ieee_invalid' in log.lower():
        raise ValueError('solver did not complete cleanly')
    if sanitizer:
        logs = list((case/'validation').glob('memcheck.*.log'))
        if not logs or any('ERROR SUMMARY: 0 errors' not in p.read_text() for p in logs):
            raise ValueError('memcheck failed')
    state_gate(case, Air5RadauReference(ROOT/'chemMech/air5_kimjo12.json'))
    data = fields(case)
    for index in (1, 2, 3):
        load_monitor(case, meta, index)
    for key, array in data.items():
        row = metrics(array, (0, 0, 0))
        if row['transport_sum_violations'] or row['min_species_density'] < 0:
            raise ValueError(f'{key}: strict composition failure')
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--topology', default='1,1,1')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--sanitizer', action='store_true')
    parser.add_argument('--executable', type=Path, default=ROOT/'build_gpu_probe/bin/astr')
    args = parser.parse_args()
    data = execute(args.output.resolve(), args.executable.resolve(), args.gpu, args.topology, args.sanitizer)
    report = dict(passed=True, scope='frozen inviscid short pulse; spanwise homogeneous')
    if args.reference:
        report['comparison'] = compare(fields(args.reference), data)
        old = json.loads((args.reference/'gate.json').read_text())
        new = json.loads((args.output/'gate.json').read_text())
        monitor_error = 0.
        for index in (1, 2, 3):
            a = load_monitor(args.reference, old, index)[:, :6]
            b = load_monitor(args.output, new, index)[:, :6]
            monitor_error = max(monitor_error, float((abs(a-b)/np.maximum(1.,abs(a))).max()))
        report['monitor_max_scaled'] = monitor_error
        if monitor_error > TOLERANCE:
            raise ValueError(f'monitor phase mismatch: {monitor_error}')
    (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))
