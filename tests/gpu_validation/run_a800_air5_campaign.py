#!/usr/bin/env python3
"""Approved P0-P3 only; fail closed and never advance to P4/P5."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import numpy as np

from check_air5_compensation_checkpoint import compare, read
from check_air5_mass_closure_stages import run as closure
from check_air5_mach4_error_replay import matched_window
from check_air5_sbli_startup import check, compare_sensors
from compare_q_validation_snapshots import compare_snapshot_sets
from run_air5_performance_diagnosis import step_samples


def live_problem(text):
    for value in re.findall(r'current CFL:\s*(\S+)', text):
        if not np.isfinite(float(value)) or float(value) >= 1:
            return 'CFL limit'
    for line in text.splitlines():
        if 'AIR5_CHEMISTRY half=' in line and 'max_Tv=' in line:
            values = line.split('max_Tv=', 1)[1].split()[:5]
            if len(values) != 5:
                return 'invalid chemistry diagnostic'
            values = np.array([float(v) for v in values])
            if not np.isfinite(values).all() or values[0] < 0 or values[1] < 1000:
                return 'chemistry/transport domain gate'
    return None


def verify_mapping(text, ranks=4):
    rows = re.findall(r'MPI rank\s+(\d+)\s+bound to GPU device\s+(\d+)\s+on local rank\s+(\d+)', text)
    if len(rows) != ranks or {int(r) for r, _, _ in rows} != set(range(ranks)):
        raise ValueError('missing or duplicate GPU rank mapping')
    if {int(d) for _, d, _ in rows} != set(range(ranks)):
        raise ValueError('MPI ranks are not using distinct visible GPUs')


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def campaign(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    status = {'status': 'running', 'scope': 'P0-P3 only; not physical acceptance', 'completed': []}
    dump(root/'status.json', status)
    driver = Path(__file__).with_name('run_air5_sbli_domain_replay.py')
    seed = args.seed.resolve()
    expected = 'a4dc04b728d9aaec925c9b05b6b8fff101f170e69d790dbac51a74e4448dd7dc'
    if hashlib.sha256((seed/'outdat/flowfield.h5').read_bytes()).hexdigest() != expected:
        raise ValueError('unexpected seed checkpoint')

    def launch(name, topology, updates, dt=2.5e-10, backend='gpu', baseline=seed,
               interval=1, memcheck=False, timing=False):
        _, _, phase, _ = read(baseline/'outdat/flowfield.h5')
        start = phase[0]
        out = root/name
        command = [sys.executable, str(driver), '--baseline', str(baseline),
            '--output', str(out), '--executable', str(args.executable.resolve()),
            '--backend', backend, '--np', '4', '--topology', topology,
            '--dt', str(dt), '--updates', str(updates), '--checkpoint-interval', str(interval),
            '--snapshot-step', str(start+updates-1), '--snapshot-step-secondary', str(start),
            '--compensation', 'on', '--compensation-restart', 'restore',
            '--convection-limiter', 'symmetric_species', '--diffusion-limiter', 'layered',
            '--primitive-reuse', 'chemistry', '--inherit-visible-devices',
            '--timeout-seconds', '21600']
        if memcheck:
            command.append('--memcheck')
        if timing:
            command.append('--complete-step-timing')
        status['active'] = name
        dump(root/'status.json', status)
        with (root/(name+'.driver.log')).open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while process.poll() is None:
                    logfile = out/backend/'run.log'
                    if logfile.exists():
                        problem = live_problem(logfile.read_text(errors='replace'))
                        if problem:
                            raise ValueError(f'{name}: {problem}')
                    time.sleep(2)
                if process.returncode != 0:
                    raise ValueError(f'{name}: driver exited {process.returncode}')
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
        result = json.loads((out/'result.json').read_text())
        if result['status'] != 'bounded-replay-completed-not-physical-pass':
            raise ValueError(f'{name}: numerical failure')
        case = out/backend
        text = (case/'run.log').read_text()
        if live_problem(text):
            raise ValueError(live_problem(text))
        if backend == 'gpu':
            verify_mapping(text)
        boundary = check(case, tuple(map(int, topology.split(','))))
        if boundary['minimum_temperature'] < 1000:
            raise ValueError('snapshot temperature below transport gate')
        dump(out/'boundary_gate.json', boundary)
        rows = [row for rank in range(4) for row in closure(case, rank, [1,1,1],
                sorted(set([start, start+updates-1])))['entries']]
        if not all(r['saved'] and r['min_species_density'] >= 0 and r['transport_sum_violations'] == 0 for r in rows):
            raise ValueError('strict stage composition gate')
        dump(out/'closure_gate.json', {'passed': True, 'samples': len(rows),
             'max_sum_error': max(abs(r['max_sum_error']) for r in rows)})
        q, carry, _, _ = read(case/'outdat/flowfield.h5')
        if np.any(abs(carry) > 4*np.maximum(abs(np.spacing(abs(q))), np.finfo(float).tiny)):
            raise ValueError('checkpoint carry is not a low part')
        if not (case/'bakup/flowfield.h5').exists():
            raise ValueError('missing previous checkpoint backup')
        read(case/'bakup/flowfield.h5')
        status['completed'].append(name)
        dump(root/'status.json', status)
        return case

    def checkpoint_gate(a, b, name, exact=False):
        result = compare(a/'outdat/flowfield.h5', b/'outdat/flowfield.h5', exact=exact)
        dump(root/(name+'.json'), result)
        if not result['passed']:
            raise ValueError(name)

    try:
        cpu = launch('p0_cpu', '4,1,1', 2, backend='cpu')
        gpu = launch('p0_gpu', '4,1,1', 2, memcheck=True)
        compare_sensors(cpu, gpu)
        comparison = compare_snapshot_sets(cpu/'validation/air5', gpu/'validation/air5',
            ('post_chemistry', 'pre_rhs', 'post_update', 'post_transport'), 1e-9, 1e-10, active_only=True)
        dump(root/'cpu_gpu.json', asdict(comparison))
        if not comparison.passed:
            raise ValueError('CPU/GPU field mismatch')
        continuous = launch('p1_continuous', '4,1,1', 4)
        restart = launch('p1_restart', '4,1,1', 3, baseline=gpu)
        checkpoint_gate(continuous, restart, 'restart_exact', exact=True)
        timings = {}
        for topology in ('4,1,1', '2,2,1', '1,4,1'):
            admission = launch('p1_'+topology.replace(',', 'x'), topology, 2)
            checkpoint_gate(gpu, admission, 'topology_'+topology.replace(',', 'x'))
            case = launch('p2_'+topology.replace(',', 'x'), topology, 20, interval=20, timing=True)
            samples = step_samples((case/'run.log').read_text(), 4, 3000, 20)
            # The final step writes snapshots and is not a timing sample.
            timings[topology] = float(np.median(samples[3:-1]))
        selected = min(timings, key=timings.get)
        dump(root/'topology_timing.json', {'median_seconds': timings, 'selected': selected,
                                         'scope': 'single short screening, not scaling publication'})
        launch('p3_coarse', selected, 400, dt=5e-10, interval=50)
        launch('p3_fine', selected, 800, interval=100)
        matched_window(root, ('p3_coarse','p3_fine'), root/'p3_matched.json')
        status.update(status='P0-P3-completed-review-required', selected_topology=selected,
                      next='P4/P5 not authorized by this job')
    except Exception as error:
        status.update(status='stopped-on-gate', error=str(error))
        dump(root/'status.json', status)
        raise
    dump(root/'status.json', status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--executable', type=Path, required=True)
    campaign(parser.parse_args())
