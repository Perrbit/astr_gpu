#!/usr/bin/env python3
"""Run the approved acoustic controls in order, stopping at any failed gate."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from check_air5_characteristic_acoustic_matrix import CASES, compare, load_case
from run_air5_characteristic_acoustic import prepare, run


def wait_existing_service(unit):
    deadline = time.monotonic() + 14500
    while True:
        result = subprocess.run(['systemctl', '--user', 'show', unit,
                                 '-p', 'ActiveState', '-p', 'ExecMainStatus'],
                                check=True, capture_output=True, text=True)
        state = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
        if state.get('ActiveState') == 'failed' or state.get('ExecMainStatus', '0') != '0':
            raise RuntimeError(f'existing service failed: {state}')
        if state.get('ActiveState') == 'inactive':
            return
        if state.get('ActiveState') not in ('active', 'activating', 'deactivating'):
            raise RuntimeError(f'unexpected service state: {state}')
        if time.monotonic() > deadline:
            raise TimeoutError('existing service still running; do not restart it')
        time.sleep(60)


def sequence(baseline, half_tau, output, wait_unit=None):
    output.mkdir(parents=True, exist_ok=False)
    entries = {'baseline': load_case(baseline)}
    compare(entries, partial=True)
    if wait_unit:
        wait_existing_service(wait_unit)
    entries['half_tau'] = load_case(half_tau)
    report = compare(entries, partial=True)
    (output/'progress.json').write_text(json.dumps(report, indent=2)+'\n')
    executable = Path(entries['baseline'][0]['executable'])
    paths = {'baseline': str(baseline), 'half_tau': str(half_tau)}
    for name in CASES[2:]:
        case = output/name
        options = {'double_tau': {'tau_factor': 2.}, 'extended': {'extended': True},
                   'coarse': {'ny': 128}, 'half_dt': {'dt': 2.5e-9},
                   'prescribed': {'mode': 'prescribed'}}[name]
        print(f'Preparing {name}', flush=True)
        meta = prepare(case, executable, use_gpu=entries['baseline'][0].get('use_gpu', False), **options)
        if meta['executable_sha256'] != entries['baseline'][0]['executable_sha256']:
            raise ValueError('baseline executable changed; refuse new solver run')
        run(case, meta)
        entries[name] = load_case(case)
        report = compare(entries, partial=True)
        paths[name] = str(case)
        (output/'progress.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report), flush=True)
    report = compare(entries)
    report['case_paths'] = paths
    (output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--half-tau', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--wait-unit')
    args = parser.parse_args()
    print(json.dumps(sequence(args.baseline.resolve(), args.half_tau.resolve(),
                              args.output.resolve(), args.wait_unit), indent=2))
