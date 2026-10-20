#!/usr/bin/env python3
"""Independent host-debugger initialization/transport observations, not RK timing."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from prepare_m12_local_case import replace_value

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    reports = []
    for filtering in ('f', 't'):
        for ranks in (1, 2):
            name = f'643c_filter{filtering}_np{ranks}_r1'
            source = args.preflight_root.resolve() / name
            case = output / name
            case.mkdir()
            shutil.copytree(source / 'datin', case / 'datin', ignore=shutil.ignore_patterns('grid.h5'))
            path = case / 'datin/controller'
            lines = path.read_text().splitlines()
            replace_value(lines, 'maxstep,feqchkpt', '0,1,9999,9999,1,9999')
            path.write_text('\n'.join(lines) + '\n')
            launch = json.loads((source / 'launch.json').read_text())
            env = {k: v for k, v in os.environ.items()
                   if not k.startswith(('ASTR_', 'OMPI_MCA_', 'UCX_'))}
            env.update(launch['environment'])
            env.pop('ASTR_VALIDATION_RK_STATE', None)
            command = launch['command'][:-3] + ['gdb', '--batch', '--return-child-result',
                '-x', str(ROOT / 'tests/gpu_validation/compact_initialization.gdb'), '--args'] + launch['command'][-3:]
            with (case / 'debugger.log').open('w') as stream:
                subprocess.run(command, cwd=case, env=env, stdout=stream,
                               stderr=subprocess.STDOUT, check=True)
            text = (case / 'debugger.log').read_text()
            setups = [json.loads(row) for row in re.findall(r'COMPACT_INIT_DIAGNOSTIC (\{[^\n]+\})', text)]
            traffic = [json.loads(row) for row in re.findall(r'COMPACT_TRANSPORT_DIAGNOSTIC (\{[^\n]+\})', text)]
            expected = {(r, phase) for r in range(ranks) for phase in
                        (('direction_groups', 'derivative_setup', 'filter_setup') if filtering == 't'
                         else ('direction_groups', 'derivative_setup'))}
            if {(row['rank'], row['phase']) for row in setups} != expected or len(setups) != len(expected):
                raise ValueError(f'missing/duplicate initialization observation: {name}')
            if len(traffic) != ranks or {row['rank'] for row in traffic} != set(range(ranks)):
                raise ValueError(f'missing transport observations: {name}')
            if ranks == 2 and not sum(item['calls'] for row in traffic for item in row['counters'].values()):
                raise ValueError('no compact transport calls observed')
            report = dict(configuration=name, initialization=setups, transport=traffic,
                          scope='one complete step plus startup/statistics; host debugger overhead included')
            reports.append(report)
            (case / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
            print(name, 'PASS', flush=True)
    (output / 'summary.json').write_text(json.dumps(reports, indent=2) + '\n')


if __name__ == '__main__':
    main()
