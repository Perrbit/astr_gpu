"""Validate initial profiles across actual x-rank groups before halo seeding."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'scope': 'CPU initial profile admission, not GPU/MPI flow advancement',
              'checks': []}
    summary = out / 'summary.json'
    for kind in ('cpu', 'gpu'):
        exe = ROOT / f'build_{kind}_probe/bin/astr'
        digest = hashlib.sha256(exe.read_bytes()).hexdigest()
        for np in (1, 2, 4):
            for mode in ('valid', 'nonuniform', 'rank_offset', 'negative_density',
                         'restart_nonuniform', 'restart_rank_offset',
                         'restart_negative_density'):
                if np == 1 and mode in ('rank_offset', 'restart_rank_offset'):
                    continue
                env = dict(os.environ, OMPI_MCA_sharedfp='individual',
                           ASTR_BOUNDARY_STAGE_INITIAL_TEST=mode)
                command = ['timeout', '--kill-after=5s', '45s', 'mpirun', '--oversubscribe',
                           '-np', str(np), str(exe), 'test', 'bciv']
                name = f'{kind}_np{np}_{mode}'
                log = out / f'{name}.log'
                with log.open('w') as stream:
                    result = subprocess.run(command, cwd=out, env=env, stdout=stream,
                                            stderr=subprocess.STDOUT, check=False)
                output = log.read_text()
                if mode in ('valid', 'restart_nonuniform', 'restart_rank_offset'):
                    passed = result.returncode == 0 and output.count('BOUNDARY_INITIAL_MPI_PASS') == 1
                else:
                    error = ('Invalid conservative physical state' if 'negative_density' in mode
                             else 'Conservative initial profile must be x-uniform')
                    passed = result.returncode == 1 and error in output and 'BOUNDARY_INITIAL_MPI_PASS' not in output
                report['checks'].append({'name': name, 'pass': passed, 'command': command,
                                         'environment_mode': mode, 'binary_sha256': digest,
                                         'returncode': result.returncode, 'log': str(log)})
                report['status'] = 'running' if passed else 'fail'
                summary.write_text(json.dumps(report, indent=2) + '\n')
                if not passed:
                    raise RuntimeError(f'Failed {name}: {log}')
    report['status'] = 'pass'
    summary.write_text(json.dumps(report, indent=2) + '\n')
    print(summary)


if __name__ == '__main__':
    main()
