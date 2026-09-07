"""Verify root-owned boundary configuration distribution with real MPI ranks."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
KEY = 'ASTR_CONSERVATIVE_BOUNDARY_FILE'
VALUES = [40., 1., .2, 0., 0., 2.5, 1.2, .4, .1, 0., 3.]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = out / 'valid.nml'
    config.write_text('&conservative_boundary\n schema=1, split_x=40,\n'
                      ' q_left=1,.2,0,0,2.5, q_right=1.2,.4,.1,0,3\n/\n')
    bad = out / 'invalid.nml'
    bad.write_text('&conservative_boundary schema=1 /\n')
    report = {'status': 'running', 'scope': 'configuration broadcast only', 'checks': []}
    summary = out / 'summary.json'
    for kind in ('cpu', 'gpu'):
        exe = ROOT / f'build_{kind}_probe/bin/astr'
        digest = hashlib.sha256(exe.read_bytes()).hexdigest()
        for np in (1, 2, 4):
            for mode in ('unset', 'empty', 'valid', 'root_only', 'invalid', 'missing', 'blank'):
                if np == 1 and mode == 'root_only':
                    continue
                env = dict(os.environ, OMPI_MCA_sharedfp='individual')
                env.pop(KEY, None)
                if mode not in ('unset', 'root_only'):
                    env[KEY] = {'empty': '', 'valid': str(config), 'invalid': str(bad),
                                'missing': str(out / 'absent.nml'), 'blank': '  '}[mode]
                command = ['mpirun', '--oversubscribe']
                if mode == 'root_only':
                    command += ['-np', '1', 'env', f'{KEY}={config}', str(exe), 'test', 'bcfg',
                                ':', '-np', str(np-1), 'env', f'{KEY}={out / "absent.nml"}',
                                str(exe), 'test', 'bcfg']
                else:
                    command += ['-np', str(np), str(exe), 'test', 'bcfg']
                command = ['timeout', '--kill-after=5s', '45s', *command]
                name = f'{kind}_np{np}_{mode}'
                log = out / f'{name}.log'
                with log.open('w') as stream:
                    result = subprocess.run(command, cwd=out, env=env, stdout=stream,
                                            stderr=subprocess.STDOUT, check=False)
                output = log.read_text()
                rejected = mode in ('invalid', 'missing', 'blank')
                rows = [line.split()[1:] for line in output.splitlines()
                        if line.startswith('BOUNDARY_CONFIG_RANK ')]
                passed = False
                if rejected:
                    passed = (result.returncode == 1 and not rows and
                              'Invalid ASTR_CONSERVATIVE_BOUNDARY_FILE configuration' in output)
                elif result.returncode == 0 and len(rows) == np:
                    enabled = mode in ('valid', 'root_only')
                    expected = VALUES if enabled else [0.] * 11
                    passed = (sorted(int(row[0]) for row in rows) == list(range(np)) and
                              all(len(row) == 13 and row[1] == ('T' if enabled else 'F') and
                                  all(abs(float(s)-v) <= 1e-15 for s, v in zip(row[2:], expected))
                                  for row in rows))
                report['checks'].append({'name': name, 'pass': passed, 'returncode': result.returncode,
                                         'command': command, 'binary_sha256': digest, 'log': str(log)})
                report['status'] = 'running' if passed else 'fail'
                summary.write_text(json.dumps(report, indent=2) + '\n')
                if not passed:
                    raise RuntimeError(f'Failed {name}: {log}')
    report['status'] = 'pass'
    summary.write_text(json.dumps(report, indent=2) + '\n')
    print(summary)


if __name__ == '__main__':
    main()
