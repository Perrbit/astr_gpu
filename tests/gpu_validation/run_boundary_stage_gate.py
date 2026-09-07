"""Check repeated CPU conservative-boundary stages in both ASTR builds."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[2]
MARKER = re.compile(r'^\s*BOUNDARY_STAGE_PASS max_abs=\s*(\S+)\s*$', re.MULTILINE)


def validate_output(returncode, output, gpu=False):
    pattern = (re.compile(r'^\s*BOUNDARY_STAGE_GPU_PASS max_abs=\s*(\S+)\s*$', re.MULTILINE)
               if gpu else MARKER)
    matches = pattern.findall(output)
    if returncode != 0 or len(matches) != 1:
        raise ValueError('Expected exit zero and exactly one boundary-stage PASS marker')
    error = float(matches[0].replace('D', 'E').replace('d', 'e'))
    if not math.isfinite(error) or error < 0 or error > 1e-12:
        raise ValueError('Boundary-stage error is nonfinite or outside tolerance')
    return error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = out / 'valid.nml'
    config.write_text('&conservative_boundary\n schema=1, split_x=40,\n'
                      ' q_left=1,.2,0,0,2.5, q_right=1.2,.4,.1,0,3\n/\n')
    report = {'status': 'running', 'scope': 'NP1 CPU/GPU boundary stage; '
              'fresh and restart initialization, no RHS or RK advancement', 'checks': []}
    summary = out / 'summary.json'
    summary.write_text(json.dumps(report, indent=2) + '\n')
    for kind, sanitizer, mode in (('cpu', False, 'fresh'),
                                  ('cpu', False, 'restart_nonuniform'),
                                  ('gpu', False, 'fresh'),
                                  ('gpu', False, 'restart_nonuniform'),
                                  ('gpu', True, 'restart_nonuniform')):
        exe = ROOT / f'build_{kind}_probe/bin/astr'
        command = ['timeout', '--kill-after=5s', '120s', 'mpirun', '-np', '1']
        if sanitizer:
            command += ['compute-sanitizer', '--tool', 'memcheck', '--error-exitcode', '1']
        command += [str(exe), 'test', 'bcst']
        env = dict(os.environ, OMPI_MCA_sharedfp='individual',
                   ASTR_CONSERVATIVE_BOUNDARY_FILE=str(config))
        if mode != 'fresh':
            env['ASTR_BOUNDARY_STAGE_INITIAL_TEST'] = mode
        if sanitizer:
            # NP1 exchanges only host MPI values. Disable MPI's GPU-pointer
            # probing, not application CUDA calls or sanitizer error reporting.
            env.update(OMPI_MCA_pml='ob1', OMPI_MCA_osc='pt2pt', OMPI_MCA_btl='self,vader,tcp',
                       OMPI_MCA_opal_cuda_support='false')
        log = out / f'{kind}_{mode}_stage{"_memcheck" if sanitizer else ""}.log'
        check = {'build': kind, 'execution': 'CPU/GPU' if kind == 'gpu' else 'CPU',
                 'initialization': mode, 'sanitizer': sanitizer, 'pass': False, 'command': command,
                 'environment_overrides': {key: value for key, value in env.items()
                                           if key.startswith('OMPI_MCA_') or key.startswith('ASTR_')},
                 'binary_sha256': hashlib.sha256(exe.read_bytes()).hexdigest(),
                 'config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
                 'log': str(log)}
        report['checks'].append(check)
        try:
            with log.open('w') as stream:
                result = subprocess.run(command, cwd=out, env=env, stdout=stream,
                                        stderr=subprocess.STDOUT, check=False)
            check['returncode'] = result.returncode
            check['max_abs'] = validate_output(result.returncode, log.read_text())
            if kind == 'gpu':
                check['gpu_max_abs'] = validate_output(result.returncode, log.read_text(), gpu=True)
            if sanitizer and 'ERROR SUMMARY: 0 errors' not in log.read_text():
                raise ValueError('Missing clean Compute Sanitizer summary')
            check['pass'] = True
        except (ValueError, OSError) as exc:
            check['error'] = str(exc)
            report['status'] = 'fail'
            raise
        finally:
            summary.write_text(json.dumps(report, indent=2) + '\n')
    report['status'] = 'pass'
    summary.write_text(json.dumps(report, indent=2) + '\n')
    print(summary)


if __name__ == '__main__':
    main()
