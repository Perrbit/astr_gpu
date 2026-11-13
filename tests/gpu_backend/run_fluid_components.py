"""Run native component checks against immutable CPU/CUDA Fortran TGV states.

This only launches the Fortran probe and parses its scalar errors; it does not
implement a fluid solver or regenerate the frozen input states.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--rhs-binary', type=Path)
    parser.add_argument('--mpiexec', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    options.output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,vader,tcp', OMPI_MCA_osc='pt2pt',
               OMPI_MCA_coll='^cuda,hcoll,ucc', OMPI_MCA_opal_cuda_support='0')
    cases = [('manufactured_np1', 1, options.binary, [], 80),
             ('manufactured_np2', 2, options.binary, [], 160)]
    for backend in ('cpu', 'gpu'):
        directory = options.reference / f'{backend}_np1_keep_x' / 'outdat'
        for stage in range(1, 4):
            suffix = f'.step00000000.rk{stage:02d}.rank00000000.bin'
            cases.append((f'frozen_{backend}_rk{stage}', 1, options.binary,
                          [directory / ('rhs.pre_rhs' + suffix),
                           directory / ('rhs.pre_rhs_primitives' + suffix)], 22))
            if options.rhs_binary:
                cases.append((f'rhs_{backend}_rk{stage}', 1, options.rhs_binary,
                              [directory, str(stage)], 60 if stage == 1 else 30))
    report = {'reference': str(options.reference.resolve()), 'cases': {}}
    pattern = re.compile(r'(?:FLUID|RHS) rank=(\d+) debug=(\d+)\s+(.*?)\s+([0-9.+\-Ee]+)$')
    for name, ranks, binary, inputs, expected in cases:
        command = [str(options.mpiexec), '--bind-to', 'none', '-n', str(ranks),
                   str(binary.resolve()), *map(str, inputs)]
        result = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        (options.output / f'{name}.log').write_text(result.stdout)
        result.check_returncode()
        errors = {}
        for line in result.stdout.splitlines():
            match = pattern.fullmatch(line)
            if match:
                rank, debug, quantity, error = match.groups()
                errors[f'rank{rank}/debug{debug}/{quantity}'] = float(error)
        if len(errors) != expected:
            raise AssertionError(f'{name}: expected {expected} scalar comparisons, found {len(errors)}')
        report['cases'][name] = {'command': command, 'errors': errors,
                                 'max_scaled': max(errors.values())}
        print(name, report['cases'][name]['max_scaled'], flush=True)
    report['max_scaled'] = max(case['max_scaled'] for case in report['cases'].values())
    (options.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
