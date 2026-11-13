"""Exact pinned-host halo checks against independent Fortran MPI packets."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--mpiexec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    options.output.mkdir(parents=True, exist_ok=False)
    env = os.environ | dict(OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,vader,tcp', OMPI_MCA_osc='pt2pt',
                           OMPI_MCA_coll='^cuda,hcoll,ucc', OMPI_MCA_opal_cuda_support='0')
    report = {}
    for ranks, axis in [(1, 0), (2, 0), (2, 1), (2, 2)]:
        name = f'np{ranks}_axis{axis}'
        command = [str(options.mpiexec), '--bind-to', 'none', '-n', str(ranks),
                   str(options.binary.resolve()), str(axis)]
        result = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (options.output/f'{name}.log').write_text(result.stdout)
        result.check_returncode()
        count = sum(line.startswith('HALO PASS ') for line in result.stdout.splitlines())
        if count != 4*ranks:
            raise AssertionError(f'{name}: expected {4*ranks} complete layout checks, got {count}')
        report[name] = dict(command=command, exact_phase_checks=count)
        print(name, count, 'exact', flush=True)
    (options.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
