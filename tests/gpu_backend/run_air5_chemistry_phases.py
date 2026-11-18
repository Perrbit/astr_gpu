"""AIR5 local chemistry and MPI release/report phases; no fluid RK or boundary update."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
FILES = [
    'src_gpu/chemistry_coupling.F90', 'src_gpu/chemistry_coupling.cu',
    'src_gpu/chemistry_coupling.h', 'src_gpu/chemistry_ros2_math.cuh',
    'src_gpu/chemistry_compensation_math.cuh', 'src_gpu/chemistry_kinetics_math.cuh',
    'src_gpu/chemistry_properties_math.cuh', 'src_gpu/chemistry_air5_data.h',
    'src_gpu/device_state.F90', 'src_gpu/device_state_internal.cuh',
    'src_gpu/CMakeLists.txt', 'tests/gpu_backend/CMakeLists.txt',
    'tests/gpu_backend/air5_chemistry_phase_probe.F90',
    'tests/gpu_backend/run_air5_chemistry_phases.py',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate', 'input', 'mpirun', 'memcheck', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name in FILES:
        target = output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    binary = output / 'candidate'
    shutil.copy2(args.candidate.resolve(), binary)
    shutil.copy2(args.input.resolve(), output / 'input.bin')
    receipt = dict(source_sha256=hashes,
                   candidate_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                   input_sha256=hashlib.sha256((output / 'input.bin').read_bytes()).hexdigest())
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    env = dict(os.environ, OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,vader,tcp',
               OMPI_MCA_osc='pt2pt', OMPI_MCA_coll='^cuda,hcoll,ucc', OMPI_MCA_opal_cuda_support='0')
    pattern = re.compile(r'AIR5_CHEMISTRY_PHASE_PASS rank=(\d+) cases=9 global_failures=2 device_words=exact')
    commands, runs = [], []
    for topology, tool in [((1, 1, 1), False), ((2, 1, 1), False), ((1, 2, 1), False),
                           ((1, 1, 2), False), ((2, 2, 1), False), ((2, 1, 1), True)]:
        ranks = topology[0] * topology[1] * topology[2]
        name = ''.join(map(str, topology)) + ('_memcheck' if tool else '')
        command = [str(args.mpirun.resolve()), '--bind-to', 'none', '-np', str(ranks)]
        if tool:
            command += [str(args.memcheck.resolve()), '--tool', 'memcheck', '--leak-check', 'full', '--error-exitcode', '99']
        command += [str(binary), str(output / 'input.bin'), *map(str, topology)]
        commands.append(dict(command=command, environment={k: v for k, v in env.items() if k.startswith('OMPI_MCA_')},
                             log=name + '.log'))
        (output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        with (output / (name + '.log')).open('w') as log:
            subprocess.run(command, cwd=output, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        text = (output / (name + '.log')).read_text()
        assert sorted(map(int, pattern.findall(text))) == list(range(ranks))
        if tool:
            assert text.count('ERROR SUMMARY: 0 errors') == ranks
            assert text.count('LEAK SUMMARY: 0 bytes leaked') == ranks
        runs.append(dict(topology=topology, ranks=ranks, cases=9 * ranks, failure_observations=2 * ranks, memcheck=tool))
        (output / 'completed_runs.json').write_text(json.dumps(runs, indent=2) + '\n')
    report = dict(**receipt, runs=runs,
                  functional_rank_cases=sum(r['cases'] for r in runs if not r['memcheck']),
                  functional_failure_observations=sum(r['failure_observations'] for r in runs if not r['memcheck']),
                  memcheck='two ranks, all nine cases, zero errors/leaks',
                  scope='region chemistry/state isolation and global status/report; no boundary, halo, fluid RK, G2 or ordinary AIR5')
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('source_sha256', 'runs')}, indent=2))


if __name__ == '__main__':
    main()
