"""Resident AIR5 fixed/profile targets; no flow advancement or MPI halo claim."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
FILES = [
    'src_gpu/chemistry_boundary.cu', 'src_gpu/chemistry_boundary.h',
    'src_gpu/chemistry_boundary.F90', 'src_gpu/chemistry_boundary_math.cuh',
    'src_gpu/chemistry_boundary_gpu.cuf', 'src/chemistry_boundary.F90',
    'src_gpu/device_state.cu', 'src_gpu/device_state_internal.cuh',
    'src_gpu/chemistry_properties_math.cuh', 'src_gpu/chemistry_air5_data.h',
    'src_gpu/CMakeLists.txt', 'tests/gpu_backend/CMakeLists.txt',
    'tests/gpu_backend/air5_boundary_targets_probe.cu',
    'tests/gpu_backend/air5_convection_abi.F90',
    'tests/gpu_backend/run_air5_boundary_targets.py',
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--mpirun', type=Path, required=True)
    parser.add_argument('--memcheck', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source_hashes = {}
    for name in FILES:
        target = out / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        source_hashes[name] = sha(target)
    binary = out / 'candidate'
    shutil.copy2(args.candidate.resolve(), binary)
    reference = args.reference.resolve()
    reference_hashes = {}
    for name in ['input.bin', 'cpu.bin', 'old_gpu.bin', 'report.json']:
        shutil.copy2(reference / name, out / name)
        reference_hashes[name] = sha(out / name)
    env = dict(os.environ, OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,vader,tcp',
               OMPI_MCA_osc='pt2pt', OMPI_MCA_coll='^cuda,hcoll,ucc',
               OMPI_MCA_opal_cuda_support='0')
    commands, results = [], []
    for topology in [(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2), (2, 2, 1)]:
        ranks = topology[0] * topology[1] * topology[2]
        tag = ''.join(map(str, topology))
        cmd = [args.mpirun.resolve(), '--bind-to', 'none', '-np', ranks, binary,
               out / 'input.bin', out / 'cpu.bin', out / 'old_gpu.bin', out / f'{tag}.json',
               *topology, min(2, ranks)]
        commands.append(list(map(str, cmd)))
        with (out / f'{tag}.log').open('w') as log:
            subprocess.run(commands[-1], env=env, cwd=out, stdout=log, stderr=subprocess.STDOUT, check=True)
        results.append(dict(topology=topology, **json.loads((out / f'{tag}.json').read_text())))
    cmd = [args.mpirun.resolve(), '--bind-to', 'none', '-np', 2,
           args.memcheck.resolve(), '--tool', 'memcheck', '--leak-check', 'full', '--error-exitcode', 99,
           binary, out / 'input.bin', out / 'cpu.bin', out / 'old_gpu.bin', out / 'memcheck.json', 2, 1, 1, 2]
    commands.append(list(map(str, cmd)))
    with (out / 'memcheck.log').open('w') as log:
        subprocess.run(commands[-1], env=env, cwd=out, stdout=log, stderr=subprocess.STDOUT, check=True)
    text = (out / 'memcheck.log').read_text()
    assert text.count('ERROR SUMMARY: 0 errors') == 2
    assert text.count('LEAK SUMMARY: 0 bytes leaked') == 2
    report = dict(source_sha256=source_hashes, candidate_sha256=sha(binary),
                  frozen_reference=str(reference), reference_sha256=reference_hashes,
                  commands=commands, mpi_environment={k: v for k, v in env.items() if k.startswith('OMPI_MCA_')},
                  runs=results, total_calls=sum(r['calls'] for r in results),
                  memcheck='two ranks, zero errors/leaks; no device MPI halo operations',
                  scope='fixed/profile target storage, local physical ownership and ghost writes only')
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['runs', 'total_calls', 'memcheck', 'scope']}, indent=2))


if __name__ == '__main__':
    main()
