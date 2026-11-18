"""Batch primitive cache coverage, failure retention and immutable solver state."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILES = [
    'src_gpu/chemistry_properties.cu', 'src_gpu/chemistry_properties.h',
    'src_gpu/chemistry_properties.F90', 'src_gpu/chemistry_properties_math.cuh',
    'src_gpu/chemistry_air5_data.h', 'src_gpu/device_state_internal.cuh',
    'src_gpu/chemistry_coupling.h', 'src_gpu/CMakeLists.txt',
    'tests/gpu_backend/CMakeLists.txt',
    'tests/gpu_backend/air5_primitives_regions_probe.cu',
    'tests/gpu_backend/run_air5_primitives_regions.py',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--memcheck', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
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
    commands = [
        [str(binary), str(output / 'input.bin')],
        [str(args.memcheck.resolve()), '--tool', 'memcheck', '--leak-check', 'full',
         '--error-exitcode', '99', str(binary), str(output / 'input.bin')],
    ]
    receipt = dict(source_sha256=hashes,
                   candidate_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                   input_sha256=hashlib.sha256((output / 'input.bin').read_bytes()).hexdigest(),
                   commands=commands)
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    marker = ('AIR5_PRIMITIVES_REGIONS_PASS contexts=4 regions=5 failure_checks=16 '
              'sequential_words=exact q_carry_history=unchanged')
    for command, name in zip(commands, ['functional.log', 'memcheck.log']):
        with (output / name).open('w') as log:
            subprocess.run(command, cwd=output, stdout=log, stderr=subprocess.STDOUT, check=True)
        text = (output / name).read_text()
        assert text.count(marker) == 1
        assert text.count('AIR5_PRIMITIVE_SHELL_PASS cases=24 asymmetric_and_singleton=checked') == 1
        if name == 'memcheck.log':
            assert text.count('ERROR SUMMARY: 0 errors') == 1
            assert text.count('LEAK SUMMARY: 0 bytes leaked') == 1
    report = dict(contexts=4, regions=5, failure_checks=16, shell_checks=24,
                  cache_words='batch equals separate checked calls',
                  authoritative_words='q/carry/history unchanged',
                  coverage='physical, two face halos, skipped exterior corner, duplicate node',
                  memcheck='zero errors/leaks; no MPI in this probe')
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
