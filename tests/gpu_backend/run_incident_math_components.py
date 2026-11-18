"""Freeze original GRCBC algebra and check the native C++ device implementation."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-probe', type=Path, required=True)
    parser.add_argument('--old-gpu-probe', type=Path, required=True)
    parser.add_argument('--cpp-probe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sanitizer', type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = dict(scope='FP64 GRCBC characteristic components only; no main-program or MPI admission',
                  tolerance=2e-10, cases=9, commands={}, binaries={})

    def run(name, command):
        report['commands'][name] = [str(x) for x in command]
        with (out / f'{name}.log').open('w') as log:
            subprocess.run(report['commands'][name], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        return (out / f'{name}.log').read_text()

    for name, binary in [('cpu', args.cpu_probe), ('old_gpu', args.old_gpu_probe), ('cpp', args.cpp_probe)]:
        binary = binary.resolve()
        report['binaries'][name] = dict(path=str(binary), sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    for name in ('cpu', 'old_gpu'):
        log = run(name, [report['binaries'][name]['path'], out / f'{name}.bin'])
        if 'NSCBC_INCIDENT_PROBE_PASS' not in log:
            raise AssertionError(f'{name} missing completed component assertions')
    if (out / 'cpu.bin').read_bytes() != (out / 'old_gpu.bin').read_bytes():
        raise AssertionError('CPU packet changed between original CPU and CPU/old-GPU probes')
    log = run('cpp', [report['binaries']['cpp']['path'], out / 'cpu.bin'])
    if 'INCIDENT_CPP_COMPONENT_PASS cases=9' not in log:
        raise AssertionError('C++ component assertions did not complete')
    errors = {name: float(value) for name, value in re.findall(r'INCIDENT_(\w+)_MAX_ABS\s+(\S+)', log)}
    if len(errors) != 7 or any(not (0 <= value <= 2e-10) for value in errors.values()):
        raise AssertionError(f'component errors: {errors}')
    report['max_absolute_errors'] = errors
    report['same_state_masks_and_rejection_contract'] = True
    report['incoming_outgoing_late_source_idempotence_reference_preservation'] = True
    if args.sanitizer:
        log = run('cpp_memcheck', [args.sanitizer.resolve(), '--tool', 'memcheck', '--leak-check', 'full',
                  '--error-exitcode', '99', '--log-file', out / 'memcheck.log',
                  report['binaries']['cpp']['path'], out / 'cpu.bin'])
        tool = (out / 'memcheck.log').read_text()
        if 'INCIDENT_CPP_COMPONENT_PASS cases=9' not in log or 'ERROR SUMMARY: 0 errors' not in tool or \
                'LEAK SUMMARY: 0 bytes leaked in 0 allocations' not in tool:
            raise AssertionError('component memcheck did not report zero errors and zero leaks')
        report['memcheck_zero_errors_and_leaks'] = True
    sources = [ROOT / 'src/nscbc_incident_body.inc', ROOT / 'src/astr_precision.h',
               ROOT / 'src_gpu/nscbc_incident_math.cuh', ROOT / 'tests/gpu_validation/nscbc_incident_probe.F90',
               Path(__file__), ROOT / 'tests/gpu_backend/nscbc_incident_probe.cu',
               ROOT / 'tests/gpu_backend/CMakeLists.txt']
    report['sources'] = {}
    for source in sources:
        relative = source.relative_to(ROOT)
        destination = out / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        report['sources'][str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    report['reference_packet_sha256'] = hashlib.sha256((out / 'cpu.bin').read_bytes()).hexdigest()
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['max_absolute_errors'], indent=2))


if __name__ == '__main__':
    main()
