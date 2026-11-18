"""Original AIR5 SUM44/ENDPOINT14 device statistics; no CFD or file-level restart."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from run_air5_flux_components import extract
from run_air5_properties_components import read as read_properties

ROOT = Path(__file__).resolve().parents[2]
BASE = ['src/astr_precision.F90', 'src/constdef.F90', 'src/mean_statistics_math.F90',
        'src/chemistry_air5_data.F90', 'src/chemistry_core.F90', 'src/chemistry_properties.F90']
GPU = ['src_gpu/chemistry_core_gpu.cuf', 'src_gpu/chemistry_flow_state_gpu.cuf',
       'src_gpu/chemistry_transport_gpu.cuf']
FILES = BASE + GPU + [
    'src/chemistry_pure_vt_body.inc', 'src/chemistry_vibrational_difference_body.inc',
    'src_gpu/chemistry_monitor_gpu.cuf', 'src_gpu/chemistry_statistics.cu',
    'src_gpu/chemistry_statistics.h', 'src_gpu/chemistry_statistics.F90',
    'src_gpu/chemistry_properties_math.cuh', 'src_gpu/chemistry_air5_data.h',
    'src_gpu/device_state.h', 'src_gpu/device_state.F90', 'src_gpu/device_state.cu',
    'src_gpu/device_state_internal.cuh', 'src_gpu/CMakeLists.txt',
    'tests/gpu_backend/CMakeLists.txt', 'tests/gpu_backend/air5_mean_probe.F90',
    'tests/gpu_backend/run_air5_mean_components.py',
]


def run(command, work, log):
    with (work / log).open('w') as stream:
        subprocess.run(list(map(str, command)), cwd=work, stdout=stream, stderr=subprocess.STDOUT, check=True)


def build_reference(output, compiler, gpu):
    work = output / ('old_gpu_build' if gpu else 'cpu_build')
    work.mkdir()
    source = output / 'source'
    flags = [compiler, '-O2', '-Mpreprocess', '-Mextend', '-I' + str(source / 'src'), '-I' + str(work)]
    if gpu:
        flags += ['-cuda', '-gpu=rdc,maxregcount:128', '-D_CUDA', '-DASTR_MEAN_REFERENCE_GPU']
        kernel = extract((source / 'src_gpu/chemistry_monitor_gpu.cuf').read_text(), 'accumulate_mean_kernel')
        (work / 'frozen_mean_kernel.inc').write_text(kernel)
    else:
        flags += ['-DASTR_MEAN_REFERENCE_CPU']
    commands, objects = [], []
    names = BASE + (GPU if gpu else []) + ['tests/gpu_backend/air5_mean_probe.F90']
    for n, name in enumerate(names):
        path = source / name
        if gpu and name.endswith('air5_mean_probe.F90'):
            path = work / 'reference.cuf'
            shutil.copy2(source / name, path)
        obj = work / f'unit{n}.o'
        command = flags + ['-c', path, '-o', obj]
        commands.append(list(map(str, command)))
        run(command, work, f'compile{n}.log')
        objects.append(obj)
    command = flags + objects + ['-o', work / 'reference']
    commands.append(list(map(str, command)))
    run(command, work, 'link.log')
    return work / 'reference', commands


def fixture(path, properties):
    status, values = read_properties(properties / 'cpu_all.bin')
    pool = [120, 121, 122, 123, 124, 125, 126, 215, 293, 371]
    assert np.all(status[pool] == 0)
    dims = (4, 3, 2)
    frames = 8
    data = np.empty(dims + (32, frames))
    for sample in range(frames):
        for i, j, k in np.ndindex(dims):
            row = values[pool[(i + 2*j + 3*k + 3*sample) % len(pool)]]
            data[i, j, k, :12, sample] = np.r_[row[11:15], row[22], row[15], row[21], row[16:21]]
            data[i, j, k, 12:21, sample] = .025 * (1+i+2*j+3*k) * (sample+1) * np.arange(-4, 5)
            data[i, j, k, 21:32, sample] = row[:11]
    data[0, 0, 0, 4, 6] = -1.
    data[1, 0, 0, 12, 7] = np.nan
    intervals = np.array([0., 1e-7, 1.6e-7, 1.1e-7, 3.5e-7, 2.2e-7, .5e-7, .5e-7], '<f8')
    with path.open('wb') as stream:
        np.array([3, 2, 1, frames], '<i4').tofile(stream)
        intervals.tofile(stream)
        data.astype('<f8').ravel(order='F').tofile(stream)
    return dict(property_rows=pool, nodes=int(np.prod(dims)), intervals=intervals.tolist(),
                valid_frames=6, failure_frames=[dict(frame=6, node=[0, 0, 0], fault='negative pressure'),
                                               dict(frame=7, node=[1, 0, 0], fault='NaN velocity gradient')],
                description='frozen physical point states and manufactured gradients, not a flow solution')


def read(path):
    with path.open('rb') as stream:
        header = np.fromfile(stream, '<i4', 4)
        assert len(header) == 4
        dims = tuple(header[:3] + 1)
        nodes = int(np.prod(dims))
        result = []
        for _ in range(header[3]):
            status = np.fromfile(stream, '<i4', 1)
            assert len(status) == 1
            sums = np.fromfile(stream, '<f8', nodes*44).reshape(dims + (44,), order='F')
            endpoints = np.fromfile(stream, '<f8', nodes*14).reshape(dims + (14,), order='F')
            result.append((int(status[0]), sums, endpoints))
        assert stream.read() == b''
    return result


def compare(actual, reference):
    assert len(actual) == len(reference) == 8
    maximum = {'sum44': 0., 'endpoint14': 0.}
    for n, (a, b) in enumerate(zip(actual, reference)):
        assert a[0] == b[0] and (a[0] == 0 if n < 6 else a[0] != 0), (n, a[0], b[0])
        for name, x, y in zip(maximum, a[1:], b[1:]):
            assert np.isfinite(x).all() and np.isfinite(y).all()
            # Existing test_meanflow_air5_transport per-field spatial scale.
            scale = np.maximum(1., np.max(abs(y), axis=(0, 1, 2)))
            ratio = float(np.max(abs(x-y) / (1e-14*scale)))
            assert ratio <= 1., (n, name, ratio)
            maximum[name] = max(maximum[name], ratio)
    for frame, node in [(6, (0, 0, 0)), (7, (1, 0, 0))]:
        for index in (1, 2):
            assert actual[frame][index][node].tobytes() == actual[frame-1][index][node].tobytes()
    return maximum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate', 'compiler', 'properties-reference', 'memcheck', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name in FILES:
        path = output / 'source' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, path)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    shutil.copy2(args.candidate.resolve(), output / 'candidate')
    description = fixture(output / 'input.bin', args.properties_reference.resolve())
    receipt = dict(source_sha256=hashes,
                   candidate_sha256=hashlib.sha256((output / 'candidate').read_bytes()).hexdigest(),
                   input_sha256=hashlib.sha256((output / 'input.bin').read_bytes()).hexdigest(), fixture=description)
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    builds, commands, references = {}, [], {}
    for backend, gpu in [('cpu', False), ('old_gpu', True)]:
        binary, builds[backend] = build_reference(output, args.compiler.resolve(), gpu)
        (output / 'build_commands.json').write_text(json.dumps(builds, indent=2) + '\n')
        command = [binary, output / 'input.bin', output / (backend + '.bin'), 1, 0]
        commands.append(list(map(str, command)))
        (output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        run(command, output, backend + '.log')
        references[backend] = read(output / (backend + '.bin'))
    reference_ratio = compare(references['old_gpu'], references['cpu'])
    results = []
    for debug, resumed in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        name = f'cpp_d{debug}_r{resumed}'
        command = [output / 'candidate', output / 'input.bin', output / (name + '.bin'), debug, resumed]
        commands.append(list(map(str, command)))
        (output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        run(command, output, name + '.log')
        assert 'AIR5_MEAN_PASS fields_unchanged=10 reset=exact optional_allocation=checked' in (output / (name + '.log')).read_text()
        current = read(output / (name + '.bin'))
        ratios = {backend: compare(current, reference) for backend, reference in references.items()}
        if name != 'cpp_d0_r0':
            assert (output / (name + '.bin')).read_bytes() == (output / 'cpp_d0_r0.bin').read_bytes()
        results.append(dict(debug=debug, restored_arrays=bool(resumed), ratios=ratios))
        (output / 'completed_runs.json').write_text(json.dumps(results, indent=2) + '\n')
    command = [args.memcheck.resolve(), '--tool', 'memcheck', '--leak-check', 'full', '--error-exitcode', 99,
               output / 'candidate', output / 'input.bin', output / 'memcheck.bin', 1, 1]
    commands.append(list(map(str, command)))
    (output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    run(command, output, 'memcheck.log')
    text = (output / 'memcheck.log').read_text()
    assert 'ERROR SUMMARY: 0 errors' in text and 'LEAK SUMMARY: 0 bytes leaked' in text
    assert (output / 'memcheck.bin').read_bytes() == (output / 'cpp_d0_r0.bin').read_bytes()
    report = dict(**receipt, builds=builds, reference_budget_ratio=reference_ratio, runs=results,
                  budget='1e-14 * max(1, spatial maxabs(reference field)), unchanged existing AIR5 mean-statistics budget',
                  memcheck='single rank, all frames, restored component arrays; zero errors/leaks',
                  scope='optional device statistics and raw SUM44/ENDPOINT14 restore only; no sample clock, checkpoint file, MPI, ordinary AIR5 or G2')
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['reference_budget_ratio', 'runs', 'budget', 'memcheck', 'scope']}, indent=2))


if __name__ == '__main__':
    main()
