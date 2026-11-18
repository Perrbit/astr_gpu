"""Original AIR5 mean clock/HDF5 contract on C++ device state; no reacting CFD."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import h5py
import numpy as np
from run_air5_properties_components import read as read_properties

ROOT = Path(__file__).resolve().parents[2]
BASE = ['src/astr_precision.F90', 'src/constdef.F90', 'src/mean_statistics_math.F90',
        'src/chemistry_air5_data.F90', 'src/chemistry_core.F90', 'src/chemistry_properties.F90']
GPU = ['src_gpu/chemistry_core_gpu.cuf', 'src_gpu/chemistry_flow_state_gpu.cuf', 'src_gpu/chemistry_transport_gpu.cuf']
FILES = BASE + GPU + ['src/chemistry_pure_vt_body.inc', 'src/chemistry_vibrational_difference_body.inc',
    'src/checkpoint_state_io.F90', 'src_gpu/chemistry_monitor_gpu.cuf', 'src_gpu/chemistry_statistics_runtime.F90',
    'src_gpu/chemistry_statistics.cu', 'src_gpu/chemistry_statistics.h', 'src_gpu/chemistry_statistics.F90',
    'src_gpu/chemistry_properties_math.cuh', 'src_gpu/chemistry_air5_data.h', 'src_gpu/device_state.h',
    'src_gpu/device_state.F90', 'src_gpu/device_state.cu', 'src_gpu/device_state_internal.cuh',
    'src_gpu/CMakeLists.txt', 'src/CMakeLists.txt', 'tests/gpu_backend/CMakeLists.txt',
    'tests/gpu_backend/air5_mean_runtime_storage.F90', 'tests/gpu_backend/air5_mean_runtime_probe.F90',
    'tests/gpu_backend/run_air5_mean_runtime.py']
ENV = dict(os.environ, OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,vader,tcp', OMPI_MCA_osc='pt2pt',
           OMPI_MCA_coll='^cuda,hcoll,ucc', OMPI_MCA_opal_cuda_support='0')
TIMES = np.array([0., 1e-7, 2.6e-7, 3.7e-7, 7.2e-7, 9.4e-7], '<f8')
STEPS = np.array([0, 2, 4, 5, 8, 10], '<i8')
PRODUCTS = ['empty', 'first', 'seed', 'final', 'reset_empty', 'reset_first', 'reset_final']


def execute(command, work, log):
    with (work / log).open('w') as stream:
        subprocess.run(list(map(str, command)), cwd=work, env=ENV, stdout=stream, stderr=subprocess.STDOUT, check=True)


def build_reference(output, compiler, hdf5):
    source = output / 'source'
    work = output / 'old_gpu_build'
    work.mkdir()
    flags = [compiler, '-O2', '-Mpreprocess', '-Mextend', '-cuda', '-gpu=rdc,maxregcount:128',
             '-D_CUDA', '-DASTR_MEAN_RUNTIME_REFERENCE', '-I' + str(source / 'src'), '-I' + str(hdf5 / 'include')]
    names = BASE + GPU + ['src/checkpoint_state_io.F90', 'tests/gpu_backend/air5_mean_runtime_storage.F90',
                         'src_gpu/chemistry_monitor_gpu.cuf', 'tests/gpu_backend/air5_mean_runtime_probe.F90']
    commands, objects = [], []
    for n, name in enumerate(names):
        path = source / name
        if name.startswith('tests/'):
            path = work / (Path(name).stem + '.cuf')
            shutil.copy2(source / name, path)
        obj = work / f'unit{n}.o'
        command = flags + ['-c', path, '-o', obj]
        commands.append(list(map(str, command)))
        (output / 'reference_build_commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        execute(command, work, f'compile{n}.log')
        objects.append(obj)
    command = flags + objects + [hdf5 / 'lib/libhdf5_fortran.so', hdf5 / 'lib/libhdf5.so',
                                 '-Wl,-rpath,' + str(hdf5 / 'lib'), '-o', work / 'reference']
    commands.append(list(map(str, command)))
    (output / 'reference_build_commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    execute(command, work, 'link.log')
    return work / 'reference'


def fixture(directory, topology, properties):
    directory.mkdir()
    status, values = read_properties(properties / 'cpu_all.bin')
    pool = [120, 121, 122, 123, 124, 125, 126, 215, 293, 371]
    assert np.all(status[pool] == 0)
    hashes = []
    for rank in range(int(np.prod(topology))):
        coord = np.unravel_index(rank, topology)
        data = np.empty((4, 3, 3, 32, 6))
        for sample in range(6):
            for i, j, k in np.ndindex((4, 3, 3)):
                gi, gj, gk = 3*coord[0]+i, 2*coord[1]+j, 2*coord[2]+k
                row = values[pool[(gi+2*gj+3*gk+3*sample) % len(pool)]]
                data[i, j, k, :12, sample] = np.r_[row[11:15], row[22], row[15], row[21], row[16:21]]
                data[i, j, k, 12:21, sample] = .025*(1+gi+2*gj+3*gk)*(sample+1)*np.arange(-4, 5)
                data[i, j, k, 21:32, sample] = row[:11]
        path = directory / f'input.rank{rank:04d}.bin'
        with path.open('wb') as stream:
            np.array([3, 2, 2, 6], '<i4').tofile(stream)
            TIMES.tofile(stream); STEPS.tofile(stream)
            data.astype('<f8').ravel(order='F').tofile(stream)
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    return dict(topology=topology, rank_input_sha256=hashes, times=TIMES.tolist(), steps=STEPS.tolist(),
                scope='frozen physical point caches/manufactured gradients, same values at MPI shared nodes; no CFD')


def compare(a, b, exact):
    ratios = {}
    with h5py.File(a) as left, h5py.File(b) as right:
        assert set(left) == set(right)
        for name in left:
            x, y = left[name][...], right[name][...]
            assert x.shape == y.shape and x.dtype == y.dtype, (a, name)
            if exact or not (name.startswith('q') or name == 'rank_extras'):
                assert x.tobytes() == y.tobytes(), (a, name, 'word mismatch')
            else:
                assert np.isfinite(x).all() and np.isfinite(y).all(), (a, name)
                ratio = float(np.max(abs(x-y), initial=0) / (1e-14*max(1., float(np.max(abs(y), initial=0)))))
                assert ratio <= 1., (a, name, ratio)
                ratios[name] = ratio
    return max(ratios.values(), default=0.)


def check_clock(directory, inherited=False):
    expected = {'empty': (0, 0, 0), 'first': (1, 0, 0), 'seed': (3, 0, 4), 'final': (6, 0, 10),
                'reset_empty': (0, 0, 0), 'reset_first': (1, 12, 12), 'reset_final': (2, 12, 13)}
    for name, (count, first, last) in expected.items():
        if inherited and name in ('first', 'seed'):
            continue
        with h5py.File(directory / (name + '.h5')) as data:
            meta = data['metadata'][...]
            assert meta.dtype.kind == 'i' and meta.itemsize == 8 and meta.size == 9
            assert meta[0] == 3 and tuple(map(int, meta[1:4])) == (count, first, last), (directory, name, meta)
            assert np.array_equal(meta[5:7], [int(count > 0)]*2)
            first_time, last_time, weight = meta[[4, 7, 8]].astype('<i8').view('<f8')
            if count == 0:
                assert np.all(meta[2:] == 0)
                assert all(np.all(data[key][...] == 0) for key in data if key.startswith('q') or key == 'rank_extras')
            if name in ('first', 'reset_first'):
                assert first_time == last_time and weight == 0
            if name == 'final':
                total = np.float64(0)
                for i in range(1, len(TIMES)):
                    total = np.float64(total + (TIMES[i]-TIMES[i-1]))
                assert first_time == TIMES[0] and last_time == TIMES[-1] and weight == total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('candidate', 'compiler', 'hdf5-root', 'properties-reference', 'mpirun', 'memcheck', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name in FILES:
        path = output / 'source' / name; path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, path); hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    candidate = output / 'candidate'; shutil.copy2(args.candidate.resolve(), candidate)
    receipt = dict(source_sha256=hashes, candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest())
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    reference = build_reference(output, args.compiler.resolve(), args.hdf5_root.resolve())
    commands, runs = [], []
    pattern = re.compile(r'AIR5_MEAN_RUNTIME_PASS rank=(\d+) samples=6 clock=checked duplicate=skipped reset=checked')

    def launch(binary, inputs, directory, topology, mode, source, tool=False):
        directory.mkdir(); ranks = int(np.prod(topology))
        command = [args.mpirun.resolve(), '--bind-to', 'none', '-np', ranks]
        if tool:
            command += [args.memcheck.resolve(), '--tool', 'memcheck', '--leak-check', 'full', '--error-exitcode', 99]
        command += [binary, inputs, directory, *topology, mode, source]
        commands.append(dict(command=list(map(str, command)), log=str(directory.relative_to(output) / 'run.log'),
                             environment={k: v for k, v in ENV.items() if k.startswith('OMPI_MCA_')}))
        (output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        execute(command, directory, 'run.log')
        text = (directory / 'run.log').read_text()
        assert sorted(map(int, pattern.findall(text))) == list(range(ranks))
        if tool:
            assert text.count('ERROR SUMMARY: 0 errors') == ranks and text.count('LEAK SUMMARY: 0 bytes leaked') == ranks
        check_clock(directory, inherited=mode == 'restore-old')

    for topology in [(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2), (2, 2, 1)]:
        case = output / ''.join(map(str, topology)); case.mkdir()
        description = fixture(case / 'inputs', topology, args.properties_reference.resolve())
        (case / 'fixture.json').write_text(json.dumps(description, indent=2) + '\n')
        paths = {name: case / name for name in ('old_cont', 'old_resume', 'cpp_cont', 'cpp_resume', 'cpp_from_old')}
        for name, binary, mode in [('old_cont', reference, 'continuous'), ('old_resume', reference, 'resume'),
                                   ('cpp_cont', candidate, 'continuous'), ('cpp_resume', candidate, 'resume'),
                                   ('cpp_from_old', candidate, 'restore-old')]:
            launch(binary, case / 'inputs', paths[name], topology, mode, paths['old_cont'])
        ratio = 0.; files = 0
        for name in PRODUCTS:
            for a, b, exact in [('old_resume', 'old_cont', True), ('cpp_resume', 'cpp_cont', True),
                                 ('cpp_cont', 'old_cont', False), ('cpp_from_old', 'old_cont', False)]:
                if a == 'cpp_from_old' and name in ('first', 'seed'):
                    continue
                ratio = max(ratio, compare(paths[a] / (name + '.h5'), paths[b] / (name + '.h5'), exact)); files += 1
        runs.append(dict(topology=topology, hdf_comparisons=files, max_original_budget_ratio=ratio,
                         clock_identity='exact', same_backend_restore='word exact', cross_backend_restore='original field budget'))
        (output / 'completed_runs.json').write_text(json.dumps(runs, indent=2) + '\n')
    case = output / '211'
    launch(candidate, case / 'inputs', output / 'memcheck', (2, 1, 1), 'resume', case / 'old_cont', True)
    for name in PRODUCTS:
        compare(output / 'memcheck' / (name + '.h5'), case / 'cpp_cont' / (name + '.h5'), True)
    report = dict(**receipt, runs=runs, max_original_budget_ratio=max(r['max_original_budget_ratio'] for r in runs),
                  numerical_budget='original AIR5 mean per-HDF-dataset 1e-14*max(1,maxabs(reference))',
                  memcheck='NP2 x, context destroy/rebuild and real HDF5 restore, zero errors/leaks',
                  scope='mean sample-clock/role6 HDF5 component; no ordinary AIR5, CFD G2, complete restart bundle or DCU')
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['runs', 'max_original_budget_ratio', 'numerical_budget', 'memcheck', 'scope']}, indent=2))


if __name__ == '__main__':
    main()
