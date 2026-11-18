"""AIR5 compact monitor packet and actual CSV consumer; no CFD advance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
from run_air5_properties_components import read as read_properties
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'gpu_validation'))
from test_air5_flow_monitor import monitor_difference

ROOT = Path(__file__).resolve().parents[2]
BASE = ['src/chemistry_air5_data.F90', 'src/chemistry_core.F90', 'src/chemistry_properties.F90']
FILES = BASE + ['src/chemistry_pure_vt_body.inc', 'src/chemistry_vibrational_difference_body.inc',
    'src/chemistry_monitor.F90', 'src/readwrite.F90', 'src_gpu/chemistry_monitor_gpu.cuf',
    'src_gpu/chemistry_statistics.cu', 'src_gpu/chemistry_statistics.h', 'src_gpu/chemistry_statistics.F90',
    'src_gpu/chemistry_properties_math.cuh', 'src_gpu/chemistry_air5_data.h', 'src_gpu/device_state.cu',
    'src_gpu/device_state_internal.cuh', 'src_gpu/device_state.h', 'src_gpu/device_state.F90',
    'src_gpu/CMakeLists.txt', 'tests/gpu_backend/CMakeLists.txt', 'tests/gpu_backend/air5_monitor_storage.F90',
    'tests/gpu_backend/air5_monitor_probe.F90', 'tests/gpu_backend/run_air5_monitor_components.py',
    'tests/gpu_validation/test_air5_flow_monitor.py']
ENV = dict(os.environ, OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,vader,tcp', OMPI_MCA_osc='pt2pt',
           OMPI_MCA_coll='^cuda,hcoll,ucc', OMPI_MCA_opal_cuda_support='0')
PLANS = [((1, 1, 1), 'raw'), ((1, 1, 1), 'both'), ((1, 1, 1), 'none'),
         ((2, 1, 1), 'both'), ((1, 2, 1), 'both'), ((1, 1, 2), 'curved-probes'),
         ((2, 2, 1), 'both'), ((2, 1, 1), 'profiles'), ((1, 2, 1), 'probes')]


def execute(command, directory, log, env):
    with (directory / log).open('w') as stream:
        subprocess.run(list(map(str, command)), cwd=directory, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def build_reference(output, compiler):
    source = output / 'source'
    work = output / 'reference_build'; work.mkdir()
    original = subprocess.run(['git', 'show', 'HEAD:src/chemistry_monitor.F90'], cwd=ROOT, check=True, capture_output=True).stdout
    (work / 'original_monitor.F90').write_bytes(original)
    gpu_text = (source / 'src_gpu/chemistry_monitor_gpu.cuf').read_text()
    monitor = gpu_text.split('module chemistry_mean_statistics_gpu', 1)[0]
    assert monitor.startswith('module chemistry_monitor_gpu') and monitor.count('end module') == 1
    (work / 'original_monitor_gpu.cuf').write_text(monitor)
    flags = [compiler, '-O2', '-Mpreprocess', '-Mextend', '-cuda', '-gpu=rdc,maxregcount:128',
             '-D_CUDA', '-DASTR_AIR5_MONITOR_REFERENCE', '-I' + str(source / 'src')]
    paths = [source / name for name in BASE]
    for name in ('air5_monitor_storage', 'air5_monitor_probe'):
        shutil.copy2(source / f'tests/gpu_backend/{name}.F90', work / (name + '.cuf'))
    paths += [work / 'air5_monitor_storage.cuf', work / 'original_monitor_gpu.cuf',
              work / 'original_monitor.F90', work / 'air5_monitor_probe.cuf']
    commands, objects = [], []
    for n, path in enumerate(paths):
        obj = work / f'unit{n}.o'
        command = flags + ['-c', path, '-o', obj]
        commands.append(list(map(str, command)))
        (output / 'reference_build_commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        execute(command, work, f'compile{n}.log', ENV)
        objects.append(obj)
    command = flags + objects + ['-o', work / 'reference']
    commands.append(list(map(str, command)))
    (output / 'reference_build_commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    execute(command, work, 'link.log', ENV)
    return work / 'reference', hashlib.sha256(original).hexdigest()


def fixture(directory, topology, mode, properties):
    directory.mkdir()
    status, values = read_properties(properties / 'cpu_all.bin')
    pool = [120, 121, 122, 123, 124, 125, 126, 215, 293, 371]
    assert np.all(status[pool] == 0)
    im, jm, km, h, frames = 8, 7, 6, 5, 4
    shape = (im+2*h+1, jm+2*h+1, km+2*h+1)
    inputs = []
    for rank in range(int(np.prod(topology))):
        coord = np.unravel_index(rank, topology)
        q = np.full(shape + (11, frames), -777.)
        x = np.full(shape + (3,), -888.)
        for i, j, k in np.ndindex((im+1, jm+1, km+1)):
            gi, gj, gk = im*coord[0]+i, jm*coord[1]+j, km*coord[2]+k
            xx, yy, zz = gi*.25, gj*.125, gk*.2
            if mode == 'curved-probes':
                yy += .005*np.sin(xx)*(gj/(jm*topology[1]))**2*np.cos(zz)
            x[i+h, j+h, k+h] = [xx, yy, zz]
            for frame in range(frames):
                q[i+h, j+h, k+h, :, frame] = values[pool[(gi+2*gj+3*gk+3*frame) % len(pool)], :11]
        if mode == 'raw':
            q[h, h, h, 1, :] = -0.
            q[h+1, h+1, h+1, 4, 0] = np.array([0x7ff8000000000042], '<u8').view('<f8')[0]
            q[0, 0, 0, 0, :] = np.array([0x7ff8000000000051], '<u8').view('<f8')[0]
        path = directory / f'input.rank{rank:04d}.bin'
        with path.open('wb') as stream:
            np.array([im, jm, km, h, frames], '<i4').tofile(stream)
            np.array([1e-7, 2.5e-7, 4e-7, 5e-7], '<f8').tofile(stream)
            x.astype('<f8').ravel(order='F').tofile(stream)
            q.astype('<f8').ravel(order='F').tofile(stream)
        inputs.append(hashlib.sha256(path.read_bytes()).hexdigest())
    return dict(topology=topology, mode=mode, source_rows=pool, input_sha256=inputs,
                scope='same physical point states across duplicate nodes; no CFD; raw mode includes NaN/negative-zero copy words')


def check_files(directory, topology, mode):
    files = {p.name for p in (directory / 'monitor').iterdir()}
    names = set()
    if mode in ('both', 'profiles'):
        names |= {'air5_profiles.dat', 'air5_wall.dat'}
    if mode in ('both', 'probes', 'curved-probes'):
        names.add('air5_probes.dat')
    assert files == names, (directory, files, names)
    for name in names:
        values = np.atleast_2d(np.loadtxt(directory / 'monitor' / name))
        assert np.isfinite(values).all()
        rows, cols = (16, 18) if name == 'air5_probes.dat' else ((6*(7*topology[1]+1), 20) if name == 'air5_profiles.dat' else (2*(8*topology[0]+1), 11))
        assert values.shape == (rows, cols), (directory, name, values.shape, rows, cols)
        assert set(values[:, 0]) == ({0., 1., 2., 3.} if name == 'air5_probes.dat' else {0., 2.})
    return sorted(names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('candidate', 'compiler', 'properties-reference', 'mpirun', 'memcheck', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name in FILES:
        path = output / 'source' / name; path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, path); hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    candidate = output / 'candidate'; shutil.copy2(args.candidate.resolve(), candidate)
    reference, reference_hash = build_reference(output, args.compiler.resolve())
    receipt = dict(source_sha256=hashes, original_monitor_sha256=reference_hash,
                   candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest())
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    commands, runs = [], []
    pattern = re.compile(r'AIR5_MONITOR_PASS rank=(\d+) mode=(\S+) packets/fields checked')

    def launch(binary, case, topology, mode, backend, name, tool=False):
        path = case / name; path.mkdir(); (path / 'monitor').mkdir()
        ranks = int(np.prod(topology))
        env = ENV.copy(); env.pop('ASTR_AIR5_FLOW_MONITOR_STRIDE', None)
        if mode in ('both', 'profiles'):
            env['ASTR_AIR5_FLOW_MONITOR_STRIDE'] = '2'
        command = [args.mpirun.resolve(), '--bind-to', 'none', '-np', ranks]
        if tool:
            command += [args.memcheck.resolve(), '--tool', 'memcheck', '--leak-check', 'full', '--error-exitcode', 99]
        command += [binary, case / 'inputs', mode, backend, *topology]
        commands.append(dict(command=list(map(str, command)), log=str(path.relative_to(output) / 'run.log'),
                             environment={k: v for k, v in env.items() if k.startswith(('OMPI_MCA_', 'ASTR_AIR5_FLOW_MONITOR'))}))
        (output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        execute(command, path, 'run.log', env)
        text = (path / 'run.log').read_text()
        assert sorted(int(r) for r, m in pattern.findall(text) if m == mode) == list(range(ranks))
        if tool:
            assert text.count('ERROR SUMMARY: 0 errors') == ranks and text.count('LEAK SUMMARY: 0 bytes leaked') == ranks
        check_files(path, topology, mode)
        return path

    for topology, mode in PLANS:
        case = output / (''.join(map(str, topology)) + '_' + mode); case.mkdir()
        description = fixture(case / 'inputs', topology, mode, args.properties_reference.resolve())
        (case / 'fixture.json').write_text(json.dumps(description, indent=2) + '\n')
        outputs = {}
        for name, binary, backend in [('old_cpu', reference, 'cpu'), ('old_gpu', reference, 'gpu'),
                                       ('cpp_cpu', candidate, 'cpu'), ('cpp_gpu', candidate, 'gpu')]:
            outputs[name] = launch(binary, case, topology, mode, backend, name)
        maximum = 0.
        names = check_files(outputs['cpp_gpu'], topology, mode)
        for name in names:
            assert (outputs['cpp_gpu'] / 'monitor' / name).read_bytes() == (outputs['cpp_cpu'] / 'monitor' / name).read_bytes()
            for backend in ('old_gpu', 'cpp_cpu', 'cpp_gpu'):
                old = np.atleast_2d(np.loadtxt(outputs['old_cpu'] / 'monitor' / name))
                new = np.atleast_2d(np.loadtxt(outputs[backend] / 'monitor' / name))
                assert (outputs[backend] / 'monitor' / name).read_text().splitlines()[0] == (outputs['old_cpu'] / 'monitor' / name).read_text().splitlines()[0]
                copied = 6 if name == 'air5_probes.dat' else (8 if name == 'air5_profiles.dat' else 6)
                assert np.array_equal(old[:, :copied], new[:, :copied])
                error = monitor_difference(old, new, name); assert error <= 2e-10, (case, name, error)
                maximum = max(maximum, error)
        if mode == 'raw':
            for rank in range(int(np.prod(topology))):
                name = f'raw.rank{rank:04d}.bin'
                assert len({(path / name).read_bytes() for path in outputs.values()}) == 1
        runs.append(dict(topology=topology, mode=mode, files=names, max_existing_scaled_error=maximum,
                         same_binary_cpu_gpu='word-exact text; raw packet words exact', conservative_field='unchanged'))
        (output / 'completed_runs.json').write_text(json.dumps(runs, indent=2) + '\n')
    case = output / '211_both'
    checked = launch(candidate, case, (2, 1, 1), 'both', 'gpu', 'cpp_gpu_memcheck', True)
    for name in check_files(checked, (2, 1, 1), 'both'):
        assert (checked / 'monitor' / name).read_bytes() == (case / 'cpp_gpu' / 'monitor' / name).read_bytes()
    report = dict(**receipt, runs=runs, max_existing_scaled_error=max(r['max_existing_scaled_error'] for r in runs),
                  reference_budget='existing test_air5_flow_monitor.monitor_difference <=2e-10; copied metadata/raw packet exact',
                  memcheck='NP2 x actual monitor consumer, zero errors/leaks',
                  scope='point packet, original ownership/CPU decoder/profile/probe/three-point wall diagnostics; no CFD, complete restart or physics')
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['runs', 'max_existing_scaled_error', 'reference_budget', 'memcheck', 'scope']}, indent=2))


if __name__ == '__main__':
    main()
