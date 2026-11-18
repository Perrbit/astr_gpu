"""Original AIR5 Cartesian characteristic operators, with fixed thermodynamic inputs."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import numpy as np
from run_air5_properties_components import read as read_properties
from run_air5_boundary_components import Air5RadauReference, ROOT

FILES = ['src/chemistry_characteristic.F90', 'src_gpu/chemistry_characteristic_math.cuh',
         'src_gpu/chemistry_properties_math.cuh', 'src_gpu/chemistry_air5_data.h',
         'tests/gpu_backend/air5_characteristic_reference.F90', 'tests/gpu_backend/air5_characteristic_probe.cu',
         'tests/gpu_backend/run_air5_characteristic_components.py', 'tests/gpu_backend/CMakeLists.txt',
         'tests/gpu_validation/air5_characteristic_thermo_probe.F90', 'chemMech/air5_kimjo12.json']


def execute(cmd, cwd, log):
    with (cwd / log).open('w') as f:
        subprocess.run(list(map(str, cmd)), cwd=cwd, stdout=f, stderr=subprocess.STDOUT, check=True)


def flux(q, model):
    z = q[5:10]
    kinetic = np.dot(q[1:4], q[1:4]) / (2. * q[0])
    temperature = (q[4] - q[10] - np.dot(z, model.formation_energy) - kinetic) / np.dot(z, model.cv_tr)
    pressure = np.dot(z, model.gas_constant) * temperature
    v = q[2] / q[0]
    result = q * v
    result[2] += pressure
    result[4] += pressure * v
    return result, pressure, temperature


def inputs(reference):
    statuses, properties = read_properties(reference / 'cpu_all.bin')
    valid = np.flatnonzero(np.all(statuses == 0, axis=1))
    valid = valid[np.linspace(0, len(valid) - 1, 24, dtype=int)]
    model = Air5RadauReference(ROOT / 'chemMech/air5_kimjo12.json')
    rows, labels, tags, derivatives = [], [], [], []
    for index in valid:
        original = properties[index, :11] / properties[index, 0]
        _, p0, _ = flux(original, model)
        beta = np.dot(original[5:10], model.gas_constant) / np.dot(original[5:10], model.cv_tr)
        a0 = math.sqrt((1. + beta) * float(p0))
        for mach in [-1.2, -1., -.2, -1e-8, 0., 1e-8, .2, 1., 1.2]:
            q = original.copy()
            q[2] = mach * a0
            q[4] += .5 * (q[2] * q[2] - original[2] * original[2])
            _, pressure, temperature = flux(q, model)
            u = q[1:4] / q[0]
            sound = abs(q[2]) if abs(mach) == 1. else math.sqrt((1. + beta) * float(pressure) / q[0])
            gradient = np.empty(11)
            gradient[0] = .5 * beta * np.dot(u, u)
            gradient[1:4] = -beta * u
            gradient[4] = beta
            gradient[5:10] = (model.gas_constant - beta * model.cv_tr) * temperature - beta * model.formation_energy
            gradient[10] = -beta
            delta = q * np.arange(1, 12) / 32.
            delta[5] = delta[0] - sum(delta[6:10])
            differential = np.imag(flux(q.astype(complex) + 1e-30j * delta, model)[0]) / 1e-30
            for rate in [0., 4000.]:
                x = np.zeros(96)
                x[:11], x[11:22], x[22:33] = q, .999 * q, gradient
                x[33:44], x[44:55] = differential, .125 * differential
                h = .125
                x[55:66] = q - h * delta + .25 * h * h * delta
                x[66:77] = q - 2. * h * delta + h * h * delta
                x[77:88] = delta
                x[88:93] = h, pressure, sound, rate, 1.1 * sound
                rows.append(x)
                labels.append(f'state-{index}-mach-{mach}-rate-{rate}')
                tags.append('supersonic-out' if mach > 1. else 'general')
                derivatives.append(differential)
        x = rows[-1].copy()
        x[11:22] = x[:11]
        x[33:88] = 0.
        x[55:66], x[66:77] = x[:11], x[:11]
        rows.append(x)
        labels.append(f'uniform-{index}')
        tags.append('uniform')
        derivatives.append(np.zeros(11))
    for label, offset, value in [('zero-spacing', 88, 0.), ('negative-rate', 91, -1.),
                                 ('zero-reference-sound', 92, 0.), ('invalid-target-density', 11, 0.),
                                 ('nonfinite-target', 12, float('nan'))]:
        x = rows[0].copy()
        x[offset] = value
        rows.append(x)
        labels.append(label)
        tags.append('invalid')
        derivatives.append(derivatives[0])
    return np.array(rows), labels, tags, np.array(derivatives)


def read(path):
    with path.open('rb') as f:
        n = int(np.fromfile(f, '<i4', 1)[0])
        flags = np.fromfile(f, '<i4', 5 * n).reshape(n, 5)
        values = np.fromfile(f, '<f8').reshape(n, 66)
    return flags, values


def compare(ref, new, labels):
    rs, rv = ref
    ns, nv = new
    assert np.array_equal(rs, ns), [(labels[i], rs[i].tolist(), ns[i].tolist()) for i in np.flatnonzero(np.any(rs != ns, axis=1))[:3]]
    assert np.isfinite(rv).all() and np.isfinite(nv).all()
    groups = {}
    names = ['normal-high', 'normal-low', 'transport', 'flux-differential', 'normal-high-rhs', 'normal-low-rhs']
    for group, name in enumerate(names):
        a, b = nv[:, 11 * group:11 * (group + 1)], rv[:, 11 * group:11 * (group + 1)]
        tolerance = 1e-13 * np.maximum(1., np.max(abs(b), axis=1))[:, None] + 5e-12 * abs(b)
        ratio = abs(a - b) / tolerance
        i, c = np.unravel_index(np.argmax(ratio), ratio.shape)
        assert ratio[i, c] <= 1., (name, labels[i], c, float(ratio[i, c]))
        groups[name] = float(ratio[i, c])
    return dict(statuses_exact=True, maximum_original_budget_ratio=groups)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--compiler', type=Path, required=True)
    p.add_argument('--properties-reference', type=Path, required=True)
    p.add_argument('--memcheck', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name in FILES:
        dst = out / 'source' / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dst)
        hashes[name] = hashlib.sha256(dst.read_bytes()).hexdigest()
    candidate = out / 'candidate'
    shutil.copy2(args.candidate.resolve(), candidate)
    rows, labels, tags, derivative = inputs(args.properties_reference.resolve())
    with (out / 'input.bin').open('wb') as f:
        np.array([len(rows)], '<i4').tofile(f)
        rows.astype('<f8').tofile(f)
    (out / 'labels.json').write_text(json.dumps(labels, indent=2) + '\n')
    commands = {}
    for backend in ['cpu', 'old_gpu']:
        work = out / backend
        work.mkdir()
        gpu = backend == 'old_gpu'
        flags = [args.compiler.resolve(), '-O2', '-Mpreprocess', '-Mextend', '-I' + str(work)]
        if gpu:
            flags += ['-cuda', '-gpu=rdc,maxregcount:128', '-D_CUDA']
        commands[backend] = []
        sources = [out / 'source/src/chemistry_characteristic.F90', out / 'source/tests/gpu_backend/air5_characteristic_reference.F90']
        if gpu:
            shutil.copy2(sources[1], work / 'reference.cuf')
            sources[1] = work / 'reference.cuf'
        objects = []
        for index, source in enumerate(sources):
            obj = work / f'unit{index}.o'
            cmd = flags + ['-c', source, '-o', obj]
            execute(cmd, work, f'build{index}.log')
            commands[backend].append(list(map(str, cmd)))
            objects.append(obj)
        cmd = flags + objects + ['-o', work / 'probe']
        execute(cmd, work, 'link.log')
        commands[backend].append(list(map(str, cmd)))
    (out / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    products = {}
    for name, binary in [('cpu', out / 'cpu/probe'), ('old_gpu', out / 'old_gpu/probe'), ('cpp', candidate)]:
        execute([binary, out / 'input.bin', out / (name + '.bin')], out, name + '.log')
        products[name] = read(out / (name + '.bin'))
    comparisons = {name: compare(products[name], products['cpp'], labels) for name in ['cpu', 'old_gpu']}
    _, values = products['cpp']
    tolerance = 1e-13 * np.maximum(1., np.max(abs(derivative), axis=1))[:, None] + 5e-12 * abs(derivative)
    derivative_ratio = float(np.max(abs(values[:, 33:44] - derivative) / tolerance))
    assert derivative_ratio <= 1., ('complex-step flux differential', derivative_ratio)
    uniform = supersonic = 0
    for i, tag in enumerate(tags):
        if tag == 'uniform':
            assert np.all(values[i] == 0.), labels[i]
            uniform += 1
        if tag == 'supersonic-out':
            expected = -rows[i, 33:44] + rows[i, 44:55]
            assert np.array_equal(values[i, 22:33], expected), labels[i]
            supersonic += 1
    execute([args.memcheck.resolve(), '--tool', 'memcheck', '--leak-check', 'full', '--error-exitcode', 99,
             candidate, out / 'input.bin', out / 'memcheck.bin'], out, 'memcheck.log')
    text = (out / 'memcheck.log').read_text()
    assert 'ERROR SUMMARY: 0 errors' in text and 'LEAK SUMMARY: 0 bytes leaked' in text
    report = dict(source_sha256=hashes, candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
                  inputs=len(rows), comparisons=comparisons, complex_step_flux_budget_ratio=derivative_ratio,
                  uniform_exact_zero=uniform, supersonic_out_exact_outgoing=supersonic, memcheck='zero errors/leaks')
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'source_sha256'}, indent=2))


if __name__ == '__main__':
    main()
