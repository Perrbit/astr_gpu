"""Frozen pre-limiter top assembly and real C++ device-field writes, not full CFD."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
from run_air5_boundary_components import BASE, ROOT
from run_air5_flux_components import extract
from run_air5_properties_components import read as read_properties

FILES = BASE + ['src/chemistry_pure_vt_body.inc', 'src/chemistry_vibrational_difference_body.inc',
    'src/chemistry_characteristic.F90', 'src/chemistry_solver.F90', 'src_gpu/chemistry_solver_gpu.cuf',
    'src_gpu/chemistry_core_gpu.cuf', 'src_gpu/chemistry_flow_state_gpu.cuf',
    'src_gpu/chemistry_boundary.cu', 'src_gpu/chemistry_boundary.h', 'src_gpu/chemistry_boundary.F90',
    'src_gpu/chemistry_boundary_math.cuh', 'src_gpu/chemistry_characteristic_math.cuh',
    'src_gpu/chemistry_properties_math.cuh', 'src_gpu/chemistry_air5_data.h',
    'src_gpu/chemistry_convection.cu', 'src_gpu/CMakeLists.txt', 'tests/gpu_backend/CMakeLists.txt',
    'tests/gpu_backend/air5_characteristic_state_reference.F90',
    'tests/gpu_backend/air5_characteristic_field_probe.cu', 'tests/gpu_backend/run_air5_characteristic_fields.py']
ENV = dict(os.environ, OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,vader,tcp', OMPI_MCA_osc='pt2pt',
           OMPI_MCA_coll='^cuda,hcoll,ucc', OMPI_MCA_opal_cuda_support='0')


def execute(cmd, cwd, log):
    with (cwd / log).open('w') as f:
        subprocess.run(list(map(str, cmd)), cwd=cwd, env=ENV, stdout=f, stderr=subprocess.STDOUT, check=True)


def inputs(reference):
    statuses, p = read_properties(reference / 'cpu_all.bin')
    valid = np.flatnonzero(np.all(statuses == 0, axis=1))
    valid = valid[np.linspace(0, len(valid) - 1, 24, dtype=int)]
    rows, labels = [], []
    gas = 8.31446261815324 / np.array([.028, .032, .014, .016, .030])
    cv = np.array([2.5, 2.5, 1.5, 1.5, 2.5]) * gas
    for index in valid:
        base = p[index, :11]
        y = base[5:10] / base[0]
        sound = math.sqrt((1. + np.dot(y, gas) / np.dot(y, cv)) * p[index, 22] / base[0])
        for preserve in [0, 1]:
            for mach in [-1.2, -1., -.2, -1e-8, 0., 1e-8, .2, 1., 1.2]:
                q = base.copy()
                q[2] = q[0] * mach * sound
                q[4] += (q[2] * q[2] - base[2] * base[2]) / (2. * q[0])
                for rate in [0., 4000.]:
                    x = np.zeros(96)
                    x[:11], x[11:22] = q, .999 * q
                    x[55:66] = (1. - 1. / 64. + 1. / 4096.) * q
                    x[66:77] = (1. - 1. / 32. + 1. / 1024.) * q
                    x[88], x[91], x[92], x[93], x[94] = .125, rate, 1.1 * sound, preserve, .75 + .25 * (len(rows) % 4)
                    rows.append(x)
                    labels.append(f'q-{index}-cache-{preserve}-mach-{mach}-rate-{rate}')
            x = rows[-1].copy()
            x[11:22] = x[55:66] = x[66:77] = x[:11]
            rows.append(x)
            labels.append(f'uniform-{index}-{preserve}')
    return p[0, 109:114], np.array(rows), labels


def read(path):
    with path.open('rb') as f:
        n = int(np.fromfile(f, '<i4', 1)[0])
        status = np.fromfile(f, '<i4', n)
        out = np.fromfile(f, '<f8').reshape(n, 22)
    return status, out


def compare(a, b, labels):
    sa, va = a
    sb, vb = b
    assert np.array_equal(sa, sb), 'original status mismatch'
    assert np.all(sa == 0) and np.isfinite(va).all() and np.isfinite(vb).all()
    maxima = []
    for group in [0, 1]:
        x, y = va[:, 11*group:11*(group+1)], vb[:, 11*group:11*(group+1)]
        tolerance = 1e-13 * np.maximum(1., np.max(abs(y), axis=1))[:, None] + 5e-12 * abs(y)
        ratio = abs(x-y) / tolerance
        i, c = np.unravel_index(np.argmax(ratio), ratio.shape)
        assert ratio[i, c] <= 1., (labels[i], group, c, float(ratio[i, c]))
        maxima.append(float(ratio[i, c]))
    return dict(statuses_exact=True, high_low_original_budget_ratios=maxima)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--compiler', type=Path, required=True)
    p.add_argument('--mpirun', type=Path, required=True)
    p.add_argument('--properties-reference', type=Path, required=True)
    p.add_argument('--memcheck', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference-run', type=Path)
    p.add_argument('--diagnose-rows', nargs='+', type=int)
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    hashes = {}
    original_files = BASE + ['src/chemistry_pure_vt_body.inc', 'src/chemistry_vibrational_difference_body.inc',
        'src/chemistry_characteristic.F90', 'src/chemistry_solver.F90', 'src_gpu/chemistry_solver_gpu.cuf',
        'src_gpu/chemistry_core_gpu.cuf', 'src_gpu/chemistry_flow_state_gpu.cuf']
    for name in FILES:
        dst = out / 'source' / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        source_root = args.reference_run.resolve() / 'source' if args.diagnose_rows and args.reference_run and name in original_files else ROOT
        shutil.copy2(source_root / name, dst)
        hashes[name] = hashlib.sha256(dst.read_bytes()).hexdigest()
    candidate = out / 'candidate'
    shutil.copy2(args.candidate.resolve(), candidate)
    (out / 'source_receipt.json').write_text(json.dumps(dict(source_sha256=hashes,
        candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest()), indent=2) + '\n')
    if args.diagnose_rows:
        assert args.reference_run, 'diagnostics require an explicit frozen reference run'
        reference_run = args.reference_run.resolve()
        labels = json.loads((reference_run / 'labels.json').read_text())
        with (reference_run / 'input.bin').open('rb') as f:
            n = int(np.fromfile(f, '<i4', 1)[0]);floor = np.fromfile(f, '<f8', 5)
            rows = np.fromfile(f, '<f8').reshape(n, 96)
        assert all(0 <= i < n for i in args.diagnose_rows)
        labels = [labels[i] for i in args.diagnose_rows];rows = rows[args.diagnose_rows]
    else:
        assert not args.reference_run, 'reference-run is only used for the explicit diagnostic mode'
        floor, rows, labels = inputs(args.properties_reference.resolve())
    with (out / 'input.bin').open('wb') as f:
        np.array([len(rows)], '<i4').tofile(f)
        floor.astype('<f8').tofile(f)
        rows.astype('<f8').tofile(f)
    (out / 'labels.json').write_text(json.dumps(labels, indent=2) + '\n')
    commands = {}
    for backend in ['cpu', 'old_gpu']:
        work = out / backend
        work.mkdir()
        gpu = backend == 'old_gpu'
        original = out / 'source' / ('src_gpu/chemistry_solver_gpu.cuf' if gpu else 'src/chemistry_solver.F90')
        names = ['record_air5_status', 'air5_characteristic_top_rhs_kernel'] if gpu else ['air5_characteristic_top_rhs']
        body = '\n'.join(extract(original.read_text(), name) for name in names)
        (work / 'frozen_top_rhs_original.inc').write_text(body)
        if args.diagnose_rows:
            assert body.count('gradient(11)=-beta') == 1
            dest, density = ('diagnostic_d', 'rho') if gpu else ('diagnostic', 'density')
            observation = (f'\n    {dest}(1:11)=[{density},velocity,temperature,tv,pressure,beta,sound,'
                           f'velocity(2)-sound,velocity(2)+sound]\n    {dest}(12:22)=gradient')
            body = body.replace('gradient(11)=-beta', 'gradient(11)=-beta' + observation)
        (work / 'frozen_top_rhs.inc').write_text(body)
        flags = [args.compiler.resolve(), '-O2', '-Mpreprocess', '-Mextend', '-I' + str(work), '-I' + str(out/'source/src')]
        if gpu:
            flags += ['-cuda', '-gpu=rdc,maxregcount:128', '-D_CUDA']
        names = BASE + (['src_gpu/chemistry_core_gpu.cuf', 'src_gpu/chemistry_flow_state_gpu.cuf'] if gpu else [])
        names += ['src/chemistry_characteristic.F90', 'tests/gpu_backend/air5_characteristic_state_reference.F90']
        objects = []
        commands[backend] = []
        for i, name in enumerate(names):
            source = out / 'source' / name
            if gpu and name.endswith('state_reference.F90'):
                shutil.copy2(source, work / 'reference.cuf')
                source = work / 'reference.cuf'
            obj = work / f'unit{i}.o'
            cmd = flags + ['-c', source, '-o', obj]
            execute(cmd, work, f'build{i}.log')
            objects.append(obj)
            commands[backend].append(list(map(str, cmd)))
        cmd = flags + objects + ['-o', work / 'probe']
        execute(cmd, work, 'link.log')
        commands[backend].append(list(map(str, cmd)))
    (out / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    products = {}
    for name in ['cpu', 'old_gpu']:
        cmd = [args.mpirun.resolve(), '--bind-to', 'none', '-np', 1, out / name / 'probe',
               out / 'input.bin', out / (name + '.bin')]
        if args.diagnose_rows:
            cmd.append(out / (name + '_diagnostic.bin'))
        execute(cmd, out, name + '.log')
        products[name] = read(out / (name + '.bin'))
    if args.diagnose_rows:
        diagnostic = {}
        for name in ['cpu', 'old_gpu']:
            status, values = read(reference_run / (name + '.bin'))
            assert np.array_equal(status[args.diagnose_rows], products[name][0]), name
            assert np.array_equal(values[args.diagnose_rows].view('u8'), products[name][1].view('u8')), (name, 'instrumentation changed RHS')
            with (out / (name + '_diagnostic.bin')).open('rb') as f:
                n = int(np.fromfile(f, '<i4', 1)[0]);d = np.fromfile(f, '<f8').reshape(n, 22)
            diagnostic[name] = [dict(label=label, rho=float(a[0]), velocity=a[1:4].tolist(), temperature=float(a[4]), tv=float(a[5]),
                                     pressure=float(a[6]), beta=float(a[7]), sound=float(a[8]), lambda_minus=float(a[9]), lambda_plus=float(a[10]),
                                     incoming_acoustic=[bool(a[9]<0), bool(a[10]<0)], pressure_gradient=a[11:22].tolist()) for label, a in zip(labels, d)]
        report = dict(source_sha256=hashes, commands=commands, reference_run=str(reference_run), rows=args.diagnose_rows,
                      instrumented_rhs_words_match_original=True, diagnostics=diagnostic, scope='diagnosis only; no C++ field acceptance run')
        (out / 'diagnosis.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k:v for k,v in report.items() if k not in ['source_sha256', 'commands']}, indent=2))
        return
    comparison = compare(products['cpu'], products['old_gpu'], labels)
    (out / 'reference_comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
    execute([candidate, out / 'input.bin', out / 'cpu.bin', out / 'old_gpu.bin', out / 'fields.json'], out, 'fields.log')
    execute([args.memcheck.resolve(), '--tool', 'memcheck', '--leak-check', 'full', '--error-exitcode', 99,
             candidate, out / 'input.bin', out / 'cpu.bin', out / 'old_gpu.bin', out / 'memcheck.json', 'sample'], out, 'memcheck.log')
    text = (out / 'memcheck.log').read_text()
    assert 'ERROR SUMMARY: 0 errors' in text and 'LEAK SUMMARY: 0 bytes leaked' in text
    report = dict(source_sha256=hashes, candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
                  inputs=len(rows), original_reference_comparison=comparison,
                  fields=json.loads((out / 'fields.json').read_text()), memcheck_sample=json.loads((out / 'memcheck.json').read_text()),
                  memcheck='zero errors/leaks in explicit sampled field matrix',
                  scope='pre-limiter normal top assembly; constant target supplier; no full CFD or MPI halo')
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'source_sha256'}, indent=2))


if __name__ == '__main__':
    main()
