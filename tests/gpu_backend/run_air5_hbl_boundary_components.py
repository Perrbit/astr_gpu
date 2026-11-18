"""Freeze and compare existing AIR5 HBL outflow and similarity-top mathematics."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from run_air5_boundary_components import BASE, FILES, ROOT, build, compare, read, run
from run_air5_properties_components import read as read_properties
from air5_radau_reference import Air5RadauReference


def inputs(reference):
    statuses, p = read_properties(reference / 'cpu_all.bin')
    valid = np.flatnonzero(np.all(statuses == 0, axis=1) & (p[:, 15] > 500.) & (p[:, 15] < 7500.) &
                           (p[:, 21] > 500.) & (p[:, 21] < 7500.))
    valid = valid[np.linspace(0, len(valid) - 1, min(24, len(valid)), dtype=int)]
    model = Air5RadauReference(ROOT / 'chemMech/air5_kimjo12.json')
    rows, labels = [], []

    def append(one, two, label, preserve, kind=2, x=0., origin=1.):
        a = np.zeros(32)
        a[:11], a[11:22], a[22], a[23], a[24], a[31] = one, two, x, preserve, origin, kind
        rows.append(a)
        labels.append(label)

    for index in valid:
        one = p[index, :11].copy()
        for preserve in [0, 1]:
            prefix = f'{index}-{preserve}'
            for factor in [1., 1.03, .25, 6.]:
                append(one, factor * one, f'outflow-density-{prefix}-{factor}', preserve)
            for temp, tv in [(300., 300.), (8000., 8000.)]:
                two = one.copy()
                two[10] = model.ev_from_tv(two[5:10], tv)
                two[4] = model.q5_from_state(two[0], two[1:4], two[5:10], two[10], temp)
                append(one, two, f'outflow-temperature-{prefix}-{temp}', preserve)
            two = one.copy()
            donor = int(np.argmax(two[5:10]))
            trace = (donor + 1) % 5
            amount = .25 * two[5 + donor]
            two[5 + donor] -= amount
            two[5 + trace] += amount
            two[10] = model.ev_from_tv(two[5:10], 2000.)
            two[4] = model.q5_from_state(two[0], two[1:4], two[5:10], two[10], 3000.)
            append(one, two, f'outflow-composition-{prefix}', preserve)
            bad = one.copy()
            bad[0] = -1.
            append(bad, one, f'outflow-invalid-inner-{prefix}', preserve)
            for x, origin in [(0., 1.), (.25, 1.), (4., 1.), (1e4, .01), (-.01, 1.), (0., 0.)]:
                append(one, one, f'similarity-{prefix}-{x}-{origin}', preserve, 3, x, origin)
    return np.array(rows), labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--compiler', type=Path, required=True)
    parser.add_argument('--properties-reference', type=Path, required=True)
    parser.add_argument('--memcheck', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-run', type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name in FILES + ['tests/gpu_backend/run_air5_hbl_boundary_components.py']:
        dst = out / 'source' / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dst)
        hashes[name] = hashlib.sha256(dst.read_bytes()).hexdigest()
    candidate = out / 'candidate'
    shutil.copy2(args.candidate.resolve(), candidate)
    rows, labels = inputs(args.properties_reference.resolve())
    with (out / 'input.bin').open('wb') as f:
        np.array([len(rows)], '<i4').tofile(f)
        rows.astype('<f8').tofile(f)
    (out / 'labels.json').write_text(json.dumps(labels, indent=2) + '\n')
    if args.reference_run:
        original = args.reference_run.resolve()
        assert (original / 'input.bin').read_bytes() == (out / 'input.bin').read_bytes()
        for name in BASE + ['src/chemistry_pure_vt_body.inc', 'src/chemistry_vibrational_difference_body.inc',
                            'src/chemistry_boundary_state.F90', 'src_gpu/chemistry_core_gpu.cuf',
                            'src_gpu/chemistry_flow_state_gpu.cuf', 'src_gpu/chemistry_boundary_gpu.cuf',
                            'tests/gpu_backend/air5_boundary_reference.F90']:
            assert (original / 'source' / name).read_bytes() == (out / 'source' / name).read_bytes(), name
        for name in ['cpu.bin', 'old_gpu.bin']:
            shutil.copy2(original / name, out / name)
        commands = dict(reused_original_run=str(original), original_sources_and_input_exact=True,
                        sha256={name: hashlib.sha256((original / name).read_bytes()).hexdigest()
                                for name in ['input.bin', 'cpu.bin', 'old_gpu.bin', 'cpu/probe', 'old_gpu/probe']})
    else:
        commands = build(out, args.compiler.resolve())
    products = {}
    for name, binary in [('cpu', out / 'cpu/probe'), ('old_gpu', out / 'old_gpu/probe'), ('cpp', candidate)]:
        if not args.reference_run or name == 'cpp':
            run([binary, out / 'input.bin', out / (name + '.bin')], out, name + '.log')
        products[name] = read(out / (name + '.bin'))
    comparisons = {}
    for name in ['cpu', 'old_gpu']:
        rs, rv = products[name]
        ns, nv = products['cpp']
        assert np.array_equal(rs, ns), (name, 'status mismatch')
        valid = rs == 0
        comparisons[name] = compare((rs[valid], rv[valid]), (ns[valid], nv[valid]),
                                    [label for label, use in zip(labels, valid) if use])
        # Failed helper outputs are scratch, not boundary q; keep the original raw differences.
        comparisons[name]['failed_scratch_rows_differing'] = int(np.sum(np.any(rv[~valid].view('u8') != nv[~valid].view('u8'), axis=1)))
    status, values = products['cpp']
    raw, limited = 0, 0
    for row, q, s in zip(rows, values, status):
        if s != 0:
            continue
        if row[31] == 2:
            high = (4. * row[:11] - row[11:22]) / 3.
            is_raw = np.array_equal(q.view('u8'), high.view('u8'))
            raw += is_raw
            limited += not is_raw
            assert q[0] > 0. and np.all(q[5:10] >= 0.)
            assert abs(sum(q[5:10]) - q[0]) <= 128 * np.finfo(float).eps * max(1., q[0])
        else:
            base = row[:11]
            scale = np.sqrt(row[24] / (row[24] + row[22]))
            assert np.array_equal(q[[0, 1, 3, 5, 6, 7, 8, 9, 10]].view('u8'), base[[0, 1, 3, 5, 6, 7, 8, 9, 10]].view('u8'))
            assert abs(q[2] - base[2] * scale) <= 1e-13 * max(1., abs(base[2]))
    run([candidate, out / 'input.bin', out / 'fields.bin', out / 'cpu.bin'], out, 'fields.log')
    run([args.memcheck.resolve(), '--tool', 'memcheck', '--leak-check', 'full', '--error-exitcode', 99,
         candidate, out / 'input.bin', out / 'memcheck.bin', out / 'cpu.bin'], out, 'memcheck.log')
    text = (out / 'memcheck.log').read_text()
    assert 'ERROR SUMMARY: 0 errors' in text and 'LEAK SUMMARY: 0 bytes leaked' in text
    report = dict(source_sha256=hashes, candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
                  commands=commands, inputs=len(rows), comparisons=comparisons,
                  statuses={str(int(s)): int(np.sum(status == s)) for s in np.unique(status)},
                  outflow_raw=int(raw), outflow_existing_limiter=int(limited),
                  field_observation=(out / 'fields.log').read_text().strip(), memcheck='zero errors/leaks')
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ['source_sha256', 'commands']}, indent=2))


if __name__ == '__main__':
    main()
