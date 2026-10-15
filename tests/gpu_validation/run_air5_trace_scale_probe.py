#!/usr/bin/env python3
"""Approved nine-state T1 source-only diagnostic; never a CFD acceptance gate."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from air5_radau_reference import Air5RadauReference
from check_air5_actual_error import compare_case, require_g1
from run_air5_performance_diagnosis import digest
from run_air5_reference_convergence import LEVELS, solve_case
from run_air5_state_workload import TOLERANCES, parse_results

ROOT = Path(__file__).resolve().parents[2]


def trace_scales(model, case):
    initial = np.asarray(case['initial'])
    duration, rho = case['duration'], case['rho']
    source = model.rhs(0., initial, rho, case['momentum'], case['q5'])
    predictor = initial + duration*source
    if not np.isfinite(predictor).all() or np.any(predictor < 0.):
        raise ValueError('T1 source predictor is not admissible; no clipping or fallback')
    next_source = model.rhs(duration, predictor, rho, case['momentum'], case['q5'])
    scale = np.maximum.reduce([abs(initial[:5]), abs(duration*source[:5]),
                               abs(duration*next_source[:5])])
    scale = np.where(scale == 0., rho, np.minimum(rho, scale))
    scale = np.r_[scale, max(abs(initial[5]), 1.)]
    if not np.isfinite(scale).all() or np.any(scale <= 0.):
        raise ValueError('T1 requires finite positive component scales')
    return scale


def absolute_tolerances(scale, factor):
    values = factor*np.asarray(scale)
    if not np.isfinite(values).all() or np.any(values <= 0.):
        raise ValueError('T1 absolute tolerance is nonpositive or nonfinite; no floor')
    return values


def reusable_reference(reference, case):
    return (reference['rho'] == case['rho'] and reference['q5'] == case['q5'] and
            reference['duration'] == case['duration'] and reference['mode'] == case['mode'] and
            np.array_equal(reference['initial'], case['initial']) and
            np.array_equal(reference['momentum'], case['momentum']) and
            [(row['rtol'], row['absolute_scale_factor']) for row in reference['levels']] == list(LEVELS) and
            np.array_equal(reference['reference_atol_scale_used'], case['reference_atol_scale']))


def invoke(probe, backend, input_path, output_path, rtol, factor, tolerance_path=None):
    command = [str(probe), 'batch-cpu' if backend == 'cpu' else 'batch', str(input_path),
               str(output_path), f'{rtol:.17e}', f'{factor:.17e}']
    if tolerance_path is not None:
        command.append(str(tolerance_path))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if backend == 'cpu' else '0',
               ASTR_AIR5_SOURCE_EVALUATION='joint', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    started = time.perf_counter()
    with output_path.with_suffix('.log').open('w') as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
    return command, time.perf_counter()-started


def failure_record(output_path, error):
    result = dict(command=error.cmd, returncode=error.returncode, output=str(output_path),
                  log=str(output_path.with_suffix('.log')), endpoints_not_accepted=True)
    if output_path.exists():
        rows = np.loadtxt(output_path, ndmin=2)
        result.update(output_sha256=digest(output_path), native_identity_status_counts=rows[:, :6].tolist())
    return result


def run(args):
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    args.probe, args.legacy_probe = [p.resolve(strict=True) for p in (args.probe, args.legacy_probe)]
    evidence = args.evidence.resolve(strict=True)
    source_input = evidence/'same_state.in'
    original = json.loads((evidence/'same_state.json').read_text())
    if digest(source_input) != original['input_sha256']:
        raise ValueError('captured source input changed')
    count, duration, mode = np.loadtxt(source_input, max_rows=1)
    rows = np.loadtxt(source_input, skiprows=1, ndmin=2)
    identities = [row for row in original['comparisons'] if row['backend'] == 'cpu']
    if rows.shape != (9, 17) or count != 9 or mode != 0 or len(identities) != 9:
        raise ValueError('T1 approval covers exactly the captured nine coupled-source states')
    mechanism = ROOT/'chemMech/air5_kimjo12.json'
    model = Air5RadauReference(mechanism)
    old_reference_path = evidence/'initial_node_radau.json'
    old_reference_hash = digest(old_reference_path)
    old_reference = json.loads(old_reference_path.read_text())
    input_path = args.output/'states.in'
    input_path.write_bytes(source_input.read_bytes())
    report = dict(status='running', scope='approved T1 source probes only; not filter/full-flow/performance acceptance',
        input_sha256=digest(source_input), probe_sha256=digest(args.probe),
        legacy_probe_sha256=digest(args.legacy_probe), mechanism_sha256=digest(mechanism),
        reference_code_sha256=digest(ROOT/'tests/gpu_validation/air5_radau_reference.py'),
        reused_reference_sha256=old_reference_hash,
        driver_sha256=digest(Path(__file__)), references=[], runs=[])
    try:
        for index, (row, identity) in enumerate(zip(rows, identities)):
            case = dict(name=f"{identity['source']}-node-"+'-'.join(map(str, identity['node_zyx'][::-1])),
                rho=row[0], momentum=row[1:4], q5=row[4], initial=row[5:11], carry=row[11:17],
                duration=float(duration), mode='coupled', provenance=identity)
            scale = trace_scales(model, case)
            case['reference_atol_scale'] = scale
            if reusable_reference(old_reference, case):
                reference = dict(old_reference, name=case['name'], provenance=identity, carry=case['carry'].tolist())
                reference['reused_artifact'] = str(old_reference_path)
                reference['reused_sha256'] = digest(old_reference_path)
            else:
                reference = solve_case(model, case)
            reference['trial_absolute_scale'] = scale.tolist()
            report['references'].append(reference)
            require_g1(compare_case(model, reference, reference['endpoints'][-1], np.zeros(6)))
            print('reference', index+1, case['name'], flush=True)
        for tolerance in ('baseline', 'loose', 'tight'):
            rtol, factor = TOLERANCES[tolerance]
            for backend in ('cpu', 'gpu'):
                for scale_mode in ('density', 'birth'):
                    stem = args.output/f'{backend}_{tolerance}_{scale_mode}'
                    tolerance_path = None
                    if scale_mode == 'birth':
                        tolerance_path = stem.with_suffix('.atol')
                        with tolerance_path.open('w') as stream:
                            stream.write('9\n')
                            np.savetxt(stream, [absolute_tolerances(row['trial_absolute_scale'], factor)
                                               for row in report['references']], fmt='%.17e')
                    output_path = stem.with_suffix('.out')
                    try:
                        command, elapsed = invoke(args.probe, backend, input_path, output_path,
                                                  rtol, factor, tolerance_path)
                    except subprocess.CalledProcessError as exc:
                        report['failed_probe'] = failure_record(output_path, exc)
                        raise
                    work = parse_results(output_path, 9)
                    values = np.loadtxt(output_path, ndmin=2)
                    entry = dict(backend=backend, tolerance=tolerance, scale_mode=scale_mode,
                        rtol=rtol, atol_factor=factor, command=command, work=work,
                        cli_wall_seconds_including_startup=elapsed, output_sha256=digest(output_path), cases=[])
                    report['runs'].append(entry)
                    for reference, value in zip(report['references'], values):
                        result = compare_case(model, reference, value[6:12], value[12:18])
                        result['work_counts'] = value[2:6].astype(int).tolist()
                        entry['cases'].append(result)
                        require_g1(result)
                    if scale_mode == 'density':
                        legacy_output = stem.with_suffix('.legacy.out')
                        invoke(args.legacy_probe, backend, input_path, legacy_output, rtol, factor)
                        if legacy_output.read_bytes() != output_path.read_bytes():
                            raise ValueError('legacy batch changed after test-only vector input')
                        entry['legacy_exact'] = True
                    print(backend, tolerance, scale_mode, work, flush=True)
        if digest(source_input) != original['input_sha256'] or digest(old_reference_path) != old_reference_hash:
            raise ValueError('read-only source evidence changed')
        report['status'] = 't1-source-probes-passed-not-filter-or-full-flow-acceptance'
    except BaseException as exc:
        report.update(status='failed', error=str(exc))
        raise
    finally:
        (args.output/'trace_scale.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('evidence', 'output', 'probe', 'legacy-probe'):
        parser.add_argument('--'+name, type=Path, required=True)
    run(parser.parse_args())
