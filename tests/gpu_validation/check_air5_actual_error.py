#!/usr/bin/env python3
"""U2/U3 zero-dimensional actual-error gate approved by the user on 2026-10-13."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np

from air5_radau_reference import Air5RadauReference
from run_air5_performance_diagnosis import digest
from run_air5_state_workload import parse_results, TOLERANCES

ROOT = Path(__file__).resolve().parents[2]
MODES = {'coupled': 0, 'chemical': 1, 'vt': 2}
NAMES = ('Y_N2', 'Y_O2', 'Y_N', 'Y_O', 'Y_NO', 'ev_J_per_kg', 'T_K', 'Tv_K')


def g1_scales(case):
    initial_ev = case['initial'][5] / case['rho']
    return (np.array([1e-12] * 5 + [1e-10 * max(abs(initial_ev), 1.), 1e-5, 1e-5]),
            np.array([1e-6] * 5 + [1e-7] * 3))


def error_values(actual, target, absolute, relative):
    values = [np.asarray(v, dtype=float) for v in (actual, target, absolute, relative)]
    actual, target, absolute, relative = values
    if any(v.shape != target.shape or not np.isfinite(v).all() for v in values):
        raise ValueError('error comparison requires matching finite components')
    if np.any(absolute <= 0.) or np.any(relative < 0.):
        raise ValueError('error scales require positive absolute and nonnegative relative values')
    difference = abs(actual - target)
    relative_error = [float(d / abs(t)) if t != 0. else None
                      for d, t in zip(difference, target)]
    return dict(actual=actual.tolist(), target=target.tolist(),
                absolute_error=difference.tolist(), relative_error=relative_error,
                normalized_error=(difference / (absolute + relative * abs(target))).tolist())


def state_metrics(model, case, state):
    state = np.asarray(state, dtype=float)
    if state.shape != (6,) or not np.isfinite(state).all() or np.any(state < 0.):
        raise ValueError('endpoint must contain six finite nonnegative values')
    rho, momentum, q5 = case['rho'], case['momentum'], case['q5']
    temperature, tv = model.temperatures(rho, momentum, q5, state[:, None])
    temperature, tv = float(temperature[0]), float(tv[0])
    pressure = model.pressure(state[:5], temperature)
    observations = np.r_[state / rho, temperature, tv]
    drift = model.invariant_drifts(np.asarray(case['initial']), state)
    closure = abs(math.fsum(state[:5] / rho) - 1.)
    reconstructed = model.q5_from_state(rho, momentum, state[:5], state[5], temperature)
    energy_error = abs(reconstructed - q5) / max(abs(q5), 1.)
    invariants = dict(relative_mass_drift=drift[0], relative_N_drift=drift[1],
        relative_O_drift=drift[2], species_closure=closure, q5_eos_roundtrip=energy_error,
        pressure_pa=pressure)
    within = (max(drift) <= 1e-14 and closure <= 128 * np.finfo(float).eps
              and energy_error <= 32 * np.finfo(float).eps)
    return observations, invariants, bool(within)


def compare_case(model, case, high, carry):
    high, carry = np.asarray(high, dtype=float), np.asarray(carry, dtype=float)
    if high.shape != (6,) or carry.shape != (6,) or not np.isfinite(carry).all():
        raise ValueError('endpoint and carry require six finite components')
    endpoints = np.asarray(case['endpoints'], dtype=float)
    if endpoints.shape != (3, 6):
        raise ValueError('reference requires exactly three endpoint levels')
    absolute, relative = g1_scales(case)
    target, target_invariants, reference_invariants_ok = state_metrics(model, case, endpoints[-1])
    preceding, _, _ = state_metrics(model, case, endpoints[-2])
    margin = error_values(preceding, target, absolute, relative)
    comparisons = {}
    for name, state in (('q', high), ('q-minus-carry', high - carry)):
        observations, invariants, invariants_ok = state_metrics(model, case, state)
        errors = error_values(observations, target, absolute, relative)
        # Preserve a small carry in the state/reference difference even when
        # rounding high-carry back to FP64 would absorb it. Thermodynamics stays FP64.
        if name == 'q-minus-carry':
            difference = np.array([abs(math.fsum((float(h), -float(c), -float(t)))) / case['rho']
                                   for h, c, t in zip(high, carry, endpoints[-1])])
            errors['absolute_error'][:6] = difference.tolist()
            errors['relative_error'][:6] = [float(d / abs(t)) if t != 0. else None
                                          for d, t in zip(difference, target[:6])]
            errors['normalized_error'][:6] = (difference / (absolute[:6] + relative[:6] * abs(target[:6]))).tolist()
        comparisons[name] = dict(**errors, invariants=invariants,
            within_g1_invariants=invariants_ok,
            within_g1_error=bool(max(errors['normalized_error']) <= 1.))
    return dict(name=case['name'], source_mode=case['mode'], duration=case['duration'],
        provenance=case['provenance'], components=list(NAMES),
        g1_absolute_scales=absolute.tolist(), g1_relative_scales=relative.tolist(),
        reference_margin=margin, reference_invariants=target_invariants,
        within_g1_reference_margin=bool(max(margin['normalized_error']) <= .01
                                       and reference_invariants_ok),
        endpoint=high.tolist(), carry=carry.tolist(), comparisons=comparisons)


def require_g1(result):
    if not result['within_g1_reference_margin']:
        raise ValueError(f"{result['name']}: unresolved independent reference")
    for representation, comparison in result['comparisons'].items():
        if not comparison['within_g1_invariants']:
            raise ValueError(f"{result['name']}: {representation} violates physical invariants")
        if not comparison['within_g1_error']:
            raise ValueError(f"{result['name']}: {representation} exceeds G1 error budget")


def run(args):
    if getattr(args,'backend','gpu')=='cpu' and args.source_evaluation!='joint':
        raise ValueError('CPU probe supports joint source evaluation only')
    reference = json.loads(args.reference.read_text())
    if reference['status'] != 'reference-convergence-collected-awaiting-G1':
        raise ValueError('require successful independent reference preparation')
    mechanism = ROOT / 'chemMech/air5_kimjo12.json'
    reference_code = ROOT / 'tests/gpu_validation/air5_radau_reference.py'
    if (digest(mechanism) != reference['mechanism_sha256'] or
            digest(reference_code) != reference['reference_code_sha256']):
        raise ValueError('reference mechanism or implementation changed')
    model = Air5RadauReference(mechanism)
    selection=getattr(args,'case_names',None)
    if selection is not None and (not selection or set(selection)-{case['name'] for case in reference['cases']}):
        raise ValueError('unknown or empty G1 case selection')
    cases=[case for case in reference['cases'] if selection is None or case['name'] in selection]
    low_activity=getattr(args,'low_activity','off')
    threshold=0.0 if low_activity=='off' else float(low_activity)
    if not np.isfinite(threshold) or threshold<0:
        raise ValueError('invalid activity threshold')
    args.output.mkdir(parents=True, exist_ok=False)
    rtol, atol_factor = TOLERANCES[args.tolerance]
    report = dict(status='running', scope='approved 11-state zero-dimensional G1 gate; not full-flow acceptance',
        g1_approved=True, approval_date='2026-10-13', reference_sha256=digest(args.reference),
        mechanism_sha256=digest(mechanism), reference_code_sha256=digest(reference_code),
        checker_sha256=digest(Path(__file__)), probe_sha256=digest(args.probe),
        probe_source_sha256=digest(ROOT / 'tests/gpu_validation/chemistry_ros2_compensation_gpu_probe.cuf'),
        cpu_integrator_sha256=digest(ROOT / 'src/chemistry_kinetics.F90'),
        gpu_integrator_sha256=digest(ROOT / 'src_gpu/chemistry_kinetics_gpu.cuf'),
        tolerance=args.tolerance, rtol=rtol, atol_factor=atol_factor, source_evaluation=args.source_evaluation,
        chemistry_calls=args.segments, step_history=args.step_history,
        case_selection=[case['name'] for case in cases],complete_11_state_gate=len(cases)==len(reference['cases']),
        activity_threshold=threshold,backend=getattr(args,'backend','gpu'),
        carry_convention='reference starts from rounded q; check q and q-carry separately. '
            'State error uses fsum; recovered thermodynamics remains FP64.', cases=[])
    try:
        for case in cases:
            stem = args.output / case['name']
            row = np.r_[case['rho'], case['momentum'], case['q5'], case['initial'], case['carry']]
            input_path, output_path = stem.with_suffix('.in'), stem.with_suffix('.out')
            with input_path.open('w') as stream:
                stream.write(f"1 {case['duration']:.17e} {MODES[case['mode']]}\n")
                np.savetxt(stream, row[None, :], fmt='%.17e')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu),
                       ASTR_AIR5_SOURCE_EVALUATION=args.source_evaluation,ASTR_AIR5_LOW_ACTIVITY=str(low_activity))
            command = [str(args.probe.resolve()), 'batch-cpu' if getattr(args,'backend','gpu')=='cpu' else 'batch',
                       str(input_path.resolve()), str(output_path.resolve()),
                       f'{rtol:.17e}', f'{atol_factor:.17e}', str(args.segments),
                       'history' if args.step_history else 'reset']
            with stem.with_suffix('.log').open('w') as log:
                subprocess.run(command,
                    env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
            parse_results(output_path, 1)
            result = np.loadtxt(output_path, ndmin=2)[0]
            diagnostics = compare_case(model, case, result[6:12], result[12:18])
            diagnostics.update(work_counts=result[2:6].astype(int).tolist(),
                suggested_step=float(result[18]), command=command,
                input_sha256=digest(input_path), output_sha256=digest(output_path))
            activity_rows=[line for line in output_path.read_text().splitlines() if line.startswith('# AIR5_ACTIVITY ')]
            if activity_rows:
                activity=dict(word.split('=',1) for word in activity_rows[0].split()[2:])
                diagnostics['activity_calls']={key:int(activity[key]) for key in ('skipped_calls','total_calls')}
            report['cases'].append(diagnostics)
            require_g1(diagnostics)
            print(case['name'], max(diagnostics['comparisons']['q']['normalized_error']), flush=True)
        report['status'] = 'g1-actual-error-passed'
    except BaseException as exc:
        report.update(status='failed', failed_case=case['name'], error=str(exc))
        raise
    finally:
        (args.output / 'actual_error.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--probe', type=Path, default=ROOT / 'build_gpu_probe/bin/chemistry_ros2_compensation_gpu_probe')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--tolerance', choices=tuple(TOLERANCES), default='baseline')
    parser.add_argument('--source-evaluation', choices=('joint', 'source-only'), default='joint')
    parser.add_argument('--low-activity',default='off',help='off or a fixed positive screening threshold')
    parser.add_argument('--backend',choices=('cpu','gpu'),default='gpu')
    parser.add_argument('--case-names',nargs='+',help='explicit calibration or holdout subset; default is all 11')
    parser.add_argument('--segments', type=int, choices=(1, 4), default=1,
                        help='equal chemistry calls over the same total reference duration')
    parser.add_argument('--step-history', action='store_true')
    run(parser.parse_args())
