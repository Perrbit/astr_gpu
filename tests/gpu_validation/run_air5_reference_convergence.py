#!/usr/bin/env python3
"""U2 independent zero-dimensional reference preparation; no candidate ranking."""
import argparse
import json
from pathlib import Path

import numpy as np

from air5_radau_reference import Air5RadauReference
from run_air5_performance_diagnosis import digest

ROOT = Path(__file__).resolve().parents[2]
LEVELS = ((1e-9, 1e-12), (1e-11, 1e-14), (1e-12, 1e-15))


def synthetic_states(model):
    air = np.array([.767, .233, 0., 0., 0.])
    mixed = np.array([.55, .15, .10, .12, .08])
    trace = np.array([.767, .233, 1e-20, 1e-20, 1e-20])
    specifications = [('cold-air', 800., 800., air, 1e-8, 'coupled'),
        ('exact-zero-air', 3000., 1500., air, 5e-10, 'coupled'),
        ('hot-dissociated', 6000., 1000., mixed, 2e-8, 'coupled'),
        ('hot-chemical', 6000., 1000., mixed, 2e-8, 'chemical'),
        ('hot-vt', 6000., 1000., mixed, 2e-8, 'vt'),
        ('trace-air', 3000., 1500., trace, 5e-10, 'coupled'),
        ('vt-equilibrium', 4000., 4000., mixed, 1e-8, 'vt')]
    for name, temperature, tv, fractions, duration, mode in specifications:
        rho = 20000. / (temperature * (fractions @ model.gas_constant))
        partial = rho * fractions
        momentum = rho * np.array([2000., 0., 0.])
        ev = model.ev_from_tv(partial, tv)
        q5 = model.q5_from_state(rho, momentum, partial, ev, temperature)
        initial = np.r_[partial, ev]
        case = dict(name=name, rho=rho, momentum=momentum, q5=q5,
                    initial=initial, carry=np.zeros(6), duration=duration,
                    mode=mode, provenance='explicit synthetic zero-dimensional state')
        if name == 'cold-air':
            # Reference-only scales resolve newborn species far below rho*1e-15.
            source = model.rhs(0., initial, rho, momentum, q5)
            predictor = initial + duration * source
            next_source = model.rhs(duration, predictor, rho, momentum, q5)
            case['reference_atol_scale'] = np.r_[np.maximum.reduce([
                initial[:5], abs(duration * source[:5]), abs(duration * next_source[:5])]),
                max(abs(ev), 1.)]
            case['reference_scale_origin'] = 'initial species and two RHS production estimates; no state clipping'
        yield case


def workload_states(workload):
    report = json.loads((workload / 'workload.json').read_text())
    if report['status'] != 'bounded-state-workload-collected':
        raise ValueError('require a successful immutable workload replay')
    case = report['cases']['gpu_np2']
    for rank, regions in case['ranks'].items():
        entry = regions['interior']
        stem = workload / f'gpu_np2_rank{rank}_interior'
        if digest(stem.with_suffix('.in')) != entry['input_sha256'] or digest(stem.with_suffix('.out')) != entry['output_sha256']:
            raise ValueError('workload replay files changed')
        inputs = np.loadtxt(stem.with_suffix('.in'), skiprows=1, ndmin=2)
        outputs = np.loadtxt(stem.with_suffix('.out'), ndmin=2)
        attempts = outputs[:, 2] + outputs[:, 3]
        # Deterministic extrema, not a search for favorable solver tolerances.
        for label, index in (('min-work', int(np.argmin(attempts))), ('max-work', int(np.argmax(attempts)))):
            row = inputs[index]
            yield dict(name=f'rank{rank}-{label}', rho=row[0], momentum=row[1:4], q5=row[4],
                       initial=row[5:11], carry=row[11:17], duration=case['duration'], mode='coupled',
                       provenance=dict(checkpoint=case['checkpoint'], step=case['step'], rank=int(rank),
                           replay_index=index + 1, input_sha256=entry['input_sha256']))


def solve_case(model, case):
    rho, initial = case['rho'], case['initial']
    scale = np.asarray(case.get('reference_atol_scale',
        np.r_[np.full(5, rho), max(abs(initial[5]), 1.)]))
    solutions, costs = [], []
    for rtol, factor in LEVELS:
        solution = model.integrate_radau(rho, case['momentum'], case['q5'], initial,
            case['duration'], rtol=rtol, atol=factor * scale, source_mode=case['mode'])
        if solution.t[-1] != case['duration']:
            raise ValueError('reference did not reach the requested time')
        solutions.append(solution.y[:, -1])
        costs.append(dict(rtol=rtol, absolute_scale_factor=factor,
                          rhs=solution.nfev, jacobian=solution.njev, lu=solution.nlu))
    temperatures = [np.array(model.temperatures(rho, case['momentum'], case['q5'], v[:, None])).ravel()
                    for v in solutions]
    final = solutions[-1]
    reconstructed = model.q5_from_state(rho, case['momentum'], final[:5], final[5], temperatures[-1][0])
    return dict(**{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in case.items()},
        levels=costs, reference_atol_scale_used=scale.tolist(), endpoints=[v.tolist() for v in solutions],
        temperatures=[v.tolist() for v in temperatures],
        adjacent_absolute_differences=[abs(b - a).tolist() for a, b in zip(solutions, solutions[1:])],
        adjacent_temperature_differences=[abs(b - a).tolist() for a, b in zip(temperatures, temperatures[1:])],
        tight_invariant_drifts=list(model.invariant_drifts(initial, final)),
        tight_species_closure=abs(float(final[:5].sum() / rho - 1)),
        tight_q5_roundtrip=abs(reconstructed - case['q5']) / max(abs(case['q5']), 1.),
        status='reference-convergence-collected-not-G1-approved')


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    mechanism = ROOT / 'chemMech/air5_kimjo12.json'
    model = Air5RadauReference(mechanism)
    report = dict(status='running', mechanism_sha256=digest(mechanism),
        reference_code_sha256=digest(ROOT / 'tests/gpu_validation/air5_radau_reference.py'),
        scope='independent 0D Radau convergence preparation, not an error bound or a performance ranking',
        carry_convention='reference starts from rounded q; input carry is preserved in metadata. '
            'G1 must freeze q versus q-carry endpoint comparison before ranking compensated candidates.', cases=[])
    cases = [*synthetic_states(model), *workload_states(args.workload)]
    try:
        for case in cases:
            print(case['name'], flush=True)
            report['cases'].append(solve_case(model, case))
        report['status'] = 'reference-convergence-collected-awaiting-G1'
    except BaseException as exc:
        report.update(status='failed', failed_case=case['name'], error=str(exc))
        raise
    finally:
        (args.output / 'reference.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workload', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
