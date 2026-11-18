"""Exact box/two-moment geometry of captured inputs; no limiter modification."""
import argparse
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import shutil

import numpy as np
from run_air5_flux_components import read as read_result


def rational(x):
    return Fraction.from_float(float(x))


def packed(x):
    return dict(value=float(x), exact=str(x))


def geometry(lower, upper, mass, weights, moment):
    lo, hi, g = [[rational(x) for x in a] for a in (lower, upper, weights)]
    m, target = rational(mass), rational(moment)
    assert all(a <= b for a, b in zip(lo, hi))
    low_mass, high_mass = sum(lo), sum(hi)
    result = dict(mass_lower_margin=packed(m-low_mass), mass_upper_margin=packed(high_mass-m))
    if not low_mass <= m <= high_mass:
        return dict(**result, mass_feasible=False, feasible=False)

    def extremum(reverse):
        point = lo.copy()
        remainder = m-low_mass
        for s in sorted(range(5), key=lambda i: g[i], reverse=reverse):
            add = min(remainder, hi[s]-lo[s])
            point[s] += add
            remainder -= add
        assert remainder == 0 and sum(point) == m
        return sum(a*b for a, b in zip(g, point)), point

    minimum, low = extremum(False)
    maximum, high = extremum(True)
    vertices = []
    for free in range(5):
        fixed = [s for s in range(5) if s != free]
        for choices in itertools.product((0, 1), repeat=4):
            point = lo.copy()
            for s, choice in zip(fixed, choices):
                point[s] = hi[s] if choice else lo[s]
            point[free] = m-sum(point[s] for s in fixed)
            if lo[free] <= point[free] <= hi[free]:
                vertices.append(sum(a*b for a, b in zip(g, point)))
    assert vertices and min(vertices) == minimum and max(vertices) == maximum
    gap = max(Fraction(0), minimum-target, target-maximum)
    result.update(mass_feasible=True, moment_lower_margin=packed(target-minimum),
                  moment_upper_margin=packed(maximum-target), moment_distance=packed(gap),
                  feasible=gap == 0, independent_vertex_checks=80)
    if gap != 0:
        point = low if target < minimum else high
        result['nearest_moment_witness'] = [packed(x) for x in point]
        result['rounded_nearest_moment_witness'] = [float(x) for x in point]
    if gap == 0:
        theta = Fraction(0) if maximum == minimum else (target-minimum)/(maximum-minimum)
        point = [a+theta*(b-a) for a, b in zip(low, high)]
        assert sum(point) == m and sum(a*b for a, b in zip(g, point)) == target
        assert all(a <= x <= b for x, a, b in zip(point, lo, hi))
        result['exact_witness'] = [packed(x) for x in point]
        result['rounded_witness'] = [float(x) for x in point]
    return result


def sequential_sum(a):
    total = 0.
    for value in a:
        total += float(value)
    return total


def sequential_dot(a, b):
    return sequential_sum([float(x)*float(y) for x, y in zip(a, b)])


def analyze(x):
    c, lo, hi, mass, weights, moment = x[:5], x[5:10], x[10:15], x[15], x[16:21], x[21]
    sf = max(abs(mass), *abs(c), *abs(lo), *abs(hi))
    sw = max(abs(weights))
    assert sf > 0 and sw > 0
    cn, ln, hn, g = [float(a)/sf for a in c], [float(a)/sf for a in lo], [float(a)/sf for a in hi], [float(a)/sw for a in weights]
    tm, te = float(mass)/sf, (float(moment)/sw)/sf
    tolerance = 64*np.finfo(float).eps*max(abs(tm), abs(te), sequential_sum(abs(a) for a in cn))
    dl, dh = [(float(a)-float(b))/sf for a, b in zip(lo, c)], [(float(a)-float(b))/sf for a, b in zip(hi, c)]
    dm = (float(mass)-sequential_sum(c))/sf
    de = ((float(moment)-sequential_dot(weights, c))/sw)/sf
    original_dm, original_de = dm, de
    if abs(dm) <= tolerance:
        dm = 0.
    if abs(de) <= tolerance:
        de = 0.
    physical = geometry(lo, hi, mass, weights, moment)
    normalized = geometry(ln, hn, tm, g, te)
    increments = geometry(dl, dh, dm, g, de)
    result = dict(scale_flux=sf, scale_weight=sw, tolerance=tolerance,
                  delta_before_zero=[original_dm, original_de], delta_after_zero=[dm, de],
                  physical=physical, normalized=normalized, increments=increments)
    if physical.get('mass_feasible'):
        witness = physical.get('rounded_witness', physical.get('rounded_nearest_moment_witness'))
        scaled = [a/sf for a in witness]
        inside = all(a <= b <= d for a, b, d in zip(lo, witness, hi))
        mass_error = abs(sequential_sum(scaled)-tm)
        moment_error = abs(sequential_dot(g, scaled)-te)
        mass_limit = tolerance+64*np.finfo(float).eps*sequential_sum(abs(a) for a in scaled)
        moment_limit = tolerance+64*np.finfo(float).eps*sequential_sum(abs(a*b) for a, b in zip(g, scaled))
        result['rounded_physical_witness'] = dict(flux=witness, box_inside=inside,
            mass_error=mass_error, mass_limit=mass_limit, moment_error=moment_error, moment_limit=moment_limit,
            passes_original_final_checks=bool(inside and mass_error <= mass_limit and moment_error <= moment_limit))
    for name, problem in [('normalized', normalized), ('increments', increments)]:
        if problem.get('mass_feasible'):
            result[name+'_moment_distance_over_tolerance'] = problem['moment_distance']['value']/tolerance
    if normalized['feasible']:
        witness = [float(a*sf) for a in normalized['rounded_witness']]
        inside = all(a <= b <= d for a, b, d in zip(lo, witness, hi))
        scaled = [a/sf for a in witness]
        mass_error = abs(sequential_sum(scaled)-tm)
        moment_error = abs(sequential_dot(g, scaled)-te)
        mass_limit = tolerance+64*np.finfo(float).eps*sequential_sum(abs(a) for a in scaled)
        moment_limit = tolerance+64*np.finfo(float).eps*sequential_sum(abs(a*b) for a, b in zip(g, scaled))
        result['rounded_normalized_witness'] = dict(flux=witness, box_inside=inside,
            mass_error=mass_error, mass_limit=mass_limit, moment_error=moment_error, moment_limit=moment_limit,
            passes_original_final_checks=bool(inside and mass_error <= mass_limit and moment_error <= moment_limit))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    capture, output = args.capture.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output/'diagnostic.py')
    path = capture/'projection_endpoints.bin'
    with path.open('rb') as stream:
        count = int(np.fromfile(stream, '<i4', 1)[0])
        x = np.fromfile(stream, '<f8', count*80).reshape(count, 80)
        assert stream.read() == b'' and np.all(x[:, 79] == 1)
    status_cpu, _ = read_result(capture/'projection_cpu.bin')
    status_cpp, _ = read_result(capture/'projection_cpp.bin')
    rows = []
    for n, row in enumerate(x):
        rows.append(dict(index=n, original_cpu_status=int(status_cpu[n]), original_cpp_status=int(status_cpp[n]), **analyze(row)))
    report = dict(scope='exact feasibility of two frozen endpoint projection inputs only; no production change or repaired CFD',
                  input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  note='exact equality feasibility is diagnostic, not a replacement for the existing finite-arithmetic acceptance budgets',
                  rows=rows)
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    summary = [dict(index=r['index'], status=r['original_cpu_status'],
                    physical_exact_feasible=r['physical']['feasible'], normalized_exact_feasible=r['normalized']['feasible'],
                    delta_exact_feasible=r['increments']['feasible'],
                    delta_moment_gap_over_tolerance=r.get('increments_moment_distance_over_tolerance'),
                    physical_witness_passes=r.get('rounded_physical_witness', {}).get('passes_original_final_checks'),
                    normalized_witness_passes=r.get('rounded_normalized_witness', {}).get('passes_original_final_checks')) for r in rows]
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
