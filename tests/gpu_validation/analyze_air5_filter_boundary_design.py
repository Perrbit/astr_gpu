#!/usr/bin/env python3
"""Bounded linear-operator design analysis; no CFD, chemistry, MPI or GPU run."""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def current_increment(intervals):
    matrix = np.zeros((intervals+1, intervals+1), dtype=np.int64)
    one_sided = ([16, 928, 240, -320, 240, -96, 16],
                 [-16, 96, 784, 320, -240, 96, -16])
    for row, coefficients in enumerate(one_sided, 1):
        matrix[row, :7] = coefficients
        matrix[intervals-row, intervals-6:] = coefficients[::-1]
    coefficients = {3: [352, 240, -96, 16], 4: [372, 224, -112, 32, -4],
                    5: [386, 210, -120, 45, -10, 1]}
    for row in range(3, intervals-2):
        for offset, value in enumerate(coefficients[min(row, intervals-row, 5)]):
            matrix[row, row-offset] += value
            matrix[row, row+offset] += value
    matrix[1:intervals, 1:intervals] -= 1024*np.eye(intervals-1, dtype=np.int64)
    return matrix


def candidates(intervals):
    old = current_increment(intervals)
    # Exact left/right column-sum residuals, supported within ten nodes of a face.
    left = np.zeros(intervals+1, dtype=np.int64)
    left[:10] = old.sum(axis=0)[:10]
    right = left[::-1]
    weights = np.zeros(intervals+1, dtype=np.int64)
    weights[1:11] = 1
    corrected = 10*old-np.outer(weights, left)-np.outer(weights[::-1], right)
    difference = np.zeros((intervals-6, intervals-1), dtype=np.int64)
    for row in range(intervals-6):
        difference[row, row:row+6] = [(-1)**(5-k)*math.comb(5, k) for k in range(6)]
    factorized = np.zeros_like(old)
    factorized[1:intervals, 1:intervals] = -(difference.T@difference)
    return [('current', old, 1024), ('local-mass-correction', corrected, 10240),
            ('factorized-free-interior', factorized, 1024)]


def row_order(matrix, row):
    support = np.flatnonzero(matrix[row])
    if not len(support):
        return None
    # Integer local offsets avoid cancellation and large global polynomial values.
    offsets = support-row
    for degree in range(11):
        moment = int(matrix[row, support]@(offsets**degree))
        if moment:
            return {'first_nonzero_degree': degree, 'moment_numerator': moment}
    raise ValueError('nonidentity stencil has no moment through degree ten')


def analyze(intervals, name, matrix, denominator):
    operator = np.eye(intervals+1)+matrix/denominator
    homogeneous = operator[1:intervals, 1:intervals]
    eigenvalues = np.linalg.eigvals(homogeneous)
    boundary = [row_order(matrix, row) for row in range(1, 11)]
    return dict(name=name, intervals=intervals, denominator=denominator,
        physical_nodes_unchanged=bool(np.all(matrix[[0, -1]] == 0)),
        constant_residual_numerator_max=int(np.max(abs(matrix.sum(axis=1)))),
        weighted_mass_residual_numerator=matrix.sum(axis=0).tolist(),
        boundary_moments=boundary, center_moment=row_order(matrix, intervals//2),
        homogeneous_spectral_radius=float(np.max(abs(eigenvalues))),
        homogeneous_norm2=float(np.linalg.norm(homogeneous, 2)),
        homogeneous_power1024_norm2=float(np.linalg.norm(np.linalg.matrix_power(homogeneous, 1024), 2)),
        left_rows=[dict(row=row, entries={str(j): int(matrix[row, j])
            for j in np.flatnonzero(matrix[row])}) for row in range(11)])


def contractivity_obstruction(width):
    # Compatibility of primal and adjoint polynomial moments in a local closure.
    first_core, last_column = width+1, width+5
    stencil = {0: -252, 1: 210, 2: -120, 3: 45, 4: -10, 5: 1}
    compatibility = [[sum(i**p*stencil.get(abs(j-i), 0)*j**q
        for i in range(first_core, last_column+6) for j in range(last_column+1))
        for q in range(6)] for p in range(6)]
    expected = [[0]*6 for _ in range(6)]
    expected[4][5] = -math.factorial(4)*math.factorial(5)
    expected[5][4] = -expected[4][5]
    if compatibility != expected:
        raise ValueError('exact local closure compatibility identity failed')
    return dict(changed_interior_rows=width, last_left_column=last_column,
        denominator=1024, compatibility_numerators=compatibility,
        compatible_with_primal_and_adjoint_degree_five=False)


def exact_unit_root_analysis():
    import sympy as sp
    from sympy.polys.matrices import DomainMatrix

    symbol = sp.Symbol('x')
    results = []
    for name, matrix, denominator in candidates(32):
        integer_matrix = DomainMatrix.from_Matrix(sp.Matrix(matrix[1:32, 1:32].tolist()))
        rank = integer_matrix.rank()
        rank_squared = (integer_matrix*integer_matrix).rank()
        nullity = integer_matrix.shape[0]-rank
        polynomial = sp.Poly.from_list([int(v) for v in integer_matrix.charpoly()],
                                       symbol, domain=sp.ZZ)
        remaining = polynomial.exquo(sp.Poly(symbol**nullity, symbol))
        if rank != rank_squared or remaining.TC() == 0:
            raise ValueError('unit-root semisimplicity is not established')
        stable_real_roots = int(remaining.count_roots(-2*denominator, 0))
        positive_roots = int(remaining.count_roots(0, sp.oo))
        positive_intervals = remaining.intervals(eps=sp.Rational(1, 10**16), inf=0)
        brackets = [dict(increment_root_bounds=[str(a), str(b)], multiplicity=int(m),
            filter_root_bounds=[str(1+a/denominator), str(1+b/denominator)],
            filter_root_midpoint=float(1+(a+b)/(2*denominator)))
            for (a, b), m in positive_intervals]
        results.append(dict(name=name, intervals=32, matrix_dimension=integer_matrix.shape[0],
            rank_increment=rank, rank_increment_squared=rank_squared, unit_root_nullity=nullity,
            nonunit_polynomial_coefficients=[str(v) for v in remaining.all_coeffs()],
            denominator=denominator, unit_roots_semisimple=True,
            real_nonunit_roots_in_stable_interval=stable_real_roots,
            positive_increment_roots=positive_roots, positive_root_brackets=brackets,
            finite_grid_power_bounded=(stable_real_roots == remaining.degree()),
            sympy_version=sp.__version__))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status='linear-design-analysis-not-production-admission',
        scope='32/64/128 intervals; integer moments, matrix diagnostics and local compatibility',
        floating_spectrum_note='Near-unit eigenvalues of nonnormal matrices are not stability proofs.',
        source='src/filter.F90::filter_coefficient_explicit/spafilter10exp',
        results=[analyze(n, name, matrix, denominator) for n in (32, 64, 128)
                 for name, matrix, denominator in candidates(n)],
        contractivity_obstructions=[contractivity_obstruction(width) for width in (5, 10, 15, 20)],
        exact_unit_root_analysis=exact_unit_root_analysis())
    (args.output/'boundary_design.json').write_text(json.dumps(report, indent=2)+'\n')
    for result in report['results']:
        boundary_order = min(row['first_nonzero_degree']
                             for row in result['boundary_moments'] if row is not None)
        mass = max(abs(v) for v in result['weighted_mass_residual_numerator'])
        print(result['intervals'], result['name'], 'boundary-order', boundary_order,
              'mass-residual-numerator', mass, 'norm2', result['homogeneous_norm2'])
    for result in report['contractivity_obstructions']:
        print('local closure width', result['changed_interior_rows'], 'exact compatibility T45',
              result['compatibility_numerators'][4][5], '/', result['denominator'])
    for result in report['exact_unit_root_analysis']:
        print('exact unit-root analysis', result['name'], 'semisimple', result['unit_roots_semisimple'],
              'positive increment roots', result['positive_increment_roots'])


if __name__ == '__main__':
    main()
