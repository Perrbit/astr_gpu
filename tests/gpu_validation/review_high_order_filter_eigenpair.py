"""High-precision selected eigenpair of the actual frozen stage map, not CFD."""
import argparse
from fractions import Fraction
import json
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.linalg import eig, lu_factor, lu_solve
import sympy as sp

from review_high_order_common_metric import assemble, exact_blocks
from review_high_order_p6_filter_compatibility import load_native
from review_high_order_wall_filter_design import exact_rows
from test_high_order_wall_filter_coupling import rk3_filtered_map


def exact_filter(original, parameters, kind):
    n = len(original)-1
    scalar = sp.Matrix(original).applyfunc(lambda value: sp.Rational(float(value)))
    rows = exact_rows(parameters)
    for row in rows:
        i = row['row']
        scalar[i, :] = sp.zeros(1, n+1)
        scalar[n-i, :] = sp.zeros(1, n+1)
        for j, value in enumerate(row['coefficients']):
            scalar[i, j] = sp.Rational(value)
            scalar[n-i, n-j] = sp.Rational(value)
    _, _, _, thermal, _, _ = exact_blocks(n, kind)
    velocity = sp.zeros(n+1, n-1)
    velocity[1:n, :] = sp.eye(n-1)
    return sp.diag(scalar, scalar[1:n, :]*velocity, scalar[1:n, :]*thermal)


def sparse_rows(matrix):
    return [[(j, mp.mpf(int(value.p))/int(value.q))
             for j in range(matrix.cols) if (value := matrix[i, j])]
            for i in range(matrix.rows)]


def matvec(rows, vector):
    return [mp.fsum(value*vector[j] for j, value in row) for row in rows]


def stage_action(a, f, dt, vector):
    def euler(v):
        rhs = matvec(a, v)
        return [x+dt*y for x, y in zip(v, rhs)]
    initial = matvec(f, vector)
    first = euler(initial)
    second_rhs = euler(matvec(f, first))
    second = [(3*x+y)/4 for x, y in zip(initial, second_rhs)]
    third_rhs = euler(matvec(f, second))
    return [(x+2*y)/3 for x, y in zip(initial, third_rhs)]


def transpose_action(at, ft, dt, vector):
    def euler(v):
        rhs = matvec(at, v)
        return [x+dt*y for x, y in zip(v, rhs)]
    one = matvec(ft, vector)
    two = matvec(ft, matvec(ft, euler(vector)))
    three = vector
    for _ in range(3):
        three = matvec(ft, euler(three))
    return [x/3+y/2+z/6 for x, y, z in zip(one, two, three)]


def refine(action, matrix, value, vector, digits):
    """Bordered Newton refinement; FP64 LU is only a residual preconditioner."""
    pivot = int(np.argmax(abs(vector)))
    v = [mp.mpc(x) for x in vector/vector[pivot]]
    lam = mp.mpc(value)
    border = np.zeros((len(v)+1, len(v)+1), complex)
    border[:-1, :-1] = matrix-value*np.eye(len(v))
    border[:-1, -1] = -np.array([complex(x) for x in v])
    border[-1, pivot] = 1
    lu = lu_factor(border)
    history = []
    for _ in range(16):
        residual = [y-lam*x for x, y in zip(v, action(v))]
        norm = max(abs(x) for x in residual)
        history.append(str(norm))
        if norm < mp.mpf(10)**(-digits+15):
            return lam, v, history
        correction = lu_solve(lu, np.array([complex(-x) for x in residual]+[0j]))
        v = [x+mp.mpc(dx) for x, dx in zip(v, correction[:-1])]
        lam += mp.mpc(correction[-1])
    raise ArithmeticError('Selected eigenpair did not converge; retain it as unresolved')


def rank_mod(matrix, prime):
    a = np.array(matrix, dtype=np.int64, copy=True)%prime
    rank = 0
    for j in range(a.shape[1]):
        pivots = np.flatnonzero(a[rank:, j])
        if not len(pivots):
            continue
        pivot = rank+int(pivots[0])
        a[[rank, pivot]] = a[[pivot, rank]]
        a[rank, j:] = a[rank, j:]*pow(int(a[rank, j]), -1, prime)%prime
        factors = a[rank+1:, j].copy()
        a[rank+1:, j:] = (a[rank+1:, j:]-factors[:, None]*a[rank, j:])%prime
        rank += 1
        if rank == a.shape[0]:
            break
    return rank


def rational_mod(matrix, prime):
    return np.array([[int(x.p)*pow(int(x.q), -1, prime)%prime for x in row]
                     for row in matrix.tolist()], dtype=np.int64)


def unit_rank_certificate(mapping, basis, prime):
    """Rank lower bounds plus an independently exact fixed basis prove ranks."""
    delta = (mapping-np.eye(len(mapping), dtype=np.int64))%prime
    fixed = basis.shape[1]
    rank = rank_mod(delta, prime)
    squared_rank = rank_mod(delta@delta%prime, prime)
    independent = rank_mod(basis, prime) == fixed
    invariant = not np.any(delta@basis%prime)
    return dict(dimension=len(mapping), explicit_fixed_dimension=fixed,
                independent_mod_prime=independent, invariant_mod_prime=invariant,
                rank_delta_mod_prime=rank, rank_delta_squared_mod_prime=squared_rank,
                unit_semisimple_certified=bool(independent and invariant and
                                               rank == squared_rank == len(mapping)-fixed))


def polynomial_fields(n, kind):
    x = [sp.Rational(j, n) for j in range(n+1)]
    velocity = sp.Matrix([[(y*(1-y)*y**p) for p in range(4)] for y in x])
    thermal = velocity if kind == 41 else sp.Matrix([
        [sp.Integer(1)]+[y**(p+2)/sp.Integer(p+2)-y**(p+3)/sp.Integer(p+3)
                         for p in range(3)] for y in x])
    density = sp.Matrix([[y**p for p in range(6)] for y in x])
    return density, velocity, thermal


def modular_stage_map(a, f, cfl, n, prime=1000003):
    root = 130104
    assert root*root%prime == 35
    dt = int(sp.Rational(cfl).p)*pow(int(sp.Rational(cfl).q)*7*n, -1, prime)*root%prime
    fm, am = rational_mod(f, prime), rational_mod(a, prime)
    euler = (np.eye(a.rows, dtype=np.int64)+dt*am)%prime
    first = euler@fm%prime
    second = (3*pow(4, -1, prime)*fm+pow(4, -1, prime)*(first@first%prime))%prime
    return (pow(3, -1, prime)*fm+2*pow(3, -1, prime)*(first@second%prime))%prime


def tangential_unit_certificate(original, parameters, kind, viscosity, cfl):
    n = len(original)-1
    f = exact_filter(original, parameters, kind)
    _, _, _, _, dv, dt = exact_blocks(n, kind)
    _, velocity, thermal = polynomial_fields(n, kind)
    nu = sp.Rational(viscosity)
    if kind == 421:
        filtering = f[2*n:, 2*n:]
        a = nu*sp.Rational(18, 35)*dt
        basis = thermal[1:n, :] if nu == 0 or sp.Rational(cfl) == 0 else sp.ones(n-1, 1)
    else:
        filtering = f[n+1:2*n, n+1:2*n]
        a = nu*sp.Rational(3, 4)*dv
        basis = velocity[1:n, :] if nu == 0 or sp.Rational(cfl) == 0 else sp.zeros(n-1, 0)
    assert filtering*basis == basis
    if sp.Rational(cfl):
        assert a*basis == sp.zeros(a.rows, basis.cols)
    prime = 1000003
    report = unit_rank_certificate(modular_stage_map(a, filtering, cfl, n, prime),
                                   rational_mod(basis, prime), prime)
    return dict(intervals=n, bctype=kind, viscosity=viscosity, sound_cfl=cfl,
                scope='one tangential block, identical for both tangential components',
                exact_fixed_basis_verified=True, prime=prime, **report)


def unit_certificates(original, parameters, kind, viscosity, cfl):
    """Specific rational design, dt in Q(sqrt(35)); not contraction or all-mode proof."""
    n = len(original)-1
    a0, diffusion = assemble(exact_blocks(n, kind))
    nu = sp.Rational(viscosity)
    a = a0+nu*diffusion
    f = exact_filter(original, parameters, kind)
    density, velocity, thermal = polynomial_fields(n, kind)
    if sp.Rational(cfl) == 0:
        basis = sp.diag(density, velocity[1:n, :], thermal[1:n, :])
    elif nu == 0:
        basis = sp.Matrix.hstack(sp.Matrix.vstack(sp.ones(n+1, 1), sp.zeros(2*n-2, 1)),
                                sp.Matrix.vstack(-thermal, sp.zeros(n-1, 4), thermal[1:n, :]))
    else:
        basis = sp.Matrix.vstack(sp.ones(n+1, 1), sp.zeros(2*n-2, 1))
        if kind != 41:
            basis = sp.Matrix.hstack(basis, sp.Matrix.vstack(sp.zeros(2*n, 1), sp.ones(n-1, 1)))
    # Exact identities give upper rank bounds in characteristic zero. Modular
    # rank alone would not establish the nullity or rule out a Jordan chain.
    assert f*basis == basis
    if sp.Rational(cfl):
        assert a*basis == sp.zeros(a.rows, basis.cols)
    prime = 1000003
    root = 130104
    mapping = modular_stage_map(a, f, cfl, n, prime)
    report = unit_rank_certificate(mapping, rational_mod(basis, prime), prime)
    return dict(intervals=n, bctype=kind, viscosity=viscosity, sound_cfl=cfl,
                prime=prime, sqrt35_mod_prime=root, exact_fixed_basis_verified=True,
                scope='normal block at this rational dt; unit eigenvalue only, not all-mode stability',
                **report)


def probe(original, parameters, kind, viscosity, cfl, digits, target_imag=None):
    n = len(original)-1
    a, d = assemble(exact_blocks(n, kind))
    a += sp.Rational(viscosity)*d
    f = exact_filter(original, parameters, kind)
    af = np.array(a, float)
    ff = np.array(f, float)
    machine = rk3_filtered_map(af, ff, float(cfl)/(n*np.sqrt(1.4)))
    values, left, right = eig(machine, left=True, right=True)
    index = int(np.argmax(abs(values))) if target_imag is None else int(np.argmin(abs(values.imag-target_imag)))
    with mp.workdps(digits):
        dt = mp.mpf(cfl)/(n*mp.sqrt(mp.mpf(7)/5))
        ar, fr = sparse_rows(a), sparse_rows(f)
        at, ft = sparse_rows(a.T), sparse_rows(f.T)
        lam, v, history = refine(lambda x: stage_action(ar, fr, dt, x),
                                 machine, values[index], right[:, index], digits)
        dual_lam, u, dual_history = refine(lambda x: transpose_action(at, ft, dt, x),
                                          machine.T, values[index], left[:, index].conj(), digits)
        product = mp.fsum(x*y for x, y in zip(u, v))
        condition = mp.sqrt(mp.fsum(abs(x)**2 for x in u)*mp.fsum(abs(x)**2 for x in v))/abs(product)
        separation = min(abs(values[index]-z) for j, z in enumerate(values) if j != index)
        return dict(intervals=n, bctype=kind, viscosity=viscosity, sound_cfl=cfl,
                    selection='largest machine modulus' if target_imag is None else 'specified historical acoustic branch',
                    target_imaginary_part=target_imag,
                    decimal_digits=digits, selected_machine_eigenvalue=[float(values[index].real),
                                                                      float(values[index].imag)],
                    high_precision_eigenvalue=[str(lam.real), str(lam.imag)],
                    radius_excess=str(abs(lam)-1), right_residual_history=history,
                    left_residual_history=dual_history, left_right_eigenvalue_difference=str(abs(lam-dual_lam)),
                    eigenvalue_condition=str(condition), machine_nearest_eigenvalue_separation=float(separation),
                    status='selected frozen eigenpair only; not all modes or stability admission')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--intervals', type=int, required=True)
    parser.add_argument('--kind', type=int, choices=(41, 42, 421), required=True)
    parser.add_argument('--viscosity', default='0')
    parser.add_argument('--cfl', default='0.0001')
    parser.add_argument('--digits', type=int, default=80)
    parser.add_argument('--unit-certificate', action='store_true')
    parser.add_argument('--target-imag', type=float)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    design = json.loads(args.design.read_text())
    parameters = [Fraction(x) for x in design['rational_parameters']]
    original = load_native(args.native)[args.intervals]
    if args.unit_certificate:
        report = unit_certificates(original, parameters, args.kind, args.viscosity, args.cfl)
    else:
        report = probe(original, parameters, args.kind, args.viscosity, args.cfl, args.digits, args.target_imag)
    report['design'] = str(args.design)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(report.get('radius_excess', report.get('unit_semisimple_certified')), flush=True)


if __name__ == '__main__':
    main()
