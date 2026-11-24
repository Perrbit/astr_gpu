"""Verify the ten-node ME6 half-point candidate for constant positive viscosity."""
import argparse
from math import comb
import json
from pathlib import Path

import mpmath as mp
import numpy as np
import sympy as sp

from check_me6_wall_halfpoint import rows
from check_me6_wall_moment_energy import constant_me6, negative_pivots


def closure():
    _, original, divergence = rows()
    gradient = original.row_join(sp.zeros(7,2))
    annihilator = sp.zeros(2,10)
    for shift in range(2):
        annihilator[shift,shift:shift+9] = sp.Matrix(1,9,[(-1)**k*comb(8,k) for k in range(9)])
    # Search the six row parameters against the symmetric energy form on 32 intervals,
    # using the inverse Dirichlet-Laplacian square root only as a coordinate change.
    # Freeze the resulting parameters to simple rationals and verify them exactly below.
    row_weights = sp.Matrix([[0,0],[72,22],[5,20],[39,28]])/1000
    lift = divergence[:,1:]
    half_weights = lift.T*(lift*lift.T).inv()*row_weights
    gradient[1:,:] += half_weights*annihilator
    stencil = divergence*gradient
    if stencil != divergence*original.row_join(sp.zeros(7,2))+row_weights*annihilator:
        raise AssertionError('half-point lifting changed the target rows')
    faces = [0]+[sp.Rational(2*j+1,2) for j in range(6)]
    for degree in range(8):
        q = sp.Matrix([j**degree for j in range(10)])
        d1 = [degree*x**(degree-1) if degree else 0 for x in faces]
        d2 = [degree*(degree-1)*j**(degree-2) if degree>=2 else 0 for j in range(4)]
        if gradient*q != sp.Matrix(d1) or stencil*q != sp.Matrix(d2):
            raise AssertionError('loss of exact half-point or second-derivative moments')
    return gradient, divergence, stencil, row_weights


def dirichlet_matrix(n, stencil, bulk):
    matrix = sp.zeros(n+1)
    matrix[1:4,:10] = stencil[1:,:]
    matrix[n-3:n,n-9:] = stencil[:0:-1,::-1]
    row = sp.Matrix(1,9,bulk)
    for j in range(4,n-3):
        matrix[j,j-4:j+5] = row
    return matrix[1:n,1:n]


def sym(matrix):
    return (matrix+matrix.T)/2


def energy_certificate(stencil, bulk):
    # For disjoint end supports, -sym(L) differs from a polynomial in
    # the Dirichlet Laplacian only within the first/last nine-node blocks.
    n, side = 19, 9
    diff = sp.zeros(n,n-1)
    for j in range(n):
        if j>0:
            diff[j,j-1] = -1
        if j<n-1:
            diff[j,j] = 1
    t = diff.T*diff
    factors = ((1,diff),(sp.Rational(1,12),t),(sp.Rational(1,90),diff*t),(-bulk[0],t*t))
    reference = sum((coef*factor.T*factor for coef,factor in factors),sp.zeros(n-1))
    z = sp.Symbol('z')
    lam = 2-z-1/z
    symbol = sp.expand(-(lam+lam**2/12+lam**3/90-bulk[0]*lam**4))
    if [symbol.coeff(z,j) for j in range(-4,5)] != list(bulk):
        raise AssertionError('Dirichlet polynomial does not reproduce the original ME6 interior')
    defect = -sym(dirichlet_matrix(n,stencil,bulk))-reference
    correction = defect[:side,:side]
    if defect != sp.diag(correction,correction[::-1,::-1]):
        raise AssertionError('energy correction has nonlocal support')
    local = correction
    for coef, factor in factors:
        kept = [i for i in range(factor.rows) if not any(factor[i,side:])]
        end_factor = factor.extract(kept,range(side))
        local += coef*end_factor.T*end_factor
    if negative_pivots(-local) != side:
        raise AssertionError('end-block energy certificate failed')
    # Some supports overlap below n=19. Do not use the disjoint-block certificate there.
    short = {}
    for n in range(9,19):
        pivots = negative_pivots(sym(dirichlet_matrix(n,stencil,bulk)))
        if pivots != n-1:
            raise AssertionError(f'energy sign failed on overlapping supports at n={n}')
        short[n] = pivots
    return dict(end_block_positive_pivots=side, disjoint_support_from=19, short_grid_negative_pivots=short,
                end_block_min_eigenvalue=float(np.linalg.eigvalsh(np.asarray(local,dtype=float))[0]))


def convergence(gradient, stencil, bulk):
    with mp.workdps(55):
        m = lambda c: mp.mpf(int(c.p))/int(c.q)
        near = [[m(c) for c in stencil[j,:]] for j in (1,2,3)]
        interior = [m(c) for c in bulk]
        wall = [m(c) for c in gradient[0,:]]
        report = []
        for n in (16,32,64,128):
            y = [mp.mpf(j)/n for j in range(n+1)]
            q = [x*(1-x)*mp.exp(x) for x in y]
            exact = [-x*(x+3)*mp.exp(x) for x in y]
            end_error = max(abs(n*n*mp.fdot(row,v)-exact[j]) for v,js in ((q[:10],(1,2,3)),(q[-1:-11:-1],(n-1,n-2,n-3)))
                            for row,j in zip(near,js))
            bulk_error = max(abs(n*n*mp.fdot(interior,q[j-4:j+5])-exact[j]) for j in range(4,n-3))
            wall_error = max(abs(n*mp.fdot(wall,q[:10])-1),abs(n*mp.fdot(wall,q[-1:-11:-1])-mp.e))
            energy = np.linalg.eigvalsh(np.asarray(sym(dirichlet_matrix(n,stencil,bulk)),dtype=float))[-1]
            report.append(dict(n=n, wall_band=float(end_error), interior=float(bulk_error),
                               wall_flux=float(wall_error), max_scaled_energy_rate=float(energy)))
    return report


def run():
    gradient, divergence, stencil, weights = closure()
    bulk = constant_me6()
    def serialize(matrix):
        return [[str(c) for c in row] for row in matrix.tolist()]
    return dict(support_nodes=10, exact_moment_degree=7, closure_weights=serialize(weights),
                halfpoint_gradient=serialize(gradient), divergence=serialize(divergence),
                exact_energy_certificate=energy_certificate(stencil,bulk),
                convergence=convergence(gradient,stencil,bulk))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run()
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2)
        stream.write('\n')
    print(report['exact_energy_certificate'])
