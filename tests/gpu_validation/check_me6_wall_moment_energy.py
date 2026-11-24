"""Test a necessary constant-viscosity energy condition without solving for new rows."""
import argparse
import json
from pathlib import Path

import sympy as sp

from check_me6_wall_halfpoint import bulk_coefficient_tables


def constant_me6():
    interp, straight = bulk_coefficient_tables()
    weights = (sp.Rational(75,64),-sp.Rational(25,384),sp.Rational(3,640))
    return [sum(w*sum(interp[h,k] for k in range(9))*(straight[h,j]+straight[h,8-j])
                for h,w in enumerate(weights)) for j in range(9)]


def fixed_action(n, width, bulk):
    # Independent degree-seven polynomials with zero wall value on the two end supports.
    # In between, allow any state, not just polynomials or zero extension.
    x = sp.Symbol('x')
    scaled = x/(width-1)
    basis = [sp.binomial(7,k)*scaled**k*(1-scaled)**(7-k) for k in range(1,8)]
    free = range(width,n-width+1)
    state = sp.zeros(n+1,14+len(free))
    action = sp.zeros(n+1,state.cols)
    samples = sp.Matrix([[p.subs(x,j) for p in basis] for j in range(width)])
    state[:width,:7] = samples
    state[-width:,7:14] = samples[::-1,:]
    d2 = sp.Matrix([[p.diff(x,2).subs(x,j) for p in basis] for j in (1,2,3)])
    action[1:4,:7] = d2
    action[n-3:n,7:14] = d2[::-1,:]
    for k,j in enumerate(free,start=14):
        state[j,k] = 1
    interior = sp.Matrix(1,9,bulk)
    for j in range(4,n-3):
        action[j,:] = interior*state[j-4:j+5,:]
    energy = state.T*action
    return (energy+energy.T)/2


def negative_pivots(matrix):
    """Exact Sylvester check using fraction-free elimination of -matrix."""
    scale = sp.ilcm(*(value.q for value in matrix))
    a = [[int(-scale*matrix[i,j]) for j in range(matrix.cols)] for i in range(matrix.rows)]
    previous = 1
    for k in range(matrix.rows):
        pivot = a[k][k]
        if pivot <= 0:
            return k
        for i in range(k+1,matrix.rows):
            for j in range(i,matrix.cols):
                a[i][j] = a[j][i] = (pivot*a[i][j]-a[i][k]*a[k][j])//previous
        previous = pivot
    return matrix.rows


def run():
    bulk = constant_me6()
    for degree in range(8):
        if sum(c*x**degree for x,c in zip(range(-4,5),bulk)) != (2 if degree==2 else 0):
            raise AssertionError('current constant-viscosity interior does not match degree-seven moments')
    x = sp.Symbol('x')
    witness = [0,1,0,1,0,0,0,0]
    polynomial = sp.interpolate(list(enumerate(witness)),x)
    energy = sum(polynomial.diff(x,2).subs(x,j) for j in (1,3))
    if energy != sp.Rational(67,45):
        raise AssertionError('previous eight-node witness not recovered by the moment constraints')
    cases = []
    for n in (32,64):
        for width in (8,10,12):
            form = fixed_action(n,width,bulk)
            pivots = negative_pivots(form)
            cases.append(dict(intervals=n, width=width, restricted_dimension=form.rows,
                              strictly_negative_pivots=pivots, negative_definite=(pivots==form.rows)))
    return dict(constant_me6_row=[str(c) for c in bulk], eight_node_rayleigh=str(energy/2), cases=cases)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    report = run()
    with p.parse_args().output.open('x') as f:
        json.dump(report,f,indent=2)
        f.write('\n')
    for case in report['cases']:
        print(case)
