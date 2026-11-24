"""Check positivity, energy and convergence for two frozen smooth viscosity profiles."""
import argparse
import json
from pathlib import Path

import mpmath as mp
import numpy as np
import sympy as sp

from check_me6_wall_halfpoint import bulk_coefficient_tables, rows
from check_me6_wall10_constant import closure


def viscosity(y, profile, math, decimal):
    if profile == 'constant':
        return 1+0*y, 0*y
    if profile == 'exp':
        a = decimal('0.4')
        value = math.exp(a*y)
        return value, a*value
    if profile != 'sutherland':
        raise ValueError(profile)
    s = decimal('110.3')/decimal('273.15')
    t = 1+decimal('0.2')*y*(1-y)+decimal('0.1')*math.sin(2*math.pi*y)
    dt = decimal('0.2')*(1-2*y)+decimal('0.2')*math.pi*math.cos(2*math.pi*y)
    value = t*math.sqrt(t)*(1+s)/(t+s)
    return value, value*(decimal('1.5')/t-1/(t+s))*dt


def matrix(b, wall, interior, weights):
    n = len(b)-1
    interpolation, gradient, divergence = wall
    a, d = interior
    result = np.zeros((n+1,n+1))
    bottom = interpolation@b[:8]
    top = interpolation@b[:-9:-1]
    result[:4,:10] = divergence@(bottom[:,None]*gradient)
    result[-1:-5:-1,-1:-11:-1] = divergence@(top[:,None]*gradient)
    min_wall = float(min(bottom.min(),top.min()))
    min_interior = np.inf
    for j in range(4,n-3):
        samples = b[j-4:j+5]
        for side in (slice(None),slice(None,None,-1)):
            face_b = a[:,side]@samples
            min_interior = min(min_interior,float(face_b.min()))
            result[j,j-4:j+5] += (weights*face_b)@d[:,side]
    return result, min_wall, min_interior


def errors(n, profile, wall, interior, weights):
    i, g, d = wall
    a, s = interior
    y = [mp.mpf(j)/n for j in range(n+1)]
    u = [x*(1-x)*mp.exp(x) for x in y]
    grad = [(1-x-x*x)*mp.exp(x) for x in y]
    u_yy = [-x*(x+3)*mp.exp(x) for x in y]
    b, b_y = zip(*(viscosity(x, profile, mp, mp.mpf) for x in y))
    exact = [b[j]*u_yy[j]+b_y[j]*grad[j] for j in range(n+1)]
    end_error = wall_flux = 0
    for end in (slice(None),slice(None,None,-1)):
        half = mp.matrix([p*q for p,q in zip(i*mp.matrix(b[end][:8]),g*mp.matrix(u[end][:10]))])
        rhs = n*n*d*half
        end_error = max(end_error,max(abs(rhs[j]-exact[end][j]) for j in (1,2,3)))
        sign = -1 if end.step == -1 else 1
        wall_flux = max(wall_flux,abs(n*half[0]-sign*b[end][0]*grad[end][0]))
    interior_error = 0
    for j in range(4,n-3):
        rhs = 0
        for side in (slice(None),slice(None,None,-1)):
            half_b = a[:,side]*mp.matrix(b[j-4:j+5])
            half_du = s[:,side]*mp.matrix(u[j-4:j+5])
            rhs += sum(w*p*q for w,p,q in zip(weights,half_b,half_du))
        interior_error = max(interior_error,abs(n*n*rhs-exact[j]))
    return dict(wall_rhs=float(end_error),interior_rhs=float(interior_error),inward_wall_flux=float(wall_flux))


def run():
    interpolation, _, _ = rows()
    gradient, divergence, _, _ = closure()
    wall = (interpolation,gradient,divergence)
    interior = bulk_coefficient_tables()
    weights = (sp.Rational(75,64),-sp.Rational(25,384),sp.Rational(3,640))
    wall_np = tuple(np.asarray(t,dtype=float) for t in wall)
    interior_np = tuple(np.asarray(t,dtype=float) for t in interior)
    weights_np = np.asarray(weights,dtype=float)
    report = {}
    for profile in ('exp','sutherland'):
        report[profile] = []
        for n in (16,32,64,128,256,512):
            b,_ = viscosity(np.arange(n+1)/n,profile,np,float)
            op,min_wall,min_interior = matrix(b,wall_np,interior_np,weights_np)
            terminal = op[1:-1,1:-1]
            energy = float(np.linalg.eigvalsh((terminal+terminal.T)/2)[-1])
            report[profile].append(dict(n=n,min_wall_half_b=min_wall,min_interior_half_b=min_interior,
                                        max_scaled_energy_rate=energy))
            if min(min_wall,min_interior)<=0 or energy>0:
                return dict(passed=False,profiles=report)
    with mp.workdps(55):
        def mp_coef(c):
            return mp.mpf(int(c.p))/int(c.q)
        def mp_table(t):
            return mp.matrix([[mp_coef(c) for c in r] for r in t.tolist()])
        wall_mp = tuple(mp_table(t) for t in wall)
        interior_mp = tuple(mp_table(t) for t in interior)
        weights_mp = tuple(mp_coef(w) for w in weights)
        for profile,cases in report.items():
            for case in cases:
                case['errors'] = errors(case['n'],profile,wall_mp,interior_mp,weights_mp)
            for before,after in zip(cases,cases[1:]):
                after['orders'] = {k:float(np.log2(before['errors'][k]/v)) for k,v in after['errors'].items()}
    return dict(passed=True,profiles=report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    report = run()
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2)
        stream.write('\n')
    if not report['passed']:
        raise SystemExit('ME6 smooth-variable half-point candidate failed positivity/energy: '+str(report))
