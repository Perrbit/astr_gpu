"""Audit a physical-side ME6 wall candidate without changing production rows."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import sympy as sp


def rows():
    nodes = list(range(8))
    faces = [0]+[sp.Rational(2*j+1,2) for j in range(6)]

    def weights(nodes, targets, derivative):
        result = sp.Matrix([sp.finite_diff_weights(derivative, nodes, x)[derivative][-1] for x in targets])
        for k in range(len(nodes)):
            exact = sp.Matrix([k*x**(k-1) if derivative and k else (0 if derivative else x**k)
                               for x in targets])
            if result*sp.Matrix([x**k for x in nodes]) != exact:
                raise AssertionError('half-point moment reproduction failed')
        return result

    interpolation = weights(nodes, faces, 0)
    gradient = weights(nodes, faces, 1)
    divergence = weights(faces, range(4), 1)
    return interpolation, gradient, divergence


def bulk_coefficient_tables():
    path = Path(__file__).resolve().parents[2]/'src/me6opti_operator.F90'
    text = path.read_text()

    def table(name):
        part = text.split(f'me6opti_{name}_table(9,3)=reshape', 1)[1].split('/),', 1)[0]
        data = [sp.Rational(a,b) for a,b in re.findall(r'(-?\d+)\.e0_real_kind\s*/\s*(\d+)\.e0_real_kind', part)]
        return sp.Matrix(3,9,data)

    return table('interp'),table('straight')


def field_matrix(b, half, bulk):
    n = len(b)-1
    mat = np.zeros((n+1,n+1))
    a,d = bulk
    interpolation,gradient,divergence = half
    for side in (slice(None),slice(None,None,-1)):
        mids = interpolation@b[side][:8]
        if np.any(mids<=0):
            raise AssertionError('nonpositive sampled half-point coefficient in this fixture')
        mat[side,side][:4,:8] = divergence@(mids[:,None]*gradient)
    for j in range(4,n-3):
        for h,w in enumerate((75/64,-25/384,3/640)):
            for side in (slice(None),slice(None,None,-1)):
                mat[j,j-4:j+5] += w*(a[h,side]@b[j-4:j+5])*d[h,side]
    return mat


def nodal_wall_baseline(b, candidate):
    """Physical 643e gradient and divergence in the existing d<4 transition."""
    n=len(b)-1
    d=np.zeros((n+1,n+1))
    d[0,:3]=[-1.5,2.,-.5]
    d[1,:3]=[-.5,0.,.5]
    d[2,:5]=[1/12,-2/3,0.,2/3,-1/12]
    d[-3:,-5:]=-d[:3,:5][::-1,::-1]
    for j in range(3,n-2):
        d[j,j-3:j+4]=[-1/60,.15,-.75,0.,.75,-.15,1/60]
    nodal=(d*b[None,:])@d
    reference=candidate.copy()
    reference[:4,:],reference[-4:,:]=nodal[:4,:],nodal[-4:,:]
    return reference[1:-1,1:-1]


def run():
    half = rows()
    interpolation, gradient, divergence = half
    # Evolved interior trace u_1=u_3=1, all other entries zero. The counterexample
    # lives entirely in the replaced rows and is independent of the bulk splice.
    support = sp.Matrix([0,1,0,1,0,0,0,0])
    energy = (support[:4,0].T*divergence*gradient*support)[0]
    norm2 = support.dot(support)
    if energy != sp.Rational(67,45):
        raise AssertionError('exact homogeneous Dirichlet counterexample changed')

    half = tuple(np.asarray(a, dtype=float) for a in half)
    bulk = tuple(np.asarray(a,dtype=float) for a in bulk_coefficient_tables())
    n_values = (16,32,64,128)
    profiles = {}
    for exponent in (0.,.4):
        cases=[]
        for n in n_values:
            y = np.linspace(0.,1.,n+1)
            h = 1/n
            b = np.exp(exponent*y)
            mat = field_matrix(b, half, bulk)
            # The end nodes are Dirichlet data, not evolved unknowns.
            evolved = mat[1:-1,1:-1]
            scaled_energy = np.linalg.eigvalsh((evolved+evolved.T)/2)[-1]
            scaled_spectral = np.max(np.linalg.eigvals(evolved).real)
            baseline=nodal_wall_baseline(b,mat)
            previous_energy=np.linalg.eigvalsh((baseline+baseline.T)/2)[-1]
            pi = np.pi
            f = np.sin(pi*y)+.2*np.sin(3*pi*y)+.1*y*(1-y)
            df = pi*(np.cos(pi*y)+.6*np.cos(3*pi*y))+.1*(1-2*y)
            ddf = -pi**2*(np.sin(pi*y)+1.8*np.sin(3*pi*y))-.2
            error = np.abs((mat@f)/h**2-b*(exponent*df+ddf))
            distance = np.minimum(np.arange(n+1),n-np.arange(n+1))
            wall_grad = np.array([half[1][0]@f[:8],-half[1][0]@f[::-1][:8]])/h
            cases.append(dict(n=n, scaled_max_energy_rate=scaled_energy, scaled_max_real_eigenvalue=scaled_spectral,
                              nodal_baseline_scaled_max_energy_rate=previous_energy,
                              errors=[max(error[(distance>=1)&(distance<=3)]), max(error[(distance>=4)&(distance<=7)]),
                                      max(error[distance>=8]), max(abs(b[[0,-1]]*(wall_grad-df[[0,-1]])))]))
        values = np.asarray([case['errors'] for case in cases])
        profiles[str(exponent)] = dict(cases=cases, orders=np.log2(values[:-1]/values[1:]).tolist())
    return dict(columns=['wall_band_rhs', 'first_me6_rows_rhs', 'interior_rhs', 'wall_flux'],
                exact_energy_polynomial=str(energy), exact_scaled_rayleigh=str(energy/norm2),
                profiles=profiles, energy_passed=bool(energy<=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run()
    with args.output.open('x') as out:
        json.dump(report,out,indent=2)
        out.write('\n')
    if not report['energy_passed']:
        raise SystemExit('ME6 half-point candidate rejected: exact positive scaled energy rate '+report['exact_scaled_rayleigh'])
