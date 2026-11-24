"""Audit the Mattsson D2(b) scalar candidate against the actual ASTR SBP6/3 D1."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re

import numpy as np


SOURCE_SHA256 = '54529672355acdfbe843770c278bb4c3909148a4993b05ba24d5dcb5a65da691'


def operators(source):
    """Each table entry names the zero-based (destination, u index, b index) after conversion."""
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError('unexpected Mattsson coefficient source version')
    text = data.decode()
    left = np.zeros((9, 12, 12))
    for i, j, k, a, b in re.findall(r'^\s+d(\d\d)(\d\d)(\d\d) = T\((-?\d+) // (\d+)\)', text, re.M):
        left[int(i)-1, int(j)-1, int(k)-1] = float(Fraction(int(a), int(b)))
    inside = np.zeros((7, 7))
    offsets = dict(im3=-3, im2=-2, im1=-1, i_0=0, ip1=1, ip2=2, ip3=3)
    for j, k, a, b in re.findall(r'^\s+di_(i[mp][123]|i_0)_(i[mp][123]|i_0) = T\((-?\d+) // (\d+)\)', text, re.M):
        jj, kk = offsets[j], offsets[k]
        inside[3+jj, 3+kk] = inside[3-jj, 3-kk] = float(Fraction(int(a), int(b)))
    return left, inside


def ast_d1(n):
    path = Path(__file__).resolve().parents[2]/'src/keep_sbp63_body.inc'
    text = path.read_text()
    scope = text.split('dleft(0:8,0:5)=reshape([', 1)[1].split('],[9,6])', 1)[0]
    left = np.array([float(x) for x in re.findall(r'(-?\d+\.\d*e[+-]?\d+)_real_kind', scope)]).reshape(6, 9)
    htext = text.split('hleft(0:5)=[', 1)[1].split(']', 1)[0]
    h = np.ones(n)
    h[:6] = [int(a)/int(b) for a, b in re.findall(r'(\d+)\.e0_real_kind/(\d+)\.e0_real_kind', htext)]
    h[-6:] = h[:6][::-1]
    d = np.zeros((n, n))
    d[:6, :9] = left
    d[-6:, -9:] = -left[::-1, ::-1]
    for i in range(6, n-6):
        d[i, i-3:i+4] = np.array([-1, 9, -45, 0, 45, -9, 1])/60
    return h, d


def d2_matrix(b, left, inside):
    n = len(b)
    d2 = np.zeros((n, n))
    d2[:9, :12] = left @ b[:12]
    d2[-9:, -12:] = (left @ b[-12:][::-1])[::-1, ::-1]
    for i in range(9, n-9):
        d2[i, i-3:i+4] = inside @ b[i-3:i+4]
    return d2


def audit(source):
    left, inside = operators(source)
    h, d = ast_d1(25)
    b0 = np.eye(25)[0]
    md2 = h[:, None]*d2_matrix(b0, left, inside)
    s = md2[:, 0]-md2[0, :]
    s[0] = -s.sum()
    boundary = np.zeros((25, 25))
    boundary[0, 0], boundary[-1, -1] = -1, 1
    d1_sbp = np.max(abs(h[:, None]*d + d.T*h-boundary))
    sbp_symmetry = 0.
    m_min, r_min = 0., 0.
    worst_basis = 0
    matrices = []
    for i, b in enumerate(np.eye(25)):
        m = -h[:, None]*d2_matrix(b, left, inside)
        m[0, :] -= b[0]*s
        m[-1, :] -= b[-1]*s[::-1]
        scale = max(1., np.linalg.norm(m, ord=2))
        matrices.append((m, scale))
        sbp_symmetry = max(sbp_symmetry, np.max(abs(m-m.T))/scale)
        m_min = min(m_min, np.linalg.eigvalsh(m)[0]/scale)
        r = m-d.T@((h*b)[:, None]*d)
        residual = np.linalg.eigvalsh(r)[0]/scale
        if residual < r_min:
            r_min, worst_basis = residual, i
    b = 1 + np.eye(25)[worst_basis]
    m = -h[:, None]*d2_matrix(b, left, inside)
    m[0, :] -= b[0]*s
    m[-1, :] -= b[-1]*s[::-1]
    positive_r_min = np.linalg.eigvalsh(m-d.T@((h*b)[:, None]*d))[0]/np.linalg.norm(m, ord=2)

    # The two shifted fourth differences annihilate cubic data. Their skew
    # product shifts the one free SBP boundary parameter without changing H.
    v = np.array([1, -4, 6, -4, 1, 0])
    w = np.roll(v, 1)
    direction = np.zeros_like(d)
    direction[:6, :6] = (np.outer(v, w)-np.outer(w, v))/h[:6, None]
    direction[-6:, -6:] = -direction[:6, :6][::-1, ::-1]
    eigenvalues, vectors = np.linalg.eigh(matrices[0][0])
    null = vectors[:, eigenvalues<1.e-12].T
    projected = null@direction[0]
    shift = -np.dot(projected, null@d[0])/np.dot(projected, projected)
    candidate = d+shift*direction
    matched_min = min(np.linalg.eigvalsh(m-h[i]*np.outer(candidate[i], candidate[i]))[0]/scale
                      for i, (m, scale) in enumerate(matrices))
    matched_sbp = np.max(abs(h[:, None]*candidate + candidate.T*h-boundary))
    # A quartic Dirichlet field exposes the original O(h^2) term without
    # relying on a smooth test whose fourth derivative can vanish at the wall.
    yy = np.linspace(0., 1., 25)
    quartic = yy**4-yy
    quartic_exact = 12*yy**2
    quartic_errors = [float(abs(result[0]-quartic_exact[0])) for result in (
        d2_matrix(np.ones(25), left, inside)@quartic*24**2,
        d@(d@quartic)*24**2)]

    def manufactured(n):
        y = np.linspace(0., 1., n)
        step = 1/(n-1)
        extra = .15*np.exp(y)
        value = np.sin(np.pi*y) + .2*np.sin(3*np.pi*y) + extra - .15*(1+(np.e-1)*y)
        uy = np.pi*(np.cos(np.pi*y)+.6*np.cos(3*np.pi*y)) + extra - .15*(np.e-1)
        uyy = -np.pi**2*(np.sin(np.pi*y)+1.8*np.sin(3*np.pi*y)) + extra
        b = 1+.2*np.exp(y)
        exact = (b-1)*uy + b*uyy
        closure = d2_matrix(b, left, inside)@value/step**2
        hh, dd = ast_d1(n)
        composed = dd@(b*(dd@value))/step**2
        width = np.flatnonzero(abs(s)>1.e-13)[-1]+1
        flux = np.array([b[0]*(s[:width]@value[:width]), -b[-1]*(s[:width]@value[-1:-width-1:-1])])/step
        flux_error = max(abs(flux-b[[0,-1]]*uy[[0,-1]]))
        e = np.eye(n)[:, [0,-1]]
        s0 = np.pad(s[:width], (0, n-width))
        normal_flux = np.array([-b[0]*s0, -b[-1]*s0[::-1]])
        d2 = d2_matrix(b, left, inside)
        m = -hh[:, None]*d2 + e@normal_flux
        # In this dense scalar reference only, use an exact matrix borrowing
        # bound. This is not a production prescription for the SAT strength.
        bound = np.linalg.eigvalsh(normal_flux@np.linalg.pinv(m, hermitian=True)@normal_flux.T)[-1]
        penalty = 2*bound
        weak = -m + e@normal_flux + normal_flux.T@e.T - penalty*(e@e.T)
        if np.linalg.eigvalsh(weak)[-1] > 1.e-12:
            raise AssertionError('homogeneous scalar Dirichlet SAT grows discrete energy')
        solved = np.linalg.solve(weak, hh*exact*step**2)
        numerical_flux = (normal_flux@solved - penalty*(e.T@solved))/step
        physical_flux = np.array([-b[0]*uy[0], b[-1]*uy[-1]])
        return [max(abs(closure[:9]-exact[:9])), max(abs(closure[9:-9]-exact[9:-9])),
                max(abs(composed[:9]-exact[:9])), flux_error, max(abs(solved-value)),
                max(abs(numerical_flux-physical_flux))]

    n = (25, 49, 97, 193)
    errors = np.array([manufactured(k) for k in n])
    return dict(source_sha256=SOURCE_SHA256, norm_operator_sbp_error=d1_sbp,
                variable_coefficient_sbp_symmetry=sbp_symmetry,
                scaled_minimum_m_eigenvalue=m_min, scaled_minimum_compatibility_r_eigenvalue=r_min,
                strictly_positive_coefficient_r_eigenvalue=positive_r_min,
                separate_viscous_d1_shift=shift, matched_r_eigenvalue=matched_min,
                matched_first_derivative_sbp_error=matched_sbp, quartic_wall_errors_d2_and_composed=quartic_errors,
                worst_coefficient_index=worst_basis, wall_derivative=s[:9].tolist(), samples=n,
                columns=['lower_D2_linf', 'interior_D2_linf', 'lower_D1_B_D1_linf', 'wall_flux_linf',
                         'scalar_Dirichlet_solution_linf', 'scalar_Dirichlet_numerical_flux_linf'],
                errors=errors.tolist(), orders=np.log2(errors[:-1]/errors[1:]).tolist())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('coefficients', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.coefficients), indent=2))
