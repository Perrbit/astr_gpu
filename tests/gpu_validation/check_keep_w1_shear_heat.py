"""W1 matched variable-coefficient SBP reference: mapped shear, heat and wall power."""
import argparse
import json
from pathlib import Path

import numpy as np

from check_keep_w1_scalar import ast_d1, audit, d2_matrix, operators


def normal_operators(n, left, inside, shift):
    h, d = ast_d1(n)
    v = np.array([1, -4, 6, -4, 1, 0])
    w = np.roll(v, 1)
    q = (np.outer(v, w) - np.outer(w, v))/h[:6, None]
    d[:6, :6] += shift*q
    d[-6:, -6:] -= shift*q[::-1, ::-1]
    e = np.eye(n)[:, [0, -1]]
    s = np.zeros(n)
    s[:5] = np.array([-25, 48, -36, 16, -3])/12
    outward = np.array([-s, -s[::-1]])
    patch = np.zeros(n)
    patch[:6] = 1
    a = -h[:, None]*d2_matrix(patch, left, inside) + e@(patch[[0, -1], None]*outward)
    r = a - d.T@((h*patch)[:, None]*d)
    defect = s-d[0]
    inverse = np.linalg.pinv(r, hermitian=True, rcond=1.e-12)
    if np.max(abs(r@inverse@defect-defect)) > 2.e-12:
        raise AssertionError('the six-coefficient patch does not bound the wall derivative defect')
    gamma = float(defect@inverse@defect)
    return h, d, e, outward, gamma


def run_grid(n, left, inside, shift):
    h, _, e, outward, gamma = normal_operators(n, left, inside, shift)
    eta = np.linspace(0., 1., n)
    dy = eta*(1-eta)*.35
    y = eta+dy
    jac = 1+.35*(1-2*eta)
    step = 1/(n-1)
    wv = step*h*jac
    mu = 1+.2*np.exp(y)
    kappa = .7+.3*np.exp(.6*y)
    du_mu = mu-1
    dkappa = .6*(kappa-.7)
    velocity = np.array([np.sin(np.pi*y), .2*np.sin(2*np.pi*y)])
    velocity_y = np.pi*np.array([np.cos(np.pi*y), .4*np.cos(2*np.pi*y)])
    velocity_yy = -np.pi**2*np.array([np.sin(np.pi*y), .8*np.sin(2*np.pi*y)])
    temp_delta = .2*y*(1-y)+.05*np.sin(2*np.pi*y)
    temp_y = .2*(1-2*y)+.1*np.pi*np.cos(2*np.pi*y)
    temp_yy = -.4-.2*np.pi**2*np.sin(2*np.pi*y)

    def scalar(coef):
        b = coef/jac
        q = b[[0, -1], None]*outward
        m = -h[:, None]*d2_matrix(b, left, inside)+e@q
        bmin = np.array([b[:6].min(), b[-6:].min()])
        bw = b[[0, -1]]
        penalty = 2*(bw/h[[0, -1]] + gamma*bw**2/bmin)
        a = -m + e@q + q.T@e.T - (e*penalty)@e.T
        eigmax = np.linalg.eigvalsh(a)[-1]
        if eigmax > 2.e-12:
            raise AssertionError('scalar weak wall matrix has a growing mode')
        return a, m, q, penalty, eigmax

    a, m, q, tau, eig_u = scalar(mu)
    a_t, _, q_t, tau_t, eig_t = scalar(kappa)
    exact_rhs_u = du_mu*velocity_y + mu*velocity_yy
    exact_rhs_t = dkappa*temp_y + kappa*temp_yy
    solved_u = np.linalg.solve(a, (step*wv*exact_rhs_u).T)
    solved_t = np.linalg.solve(a_t, step*wv*exact_rhs_t)
    coeff_trace = np.array([-1, 1])
    wall_shear = (q@solved_u - tau[:, None]*(e.T@solved_u))/step
    expected_shear = (coeff_trace*mu[[0, -1]])[:, None]*velocity_y[:, [0, -1]].T
    wall_heat = (q_t@solved_t - tau_t*(e.T@solved_t))/step
    exact_heat = coeff_trace*kappa[[0, -1]]*temp_y[[0, -1]]

    # At stationary walls the viscous work flux is zero: remove the trace
    # flux of the b*d(|u|^2/2) strong operator consistently on both ends.
    kinetic = .5*np.sum(velocity**2, axis=0)
    work = -m@kinetic/step
    heat = a_t@temp_delta/step
    prescribed_heat = (q_t@temp_delta - tau_t*(e.T@temp_delta))/step
    kinetic_rate = np.sum(velocity.T*(a@velocity.T))/step
    balance = abs(np.sum(work+heat)-prescribed_heat.sum())
    if balance > 2.e-12 or kinetic_rate > 0:
        raise AssertionError('stationary-wall energy or kinetic-work balance')
    exact_work = np.sum(du_mu*velocity*velocity_y+mu*(velocity_y**2+velocity*velocity_yy), axis=0)
    rhs_error = work/wv+heat/wv-exact_work-exact_rhs_t
    return dict(gamma=gamma, largest_wall_matrix_eigenvalue=max(eig_u, eig_t),
                kinetic_rate=kinetic_rate, net_work=float(work.sum()), energy_balance=balance,
                errors=[np.max(abs(solved_u-velocity.T)), np.max(abs(solved_t-temp_delta)),
                        np.max(abs(wall_shear-expected_shear)), np.max(abs(wall_heat-exact_heat)),
                        np.max(abs(rhs_error[:9])), np.max(abs(rhs_error[9:-9]))])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('coefficients', type=Path)
    args = parser.parse_args()
    src = args.coefficients
    left, inside = operators(src)
    shift = audit(src)['separate_viscous_d1_shift']
    sizes = [25, 49, 97, 193]
    cases = [run_grid(n, left, inside, shift) for n in sizes]
    err = np.array([case['errors'] for case in cases])
    print(json.dumps(dict(nodes=sizes, cases=cases, columns=['velocity', 'temperature', 'wall_shear', 'wall_heat',
                                                              'wall_energy_rhs', 'interior_energy_rhs'],
                          orders=np.log2(err[:-1]/err[1:]).tolist()), indent=2))


if __name__ == '__main__':
    main()
