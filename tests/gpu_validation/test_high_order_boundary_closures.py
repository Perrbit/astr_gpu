"""HB1 exact algebra and rejection tests, not a CFD time integrator.

The bare high-order advection candidates are intentionally NOT admitted.
Passing these tests means that their known instability remains detectable.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
import sympy as sp


def derivative_rows(matched):
    rows = []
    for row in range(3):
        if matched:
            matrix = sp.Matrix([[(j-row)**p for j in range(8)] for p in range(8)])
            moments = sp.Matrix([36 if p == 7 else int(p == 1) for p in range(8)])
            weights = matrix.inv()*moments
        else:
            weights = sp.finite_diff_weights(1, list(range(7)), row)[1][-1]
        rows.append(list(weights))
    return rows


def derivative_matrix(intervals, matched):
    matrix = np.zeros((intervals+1, intervals+1))
    for row, weights in enumerate(derivative_rows(matched)):
        count = len(weights)
        matrix[row, :count] = np.array(weights, dtype=float)
        matrix[intervals-row, -count:] = -np.array(weights, dtype=float)[::-1]
    for row in range(3, intervals-2):
        matrix[row, row-3:row+4] = np.array([-1, 9, -45, 0, 45, -9, 1])/60
    return matrix


def boundary_flux_rows():
    # Numerical FD flux, not ordinary pointwise interpolation at j+1/2.
    # B_p(j+1)-B_p(j)=p*j**(p-1) gives the correct flux-difference moments.
    matrix = sp.Matrix([[j**p for j in range(7)] for p in range(7)])
    return [list(matrix.inv()*sp.Matrix([sp.bernoulli(p, j+1) for p in range(7)]))
            for j in range(3)]


def mp_ld_linear_matrix(intervals, weight):
    flux = np.zeros((intervals, intervals+1))
    boundary = boundary_flux_rows()
    interior = np.array([-1/140, 5/84, -101/420, 319/420, 107/210, -19/210, 1/105, 0.])
    interior += weight/280*np.array([1, -7, 21, -35, 35, -21, 7, -1])
    for row in range(intervals):
        if row < 3:
            flux[row, :7] = np.array(boundary[row], dtype=float)
        elif row >= intervals-3:
            flux[row, -7:] = np.array(boundary[intervals-1-row], dtype=float)[::-1]
        else:
            flux[row, row-3:row+5] = interior
    derivative = np.zeros((intervals+1, intervals+1))
    derivative[1:-1] = flux[1:]-flux[:-1]
    derivative[-1, -7:] = -np.array(derivative_rows(False)[0], dtype=float)[::-1]
    return derivative


def stabilized_flux_rows():
    difference7 = [-1, 7, -21, 35, -35, 21, -7, 1]
    return [[w-sp.Rational(a, 8)*d for w, d in zip(row+[0], difference7)]
            for a, row in zip((1, 2, 1), boundary_flux_rows())]


def stabilized_mp_matrix(intervals, weight):
    derivative = mp_ld_linear_matrix(intervals, weight)
    difference7 = np.array([-1, 7, -21, 35, -35, 21, -7, 1.])
    for row, strength in enumerate((-1/8, -1/4, -1/8)):
        if row > 0:
            derivative[row, :8] += strength*difference7
        derivative[row+1, :8] -= strength*difference7
    derivative[-1] = 0
    derivative[-1, -8:] = -np.array(derivative_rows(True)[0], dtype=float)[::-1]
    return derivative


def filter6_matrix(intervals):
    matrix = np.eye(intervals+1)
    difference6 = np.array([1, -6, 15, -20, 15, -6, 1.])/64
    for row in range(3, intervals-2):
        matrix[row, row-3:row+4] += difference6
    matrix[1, :7] += difference6
    matrix[2, :7] -= difference6
    matrix[-2, -7:] += difference6[::-1]
    matrix[-3, -7:] -= difference6[::-1]
    return matrix


def characteristic_symbol_report():
    # Frozen two-dimensional Euler symbol. This is not nonlinear CFD validation.
    states = [(1.4905551784, .864020074, -.1620545653, 1.7614875854/5.6),
              (1.9173870077, .977070832, -.1022001539, 2.5729409244/68.6)]
    worst = -np.inf
    count = 0
    for rho, u, v, pressure in states:
        ax = np.array([[u,rho,0,0,0], [0,u,0,0,1/rho], [0,0,u,0,0],
                       [0,0,0,u,0], [0,1.4*pressure,0,0,u]])
        ay = np.array([[v,0,rho,0,0], [0,v,0,0,0], [0,0,v,0,1/rho],
                       [0,0,0,v,0], [0,0,1.4*pressure,0,v]])
        eigenvalues, right = np.linalg.eig(ay)
        transverse = np.linalg.solve(right, ax@right)
        weights = np.array([-1/140,5/84,-101/420,319/420,107/210,-19/210,1/105,0.])
        weights += .1/280*np.array([1,-7,21,-35,35,-21,7,-1])
        for n in (16, 32, 64):
            plus = stabilized_mp_matrix(n, weight=.1)
            # The seven-exterior-point derivative gives this incoming decay.
            plus[0, 0] = float(sp.harmonic(7))
            minus = -plus[::-1, ::-1]
            normal = np.zeros((5*(n+1), 5*(n+1)), dtype=complex)
            for mode, speed in enumerate(eigenvalues):
                section = slice(mode*(n+1), (mode+1)*(n+1))
                normal[section, section] = -speed*(plus if speed >= 0 else minus)
            for ratio in (.25, 1., 4.):
                for theta in (.1*np.pi, .3*np.pi, .6*np.pi):
                    symbol = (1-np.exp(-1j*theta))*np.dot(weights, np.exp(1j*theta*np.arange(-3,5)))
                    operator = normal-ratio*symbol*np.kron(transverse, np.eye(n+1))
                    worst = max(worst, float(np.linalg.eigvals(operator).real.max()))
                    count += 1
    return {'max_real_h_lambda': worst, 'samples': count,
            'scope': 'frozen Euler, HCW MP7-LD weight 0.1, candidate boundary rows and full transverse source'}


def steger_warming_physical_split(state, axis, spacing):
    """Physical flux from riemann::flux_steger_warming, not a flow integrator."""
    gamma = 1.4
    rho = state[0]; velocity = state[1:4]/rho
    pressure = (gamma-1)*(state[4]-.5*np.dot(state[1:4], velocity))
    sound = np.sqrt(gamma*pressure/rho)
    un = velocity[axis]
    total = state*un
    total[axis+1] += pressure
    total[4] += pressure*un
    # The production epsilon is 0.04 in computational eigenvalue units.
    if un.real >= sound.real:
        return total, np.zeros(5, dtype=state.dtype)
    if un.real <= -sound.real:
        return np.zeros(5, dtype=state.dtype), total
    eigenvalues = np.array([un, un+sound, un-sound])
    positive = .5*(eigenvalues+np.sqrt(eigenvalues**2+(.04*spacing)**2))
    a, b, c = positive[0], positive[1]-positive[2], 2*positive[0]-positive[1]-positive[2]
    d = a-c/(2*gamma)
    flux = np.empty(5, dtype=state.dtype)
    flux[0] = rho*d
    flux[1:4] = rho*d*velocity
    flux[axis+1] += rho*b*sound/(2*gamma)
    kinetic = .5*(gamma-1)*np.dot(velocity, velocity)
    flux[4] = a*state[4]+rho*(b*un*sound-c*(kinetic+sound**2)/(gamma-1))/(2*gamma)
    return flux, total-flux


def split_boundary_symbol_report():
    """Frozen 3D diagnostic with actual split Jacobians; no CFD propagation."""
    rho, u, v, pressure = 1.4905551784, .864020074, -.1620545653, 1.7614875854/5.6
    state = np.array([rho, rho*u, rho*v, 0., pressure/.4+.5*rho*(u*u+v*v)])
    hy, hz = 1/192, .125/8
    jacobians = []
    for axis, spacing in enumerate((hy, hy, hz)):
        plus, minus = np.zeros((5,5)), np.zeros((5,5))
        for column in range(5):
            perturbed = state.astype(complex)
            perturbed[column] += 1e-25j
            fp, fm = steger_warming_physical_split(perturbed, axis, spacing)
            plus[:,column] = fp.imag/1e-25
            minus[:,column] = fm.imag/1e-25
        jacobians.append((plus, minus))
    ay = jacobians[1][0]+jacobians[1][1]
    speeds, right = np.linalg.eig(ay)
    left = np.linalg.inv(right)
    outgoing = (right*np.where(speeds>0,speeds,0))@left
    incoming_decay = (right*np.where(speeds<0,speeds*float(sp.harmonic(7)),0))@left
    weights = np.array([-1/140,5/84,-101/420,319/420,107/210,-19/210,1/105,0.])
    weights += .1/280*np.array([1,-7,21,-35,35,-21,7,-1])
    rows = []
    for n in (16, 32, 64):
        dp = stabilized_mp_matrix(n, .1)
        dm = -dp[::-1,::-1]
        normal = -np.kron(dp[1:,1:],jacobians[1][0])-np.kron(dm[1:,1:],jacobians[1][1])
        # At upper y, replace incoming normal information, keep outgoing D6.
        normal[-5:,:] = 0.
        for s, coefficient in enumerate(derivative_rows(True)[0]):
            node = n-s
            normal[-5:,(node-1)*5:node*5] += float(coefficient)*outgoing
        normal[-5:,-5:] += incoming_decay
        for tx in (0., .2*np.pi):
            for tz in (.25*np.pi, .5*np.pi, .75*np.pi, np.pi):
                operator = normal.astype(complex)
                for axis, theta, ratio in ((0,tx,1.), (2,tz,hy/hz)):
                    symbol = (1-np.exp(-1j*theta))*np.dot(weights,np.exp(1j*theta*np.arange(-3,5)))
                    tangent = -ratio*(symbol*jacobians[axis][0]-symbol.conjugate()*jacobians[axis][1])
                    operator += np.kron(np.eye(n),tangent)
                eigenvalues, vectors = np.linalg.eig(operator)
                worst = int(eigenvalues.real.argmax())
                mode = vectors[:,worst].reshape(n,5)
                energy = np.sum(abs(mode)**2,axis=1)
                rows.append(dict(normal_intervals=n,theta_x_over_pi=tx/np.pi,theta_z_over_pi=tz/np.pi,
                                 max_real_hy_lambda=float(eigenvalues[worst].real),
                                 last_eight_nodes_mode_energy_fraction=float(energy[-8:].sum()/energy.sum())))
    return dict(scope='frozen smooth Steger-Warming split Jacobians, upper GRCBC, lower homogeneous fixed state; no sensor switching or CFD propagation',
                spacing_y=hy,spacing_z=hz,cases=rows,
                max_real_hy_lambda=max(row['max_real_hy_lambda'] for row in rows))


def advection_spectrum(derivative):
    # u_t + u_x=0, x in [0,1], u(0,t)=0 enforced strongly. Unit advection speed.
    # These are linearized trial closures; no claim about the nonlinear ASTR run.
    intervals = derivative.shape[0]-1
    eigenvalues = np.linalg.eigvals(-derivative[1:, 1:])
    z = 0.1*eigenvalues
    amplification = 1+z+z*z/2+z*z*z/6
    return {
        'max_real_h_lambda': float(eigenvalues.real.max()),
        'max_real_lambda_unit_domain': float(intervals*eigenvalues.real.max()),
        'rk3_spectral_amplification_at_cfl_0_1': float(np.abs(amplification).max()),
        'admitted': bool(eigenvalues.real.max() <= 1e-11),
    }


def rejection_report():
    candidates = {}
    for n in (32, 64, 128):
        for name, matched in [('plain_d6', False), ('matched_d6', True)]:
            candidates[f'{name}_{n}'] = advection_spectrum(derivative_matrix(n, matched))
        for weight in (0., 0.5, 1.):
            candidates[f'polynomial_mp_flux_w{weight}_{n}'] = advection_spectrum(mp_ld_linear_matrix(n, weight))
    return {
        'scope': 'HB1 algebraic trial operators; no ASTR propagation or production admission',
        'nodes': [33, 65, 129],
        'boundary': 'strong zero inflow; outgoing endpoint has backward D6',
        'derivative_rows': [[str(w) for w in row] for row in derivative_rows(True)],
        'flux_rows': [[str(w) for w in row] for row in boundary_flux_rows()],
        'candidates': candidates,
    }


def candidate_report():
    scalar, filters, diffusion, errors = {}, {}, {}, []
    for n in (32, 64, 128):
        matrix = stabilized_mp_matrix(n, weight=.1)
        scalar[n] = advection_spectrum(matrix)
        scalar[n]['reflected_max_real_h_lambda'] = float(np.linalg.eigvals(
            -matrix[1:, 1:][::-1, ::-1]).real.max())
        x = np.linspace(0, 1, n+1)
        indices = np.r_[1:5, n-4:n+1]
        error = float(np.max(np.abs((n*matrix@np.exp(4*x)-4*np.exp(4*x))[indices])))
        scalar[n]['boundary_strip_exp4_derivative_error'] = error
        errors.append(error)
        filtered = filter6_matrix(n)
        filters[n] = {
            'spectral_radius': float(np.abs(np.linalg.eigvals(filtered)).max()),
            'power_1000_Linf': float(np.linalg.norm(np.linalg.matrix_power(filtered, 1000), ord=np.inf)),
        }
        derivative = n*derivative_matrix(n, matched=True)
        for stretch in (0., 1.5, 3.5):
            jacobian = np.ones(n+1) if stretch == 0 else stretch*np.exp(stretch*x)/np.expm1(stretch)
            mu = 1+.2*np.cos(x)
            operator = (derivative@((mu/jacobian)[:, None]*derivative))/jacobian[:, None]
            spectrum = np.linalg.eigvals(operator[1:-1, 1:-1])
            diffusion[f'n{n}_stretch{stretch}'] = {
                'max_real_lambda': float(spectrum.real.max()),
                'max_magnitude_lambda': float(np.abs(spectrum).max()),
            }
    return {
        'scope': 'HB1-A candidate algebra and sampled linear symbols, not an ASTR propagation',
        'production_selected': False, 'mp_ld_weight': .1,
        'incoming_flux_strengths': ['1/8', '1/4', '1/8'],
        'incoming_flux_rows': [[str(value) for value in row] for row in stabilized_flux_rows()],
        'scalar': scalar, 'filters': filters, 'diffusion': diffusion,
        'boundary_strip_observed_orders': np.log2(np.array(errors[:-1])/errors[1:]).tolist(),
        'characteristic': characteristic_symbol_report(),
    }


def test_old_first_inner_mp_flux_difference_is_first_order():
    h = sp.Symbol('h', positive=True)
    values = [0, h*h, 4*h*h]
    first_flux = (values[0]+values[1])/2
    second_flux = (-values[0]+5*values[1]+2*values[2])/6
    assert sp.simplify((second_flux-first_flux)/h-2*h) == -h/3


def test_matched_derivative_has_the_same_leading_error_as_interior():
    interior = [sp.Rational(v, 60) for v in (-1, 9, -45, 0, 45, -9, 1)]
    assert sum(w*(j-3)**7 for j, w in enumerate(interior)) == 36
    for row, weights in enumerate(derivative_rows(True)):
        for p in range(8):
            expected = 36 if p == 7 else int(p == 1)
            assert sum(w*(j-row)**p for j, w in enumerate(weights)) == expected


def test_seventh_order_neumann_derivative_gives_eighth_order_state():
    weights = sp.finite_diff_weights(1, list(range(8)), 0)[1][-1]
    # Homogeneous Neumann manufactured fields satisfy f'(0)=0.
    for p in (0, 2, 3, 4, 5, 6, 7):
        state = -sum(weights[j]*j**p for j in range(1, 8))/weights[0]
        assert state == int(p == 0)
    assert -sum(weights[j]*j**8 for j in range(1, 8))/weights[0] != 0


def test_flux_difference_moments_are_not_midpoint_interpolation():
    flux = boundary_flux_rows()
    for row in (1, 2):
        for p in range(7):
            result = sum((flux[row][j]-flux[row-1][j])*j**p for j in range(7))
            assert result == (0 if p == 0 else p*row**(p-1))
    assert sum(flux[0][j]*j*j for j in range(7)) == sp.Rational(1, 6)


def test_unstabilized_candidates_are_rejected(tmp_path):
    report = rejection_report()
    (tmp_path/'unstabilized_candidates.json').write_text(json.dumps(report, indent=2)+'\n')
    for name, evidence in report['candidates'].items():
        assert evidence['max_real_h_lambda'] > 0.1, name
        assert evidence['rk3_spectral_amplification_at_cfl_0_1'] > 1, name
        assert not evidence['admitted'], name


def test_stabilized_flux_keeps_final_sixth_order_moments():
    rows = stabilized_flux_rows()
    old = boundary_flux_rows()
    for row in range(3):
        for p in range(7):
            assert sum(rows[row][j]*j**p for j in range(8)) == sp.bernoulli(p, row+1)
        assert rows[row] != old[row]+[0]
    for row in (1, 2):
        for p in range(7):
            actual = sum((rows[row][j]-rows[row-1][j])*j**p for j in range(8))
            assert actual == (0 if p == 0 else p*row**(p-1))


def test_stabilized_flux_transition_has_sixth_order_convergence():
    errors = []
    for n in (32, 64, 128):
        x = np.linspace(0, 1, n+1)
        indices = np.r_[1:5, n-4:n+1]
        errors.append(np.max(np.abs((n*stabilized_mp_matrix(n, .1)@np.exp(4*x)-4*np.exp(4*x))[indices])))
    assert np.all(np.log2(np.array(errors[:-1])/errors[1:]) >= 5.5)


@pytest.mark.parametrize('intervals', [32, 64, 128])
def test_hcw_weight_stabilized_scalar_spectrum(intervals):
    matrix = stabilized_mp_matrix(intervals, weight=0.1)
    evidence = advection_spectrum(matrix)
    assert evidence['max_real_h_lambda'] < 0
    assert evidence['rk3_spectral_amplification_at_cfl_0_1'] < 1
    # Reflection changes the wind, not the physical derivative sign convention.
    reverse = -matrix[::-1, ::-1]
    np.testing.assert_array_equal(reverse[:-1, :-1][::-1, ::-1], -matrix[1:, 1:])
    assert np.linalg.eigvals(reverse[:-1, :-1]).real.max() < 0


def test_sixth_order_filter_closure_moments_and_powers():
    for n in (32, 64, 128):
        matrix = filter6_matrix(n)
        x = np.arange(n+1, dtype=float)/n
        for p in range(6):
            np.testing.assert_allclose(matrix@x**p, x**p, atol=2e-15, rtol=2e-15)
        assert np.max(np.abs(np.linalg.eigvals(matrix))) <= 1+2e-12
        # A unit spectral radius is not enough to exclude a defective unit root.
        assert np.linalg.norm(np.linalg.matrix_power(matrix, 1000), ord=np.inf) < 4


def test_same_phase_characteristic_source_is_not_removed():
    report = characteristic_symbol_report()
    assert report['max_real_h_lambda'] < 2e-12
    assert report['samples'] == 54


def test_split_flux_jacobian_recovers_euler_characteristic_speeds():
    rho, velocity, pressure = 1.4905551784, np.array([.864020074,-.1620545653,0.]), 1.7614875854/5.6
    state = np.r_[rho,rho*velocity,pressure/.4+.5*rho*np.dot(velocity,velocity)]
    sound = np.sqrt(1.4*pressure/rho)
    for axis in range(3):
        total = np.zeros((5,5))
        for column in range(5):
            value = state.astype(complex); value[column] += 1e-25j
            plus, minus = steger_warming_physical_split(value,axis,1/192)
            total[:,column] = (plus+minus).imag/1e-25
        expected = np.sort([velocity[axis]]*3+[velocity[axis]-sound,velocity[axis]+sound])
        np.testing.assert_allclose(np.sort(np.linalg.eigvals(total)),expected,atol=2e-12,rtol=0)


@pytest.mark.skipif(not os.environ.get('ASTR_HIGH_ORDER_BOUNDARY_PROBE'),
                    reason='Set ASTR_HIGH_ORDER_BOUNDARY_PROBE to a root-CMake probe')
def test_compiled_cpu_gpu_component_accuracy():
    result = subprocess.run([os.environ['ASTR_HIGH_ORDER_BOUNDARY_PROBE']],
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout+result.stderr
    assert 'HB_COMPONENT_ACCURACY_PASS_NOT_SOLVER_STABILITY' in result.stdout
    rows = [line.split() for line in result.stdout.splitlines() if line.startswith('HB_WEIGHTS ')]
    assert len(rows) == 3
    for output, expected in zip(rows, derivative_rows(True)):
        np.testing.assert_allclose(np.array(output[2:], dtype=float), np.array(expected, dtype=float),
                                   rtol=2e-15, atol=2e-15)
    output = next(line.split()[1:] for line in result.stdout.splitlines()
                  if line.startswith('HB_NEUMANN_WEIGHTS'))
    expected = sp.finite_diff_weights(1, list(range(8)), 0)[1][-1]
    np.testing.assert_allclose(np.array(output, dtype=float), np.array(expected, dtype=float),
                               rtol=2e-15, atol=2e-15)
    flux_rows = [line.split() for line in result.stdout.splitlines() if line.startswith('HB_FLUX ')]
    assert len(flux_rows) == 6
    base = boundary_flux_rows()
    incoming = stabilized_flux_rows()
    for line in flux_rows:
        row, wind = map(int, line[1:3])
        expected = (incoming[row] if wind == 1 else base[row]+[0])
        np.testing.assert_allclose(np.array(line[3:], dtype=float), np.array(expected, dtype=float),
                                   rtol=2e-14, atol=2e-14)
    filter_rows = [line.split() for line in result.stdout.splitlines() if line.startswith('HB_FILTER ')]
    assert len(filter_rows) == 3
    for line in filter_rows:
        row = int(line[1])
        np.testing.assert_allclose(np.array(line[2:], dtype=float), filter6_matrix(16)[row, :8],
                                   rtol=2e-14, atol=2e-14)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate', action='store_true', help='Report the HB1-A candidate instead of rejected trials')
    parser.add_argument('--split-symbol', action='store_true', help='Diagnose the actual split-flux 3D boundary symbol')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = split_boundary_symbol_report() if args.split_symbol else (candidate_report() if args.candidate else rejection_report())
    args.output.write_text(json.dumps(report, indent=2)+'\n')
