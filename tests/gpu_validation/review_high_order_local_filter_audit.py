"""Single-filter accuracy and finite stage-map audit, never a CFD integrator."""
import argparse
from fractions import Fraction
import json
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.linalg import svdvals
import sympy as sp

from review_high_order_current_local_filter import local_blocks, phase_comparison
from review_high_order_filter_eigenpair import (
    exact_filter, matvec, sparse_rows, tangential_unit_certificate, unit_certificates)
from review_high_order_p6_filter_compatibility import load_native
from review_high_order_wall_filter_design import exact_rows
from test_high_order_wall_filter_coupling import rk3_filtered_map


def adiabatic_density_taylor(parameters, reference='extra'):
    """Resolve density truncation terms without changing coefficients or grids."""
    z = sp.symbols('z')
    expression = 1+sp.cos(z)/100 if reference == 'wall-manufactured' else (1+sp.cos(z)/50)/(1+sp.cos(z)/20)
    angle = 1 if reference == 'wall-manufactured' else 2
    series = sp.series(expression, z, 0, 18).removeO().expand()
    row = exact_rows(parameters)[3]
    delta = [sp.Rational(value)-int(j == 4) for j, value in enumerate(row['coefficients'])]
    samples = []
    with mp.workdps(80):
        for n in (32, 64, 128):
            terms = {}
            for k in range(6, 18, 2):
                coefficient = series.coeff(z, k)*sum(value*j**k for j, value in enumerate(delta))
                terms[k] = mp.mpf(int(coefficient.p))/int(coefficient.q)*(angle*mp.pi/n)**k
            rho = [1+mp.cos(mp.pi*j/n)/100 if reference == 'wall-manufactured' else
                   (1+mp.cos(2*mp.pi*j/n)/50)/(1+mp.cos(2*mp.pi*j/n)/20) for j in range(9)]
            actual = mp.fsum(mp.mpf(int(weight.p))/int(weight.q)*value for weight, value in zip(delta, rho))
            samples.append(dict(intervals=n, signed_error=float(actual),
                                signed_terms={k: float(value) for k, value in terms.items()},
                                series_through_16=float(mp.fsum(terms.values())),
                                remaining_series_tail=float(actual-mp.fsum(terms.values()))))
    leading = series.coeff(z, 6)*sum(value*j**6 for j, value in enumerate(delta))
    return dict(reference=reference, angle_factor_pi=angle,
                field='rho=1+.01cos(pi y)' if reference == 'wall-manufactured' else
                      'rho=(1+.02cos(2pi y))/(1+.05cos(2pi y))', row=4,
                exact_leading_coefficient_in_angle_h=str(leading),
                exact_density_series={k: str(series.coeff(z, k)) for k in range(6, 18, 2)},
                samples=samples,
                scope='single row Taylor diagnosis; no gate, field or grid change authorized')


def reference_state(y, kind, reference):
    if reference == 'wall-manufactured':
        rho = 1+mp.cos(mp.pi*y)/100
        temperature = 1+mp.sin(mp.pi*y)**2/50
        u = mp.sin(mp.pi*y)/10 if kind != 421 else mp.mpf(1)/10+mp.cos(mp.pi*y)/50
        v = mp.sin(2*mp.pi*y)/100
        w = mp.sin(mp.pi*y)/50 if kind != 421 else mp.cos(2*mp.pi*y)/50
        cv = mp.mpf(1250)/63  # const6/const2, gamma=1.4 and Ma=0.3.
    elif reference == 'extra':
        pressure = 1+mp.cos(2*mp.pi*y)/50
        temperature = 1+(mp.sin(mp.pi*y) if kind == 41 else mp.cos(2*mp.pi*y))/20
        rho = pressure/temperature
        u = mp.sin(mp.pi*y)/5 if kind != 421 else 1+mp.cos(2*mp.pi*y)/10
        v = mp.mpf(0)
        w = mp.sin(mp.pi*y)/10 if kind != 421 else mp.cos(mp.pi*y)/10
        cv = mp.mpf(5)/2
    else:
        raise ValueError('Unknown single-filter reference')
    if y in (0, 1):
        v = mp.mpf(0)
        if kind != 421:
            u = w = mp.mpf(0)
    return [rho, rho*u, rho*v, rho*w, rho*(cv*temperature+(u*u+v*v+w*w)/2)], cv


def single_mapping_accuracy(matrices, parameters, kind, reference='extra'):
    """One nonlinear conservative filter and wall projection, not a time step."""
    samples = []
    with mp.workdps(80):
        weights = sp.finite_diff_weights(1, list(range(8)), 0)[1][-1]
        weights = [mp.mpf(int(x.p))/int(x.q) for x in weights]
        for n in (32, 64, 128):
            scalar = exact_filter(matrices[n], parameters, kind)[:n+1, :n+1]
            rows = sparse_rows(scalar)
            original = []
            for j in range(n+1):
                y = mp.mpf(j)/n
                state, cv = reference_state(y, kind, reference)
                original.append(state)
            fields = [matvec(rows, [q[k] for q in original]) for k in range(5)]
            primitive_velocity = [[fields[k][j]/fields[0][j] for j in range(n+1)] for k in (1, 2, 3)]
            temperature = [(fields[4][j]/fields[0][j]-sum(v[j]**2 for v in primitive_velocity)/2)/cv
                           for j in range(n+1)]
            for wall, indices in ((0, range(1, 8)), (n, range(n-1, n-8, -1))):
                def neumann(values):
                    return -mp.fsum(weights[k]*values[j] for k, j in enumerate(indices, 1))/weights[0]
                velocity = [neumann(v) if kind == 421 and k != 1 else mp.mpf(0)
                            for k, v in enumerate(primitive_velocity)]
                tw = mp.mpf(1) if kind == 41 else neumann(temperature)
                rho = fields[0][wall]
                for k, v in enumerate(velocity, 1):
                    fields[k][wall] = rho*v
                fields[4][wall] = rho*(cv*tw+mp.fsum(v*v for v in velocity)/2)
            strip = sorted(set(range(8)) | set(range(n-7, n+1)))
            errors = [max(abs(fields[k][j]-original[j][k]) for j in strip) for k in range(5)]
            temperatures = [(fields[4][j]/fields[0][j]-
                             mp.fsum((fields[k][j]/fields[0][j])**2 for k in (1, 2, 3))/2)/cv
                            for j in range(n+1)]
            wall_normal = max(abs(fields[2][j]/fields[0][j]) for j in (0, n))
            wall_density = max(abs(fields[0][j]-original[j][0]) for j in (0, n))
            if kind == 41:
                thermal_residual = max(abs(temperatures[j]-1) for j in (0, n))
            else:
                thermal_residual = max(abs(mp.fsum(weights[k]*temperatures[j]
                                                   for k, j in enumerate(indices)))
                                       for indices in (range(8), range(n, n-8, -1)))
            samples.append(dict(intervals=n, near_strip_Linf=[float(x) for x in errors],
                                minimum_density=float(min(fields[0])), minimum_temperature=float(min(temperatures)),
                                wall_normal_velocity_maximum=float(wall_normal),
                                wall_density_change=float(wall_density), thermal_wall_constraint_residual=float(thermal_residual)))
    errors = np.array([row['near_strip_Linf'] for row in samples])
    orders = {}
    for k, name in enumerate(('rho', 'rho_u', 'rho_v', 'rho_w', 'rho_E')):
        if np.all(errors[:, k] == 0):
            orders[name] = dict(zero_field=True, orders=None)
        else:
            values = np.log2(errors[:-1, k]/errors[1:, k])
            orders[name] = dict(zero_field=False, orders=values.tolist())
    passed = all(entry['zero_field'] or min(entry['orders']) >= 5.5 for entry in orders.values())
    return dict(bctype=kind, reference=reference, arithmetic_decimal_digits=80, samples=samples, components=orders,
                passes_observed_order=passed,
                field=('rho=1+.01cos(pi y); T=1+.02sin(pi y)^2; existing manufactured wall fixture'
                       if reference == 'wall-manufactured' else
                       'p=1+.02cos(2pi y); T=1+.05sin(pi y) for41, 1+.05cos(2pi y) otherwise; rho=p/T'),
                eos=('const2=.126; const6=2.5; p=rho*T/.126' if reference == 'wall-manufactured'
                     else 'R=1; gamma=1.4; p=rho*T'),
                velocity=('existing wall fixture: u=.1sin(pi y), v=.01sin(2pi y), w=.02sin(pi y); '
                          '421 u=.1+.02cos(pi y), w=.02cos(2pi y)' if reference == 'wall-manufactured' else
                          'normal zero; no-slip sin(pi y) or slip cosines as recorded in source'),
                phase='one conservative filter, then nonlinear temperature/velocity wall reconstruction',
                scope='single mapping only; not full-stage or CFD convergence')


def transient_samples(matrices, values, kind):
    records = []
    for n in (8, 9, 32, 64, 128):
        for name, f, a0, diffusion in local_blocks(matrices[n], kind, values):
            for nu in (0., 1e-5, 1/1600):
                for cfl in (.001, .05, .2):
                    mapping = rk3_filtered_map(a0+nu*diffusion, f, cfl/(n*np.sqrt(1.4)))
                    power = mapping.copy()
                    trajectory = []
                    for exponent in range(11):
                        trajectory.append(dict(repetitions=2**exponent,
                                               euclidean_operator_norm=float(svdvals(power)[0])))
                        if exponent < 10:
                            power = power@power
                    records.append(dict(intervals=n, block=name, viscosity=nu, sound_cfl=cfl,
                                        trajectory=trajectory))
    return dict(scope='sampled 2-norms of finite matrix powers; no uniform energy-bound claim', samples=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, nargs='+', required=True)
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--kind', type=int, choices=(41, 42, 421), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--taylor-only', action='store_true')
    parser.add_argument('--reference', choices=('wall-manufactured', 'extra'), default='wall-manufactured')
    parser.add_argument('--unit-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    matrices = {}
    for path in args.native:
        matrices.update(load_native(path))
    design = json.loads(args.design.read_text())
    parameters = [Fraction(x) for x in design['rational_parameters']]
    if args.reference == 'wall-manufactured' or args.taylor_only:
        if args.reference == 'extra' and args.kind == 41:
            raise ValueError('Extra Taylor diagnosis covers the cosine adiabatic density only')
        report = adiabatic_density_taylor(parameters, args.reference)
        with (args.output/'density_taylor.json').open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
    if args.taylor_only:
        print('Taylor diagnosis saved; gate unchanged', flush=True)
        return
    values = np.array([float(x) for x in parameters[:7]])
    accuracy = single_mapping_accuracy(matrices, parameters, args.kind, args.reference)
    with (args.output/'accuracy.json').open('x') as stream:
        json.dump(accuracy, stream, indent=2, allow_nan=False)
    print('single_mapping_order', accuracy['passes_observed_order'], flush=True)
    if not accuracy['passes_observed_order']:
        raise SystemExit(1)
    certificates = []
    for n in (8, 9, 32, 64, 128):
        for nu in ('0', '1/1600'):
            for cfl in ('0', '0.0001', '0.2'):
                record = unit_certificates(matrices[n], parameters, args.kind, nu, cfl)
                record['tangential'] = tangential_unit_certificate(matrices[n], parameters, args.kind, nu, cfl)
                certificates.append(record)
                passed = record['unit_semisimple_certified'] and record['tangential']['unit_semisimple_certified']
                print('unit', n, nu, cfl, passed, flush=True)
                with (args.output/'unit_certificates.json').open('w') as stream:
                    json.dump(certificates, stream, indent=2, allow_nan=False)
                if not passed:
                    raise SystemExit('Exact unit-mode certificate unresolved; no downstream acceptance')
    if args.unit_only:
        return
    transient = transient_samples(matrices, values, args.kind)
    with (args.output/'transient.json').open('x') as stream:
        json.dump(transient, stream, indent=2, allow_nan=False)
    phases = [phase_comparison(matrices[n], args.kind, values, 0., cfl)
              for n, cfl in ((8, .2), (64, .0001), (128, .0001))]
    with (args.output/'phases.json').open('x') as stream:
        json.dump(phases, stream, indent=2, allow_nan=False)
    print('bounded audit finished; no production admission', flush=True)


if __name__ == '__main__':
    main()
