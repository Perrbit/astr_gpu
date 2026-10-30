"""Bounded local-coefficient research on current P7/P6-T7 blocks, not CFD."""
import argparse
from fractions import Fraction
import json
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.linalg import block_diag, eig, eigvals
from scipy.optimize import LinearConstraint, minimize

from review_high_order_p6_filter_compatibility import current_normal, load_native, sample
from review_high_order_wall_filter_design import slow_filter_coefficients
from test_high_order_boundary_closures import derivative_matrix
from test_high_order_explicit_wall_candidates import state_projection
from test_high_order_wall_filter_candidates import sixth_order_family
from test_high_order_wall_filter_coupling import native_filter_matrices, rk3_filtered_map


def parameters7(values):
    """One shared central row at N8; never overwrite it with a different mirror."""
    values = np.asarray(values)
    return np.r_[values[:6], values[6], values[6]]


def local_blocks(original, kind, values):
    n = len(original)-1
    filtering = sixth_order_family(original, parameters7(values))
    pd = state_projection(n, 'dirichlet')
    pt = state_projection(n, 'dirichlet' if kind == 41 else 'neumann')
    a = current_normal(n, 0., kind)
    diffusion = current_normal(n, 1., kind)-a
    result = [('normal', block_diag(filtering, filtering[1:-1]@pd,
                                   filtering[1:-1]@pt), a, diffusion)]
    d = n*derivative_matrix(n, True)
    for mode in (('neumann',) if kind == 421 else ('dirichlet',)):
        p = state_projection(n, mode)
        f = filtering[1:-1]@p
        result.append(('tangent_'+mode, f, np.zeros_like(f), (d@d@p)[1:-1]))
    return result


def affine_training(matrices, kind):
    # All dependence on the seven coefficients is affine. Cache the approved
    # space blocks, projections and filter basis, not perturbed flow states.
    training = []
    for n, original in sorted(matrices.items()):
        zero = local_blocks(original, kind, np.zeros(7))
        basis = [local_blocks(original, kind, np.eye(7)[j]) for j in range(7)]
        for index, (name, f0, a, diffusion) in enumerate(zero):
            increments = np.array([entry[index][1]-f0 for entry in basis])
            for nu in (0., 1/1600):
                space = a+nu*diffusion
                for cfl in (0., .0001, .001, .05, .2):
                    training.append((n, name, nu, cfl, f0, increments, space))
    return training


def density_accuracy_training(matrices, kind=42):
    """High-precision affine density defects for the fixed nonlinear fixture."""
    result = []
    with mp.workdps(80):
        for n in (32, 64, 128):
            rho = [(1+mp.cos(2*mp.pi*j/n)/50)/
                   (1+(mp.sin(mp.pi*j/n) if kind == 41 else mp.cos(2*mp.pi*j/n))/20)
                   for j in range(n+1)]
            strip = sorted(set(range(8)) | set(range(n-7, n+1)))
            base = sixth_order_family(matrices[n], parameters7(np.zeros(7)))
            matrices7 = [sixth_order_family(matrices[n], parameters7(np.eye(7)[j]))-base
                         for j in range(7)]
            delta = base-np.eye(n+1)
            def defects(matrix):
                return np.array([float(mp.fsum(mp.mpf(float(matrix[i, j]))*rho[j]
                                               for j in np.flatnonzero(matrix[i]))) for i in strip])
            result.append((defects(delta), np.array([defects(matrix) for matrix in matrices7])))
    return result


def density_accuracy_errors(training, values):
    return [float(np.max(abs(base+np.tensordot(values, increments, axes=1))))
            for base, increments in training]


def project_density_accuracy(training, start, selected=None):
    """Nearest coefficient vector in one explicit finite-grid accuracy chamber."""
    active = [] if selected is None else list(selected)
    if selected is None:
        for base, increments in training[:2]:
            defects = base+np.tensordot(start, increments, axes=1)
            j = int(np.argmax(abs(defects)))
            active.append((j, float(np.sign(defects[j]))))
    coefficients, constants = [], []
    # 5.6 is a design margin for rational rounding, not a changed 5.5 gate.
    factor = 2**5.6
    for level in (0, 1):
        base, increments = training[level]
        fine, fine_increments = training[level+1]
        row, sign = active[level]
        for j in range(len(fine)):
            for direction in (-1., 1.):
                coefficients.append(sign*increments[:, row]+direction*factor*fine_increments[:, j])
                constants.append(sign*base[row]+direction*factor*fine[j])
    coefficients = np.array(coefficients)
    constants = np.array(constants)
    scale = np.maximum(np.linalg.norm(coefficients, axis=1), abs(constants))
    scale[scale == 0] = 1
    constraint = LinearConstraint(coefficients/scale[:, None], -constants/scale, np.inf)
    result = minimize(lambda x: float(np.dot(x-start, x-start)), start,
                      jac=lambda x: 2*(x-start), method='SLSQP', bounds=[(-2., 2.)]*7,
                      constraints=[constraint], options=dict(maxiter=200, ftol=1e-13))
    errors = density_accuracy_errors(training, result.x)
    return result, dict(method='nearest Euclidean coefficient projection with linear inequalities',
                        optimizer_success=bool(result.success),
                        fixed_manufactured_field=True, design_order_margin=5.6, acceptance_order=5.5,
                        selected_rows_and_signs=active, density_errors=errors,
                        observed_orders=np.log2(np.array(errors[:-1])/errors[1:]).tolist(),
                        maximum_scaled_constraint_violation=float(max(0.,
                            np.max(-(coefficients@result.x+constants)/scale))))


def objective(training, values, density=None):
    worst = 0.
    for n, _, _, cfl, base, increments, a in training:
        f = base+np.tensordot(values, increments, axes=1)
        matrix = f if cfl == 0 else rk3_filtered_map(a, f, cfl/(n*np.sqrt(1.4)))
        radius = float(np.max(abs(eigvals(matrix))))
        gate = 100*np.finfo(float).eps*max(1., np.linalg.norm(matrix, np.inf))
        # This weighting guides coefficient selection only. Review below uses
        # the original raw spectral gate, not an optimizer score as acceptance.
        worst = max(worst, (radius-1-gate)/max(cfl*cfl, 1e-6))
    if density is not None:
        errors = density_accuracy_errors(density, values)
        worst = max(worst, *(2**5.5*errors[j+1]/errors[j]-1 for j in range(2)))
    return float(worst)


def review(matrices, kind, values):
    samples = []
    for n, original in sorted(matrices.items()):
        blocks = local_blocks(original, kind, values)
        for nu in (0., 1e-6, 1e-5, 1e-4, 1/1600):
            for cfl in (None, 0., .0001, .001, .005, .01, .025, .05, .1, .15, .2):
                record = dict(intervals=n, **sample(blocks, nu, cfl, n))
                samples.append(record)
                if not record['passes']:
                    return samples, record
    return samples, None


def phase_comparison(original, kind, values, viscosity, cfl):
    """Separate phase diagnostics; one end-step filter is not a proposal."""
    n = len(original)-1
    _, f, a0, diffusion = local_blocks(original, kind, values)[0]
    a = a0+viscosity*diffusion
    dt = cfl/(n*np.sqrt(1.4))
    unfiltered = rk3_filtered_map(a, np.eye(len(a)), dt)
    maps = dict(unfiltered_rk=unfiltered, filter_alone=f,
                one_filter_after_rk=f@unfiltered, actual_stage_rk=rk3_filtered_map(a, f, dt))
    records = {}
    for name, matrix in maps.items():
        spectrum, left, right = eig(matrix, left=True, right=True)
        gate = float(100*np.finfo(float).eps*max(1., np.linalg.norm(matrix, np.inf)))
        near_unit = []
        for j in np.flatnonzero(abs(abs(spectrum)-1) <= gate):
            value, lvec, rvec = spectrum[j], left[:, j], right[:, j]
            pairing = abs(np.vdot(lvec, rvec))
            near_unit.append(dict(eigenvalue=[float(value.real), float(value.imag)],
                                  radius_excess=float(abs(value)-1),
                                  right_residual=float(np.linalg.norm(matrix@rvec-value*rvec, np.inf)),
                                  left_residual=float(np.linalg.norm(matrix.T@lvec-value.conjugate()*lvec, np.inf)),
                                  individual_condition=None if pairing == 0 else
                                  float(np.linalg.norm(lvec)*np.linalg.norm(rvec)/pairing)))
        records[name] = dict(spectral_radius=float(np.max(abs(spectrum))), roundoff_reference=gate,
                             eigenvalues=[[float(z.real), float(z.imag)] for z in spectrum],
                             near_unit_eigenpairs=near_unit,
                             caveat='Multiple fixed eigenvalues require the exact subspace certificate; individual vectors are not unique')
    return dict(intervals=n, bctype=kind, viscosity=viscosity, sound_cfl=cfl,
                scope='normal frozen block; original qsave/stage map retained', maps=records)


def run(matrices, kind, start, maxfev, training_intervals, density_accuracy=False,
        density_projection=False):
    training = affine_training({n: matrices[n] for n in training_intervals}, kind)
    density = density_accuracy_training(matrices, kind) if density_accuracy or density_projection else None
    history = []

    def score(values):
        value = objective(training, values, density)
        if not history or value < history[-1]['objective']:
            history.append(dict(objective=value, parameters=np.asarray(values).tolist()))
        return value

    baseline = score(start)
    projection = None
    if density_projection:
        result, projection = project_density_accuracy(density, start)
        best = result.x
        score(best)
    else:
        result = minimize(score, start, method='Powell', bounds=[(-2., 2.)]*7,
                          options=dict(maxfev=maxfev, xtol=1e-9, ftol=1e-10))
        best = np.array(history[-1]['parameters'])
    rational = [Fraction(float(x)).limit_denominator(4096) for x in best]
    rounded = np.array([float(x) for x in rational])
    reviews = {}
    for label, values in (('floating', best), ('rational', rounded)):
        records, failure = review(matrices, kind, values)
        reviews[label] = dict(samples=records, first_failure=failure)
    f, a = local_blocks(matrices[8], kind, best)[0][1:3]
    return dict(
        status='rejected' if reviews['floating']['first_failure'] or reviews['rational']['first_failure']
        else 'finite spectra only; transient, accuracy and nonlinear design still required',
        bctype=kind, space='P7' if kind == 41 else 'P6/T7',
        scope='static local coefficient design; no solver, CFD, GPU or production admission',
        coefficient_basis='three shifted sixth differences, seventh parameter shared by row4 mirror',
        unchanged='endpoint density, deep tenth-order rows, hm=5, stage filter/qsave/RK3 phase',
        training_intervals=list(training_intervals), training_viscosities=[0., 1/1600],
        training_sound_cfl=[0., .0001, .001, .05, .2],
        declared_review_intervals=sorted(matrices),
        settings=dict(method='SLSQP density projection' if density_projection else 'Powell',
                      maximum_function_evaluations=None if density_projection else maxfev, bounds=[-2., 2.]),
        density_projection=projection,
        density_accuracy_constraint=density is not None,
        density_constraint_field=('rho=(1+.02cos(2pi y))/(1+.05sin(pi y))' if kind == 41 else
                                  'rho=(1+.02cos(2pi y))/(1+.05cos(2pi y))')+
                                 '; unchanged 32/64/128 fixture',
        density_constraint_errors=None if density is None else density_accuracy_errors(density, best),
        optimizer_termination=str(result.message), evaluations=int(result.nfev),
        baseline_objective=baseline, best_objective=objective(training, best, density),
        floating_parameters=parameters7(best).tolist(),
        rational_parameters=[str(x) for x in parameters7(rational)],
        improvement_history=history, reviews=reviews,
        n8_phase_comparison=phase_comparison(matrices[8], kind, rounded, 0., .2),
        n8_inviscid_slow_modes=slow_filter_coefficients(f, a),
        caveat='Bounded search failure is not impossibility; unvisited review samples are not passed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start', type=Path, required=True)
    parser.add_argument('--maxfev', type=int, default=1200)
    parser.add_argument('--kind', type=int, choices=(41, 42, 421), required=True)
    parser.add_argument('--training', type=int, nargs='+', default=[8, 9, 16])
    parser.add_argument('--density-accuracy', action='store_true')
    parser.add_argument('--project-density', action='store_true')
    parser.add_argument('--review-only', action='store_true')
    parser.add_argument('--intervals', type=int, nargs='+')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    matrices = load_native(args.native)
    required = tuple(args.intervals) if args.intervals else (8, 9, 10, 11, 12, 16, 24, 32, 64, 128)
    missing = tuple(n for n in required if n not in matrices)
    if missing:
        matrices.update(native_filter_matrices(args.output/'native_missing', missing))
    previous = json.loads(args.start.read_text())['floating_parameters']
    if previous[6] != previous[7]:
        raise ValueError('Starting coefficients must have the symmetric N8 central row')
    if args.review_only:
        source = json.loads(args.start.read_text())
        rational = [Fraction(x) for x in source['rational_parameters']]
        samples, failure = review({n: matrices[n] for n in required}, args.kind,
                                  np.array([float(x) for x in rational[:7]]))
        report = dict(status='rejected' if failure else 'finite held-out spectra only; not admitted',
                      bctype=args.kind, rational_parameters=[str(x) for x in rational],
                      intervals=list(required), samples=samples, first_failure=failure,
                      coefficient_search=False)
    else:
        report = run(matrices, args.kind, np.array(previous[:7]), args.maxfev, args.training,
                     args.density_accuracy, args.project_density)
    report['start_source'] = str(args.start)
    with (args.output/'report.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(report['status'], flush=True)


if __name__ == '__main__':
    main()
