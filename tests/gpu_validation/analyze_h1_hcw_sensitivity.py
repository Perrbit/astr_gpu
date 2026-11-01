#!/usr/bin/env python3
"""Directional sensitivity of the native HCW RHS along an observed state error."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from run_native_precision_tgv_gate import physical_state


def rank_rhs(case, label, step=0, rank=0):
    path = case/'diagnostics'/f'state.{label}.step{step:08d}.rk01.rank{rank:08d}.bin'
    with path.open('rb') as stream:
        header = np.fromfile(stream, np.int32, 4)
        shape = tuple(header[:3]+1)+(int(header[3]),)
        return np.fromfile(stream, np.float64).reshape(shape, order='F')


def native_rhs(case, label, step=0):
    pieces = [rank_rhs(case, label, step, rank) for rank in (0, 1)]
    return np.concatenate([pieces[0][:-1], pieces[1]], axis=0)


def native_q(case, step=0):
    pieces = [physical_state(case/'diagnostics'/
              f'state.pre_rhs.step{step:08d}.rk01.rank{rank:08d}.bin', 'fp64') for rank in (0, 1)]
    return np.concatenate([pieces[0][:-1], pieces[1]], axis=0)


def rates(case, jacobian, sign):
    raw = {label: native_rhs(case, label)[..., :-1, :]
           for label in ('conv_after_x', 'conv_after_y', 'conv_after_z', 'conv',
                         'budget_common', 'budget_low', 'full', 'limited')}
    return dict(x=sign*raw['conv_after_x']/jacobian,
                y=sign*(raw['conv_after_y']-raw['conv_after_x'])/jacobian,
                z=sign*(raw['conv_after_z']-raw['conv_after_y'])/jacobian,
                diffusion=raw['budget_common']/jacobian,
                complete=raw['full']/jacobian,
                low_reference=raw['budget_low']/jacobian,
                limiter=(raw['limited']-raw['full'])/jacobian,
                boundary=(raw['full']-raw['conv']-raw['budget_common'])/jacobian)


def analyze(root, step=840, backends=('cpu', 'gpu')):
    rows = []
    for backend in backends:
        sign = -1 if backend == 'cpu' else 1
        cases = [root/f'{source}{step}_on_{backend}' for source in ('cpu', 'gpu')]
        with h5py.File(cases[0]/'outdat/new/resources/geometry.h5') as file:
            jacobian = file['q0004'][...].transpose(2, 1, 0)[..., :-1, None]
        q = [native_q(case)[..., :-1, :] for case in cases]
        value = [rates(case, jacobian, sign) for case in cases]
        delta = q[1]-q[0]
        changes = {label: value[1][label]-value[0][label] for label in value[0]}
        advanced = []
        for case in cases:
            with h5py.File(case/'outdat/new/checkpoints/step000000000001/state.h5') as file:
                advanced.append(np.stack([file[f'q{i:04d}'][...].transpose(2,1,0)
                                          for i in range(1,6)], axis=-1)[..., :-1, :])
        masks = {}
        for name, xstop in (('inlet_three_columns', 4), ('physical_interior', 64)):
            mask = np.zeros(delta.shape[:-1], dtype=bool)
            mask[1:xstop, 1:-1, :] = True
            masks[name] = mask
        results = {}
        for region, mask in masks.items():
            assert all(np.isfinite(rate[mask]).all() for rate in changes.values())
            rho_error = delta[..., 0][mask]
            weight = jacobian[..., 0][mask]
            norm2 = np.sum(weight*rho_error*rho_error)
            assert norm2 > 0
            production = {label: float(np.sum(weight*rho_error*rate[..., 0][mask])/norm2)
                          for label, rate in changes.items()}
            norm5 = np.sum(jacobian[mask]*delta[mask]**2)
            dt = 1.919378420752319e-4
            results[region] = dict(density_error_weighted_l2=float(np.sqrt(norm2)),
                density_error_log_norm_rate=production,
                all5_error_log_norm_rate={label: float(np.sum(jacobian[mask]*delta[mask]*rate[mask])/norm5)
                                         for label, rate in changes.items()},
                native_full_step_error_norm_factor=float(np.sqrt(np.sum(
                    jacobian[mask]*(advanced[1]-advanced[0])[mask]**2)/norm5)),
                frozen_euler_error_norm_factor=float(np.sqrt(np.sum(
                    weight*(rho_error+dt*changes['complete'][..., 0][mask])**2)/norm2)),
                decomposition_residual_max=float(abs((changes['x']+changes['y']+changes['z']+
                    changes['diffusion']+changes['boundary']-changes['complete'])[mask]).max()))
        rows.append(dict(backend=backend, regions=results))
    return dict(scope='same backend, two frozen observed states; native instantaneous sensitivity, not a flow integrator',
                norm='Jacobian-weighted nodal density-error L2; unique z nodes; physical faces excluded; not physical energy',
                rate='dot(delta_rho, delta_Rrho)/norm(delta_rho)^2; inverse nondimensional time',
                initial_states=f'CPU and GPU step{step} primitives reconstructed as diagnostic inputs', cases=rows)


def endpoint_decomposition(root, step):
    cases = [root/f'endpoints_{source}{step}' for source in ('cpu', 'gpu')]
    samples = [rank_rhs(case, 'h1_inlet_flux_endpoints')[:4, 1:-1, :-1] for case in cases]
    high, low, theta = [[value[..., section] for value in samples]
                        for section in (slice(0,5), slice(5,10), slice(10,11))]
    with h5py.File(cases[0]/'outdat/new/resources/geometry.h5') as file:
        jacobian = file['q0004'][...].transpose(2,1,0)[1:4, 1:-1, :-1, None]
    delta = (native_q(cases[1])-native_q(cases[0]))[1:4, 1:-1, :-1]
    norm = np.sum(jacobian[..., 0]*delta[..., 0]**2)
    norm5 = np.sum(jacobian*delta**2)
    flux_changes = dict(high_endpoint=high[1]-high[0], limited_endpoint=low[1]-low[0],
        frozen_weight=(1-theta[0])*(high[1]-high[0])+theta[0]*(low[1]-low[0]),
        weight_feedback=(theta[1]-theta[0])*(low[1]-high[1]))
    flux_changes['total'] = flux_changes['frozen_weight']+flux_changes['weight_feedback']
    rates = {label: (value[:-1]-value[1:])/jacobian for label, value in flux_changes.items()}
    actual = -(native_rhs(cases[1], 'conv_after_x')-native_rhs(cases[0], 'conv_after_x'))[1:4,1:-1,:-1]/jacobian
    assert all(np.isfinite(value).all() for value in rates.values()) and np.isfinite(actual).all()
    project = lambda rate: float(np.sum(jacobian[..., 0]*delta[..., 0]*rate[..., 0])/norm)
    project5 = lambda rate: float(np.sum(jacobian*delta*rate)/norm5)
    return dict(scope='first three inlet-adjacent columns, native CPU flux endpoints, same frozen state pair',
        identity='deltaF = (1-thetaA)deltaH + thetaA deltaL + deltaTheta*(LB-HB)',
        density_log_norm_rates={label: project(value) for label,value in rates.items()},
        all5_log_norm_rates={label: project5(value) for label,value in rates.items()},
        actual_density_x_rate=project(actual), actual_all5_x_rate=project5(actual),
        reconstruction_residual_absolute=float(abs(rates['total']-actual).max()),
        theta_difference_max=float(abs(theta[1]-theta[0]).max()),
        note='Euclidean diagnostic norm, not a physical energy or eigenvalue stability proof')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--step', type=int, default=840)
    parser.add_argument('--backends', nargs='+', choices=('cpu', 'gpu'), default=['cpu', 'gpu'])
    parser.add_argument('--endpoints', action='store_true')
    args = parser.parse_args()
    report = endpoint_decomposition(args.root.resolve(), args.step) if args.endpoints else \
        analyze(args.root.resolve(), args.step, args.backends)
    label = 'endpoints' if args.endpoints else 'sensitivity'
    (args.root/f'{label}_{args.step}.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
