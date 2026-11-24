"""Independently verify the compact convection/explicit viscosity split in saved F0 TGV stages."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from run_keep_w1_mpi_gate import assemble
from run_cpp_tgv_gate import compare


def derivative(values, axis, compact):
    theta = 2*np.pi*np.fft.fftfreq(values.shape[axis])
    if compact:
        symbol = (14/9*np.sin(theta)+1/18*np.sin(2*theta))/(1+2/3*np.cos(theta))
    else:
        symbol = 1.5*np.sin(theta)-.3*np.sin(2*theta)+np.sin(3*theta)/30
    shape = [1]*values.ndim
    shape[axis] = len(theta)
    return np.fft.ifft(np.fft.fft(values, axis=axis)*(1j*symbol.reshape(shape)), axis=axis).real


def convection(q, u, p, jac, metric, compact):
    rhs = np.zeros_like(q)
    for a in range(3):
        normal_velocity = np.sum(u*metric[..., a, :], axis=-1)
        flux = q*normal_velocity[..., None]
        flux[..., 1:4] += p[..., None]*metric[..., a, :]
        flux[..., 4] += p*normal_velocity
        rhs -= derivative(jac[..., None]*flux, a, compact)
    return rhs


def viscosity(u, temperature, jac, metric, compact):
    computational = np.stack([derivative(u, a, compact) for a in range(3)], axis=-1)
    grad_u = np.einsum('...ma,...ab->...mb', computational, metric)
    computational_T = np.stack([derivative(temperature, a, compact) for a in range(3)], axis=-1)
    grad_T = np.einsum('...a,...ab->...b', computational_T, metric)
    # Frozen F0 nondimensional TGV: Re=1600, Ma=.1, gamma=1.4, T_ref=273.15 K.
    s = 110.3/273.15
    mu = temperature**1.5*(1+s)/(temperature+s)/1600
    strain = grad_u+np.swapaxes(grad_u, -1, -2)
    strain -= 2/3*np.trace(grad_u, axis1=-2, axis2=-1)[..., None, None]*np.eye(3)
    stress = mu[..., None, None]*strain
    work_and_heat = np.einsum('...mb,...m->...b', stress, u)
    work_and_heat += (mu/(.72*.4*.1**2))[..., None]*grad_T
    stress_energy = np.concatenate((stress, work_and_heat[..., None, :]), axis=-2)
    rhs = np.zeros((*jac.shape,5))
    for a in range(3):
        flux = np.einsum('...mb,...b->...m', stress_energy, metric[..., a, :])
        rhs[..., 1:] += derivative(jac[..., None]*flux, a, compact)
    return rhs


def check(case):
    unique = np.s_[:-1, :-1, :-1]
    with h5py.File(case/'outdat/new/resources/geometry.h5') as f:
        jac = f['q0004'][...].transpose(2,1,0)[unique]
        metric = np.empty((*jac.shape,3,3))
        for a in range(3):
            for b in range(3):
                metric[..., a, b] = f[f'q{5+a+3*b:04}'][...].transpose(2,1,0)[unique]
    evidence = []
    for rk in (1,2,3):
        q = assemble(case, 'pre_rhs', rk)[unique]
        primitives = assemble(case, 'pre_rhs_primitives', rk)[unique]
        u,p,t = primitives[..., 1:4],primitives[..., 4],primitives[..., 5]
        conv = assemble(case, 'conv', rk)[unique]
        diff = assemble(case, 'full', rk)[unique]-conv
        conv_error = compare(conv, convection(q,u,p,jac,metric,True))
        diff_error = compare(diff, viscosity(u,t,jac,metric,False))
        wrong_conv = compare(conv, convection(q,u,p,jac,metric,False), limit=None)
        wrong_diff = compare(diff, viscosity(u,t,jac,metric,True), limit=None)
        if min(max(wrong_conv), max(wrong_diff)) <= 2e-10:
            raise AssertionError('F0 snapshot does not distinguish explicit from compact')
        evidence.append(dict(compact_convection=conv_error, explicit_viscosity=diff_error,
                             wrong_explicit_convection=wrong_conv, wrong_compact_viscosity=wrong_diff))
    return evidence


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--f0-results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = {}
    for backend in ('cpu','gpu'):
        for ranks in (1,2):
            tag = f'tgv-c-np{ranks}-{backend}'
            report[tag] = check(args.f0_results/tag)
            print(f'{tag}: 643c convection and 643e viscosity matched at all three substeps')
    with args.output.open('x') as f:
        json.dump(report,f,indent=2)
        f.write('\n')
