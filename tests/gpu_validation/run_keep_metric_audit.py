"""Inspect actual solver metrics with the periodic KEEP differentiation symbols.

This is a geometry diagnostic, not another flow solver or a geometry replacement.
"""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from run_output_restart_validation import run_case

ROOT = Path(__file__).resolve().parents[2]


def periodic_derivative(values, axis, scheme):
    n = values.shape[axis]
    theta = 2*np.pi*np.fft.fftfreq(n)
    if scheme == '643e':
        symbol = 1.5*np.sin(theta)-0.3*np.sin(2*theta)+np.sin(3*theta)/30
    elif scheme == '643c':
        symbol = ((14/9)*np.sin(theta)+np.sin(2*theta)/18)/(1+(2/3)*np.cos(theta))
    else:
        raise ValueError('periodic KEEP derivative must be 643e or 643c')
    shape = [1]*values.ndim
    shape[axis] = n
    return np.fft.ifft(1j*symbol.reshape(shape)*np.fft.fft(values, axis=axis), axis=axis).real


def audit_geometry(path, scheme):
    with h5py.File(path) as state:
        fields = [state[f'q{c:04d}'][:].transpose(2, 1, 0) for c in range(1, 14)]
    if not all(np.isfinite(value).all() for value in fields) or not (fields[3] > 0).all():
        raise AssertionError('invalid production geometry')
    jacobian = fields[3][:-1, :-1, :-1]
    full_area = np.stack([fields[3]*fields[4+a+3*b] for a in range(3) for b in range(3)], axis=-1)
    area = full_area[:-1, :-1, :-1].reshape(jacobian.shape+(3, 3))
    residual = sum(periodic_derivative(area[..., a, :], a, scheme) for a in range(3))
    # For constant physical state, arithmetic KEEP reduces to F dot sum_a D_a(J*dxi).
    density, pressure, gamma = 1.0, 1.0, 1.4
    velocity = np.array([0.2, 0.13, -0.17])
    energy = pressure/(gamma-1)+0.5*density*np.dot(velocity, velocity)
    flux = np.empty((3, 5))
    for b in range(3):
        flux[b, 0] = density*velocity[b]
        flux[b, 1:4] = density*velocity[b]*velocity
        flux[b, b+1] += pressure
        flux[b, 4] = (energy+pressure)*velocity[b]
    uniform_rhs = -np.einsum('...b,bc->...c', residual, flux)/jacobian[..., None]
    seam_error = []
    for axis in range(3):
        seam_error.append(float(np.max(abs(np.take(full_area, 0, axis=axis)-np.take(full_area, -1, axis=axis)))))
    report = dict(geometry=str(path), derivative=scheme, intervals=list(jacobian.shape),
        jacobian_min=float(jacobian.min()), jacobian_max=float(jacobian.max()),
        metric_identity_max_per_component=np.max(abs(residual), axis=(0, 1, 2)).tolist(),
        constant_state_rhs_max_per_component=np.max(abs(uniform_rhs), axis=(0, 1, 2)).tolist(),
        area_periodic_seam_max_per_axis=seam_error,
        interpretation='read-only diagnostic of solver-generated metrics using the exact periodic linear symbol')
    return report


def write_stretched_grid(path, shape):
    axes=[np.linspace(0,2*np.pi,n+1) for n in shape]
    mapped=[q+a*np.sin(mode*q) for q,a,mode in zip(axes,[0.15,0.10,0.05],[1,2,1])]
    coordinates=np.meshgrid(*mapped,indexing='ij')
    with h5py.File(path,'w') as state:
        for name,values in zip(['x','y','z'],coordinates):
            state[name]=values.transpose(2,1,0)


def mapped_accuracy(path, scheme, mapping='periodic'):
    """Independent analytic mapping and physical scalar gradient; no time integration."""
    with h5py.File(path) as state:
        fields = [state[f'q{c:04d}'][:-1, :-1, :-1].transpose(2, 1, 0) for c in range(1, 14)]
    shape = fields[0].shape
    h = 2*np.pi/np.array(shape)
    xi, eta, zeta = np.meshgrid(*(np.arange(n)*spacing for n, spacing in zip(shape, h)), indexing='ij')
    basis = np.zeros(shape+(3, 3))
    if mapping=='periodic':
        basis[..., 0, 0] = 1+0.15*np.cos(xi)*np.sin(eta)
        basis[..., 0, 1] = 0.15*np.sin(xi)*np.cos(eta)
        basis[..., 1, 1] = 1+0.15*np.cos(eta)*np.sin(zeta)
        basis[..., 1, 2] = 0.15*np.sin(eta)*np.cos(zeta)
        basis[..., 2, 0] = 0.15*np.sin(zeta)*np.cos(xi)
        basis[..., 2, 2] = 1+0.15*np.cos(zeta)*np.sin(xi)
    elif mapping=='stretched':
        for a,q,amplitude,mode in zip(range(3),[xi,eta,zeta],[0.15,0.10,0.05],[1,2,1]):
            basis[...,a,a]=1+amplitude*mode*np.cos(mode*q)
    else:
        raise ValueError('unknown analytic mapping')
    exact_jacobian = np.linalg.det(basis)
    exact_inverse = np.linalg.inv(basis)
    inverse = np.stack([fields[4+a+3*b] for a in range(3) for b in range(3)], axis=-1).reshape(shape+(3, 3))
    x, y, z = fields[:3]
    scalar = np.sin(x)+np.cos(2*y)+0.2*np.sin(z)
    gradient = sum(inverse[..., a, :]*periodic_derivative(scalar, a, scheme)[..., None] for a in range(3))
    exact_gradient = np.stack([np.cos(x), -2*np.sin(2*y), 0.2*np.cos(z)], axis=-1)
    return dict(jacobian_scaled_max_error=float(np.max(abs(fields[3]/np.prod(h)-exact_jacobian))),
        inverse_scaled_max_error=float(np.max(abs(inverse*h.reshape(3, 1)-exact_inverse))),
        physical_gradient_max_errors=np.max(abs(gradient-exact_gradient), axis=(0, 1, 2)).tolist(),
        scalar='sin(x)+cos(2*y)+0.2*sin(z)',mapping=mapping)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu', type=Path, required=True)
    parser.add_argument('--mpiexec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scheme', choices=['643e', '643c'], required=True)
    parser.add_argument('--grid',default='32,32,32')
    parser.add_argument('--geometry-only',action='store_true',help='Initialize and save step zero without advancing')
    parser.add_argument('--mapping',choices=['periodic','stretched'],default='periodic')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,vader,tcp', OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc', OMPI_MCA_opal_cuda_support='0')
    fixture = SimpleNamespace(output=args.output, executable=args.cpu.resolve(), mpiexec=args.mpiexec.resolve(),
        case='tgv', mode='steps', restart_step=1, initial_dimension=0, initial_restart=True,
        legacy_statistics=False, statistics=False, filter_workspace='scalar', force='feedback', axis='x',
        no_samples=True, wall_samples=False, unrestricted=True)
    runtime=None
    if args.geometry_only:
        runtime="""&run_control max_steps=0 /
&timestep mode='fixed', fixed_dt=1.d-3 /
&logging list_steps=1 /
&repair crashfix=f /
&statistics enabled=f /
"""
    mapping_options=dict(tgv_mapping='periodic')
    if args.mapping=='stretched':
        source=args.output/'stretched.h5'
        write_stretched_grid(source,tuple(map(int,args.grid.split(','))))
        mapping_options=dict(tgv_cartesian_grid=source)
    case, _ = run_case(fixture, ROOT, 'cpu', 1, 'metric_audit', 1, grid=args.grid,**mapping_options,
        scheme=args.scheme, lfilter=False, numerics={'convection_form': 'keep'}, checkpoint_interval=1,run_config_text=runtime)
    report = audit_geometry(case/'outdat/new/resources/geometry.h5', args.scheme)
    report['accuracy']=mapped_accuracy(case/'outdat/new/resources/geometry.h5',args.scheme,args.mapping)
    (args.output/'metric_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
