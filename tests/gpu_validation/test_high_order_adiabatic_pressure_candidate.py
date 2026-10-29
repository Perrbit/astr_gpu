"""Exact P6/T7 pressure reference; native solver checks live in separate tests."""
import argparse
from functools import lru_cache
import json
from pathlib import Path

import mpmath as mp
import numpy as np
import pytest
from scipy.linalg import eigvals
import sympy as sp

from test_high_order_adiabatic_wall_candidate import adiabatic_cns_matrix
from test_high_order_boundary_closures import derivative_rows
from test_high_order_explicit_wall_candidates import dynamic_wall_cns_matrix,state_projection


@lru_cache(None)
def hermite_data(degree):
    system=sp.Matrix([[sp.Integer(j)**p for p in range(degree+1)] for j in range(degree)]+
        [[int(p==1) for p in range(degree+1)]])
    inverse=system.inv()
    return tuple(sp.Matrix([[sp.Integer(-layer)**p for p in range(degree+1)]])*inverse
        for layer in (1,2,3))


@lru_cache(None)
def pressure_rows(degree):
    ghost=hermite_data(degree)
    central=[sp.Rational(v,60) for v in (-1,9,-45,0,45,-9,1)]
    rows=[]
    for row in (1,2):
        coefficients=sp.zeros(1,degree+1)
        for offset,weight in zip(range(-3,4),central):
            location=row+offset
            if location<0:coefficients+=weight*ghost[-location-1]
            else:coefficients[0,location]+=weight
        rows.append(coefficients)
    return tuple(rows)


def pressure_gradient(n,degree):
    matrix=np.zeros((n-1,n+1))
    for row in range(3,n-2):
        matrix[row-1,row-3:row+4]=n*np.array([-1,9,-45,0,45,-9,1])/60
    for row,weights in enumerate(pressure_rows(degree),start=1):
        values=np.array(list(weights[0,:degree]),float)*n
        matrix[row-1,:degree]=values
        matrix[n-row-1,-degree:]=-values[::-1]
    return matrix


def normal_model(n,viscosity,degree=6,kind=42):
    a=dynamic_wall_cns_matrix(n,viscosity) if kind==41 else adiabatic_cns_matrix(n,viscosity,'neumann7-projection')
    if degree==7:return a
    g=pressure_gradient(n,degree)
    thermal=state_projection(n,'dirichlet' if kind==41 else 'neumann')
    a[n+1:2*n,:n+1]=-g
    a[n+1:2*n,2*n:]=-g@thermal
    return a


def exact_model(n,viscosity):
    derivative=sp.zeros(n+1)
    for row,weights in enumerate(derivative_rows(True)):
        for j,value in enumerate(weights):
            derivative[row,j]=n*value;derivative[n-row,n-j]=-n*value
    for row in range(3,n-2):
        for offset,value in zip(range(-3,4),(-1,9,-45,0,45,-9,1)):
            derivative[row,row+offset]=sp.Rational(n*value,60)
    velocity=sp.zeros(n+1,n-1);velocity[1:n,:]=sp.eye(n-1)
    thermal=velocity.copy()
    weights=sp.finite_diff_weights(1,list(range(8)),0)[1][-1]
    for j in range(7):
        thermal[0,j]=-weights[j+1]/weights[0]
        thermal[n,n-2-j]=-weights[j+1]/weights[0]
    g=sp.zeros(n-1,n+1)
    for row in range(3,n-2):
        for offset,value in zip(range(-3,4),(-1,9,-45,0,45,-9,1)):
            g[row-1,row+offset]=sp.Rational(n*value,60)
    for row,weights in enumerate(pressure_rows(6),start=1):
        for j in range(6):
            g[row-1,j]=n*weights[j];g[n-row-1,n-j]=-n*weights[j]
    gradient=derivative*velocity
    return sp.BlockMatrix([
        [sp.zeros(n+1),-gradient,sp.zeros(n+1,n-1)],
        [-g,sp.Rational(4,3)*viscosity*(derivative*derivative*velocity)[1:n,:],-g*thermal],
        [sp.zeros(n-1,n+1),-sp.Rational(2,5)*gradient[1:n,:],
            sp.Rational(35,18)*viscosity*(derivative*derivative*thermal)[1:n,:]]]).as_explicit()


def pressure_convergence():
    rows=pressure_rows(6);samples=[]
    with mp.workdps(70):
        def p(x):return 10000+mp.exp(8*x)-8*x-4*(mp.exp(8)-1)*x*x
        def dp(x):return 8*mp.exp(8*x)-8-8*(mp.exp(8)-1)*x
        for n in (32,64,128):
            errors=[]
            for side in (0,1):
                values=[p(mp.mpf(j)/n if side==0 else 1-mp.mpf(j)/n) for j in range(6)]
                for row,weights in enumerate(rows,start=1):
                    actual=n*sum(mp.mpf(str(weights[j]))*(values[j]-values[0]) for j in range(1,6))
                    if side:actual=-actual
                    exact=dp(mp.mpf(row)/n if side==0 else 1-mp.mpf(row)/n)
                    errors.append(abs(actual-exact))
            samples.append(dict(intervals=n,maximum_error=float(max(errors))))
    e=np.array([s['maximum_error'] for s in samples])
    return dict(field='10000+exp(8y)-8y-4(exp(8)-1)y^2; exact endpoint derivative zero',
        arithmetic='70 decimal digits, isolates truncation from FP64 cancellation',
        samples=samples,orders=np.log2(e[:-1]/e[1:]).tolist())


def spectral_scan():
    samples=[]
    for n in (8,9,10,11,12,14,16,20,24,28,32,40,48,64,96,128):
        for viscosity in (0.,1e-8,1e-6,1e-5,1e-4,1/1600,.01):
            a=normal_model(n,viscosity)
            value=float(np.max(eigvals(a).real))
            tolerance=float(100*np.finfo(float).eps*np.linalg.norm(a,np.inf))
            sample=dict(intervals=n,viscosity=viscosity,maximum_real=value,
                roundoff_reference=tolerance,passes=value<=tolerance)
            samples.append(sample)
            if not sample['passes']:return samples,sample
    return samples,None


def test_pressure6_exact_hermite_conditions():
    for layer,weights in enumerate(hermite_data(6),start=1):
        for power in range(7):
            value=sum(weights[j]*sp.Integer(j)**power for j in range(6))+weights[6]*int(power==1)
            assert value==sp.Integer(-layer)**power
    for row,weights in enumerate(pressure_rows(6),start=1):
        for power in (0,2,3,4,5,6):
            assert sum(weights[j]*sp.Integer(j)**power for j in range(6))==(
                0 if power==0 else power*sp.Integer(row)**(power-1))


def test_pressure6_near_gradient_observed_order():
    assert min(pressure_convergence()['orders'])>=5.5


@pytest.mark.parametrize('n',[8,9,32])
@pytest.mark.parametrize('viscosity',[0.,1e-5,1/1600])
def test_pressure6_adiabatic_frozen_spectrum(n,viscosity):
    a=normal_model(n,viscosity)
    assert np.max(eigvals(a).real)<=100*np.finfo(float).eps*np.linalg.norm(a,np.inf)


@pytest.mark.parametrize('n',[8,9])
@pytest.mark.parametrize('viscosity',[sp.Rational(0),sp.Rational(1,1600)])
def test_pressure6_exact_zero_modes_are_semisimple(n,viscosity):
    a=exact_model(n,viscosity)
    domain=a.to_DM()
    assert domain.rank()==domain.matmul(domain).rank()
    numeric=normal_model(n,float(viscosity))
    assert np.linalg.norm(np.array(a,float)-numeric,np.inf)<=100*np.finfo(float).eps*np.linalg.norm(numeric,np.inf)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    samples,failure=spectral_scan()
    exact=[]
    for n in (8,9):
        for viscosity in (sp.Rational(0),sp.Rational(1,1600)):
            domain=exact_model(n,viscosity).to_DM()
            rank=domain.rank();square=domain.matmul(domain).rank()
            exact.append(dict(intervals=n,viscosity=str(viscosity),rank=rank,square_rank=square,
                semisimple_zero=rank==square))
    report=dict(status='pressure6 candidate only, not implemented',scope='uniform Cartesian FP64 42/421 rest-state normal block',
        unchanged='Neumann7 temperature; original density RHS and viscous operators; all interior relations',
        ghost_rows=[[str(x) for x in row] for row in hermite_data(6)],
        pressure_gradient_rows=[[str(x) for x in row] for row in pressure_rows(6)],
        convergence=pressure_convergence(),spectra=samples,first_failure=failure,exact_zero_modes=exact,
        limitations='No native candidate RHS, nonlinear or long-window acceptance; 41 and filtering not admitted')
    with (args.output/'proposal_report.json').open('x') as stream:
        json.dump(report,stream,indent=2);stream.write('\n')
    print('samples',len(samples),'first_failure',failure,flush=True)
    print('convergence',json.dumps(report['convergence']),flush=True)
    print('exact_zero_modes',json.dumps(exact),flush=True)
