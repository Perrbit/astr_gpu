"""HB1-C static proposal analysis, not an ASTR implementation or time integrator."""
import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from test_high_order_boundary_closures import derivative_matrix
from test_high_order_explicit_wall_candidates import pressure_hermite_ghosts,state_projection


def neumann_gradient(intervals):
    """Centered D6 with the approved degree-seven Hermite Neumann extension."""
    ghost,_=pressure_hermite_ghosts()
    n=intervals
    extension=np.zeros((n+7,n+1));extension[3:n+4]=np.eye(n+1)
    for layer in (1,2,3):
        extension[3-layer,:7]=ghost[layer-1]
        extension[3+n+layer,-7:]=ghost[layer-1,::-1]
    weights=np.array([-1,9,-45,0,45,-9,1])/60
    return n*np.array([weights@extension[row:row+7] for row in range(n+1)])


def adiabatic_cns_matrix(intervals,viscosity,temperature_closure='dynamic-ghost'):
    """rho and T evolve at walls; only wall normal velocity is constrained."""
    n=intervals
    derivative=n*derivative_matrix(n,True)
    gradient=neumann_gradient(n)
    fixed_velocity=state_projection(n,'dirichlet')
    velocity_gradient=derivative@fixed_velocity
    velocity_laplacian=(derivative@derivative@fixed_velocity)[1:-1]
    thermal_laplacian=derivative@gradient
    zero=np.zeros((n+1,n+1))
    if temperature_closure=='neumann7-projection':
        temperature=state_projection(n,'neumann')
        return np.block([
            [zero,-velocity_gradient,np.zeros((n+1,n-1))],
            [-gradient[1:-1],(4/3)*viscosity*velocity_laplacian,-gradient[1:-1]@temperature],
            [np.zeros((n-1,n+1)),-.4*velocity_gradient[1:-1],
                (1.4/.72)*(viscosity*(derivative@derivative@temperature)[1:-1])]])
    if temperature_closure!='dynamic-ghost':
        raise ValueError('Unknown proposed thermal closure')
    return np.block([
        [zero,-velocity_gradient,zero],
        [-gradient[1:-1],(4/3)*viscosity*velocity_laplacian,-gradient[1:-1]],
        [zero,-.4*velocity_gradient,(1.4/.72)*viscosity*thermal_laplacian]])


def diffusion_sample(n,temperature_closure='dynamic-ghost'):
    x=np.linspace(0,1,n+1)
    # Both endpoint derivatives vanish, but the seventh derivative does not.
    a=4.
    f=np.exp(a*x)-a*x-a*(np.exp(a)-1)*x*x/2
    df=a*np.exp(a*x)-a-a*(np.exp(a)-1)*x
    ddf=a*a*np.exp(a*x)-a*(np.exp(a)-1)
    mu=1+.3*np.exp(2*x);dmu=.6*np.exp(2*x)
    derivative=n*derivative_matrix(n,True)
    if temperature_closure=='neumann7-projection':
        projected=state_projection(n,'neumann')@f[1:-1]
        actual=derivative@(mu*(derivative@projected))
    elif temperature_closure=='dynamic-ghost':
        projected=f
        actual=derivative@(mu*(neumann_gradient(n)@f))
    else:
        raise ValueError('Unknown proposed thermal closure')
    reference=mu*ddf+dmu*df
    strip=np.r_[0:8,n-7:n+1]
    return dict(intervals=n,full_strip_Linf=float(np.max(abs(actual[strip]-reference[strip]))),
        boundary_value_error=float(np.max(abs(projected-f))),
        full_domain_Linf=float(np.max(abs(actual-reference))))


def exact_dynamic_ghost_defect(n=8):
    import sympy as sp
    from test_high_order_boundary_closures import derivative_rows
    rows=derivative_rows(True)
    d=sp.zeros(n+1)
    central=[sp.Rational(v,60) for v in (-1,9,-45,0,45,-9,1)]
    for row in range(3):
        for j,value in enumerate(rows[row]):
            d[row,j]=value*n;d[n-row,n-j]=-value*n
    for row in range(3,n-2):
        for offset,value in enumerate(central):
            d[row,row-3+offset]=value*n
    _,ghost=pressure_hermite_ghosts()
    extension=sp.zeros(n+7,n+1)
    for row in range(n+1):
        extension[row+3,row]=1
    for layer in (1,2,3):
        for j in range(7):
            extension[3-layer,j]=ghost[layer-1][0,j]
            extension[3+n+layer,n-j]=ghost[layer-1][0,j]
    g=sp.Matrix(n+1,n+1,lambda row,col:n*sum(central[j]*extension[row+j,col] for j in range(7)))
    lap=d*g
    solution,parameters=lap.gauss_jordan_solve(sp.ones(n+1,1))
    witness=solution.subs({p:0 for p in parameters})
    return dict(intervals=n,nullity=n+1-lap.rank(),square_nullity=n+1-(lap*lap).rank(),
        exact_constant_kernel=lap*sp.ones(n+1,1)==sp.zeros(n+1,1),
        exact_jordan_relation=lap*witness==sp.ones(n+1,1),
        witness=[str(v) for v in witness],
        verdict='rejected: L*z=1 and L*1=0 exactly; the zero eigenvalue has an algebraic growth chain')


def proposal_report(temperature_closure='dynamic-ghost'):
    samples=[]
    for n in (32,64,128):
        for nu in (0.,1/1600):
            matrix=adiabatic_cns_matrix(n,nu,temperature_closure)
            eigenvalues=np.linalg.eigvals(matrix)
            norm=float(np.linalg.norm(matrix,ord=np.inf))
            low,high=0.,1/n
            for _ in range(50):
                middle=(low+high)/2;z=middle*eigenvalues
                if np.max(abs(1+z+z*z/2+z*z*z/6))<=1+1e-12:
                    low=middle
                else:
                    high=middle
            samples.append(dict(intervals=n,viscosity=nu,maximum_real=float(np.max(eigenvalues.real)),
                roundoff_reference=float(100*np.finfo(float).eps*norm),row_sum_norm=norm,
                rk3_sound_cfl_bound=low*n*np.sqrt(1.4),
                nullity=len(matrix)-int(np.linalg.matrix_rank(matrix)),
                square_nullity=len(matrix)-int(np.linalg.matrix_rank(matrix@matrix))))
    diffusion=[diffusion_sample(n,temperature_closure) for n in (32,64,128)]
    errors=np.array([row['full_strip_Linf'] for row in diffusion])
    return dict(status='proposal-only-not-approved-for-ASTR-integration',
        scope='uniform Cartesian nonreacting FP64 643e/643e, static adiabatic no-slip y walls',
        temperature_closure=temperature_closure,
        gamma=1.4,Pr=.72,normal_viscosity_factor=4/3,spectra=samples,
        scalar_diffusion=diffusion,scalar_strip_orders=np.log2(errors[:-1]/errors[1:]).tolist(),
        limitations='Frozen linear normal CNS and scalar variable diffusion only; no nonlinear, energy or physical validation.')


def test_neumann_extension_preserves_constants_and_linear_derivative_condition():
    gradient=neumann_gradient(32)
    assert np.max(abs(gradient@np.ones(33)))<1e-10
    x=np.linspace(0,1,33)
    # x^2(1-x)^2 has zero derivative at both endpoints and is exact for D6.
    f=x*x*(1-x)**2
    exact=2*x-6*x*x+4*x**3
    assert np.max(abs(gradient@f-exact))<1e-10


def test_rejected_dynamic_ghost_has_exact_algebraic_growth():
    result=exact_dynamic_ghost_defect()
    assert result['nullity']==1 and result['square_nullity']==2
    assert result['exact_jordan_relation']
    assert result['exact_constant_kernel']


@pytest.mark.parametrize('n',[32,64,128])
def test_projected_temperature_frozen_cns(n):
    for nu in (0.,1/1600):
        matrix=adiabatic_cns_matrix(n,nu,'neumann7-projection')
        eigenvalues=np.linalg.eigvals(matrix)
        assert np.max(eigenvalues.real)<=100*np.finfo(float).eps*np.linalg.norm(matrix,np.inf)
        assert np.linalg.matrix_rank(matrix)==np.linalg.matrix_rank(matrix@matrix)


def test_projected_temperature_variable_diffusion_order():
    errors=np.array([diffusion_sample(n,'neumann7-projection')['full_strip_Linf'] for n in (32,64,128)])
    assert np.min(np.log2(errors[:-1]/errors[1:]))>=5.5


@pytest.mark.parametrize('n',[32,64,128])
def test_proposed_tangential_slip_neumann_matrix(n,record_property):
    # HB1-D proposal only: each tangential velocity is a Neumann diffusion block
    # at a constant zero-velocity base; the normal CNS block is the HB1-C2 one.
    derivative=n*derivative_matrix(n,True)
    projected=state_projection(n,'neumann')
    matrix=((derivative@derivative@projected)[1:-1])/1600
    eigenvalues=np.linalg.eigvals(matrix)
    tolerance=100*np.finfo(float).eps*np.linalg.norm(matrix,np.inf)
    maximum=float(np.max(eigenvalues.real))
    nullity=len(matrix)-int(np.linalg.matrix_rank(matrix))
    squared_nullity=len(matrix)-int(np.linalg.matrix_rank(matrix@matrix))
    record_property('intervals',n)
    record_property('maximum_real',maximum)
    record_property('roundoff_reference',tolerance)
    record_property('nullity',nullity)
    record_property('squared_nullity',squared_nullity)
    assert maximum<=tolerance
    assert nullity==squared_nullity==1


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    report=dict(rejected_dynamic_temperature=proposal_report(),
        rejected_exact_defect=exact_dynamic_ghost_defect(),
        proposed_neumann_temperature=proposal_report('neumann7-projection'))
    args.report.parent.mkdir(parents=True,exist_ok=True)
    with args.report.open('x') as stream:
        json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(report,indent=2))
