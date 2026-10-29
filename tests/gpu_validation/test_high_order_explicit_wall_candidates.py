"""HB1-B pre-integration algebra: scalar diffusion and closed-wall acoustics.

These are frozen operator matrices, not a Python fluid time integrator. The
direct reuse candidate is deliberately rejected for acoustic instability.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import sympy as sp
import mpmath as mp

from test_high_order_boundary_closures import derivative_matrix,derivative_rows


def state_projection(intervals,kind):
    projection=np.zeros((intervals+1,intervals-1))
    projection[1:-1]=np.eye(intervals-1)
    if kind not in ('dirichlet','neumann','mixed'):
        raise ValueError('Unknown scalar boundary contract')
    weights=np.array(sp.finite_diff_weights(1,list(range(8)),0)[1][-1],float)
    if kind=='neumann':
        projection[0,:7]=-weights[1:]/weights[0]
    if kind in ('neumann','mixed'):
        projection[-1,-7:]=-weights[1:][::-1]/weights[0]
    return projection


def geometry(intervals,mapping,metric):
    xi=np.linspace(0,1,intervals+1)
    derivative=intervals*derivative_matrix(intervals,True)
    if mapping=='uniform':
        x=xi; jacobian=np.ones_like(x)
    elif mapping=='sinh':
        a=1.2
        x=np.sinh(a*xi)/np.sinh(a)
        jacobian=a*np.cosh(a*xi)/np.sinh(a)
    else:
        raise ValueError('Unknown manufactured mapping')
    if metric=='discrete':
        jacobian=derivative@x
    elif metric!='analytic':
        raise ValueError('Unknown metric reference')
    if np.any(jacobian<=0):
        raise ValueError('Manufactured Jacobian is not positive')
    return x,derivative/jacobian[:,None]


def manufactured_state(x,kind):
    if kind=='dirichlet':
        u=np.exp(2*x)-1-(np.exp(2)-1)*x
        du=2*np.exp(2*x)-(np.exp(2)-1)
        ddu=4*np.exp(2*x)
    elif kind=='neumann':
        u=np.cos(2*np.pi*x)
        du=-2*np.pi*np.sin(2*np.pi*x)
        ddu=-(2*np.pi)**2*u
    elif kind=='mixed':
        u=np.sin(np.pi*x/2)
        du=np.pi/2*np.cos(np.pi*x/2)
        ddu=-(np.pi/2)**2*u
    else:
        raise ValueError('Unknown manufactured state')
    return u,du,ddu


def diffusion_error(intervals,kind,mapping,metric):
    x,d=geometry(intervals,mapping,metric)
    exact,du,ddu=manufactured_state(x,kind)
    projection=state_projection(intervals,kind)
    reconstructed=projection@exact[1:-1]
    mu=1+.3*np.exp(2*x)
    dmu=.6*np.exp(2*x)
    actual=d@(mu*(d@reconstructed))
    reference=mu*ddu+dmu*du
    # Include the physical endpoints and all seven adjacent nodes on both sides.
    strip=np.r_[0:8,intervals-7:intervals+1]
    return dict(intervals=intervals,
        boundary_state_error=float(np.max(np.abs(reconstructed-exact))),
        full_strip_Linf=float(np.max(np.abs(actual[strip]-reference[strip]))),
        full_domain_Linf=float(np.max(np.abs(actual-reference))))


def legacy_derivative(intervals):
    d=np.zeros((intervals+1,intervals+1))
    d[0,:3]=[-1.5,2,-.5]
    d[1,:3]=[-.5,0,.5]
    d[2,:5]=np.array([1,-8,0,8,-1])/12
    for row in range(3,intervals-2):
        d[row,row-3:row+4]=np.array([-1,9,-45,0,45,-9,1])/60
    for row in range(3):
        d[intervals-row]=-d[row,::-1]
    return intervals*d


def acoustic_matrix(intervals,high_derivative=True,high_pressure=True):
    """u_t=-p_x, p_t=-u_x; u=0 and reconstructed p_x=0 at both walls."""
    _,d=geometry(intervals,'uniform','analytic')
    if not high_derivative:
        d=legacy_derivative(intervals)
    velocity=state_projection(intervals,'dirichlet')
    pressure=state_projection(intervals,'neumann')
    if not high_pressure:
        pressure[0]=0;pressure[-1]=0
        pressure[0,:2]=[4/3,-1/3];pressure[-1,-2:]=[-1/3,4/3]
    zero=np.zeros((intervals-1,intervals-1))
    return np.block([[zero,-(d@pressure)[1:-1]],[-(d@velocity)[1:-1],zero]])


def acoustic_rejection(intervals):
    matrix=acoustic_matrix(intervals)
    spectrum=np.linalg.eigvals(matrix)
    largest=spectrum[np.argmax(spectrum.real)]
    return dict(intervals=intervals,maximum_real=float(largest.real),
        maximum_real_times_h=float(largest.real/intervals),
        max_absolute_imaginary=float(np.max(np.abs(spectrum.imag))),
        row_sum_norm=float(np.linalg.norm(matrix,ord=np.inf)),
        verdict='rejected: positive semidiscrete eigenvalue; reducing dt cannot remove it')


def pressure_hermite_ghosts():
    # Degree seven: seven physical pressure values p0..p6 and dp/dn=0.
    matrix=sp.Matrix([[sp.Integer(j)**p for p in range(8)] for j in range(7)]+
                     [[int(p==1) for p in range(8)]])
    inverse=matrix.inv()
    rows=[(sp.Matrix([[sp.Integer(-j)**p for p in range(8)]])*inverse) for j in (1,2,3)]
    return np.array([list(row[0,:7]) for row in rows],float),rows


def dynamic_wall_cns_matrix(intervals,viscosity):
    """Frozen rho/v/T equations, isothermal walls, R*T0=rho0=1.

    Density evolves by continuity at the wall; T and v are prescribed there.
    The pressure-normal constraint supplies ghost values, not a p0 reset.
    """
    n=intervals
    ghost,_=pressure_hermite_ghosts()
    extension=np.zeros((n+7,n+1));extension[3:n+4]=np.eye(n+1)
    for j in (1,2,3):
        extension[3-j,:7]=ghost[j-1]
        extension[3+n+j,-7:]=ghost[j-1,::-1]
    central=np.array([-1,9,-45,0,45,-9,1])/60
    pressure_gradient=np.array([central@extension[i:i+7] for i in range(1,n)])*n
    fixed=state_projection(n,'dirichlet')
    derivative=derivative_matrix(n,True)*n
    gradient=derivative@fixed
    laplacian=(derivative@derivative@fixed)[1:-1]
    temperature_diffusivity=1.4/.72*viscosity
    return np.block([
        [np.zeros((n+1,n+1)),-gradient,np.zeros((n+1,n-1))],
        [-pressure_gradient,(4/3)*viscosity*laplacian,-pressure_gradient@fixed],
        [np.zeros((n-1,n+1)),-.4*gradient[1:-1],temperature_diffusivity*laplacian]])


def dynamic_wall_report():
    _,rational=pressure_hermite_ghosts()
    result=dict(status='proposal-only-not-approved-for-production',
        physical_wall='stationary no-slip, fixed T, zero pressure normal derivative',
        change='evolve wall density with continuity; use the pressure-normal condition for Hermite ghosts',
        gamma=1.4,Pr=.72,base_rho=1,base_RT=1,normal_viscosity_factor=4/3,
        ghost_rows=[dict(physical=[str(w) for w in row[0,:7]],inward_node_derivative=str(row[0,7])) for row in rational],
        scope='constant-base linearized normal CNS only; not nonlinear ASTR, CURVE or wall-family acceptance',samples=[])
    for n in (32,64,128):
        for viscosity in (0.,1/1600):
            matrix=dynamic_wall_cns_matrix(n,viscosity)
            spectrum=np.linalg.eigvals(matrix)
            norm=float(np.linalg.norm(matrix,ord=np.inf))
            tolerance=100*np.finfo(float).eps*norm
            low,high=0.,1/n
            def amplitude(dt):
                z=dt*spectrum
                return float(np.max(np.abs(1+z+z*z/2+z*z*z/6)))
            for _ in range(50):
                middle=(low+high)/2
                if amplitude(middle)<=1+1e-12:
                    low=middle
                else:
                    high=middle
            result['samples'].append(dict(intervals=n,kinematic_viscosity=viscosity,
                max_real_eigenvalue=float(spectrum.real.max()),max_abs_imaginary=float(np.max(np.abs(spectrum.imag))),
                row_sum_norm=norm,eigenvalue_roundoff_bound=tolerance,
                no_resolved_positive_eigenvalue=bool(spectrum.real.max()<=tolerance),
                frozen_RK3_dt_limit=low,frozen_RK3_sound_CFL=low*n*np.sqrt(1.4)))
    return result


def diffusion_high_precision(intervals,kind,mapping,metric):
    """80-digit static evaluation on the SAME grids, to diagnose roundoff."""
    with mp.workdps(80):
        high_rows=[[mp.mpf(int(w.p))/int(w.q) for w in row] for row in derivative_rows(True)]
        central=[mp.mpf(v)/60 for v in [-1,9,-45,0,45,-9,1]]
        def differentiate(values):
            result=[]
            for row in range(intervals+1):
                if row<3:
                    indices=range(8);weights=high_rows[row]
                elif row>intervals-3:
                    indices=range(intervals,intervals-8,-1);weights=[-w for w in high_rows[intervals-row]]
                else:
                    indices=range(row-3,row+4);weights=central
                result.append(intervals*sum(w*(values[j]-values[row]) for j,w in zip(indices,weights)))
            return result
        xi=[mp.mpf(j)/intervals for j in range(intervals+1)]
        a=mp.mpf('1.2')
        x=xi if mapping=='uniform' else [mp.sinh(a*v)/mp.sinh(a) for v in xi]
        jac=[mp.mpf(1)]*(intervals+1) if mapping=='uniform' else [a*mp.cosh(a*v)/mp.sinh(a) for v in xi]
        if metric=='discrete':
            jac=differentiate(x)
        if kind=='dirichlet':
            exact=[mp.exp(2*v)-1-(mp.exp(2)-1)*v for v in x]
            du=[2*mp.exp(2*v)-(mp.exp(2)-1) for v in x]
            ddu=[4*mp.exp(2*v) for v in x]
        else:
            wave=2*mp.pi if kind=='neumann' else mp.pi/2
            exact=[mp.cos(wave*v) if kind=='neumann' else mp.sin(wave*v) for v in x]
            du=[-wave*mp.sin(wave*v) if kind=='neumann' else wave*mp.cos(wave*v) for v in x]
            ddu=[-wave**2*v for v in exact]
        values=exact.copy()
        if kind in ('neumann','mixed'):
            w=[mp.mpf(int(v.p))/int(v.q) for v in sp.finite_diff_weights(1,list(range(8)),0)[1][-1]]
            values[-1]=values[-2]-sum(w[j]*(values[-1-j]-values[-2]) for j in range(1,8))/w[0]
            if kind=='neumann':
                values[0]=values[1]-sum(w[j]*(values[j]-values[1]) for j in range(1,8))/w[0]
        mu=[1+mp.mpf('.3')*mp.exp(2*v) for v in x]
        flux=[m*g/j for m,g,j in zip(mu,differentiate(values),jac)]
        actual=[g/j for g,j in zip(differentiate(flux),jac)]
        reference=[m*d2+mp.mpf('.6')*mp.exp(2*v)*d1 for m,d1,d2,v in zip(mu,du,ddu,x)]
        strip=list(range(8))+list(range(intervals-7,intervals+1))
        return float(max(abs(actual[j]-reference[j]) for j in strip))


def report():
    result=dict(scope='candidate matrices only; not the active ASTR wall scheme',
        derivative='H1 matched-leading-error D6, unchanged interior centered D6',
        state='existing seven-point inward data / Neumann7 derivative constraint',
        physics='homogeneous Dirichlet/Neumann scalar references; acoustic rho0=c0=1',
        diffusion_definition='Dx(mu Dx u), mu=1+0.3 exp(2x); both endpoints and seven adjacent nodes included',
        admissible_for_solver=False,diffusion={},acoustics=[])
    for kind in ('dirichlet','neumann','mixed'):
        for mapping,metric in (('uniform','analytic'),('sinh','analytic'),('sinh','discrete')):
            rows=[diffusion_error(n,kind,mapping,metric) for n in (32,64,128)]
            errors=np.array([r['full_strip_Linf'] for r in rows])
            orders=np.log2(errors[:-1]/errors[1:])
            extended=np.array([diffusion_high_precision(n,kind,mapping,metric) for n in (32,64,128)])
            result['diffusion'][f'{kind}_{mapping}_{metric}']=dict(rows=rows,orders=orders.tolist(),
                approved_grid_FP64_order_gate=bool(np.all(orders>=5.5)),
                diagnostic_80_digit_errors=extended.tolist(),
                diagnostic_80_digit_orders=np.log2(extended[:-1]/extended[1:]).tolist())
    result['acoustics']=[acoustic_rejection(n) for n in (32,64,128)]
    result['acoustic_controls']=[dict(intervals=n,high_derivative=hd,high_pressure=hp,
        maximum_real=float(np.linalg.eigvals(acoustic_matrix(n,hd,hp)).real.max()))
        for n in (32,64,128) for hd,hp in [(False,False),(False,True),(True,False),(True,True)]]
    result['dynamic_wall_proposal']=dynamic_wall_report()
    return result


@pytest.mark.parametrize('intervals',[32,64,128])
def test_legacy_acoustic_control_has_no_positive_physical_mode(intervals):
    matrix=acoustic_matrix(intervals,False,False)
    scale=np.linalg.norm(matrix,ord=np.inf)
    assert np.linalg.eigvals(matrix).real.max()<=100*np.finfo(float).eps*scale


@pytest.mark.parametrize('intervals',[32,64,128])
def test_direct_reuse_keeps_acoustic_instability_detectable(intervals):
    value=acoustic_rejection(intervals)
    # This gate rejects the candidate; it does not admit an unstable solver.
    assert value['maximum_real_times_h']>.30,value


def test_neumann_state_projection_preserves_constant():
    for n in (32,64,128):
        np.testing.assert_allclose(state_projection(n,'neumann')@np.ones(n-1),1,rtol=0,atol=3e-15)


def test_hermite_ghost_polynomial_conditions_are_exact():
    _,rows=pressure_hermite_ghosts()
    for offset,row in zip((1,2,3),rows):
        for degree in range(8):
            value=sum(row[j]*sp.Integer(j)**degree for j in range(7))
            value+=row[7]*int(degree==1)
            assert value==sp.Integer(-offset)**degree


@pytest.mark.parametrize('intervals',[32,64,128])
@pytest.mark.parametrize('viscosity',[0.,1/1600])
def test_proposed_frozen_normal_cns_has_no_resolved_positive_eigenvalue(intervals,viscosity):
    matrix=dynamic_wall_cns_matrix(intervals,viscosity)
    tolerance=100*np.finfo(float).eps*np.linalg.norm(matrix,ord=np.inf)
    assert np.linalg.eigvals(matrix).real.max()<=tolerance


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    data=report()
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data,indent=2))
