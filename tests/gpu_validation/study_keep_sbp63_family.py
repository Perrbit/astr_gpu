"""Derive a diagonal-norm SBP 6/3 candidate family without selecting or porting it.

This is exact operator algebra only. No CFD state, physical boundary condition,
time integrator, SAT parameter in ASTR, or production default is changed.
"""
import argparse
import json
from pathlib import Path

import sympy as sp


def derive():
    h=sp.symbols('h0:6')
    unknown=sp.symbols('q0:15')
    q=sp.zeros(6,9);q[0,0]=-sp.Rational(1,2)
    cursor=0
    for i in range(6):
        for j in range(i+1,6):
            q[i,j]=unknown[cursor];q[j,i]=-unknown[cursor];cursor+=1
    centered={1:sp.Rational(3,4),2:-sp.Rational(3,20),3:sp.Rational(1,60)}
    for i in range(6):
        for j in range(6,9):
            offset=i-j
            q[i,j]=-(sp.sign(offset)*centered[abs(offset)] if abs(offset) in centered else 0)
    equations=[]
    for i in range(6):
        for degree in range(4):
            equations.append(sum(q[i,j]*j**degree for j in range(9))-
                             (0 if degree==0 else h[i]*degree*i**(degree-1)))
    solution,=sp.linsolve(equations,(*h,*unknown))
    parameter=sp.Symbol('theta',real=True)
    substitution=dict(zip((*h,*unknown),solution))
    norms=[sp.factor(value) for value in solution[:6]]
    boundary=q.subs(substitution,simultaneous=True).subs(unknown[-1],parameter)
    if any(not value.is_positive for value in norms):
        raise AssertionError('Derived diagonal weights are not positive')
    return norms,boundary,centered,parameter


def verify(n,norms,boundary,centered):
    q=sp.zeros(n+1)
    for i in range(6):
        for j in range(9):
            q[i,j]=boundary[i,j]
            q[n-i,n-j]=-boundary[i,j]
    for i in range(6,n-5):
        for radius,weight in centered.items():
            q[i,i+radius]=weight;q[i,i-radius]=-weight
    weights=sp.ones(n+1,1)
    for i,h in enumerate(norms):
        weights[i]=h;weights[n-i]=h
    h=sp.diag(*weights)
    h_inverse=sp.diag(*(1/x for x in weights))
    d=h_inverse*q
    b=sp.zeros(n+1);b[0,0]=-1;b[n,n]=1
    is_zero=lambda matrix:all(sp.expand(value)==0 for value in matrix)
    if not is_zero(q+q.T-b):
        raise AssertionError('Symbolic SBP identity failed')
    for i in range(n+1):
        order=3 if min(i,n-i)<6 else 6
        for degree in range(order+1):
            exact=0 if degree==0 else degree*i**(degree-1)
            if sp.expand(sum(d[i,j]*j**degree for j in range(n+1))-exact)!=0:
                raise AssertionError('Symbolic derivative moment failed')
    for degree in range(6):
        if sum(weights[j]*j**degree for j in range(n+1))!=sp.Rational(n**(degree+1),degree+1):
            raise AssertionError('Sixth-order quadrature moment failed')
    e0=sp.zeros(n+1);e0[0,0]=1
    en=sp.zeros(n+1);en[n,n]=1
    scalar_advection=-d-h_inverse*e0
    if not is_zero(h*scalar_advection+scalar_advection.T*h+e0+en):
        raise AssertionError('Scalar homogeneous-inflow reference energy identity failed')
    frozen=check_frozen_keep(q)
    return dict(intervals=n,sbp_residual='exactly zero for all real theta',
                boundary_polynomials='degrees 0..3 exact in the first/last six rows',
                interior_polynomials='degrees 0..6 exact; existing explicit D6 coefficients',
                quadrature='degrees 0..5 exact',
                scalar_reference='L=-D-H^-1*e0*e0.T; H*L+L.T*H=-e0*e0.T-eN*eN.T exactly',
                frozen_original_keep=frozen)


def check_frozen_keep(q):
    """Rational 1D five-component wall-normal flux balance, no wall-state update."""
    n=q.rows-1
    gamma=sp.Rational(7,5)
    x=[sp.Rational(j,n) for j in range(n+1)]
    rho=[1+z*z/100 for z in x];pressure=[1+z**3/50 for z in x]
    u=[sp.Matrix([z*(1-z)/10,z*z*(1-z)/20,z*(1-z)**2/30]) for z in x]
    internal=[pressure[j]/((gamma-1)*rho[j]) for j in range(n+1)]
    mapped=[]
    for i in range(n+1):
        total=sp.zeros(5,1)
        for j in range(n+1):
            if q[i,j]==0:
                continue
            velocity=(u[i]+u[j])/2
            mass=(rho[i]+rho[j])*velocity[0]/2
            flux=sp.Matrix([mass,*(mass*velocity+sp.Matrix([(pressure[i]+pressure[j])/2,0,0])),
                            mass*(u[i].dot(u[j])/2+(internal[i]+internal[j])/2)+
                            (u[i][0]*pressure[j]+u[j][0]*pressure[i])/2])
            total-=2*q[i,j]*flux
        mapped.append(total)
    sums=[sp.expand(sum(row[c] for row in mapped)) for c in range(5)]
    boundary=sp.Matrix([0,pressure[0]-pressure[-1],0,0,0])
    if any(sp.expand(sums[c]-boundary[c])!=0 for c in range(5)):
        raise AssertionError('Candidate KEEP conservative boundary-flux identity failed')
    kinetic=sp.expand(sum(u[i].dot(mapped[i][1:4,0])-u[i].dot(u[i])*mapped[i][0]/2 for i in range(n+1)))
    pressure_work=sp.expand(sum(pressure[i]*q[i,j]*u[j][0] for i in range(n+1) for j in range(n+1)))
    if sp.expand(kinetic-pressure_work)!=0 or sp.expand(sums[4]-kinetic+pressure_work)!=0:
        raise AssertionError('Candidate KEEP pressure-work exchange identity failed')
    return dict(state='positive rational rho/p, all three velocities zero at both endpoints',
                conservative_boundary_residuals=['0']*5,kinetic_pressure_work_residual='0',
                internal_pressure_work_residual='0',parameter='all real theta',
                phase='frozen flux algebra only; no Neumann pressure or isothermal energy enforcement')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    norms,boundary,centered,parameter=derive()
    report=dict(status='candidate_family_derived_not_selected_or_ported',
                scope='uniform computational coordinate; explicit first derivative; no KEEP wall/viscous closure proof',
                representation='D=H^-1*Q; index spacing=1; mirror closure at right end',
                diagonal_weights_left=list(map(str,norms)),free_parameter=str(parameter),
                parameter_definition='Q[4,5]=theta, zero-based indices',
                q_left=[[str(boundary[i,j]) for j in range(9)] for i in range(6)],
                checks=[verify(n,norms,boundary,centered) for n in (16,32)],
                unresolved=['theta selection','KEEP boundary flux and wall-state ownership',
                            'isothermal viscous/heat boundary coupling','compact counterpart and norm',
                            'filter/geometry compatibility','physical gate before native port'])
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key!='q_left'},indent=2),flush=True)


if __name__=='__main__':
    main()
