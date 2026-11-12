"""Exact prerequisites for a future compact KEEP wall; no PDE or operator port.

The four-node banded-norm example is a counterexample to an implication, not
an ASTR stencil candidate. Native derivative witnesses reuse archived matrices.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import sympy as sp
from study_keep_sbp63_family import derive


def banded_norm_counterexample():
    r=sp.Rational
    h=sp.Matrix([[r(1,2),r(1,6),0,0],[r(1,6),r(1,2),r(1,6),0],
                 [0,r(1,6),r(1,2),r(1,6)],[0,0,r(1,6),r(1,2)]])
    q=sp.Matrix([[-r(1,2),r(1,3),r(1,6),0],[-r(1,3),0,r(1,6),r(1,6)],
                 [-r(1,6),-r(1,6),0,r(1,3)],[0,-r(1,6),-r(1,3),r(1,2)]])
    d=h.inv()*q;b=sp.diag(-1,0,0,1);one=sp.ones(4,1);x=sp.Matrix(range(4))
    minors=[h[:i,:i].det() for i in range(1,5)]
    if not all(value.is_positive for value in minors):
        raise AssertionError('The illustrative banded norm is not positive definite')
    if h*d+d.T*h!=b or d*one!=sp.zeros(4,1) or d*x!=one:
        raise AssertionError('The illustrative SBP or polynomial conditions differ')
    weights=h*one;w=sp.diag(*weights)
    if sum(weights)!=3 or (weights.T*x)[0]!=r(9,2):
        raise AssertionError('The row-sum quadrature does not integrate constants/linears')
    density=list(map(sp.Integer,[1,2,3,1]));velocity=list(map(sp.Integer,[0,1,2,0]))
    pressure=[sp.S.One]*4;gamma=r(7,5)
    rhs=[]
    for i in range(4):
        value=sp.zeros(5,1)
        for j in range(4):
            average=(velocity[i]+velocity[j])/2
            mass=(density[i]+density[j])*average/2
            internal=(pressure[i]/density[i]+pressure[j]/density[j])/(2*(gamma-1))
            flux=sp.Matrix([mass,mass*average+(pressure[i]+pressure[j])/2,0,0,
                mass*(velocity[i]*velocity[j]/2+internal)+
                (velocity[i]*pressure[j]+velocity[j]*pressure[i])/2])
            value-=2*d[i,j]*flux
        rhs.append(value)
    # Both boundary velocities vanish and their pressures coincide: wall SAT
    # flux differences are zero, and all net physical wall fluxes are zero.
    totals=sp.Matrix([sum(weights[i]*rhs[i][c] for i in range(4)) for c in range(5)])
    kinetic=sum(weights[i]*(velocity[i]*rhs[i][1]-velocity[i]**2*rhs[i][0]/2) for i in range(4))
    pressure_work=sum(weights[i]*pressure[i]*sum(d[i,j]*velocity[j] for j in range(4)) for i in range(4))
    u=sp.Matrix(velocity)
    defect=w*d+d.T*w-b
    mass_defect=-(sp.Matrix(density).T*defect*u)[0]/2
    if totals[0]!=-r(5,11) or kinetic==pressure_work or (u.T*h*d*u)[0]!=0:
        raise AssertionError('The structural counterexample no longer distinguishes the two balances')
    if mass_defect!=totals[0] or weights.T*d!=one.T*b:
        raise AssertionError('The bilinear mass-defect identity or linear flux balance differs')
    return dict(scope='four-node structural counterexample only; not an ASTR or high-order stencil',
        H=h.tolist(),Q=q.tolist(),D=d.tolist(),positive_leading_principal_minors=minors,
        HD_plus_DT_H=b.tolist(),D1=(d*one).tolist(),Dx=(d*x).tolist(),
        row_sum_quadrature=list(weights),row_sum_norm_SBP_defect=defect.tolist(),
        rho=density,u=velocity,p=pressure,net_physical_wall_flux=[0]*5,
        pointwise_original_KEEP_integral=list(totals),mass_defect_from_bilinear_identity=mass_defect,kinetic_rate=kinetic,
        pressure_work=pressure_work,linear_H_energy_flux=(u.T*h*d*u)[0],
        conclusion='banded-norm linear SBP does not alone imply pointwise original-KEEP conservation with row-sum quadrature',
        nonconclusion='does not exclude all compact diagonal-norm operators or alternative conservative banded-norm formulations')


def archived_compact_witness(directory):
    cases={}
    for n in (16,32):
        path=directory/f'matrices_n{n}.txt'
        table=np.loadtxt(path);d=np.zeros((n+1,n+1))
        d[table[:,0].astype(int),table[:,1].astype(int)]=table[:,3]
        value=float(d[1,1])
        if abs(value)<=128*np.finfo(float).eps*max(1,float(abs(d).max())):
            raise AssertionError('Archived compact matrix no longer contains the reported necessary-condition witness')
        cases[n]=dict(matrix_source=str(path),D11=value,
            implication='for positive diagonal W, the nonendpoint SBP diagonal entry 2*w1*D11 cannot be zero',
            limitation='fixed archived physical-end matrix only; not a proof that every compact closure is impossible')
    return cases


def restricted_closure_feasibility():
    """Exact linear ansatz test, not a general compact-SBP impossibility proof."""
    n=16;norms,*_=derive();weights=sp.ones(n+1,1)
    for i,h in enumerate(norms):
        weights[i]=weights[n-i]=h
    w=sp.diag(*weights);b=sp.diag(-1,*([0]*(n-1)),1)
    result={}
    for family in ('legacy_lhs','symmetric_pade_lhs'):
        p=sp.eye(n+1)
        for i in range(n):
            p[i,i+1]=p[i+1,i]=sp.Rational(1,3)
        if family=='legacy_lhs':
            p[0,1]=p[n,n-1]=2
            for i in (1,n-1):
                p[i,i-1]=p[i,i+1]=sp.Rational(1,4)
        q=sp.zeros(n+1);unknown=[]
        for i in range(6):
            for j in range(9):
                if abs(i-j)>5:
                    continue
                value=sp.Symbol(f'q_{i}_{j}');unknown.append(value)
                q[i,j]=value;q[n-i,n-j]=-value
        for i in range(6,n-5):
            q[i,i+1]=sp.Rational(7,9);q[i,i-1]=-sp.Rational(7,9)
            q[i,i+2]=sp.Rational(1,36);q[i,i-2]=-sp.Rational(1,36)
        d=p.inv()*q;sbp=w*d+d.T*w-b
        equations=[sbp[i,j] for i in range(n+1) for j in range(i,n+1)]
        for degree in range(4):
            field=sp.Matrix([i**degree for i in range(n+1)])
            exact=sp.zeros(n+1,1) if degree==0 else sp.Matrix([degree*i**(degree-1) for i in range(n+1)])
            equations.extend(q*field-p*exact)
        a,rhs=sp.linear_eq_to_matrix(equations,unknown)
        reduced,pivots=a.row_join(rhs).rref()
        inconsistent=len(unknown) in pivots
        result[family]=dict(intervals=n,unknown_coefficients=len(unknown),equations=len(equations),
            coefficient_rank=len(pivots)-int(inconsistent),augmented_rank=len(pivots),
            inconsistent=inconsistent,free_coefficients=None if inconsistent else len(unknown)-len(pivots),
            fixed_norm='the approved explicit SBP6/3 diagonal weights, not an unknown compact norm',
            fixed_lhs=family,closure='first/last six Q rows, columns 0..8/mirror, maximum offset five',
            interior='current 1/3 Pade LHS and 7/9,1/36 RHS',
            moments='degrees 0..3 in Q*f=P*f_prime; standard endpoint SBP for the complete inverse operator',
            limitation='only these fixed P/W and closure support at 16 intervals; other norms, LHS boundary rows and support are not excluded')
        if inconsistent:
            rows=[i for i in range(reduced.rows) if all(reduced[i,j]==0 for j in range(len(unknown))) and reduced[i,-1]!=0]
            if not rows:
                raise AssertionError('Exact augmented-rank test has no contradiction row')
            result[family]['reduced_contradiction']='0 = '+str(reduced[rows[0],-1])
    return result


def encode(value):
    if isinstance(value,sp.Basic):
        return str(value)
    raise TypeError(type(value).__name__)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--restricted-closure',action='store_true')
    args=parser.parse_args()
    report=dict(status='exact_structural_constraints_checked_no_operator_selected',
        archived_astr_compact=archived_compact_witness(args.native_audit.resolve()),
        banded_norm_counterexample=banded_norm_counterexample(),
        unchanged=['production coefficients','KEEP arithmetic averages','SAT formulas','PCR/MPI topology','all CFD trajectories'])
    if args.restricted_closure:
        report['restricted_linear_closure_ansatz']=restricted_closure_feasibility()
    args.output.write_text(json.dumps(report,indent=2,default=encode)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='banded_norm_counterexample'},indent=2,default=encode))
    print('banded-norm example mass balance:',report['banded_norm_counterexample']['pointwise_original_KEEP_integral'][0])


if __name__=='__main__':
    main()
