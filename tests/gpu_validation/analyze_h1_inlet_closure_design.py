"""Eight-node inlet flux design; algebra and frozen operators, never CFD evolution."""
import argparse
import json
from pathlib import Path

import numpy as np
import sympy as sp

from test_high_order_boundary_closures import stabilized_mp_matrix,steger_warming_physical_split

CURRENT=(sp.Rational(1,8),sp.Rational(1,4),sp.Rational(1,8),sp.Rational(1,2800))
SINGLE_STEP_TRIAL=(sp.Rational(1,16),sp.Rational(1,32),sp.Rational(3,32),sp.Rational(1,16))
PROPOSED=(sp.Rational(15,128),sp.Rational(1,4),sp.Rational(1,4),sp.Rational(3,32))
DELTA7=sp.Matrix([-1,7,-21,35,-35,21,-7,1])


def exact_flux_rows(strengths):
    moments=sp.Matrix([[j**m for j in range(7)] for m in range(7)])
    inverse=moments.inv()
    return [list((inverse*sp.Matrix([sp.bernoulli(m,r+1) for m in range(7)])).col_join(sp.zeros(1,1))
        -sp.sympify(beta)*DELTA7) for r,beta in enumerate(strengths)]


def scalar_generator(intervals,strengths):
    derivative=stabilized_mp_matrix(intervals,.1)
    difference=np.array(DELTA7,dtype=float).ravel()
    for r,(value,old) in enumerate(zip(strengths,CURRENT)):
        delta=float(value-old)
        if r>0:derivative[r,:8]-=delta*difference
        derivative[r+1,:8]+=delta*difference
    return -derivative[1:,1:]


def mp_weights_exact():
    base=[sp.Rational(-1,140),sp.Rational(5,84),sp.Rational(-101,420),sp.Rational(319,420),
        sp.Rational(107,210),sp.Rational(-19,210),sp.Rational(1,105),sp.S.Zero]
    return [w-CURRENT[3]*d for w,d in zip(base,DELTA7)]


def rk3_action(generator,tau,columns):
    identity=np.eye(generator.shape[0],columns,dtype=generator.dtype)
    first=tau*(generator@identity)
    second=tau*(generator@first)
    third=tau*(generator@second)
    return identity+first+second/2+third/6


def operator_metrics(generator,taus,columns):
    eigen=np.linalg.eigvals(generator)
    return dict(spectral_abscissa=float(eigen.real.max()),
        steps=[dict(tau=float(tau),spectral_radius=float(abs(1+tau*eigen+(tau*eigen)**2/2+(tau*eigen)**3/6).max()),
            localized_2norm=float(np.linalg.svd(rk3_action(generator,tau,columns),compute_uv=False)[0])) for tau in taus])


def scalar_report(strengths):
    result=[]
    for n in (32,64,128):
        generator=scalar_generator(n,strengths)
        step=rk3_action(generator,.1,n)
        horizons=sorted(set([0,1,4,8,16,24,32,48,64,96,128,256,512,10*n,20*n]))
        transient=[dict(steps=k,time_over_h=.1*k,localized_2norm=float(
            np.linalg.svd(np.linalg.matrix_power(step,k)[:,:8],compute_uv=False)[0])) for k in horizons]
        result.append(dict(intervals=n,**operator_metrics(generator,(.05,.1,.2),8),
            transient=transient,maximum_sampled_transient=max(r['localized_2norm'] for r in transient),
            steady_local_forcing_gain=float(np.linalg.svd(np.linalg.solve(-generator,np.eye(n,8)),compute_uv=False)[0])))
    return result


def derivative_accuracy(strengths):
    rows=exact_flux_rows(strengths)
    errors=[]
    for n in (32,64,128):
        h=sp.Rational(1,n)
        values=sp.Matrix([sp.exp(4*j*h) for j in range(8)])
        flux=[(sp.Matrix(row).T*values)[0] for row in rows]
        # The original interface 4 uses the same MP7-LD relation at nodes 1..8.
        bulk=mp_weights_exact()
        flux.append(sum(w*sp.exp(4*(j+1)*h) for j,w in enumerate(bulk)))
        errors.append(max(float(abs(((flux[r]-flux[r-1])/h-4*sp.exp(4*r*h)).evalf(60))) for r in range(1,5)))
    return dict(scope='exact rational coefficients and high-precision exp(4x), nodes 1..4 including interface transition',
        intervals=[32,64,128],maximum_errors=errors,orders=np.log2(np.array(errors[:-1])/errors[1:]).tolist())


def split_jacobians(mach,axis,spacing):
    pressure=1/(1.4*mach**2)
    q=np.array([1.,1.,0.,0.,pressure/.4+.5])
    plus=np.empty((5,5));minus=np.empty((5,5))
    for c in range(5):
        perturbed=q.astype(complex);perturbed[c]+=1e-25j
        fp,fm=steger_warming_physical_split(perturbed,axis,spacing)
        plus[:,c]=fp.imag/1e-25;minus[:,c]=fm.imag/1e-25
    return plus,minus


def mp_symbol(theta):
    weights=np.array(mp_weights_exact(),dtype=float)
    return (1-np.exp(-1j*theta))*np.dot(weights,np.exp(1j*theta*np.arange(-3,5)))


def euler_report(strengths):
    cases=[]
    for n in (16,32,64):
        derivative=scalar_generator(n,strengths)
        for mach in (2.,7.):
            normal,_=split_jacobians(mach,0,1/n)
            for ry,rz in ((1.,1.),(.25,4.),(4.,.25)):
                yp,ym=split_jacobians(mach,1,1/(n*ry))
                zp,zm=split_jacobians(mach,2,1/(n*rz))
                speed=1+(1+ry+rz)/mach
                for ty,tz in ((0.,0.),(.2,0.),(0.,.5),(.2,.5)):
                    sy,sz=mp_symbol(np.pi*ty),mp_symbol(np.pi*tz)
                    tangent=-ry*(sy*yp-sy.conjugate()*ym)-rz*(sz*zp-sz.conjugate()*zm)
                    generator=np.kron(derivative,normal)+np.kron(np.eye(n),tangent)
                    metrics=operator_metrics(generator,(.1/speed,.2/speed),40)
                    cases.append(dict(intervals=n,mach=mach,grid_ratios=[ry,rz],tangent_phase_over_pi=[ty,tz],**metrics))
    return dict(scope='uniform supersonic Euler/Steger-Warming x inflow; periodic Fourier tangents, not the physical HCW walls/top',
        norm='dimensionless conserved-variable Euclidean norm; not nonlinear physical perturbation energy',
        timestep='tau * (1 + (1+ry+rz)/Mach) = 0.1 or 0.2',cases=cases)


def report(include_euler):
    configurations=[('current',CURRENT),('single_step_trial',SINGLE_STEP_TRIAL),('proposed',PROPOSED)]
    rows={name:exact_flux_rows(beta) for name,beta in configurations}
    result=dict(scope='proposed coefficients only; no solver changes, native candidate run or production qualification',
        parameterization='FD numerical flux moments m=0..6; one seventh-difference nullspace coefficient per interface 0..3',
        selection='dyadic candidate assessed using finite-horizon response, steady forcing and coefficient amplification; not a global optimum or a fit to saved CFD errors',
        single_step_trial_status='not recommended: smaller one-step norm but much larger multi-step response',
        strengths={name:list(map(str,beta)) for name,beta in configurations},
        maximum_coefficient_l1={name:float(np.sum(abs(np.array(table,dtype=float)),axis=1).max()) for name,table in rows.items()},
        exact_weights={name:[[str(value) for value in row] for row in table] for name,table in rows.items()})
    for name,beta in configurations:
        result[name]=dict(scalar=scalar_report(beta),derivative=derivative_accuracy(beta))
        if include_euler and name!='single_step_trial':result[name]['euler']=euler_report(beta)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--euler',action='store_true')
    args=parser.parse_args()
    result=report(args.euler)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(args.output)
