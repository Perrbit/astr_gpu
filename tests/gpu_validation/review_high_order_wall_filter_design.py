"""Frozen design review only; keep candidate coefficients out of the solver."""
import argparse
from fractions import Fraction
import json
from math import comb
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag,eigvals,eig,null_space

from test_high_order_adiabatic_wall_candidate import adiabatic_cns_matrix
from test_high_order_boundary_closures import derivative_matrix
from test_high_order_explicit_wall_candidates import dynamic_wall_cns_matrix,state_projection
from test_high_order_wall_filter_candidates import sixth_order_family,spectral_diagnostic
from test_high_order_wall_filter_coupling import rk3_filtered_map


def exact_rows(parameters):
    records=[]
    difference=[Fraction((-1)**j*comb(6,j),64) for j in range(7)]
    for row in range(1,5):
        a0,a2=parameters[2*row-2:2*row]
        a1=a0+a2-(-1)**(row+1)
        delta=[Fraction(0) for _ in range(9)]
        for shift,weight in enumerate((a0,a1,a2)):
            for j,value in enumerate(difference):delta[j+shift]+=weight*value
        for power in range(6):
            assert sum(value*(j-row)**power for j,value in enumerate(delta))==0
        assert (-1)**row+sum(value*(-1)**j for j,value in enumerate(delta))==0
        if row==4:assert delta==delta[::-1]
        moments={power:sum(value*(j-row)**power for j,value in enumerate(delta)) for power in range(6,11)}
        order=next(power for power,value in moments.items() if value)
        coefficients=delta.copy();coefficients[row]+=1
        records.append(dict(row=row,a=[str(v) for v in (a0,a1,a2)],
            coefficients=[str(v) for v in coefficients],leading_order=order,
            leading_moment=str(moments[order])))
    return records


def blocks(original,parameters,kind):
    n=len(original)-1
    f=sixth_order_family(original,parameters)
    assert np.array_equal(f,f[::-1,::-1])
    assert np.array_equal(f[[0,-1]],original[[0,-1]])
    assert np.array_equal(f[5:n-4],original[5:n-4])
    pd=state_projection(n,'dirichlet');pn=state_projection(n,'neumann')
    thermal=pd if kind==41 else pn
    filtering=block_diag(f,f[1:-1]@pd,f[1:-1]@thermal)
    derivative=n*derivative_matrix(n,True)
    a0=dynamic_wall_cns_matrix(n,0.) if kind==41 else adiabatic_cns_matrix(n,0.,'neumann7-projection')
    a1=dynamic_wall_cns_matrix(n,1.) if kind==41 else adiabatic_cns_matrix(n,1.,'neumann7-projection')
    result=[('normal',filtering,a0,a1-a0)]
    for name,p in (('tangent_D',pd),('tangent_N',pn)):
        if kind==41 and name=='tangent_N':continue
        tf=f[1:-1]@p
        result.append((name,tf,np.zeros_like(tf),(derivative@derivative@p)[1:-1]))
    return result


def spectrum_sample(parts,nu,cfl,n):
    worst=None;norm=1.;radii=[]
    for name,f,a0,viscous in parts:
        matrix=f if cfl is None else rk3_filtered_map(a0+nu*viscous,f,cfl/(n*np.sqrt(1.4)))
        radius=float(np.max(abs(eigvals(matrix))))
        norm=max(norm,float(np.linalg.norm(matrix,np.inf)))
        radii.append(dict(block=name,spectral_radius=radius))
        if worst is None or radius>worst[0]:worst=(radius,name,matrix)
    tolerance=float(100*np.finfo(float).eps*norm)
    return dict(viscosity=nu,sound_cfl=cfl,spectral_radius=worst[0],worst_block=worst[1],
        roundoff_reference=tolerance,passes=bool(worst[0]<=1+tolerance),blocks=radii,
        diagnostic=spectral_diagnostic(worst[2]) if worst[0]>1+tolerance else None)


def rk_filter_taylor(a,f):
    m0=f/3+f@f/2+f@f@f/6
    m1=.5*a@f@f+(a@f@f@f+f@a@f@f+f@f@a@f)/6
    m2=(a@f@a@f@f+a@f@f@a@f+f@a@f@a@f)/6
    m3=(a@f)@(a@f)@(a@f)/6
    return m0,m1,m2,m3


def slow_filter_coefficients(f,a):
    """First two slow-eigenvalue coefficients for the actual filtered RK map."""
    identity=np.eye(len(f))
    right=null_space(f-identity,rcond=1e-10)
    left0=null_space(f.T-identity,rcond=1e-10)
    cross=left0.T@right
    left=left0@np.linalg.inv(cross).T
    projection=right@left.T
    m0,m1,m2,_=rk_filter_taylor(a,f)
    fast=np.linalg.solve(identity-m0+projection,(identity-projection)@m1@right)
    slow1=left.T@m1@right
    slow2=left.T@m2@right+left.T@m1@fast
    values,lvec,rvec=eig(slow1,left=True,right=True)
    modes=[]
    for j,value in enumerate(values):
        if value.imag<=1e-4:continue
        second=np.vdot(lvec[:,j],slow2@rvec[:,j])/np.vdot(lvec[:,j],rvec[:,j])
        modes.append(dict(first=[float(value.real),float(value.imag)],
            second=[float(second.real),float(second.imag)],
            amplitude_dt2=float(second.real+.5*value.imag**2)))
    return dict(unit_subspace_dimension=right.shape[1],rank_cutoff=1e-10,
        dual_basis_residual=float(np.linalg.norm(left.T@right-np.eye(right.shape[1]),np.inf)),
        unit_right_residual=float(np.linalg.norm((f-identity)@right,np.inf)),
        unit_left_residual=float(np.linalg.norm(left.T@(f-identity),np.inf)),
        unit_cross_condition=float(np.linalg.cond(cross)),slow_maximum_real=float(np.max(values.real)),modes=modes)


def small_cfl_diagnostic(f,a):
    """Perturb the unit filter eigenspace; not a replacement acceptance gate."""
    result=slow_filter_coefficients(f,a)
    if not result['modes']:return result
    worst=max(result['modes'],key=lambda m:m['amplitude_dt2'])
    target=complex(*worst['first'])
    samples=[]
    for dt in (1e-3,5e-4,2.5e-4,1e-4):
        mapping=rk3_filtered_map(a,f,dt)
        values=eigvals(mapping)
        value=values[int(np.argmin(abs((values-1)/dt-target)))]
        samples.append(dict(dt=dt,radius=float(abs(value)),
            log_amplitude_over_dt2=float(np.log(abs(value))/dt**2),
            amplitude_after_linear_over_dt2=float((abs(value)-1-target.real*dt)/dt**2)))
    return dict(**result,shrinking_dt=samples,
        caveat='A numerical eigenspace/Taylor diagnostic, not a uniform-in-grid energy proof.')


def select_dyadic(matrices,design):
    trials=[]
    for denominator in (1,2,4,8,16,32,64,128,256,512,1024,2048):
        parameters=np.round(np.array(design['floating_parameters'])*denominator)/denominator
        failures=[];maximum=1.
        for n in design['training_intervals']:
            for kind in design['training_bctype']:
                parts=blocks(matrices[n],parameters,kind)
                for cfl in (None,*design['training_sound_cfl']):
                    sample=spectrum_sample(parts,0.,cfl,n)
                    maximum=max(maximum,sample['spectral_radius'])
                    if not sample['passes']:failures.append(dict(intervals=n,bctype=kind,**sample))
        trials.append(dict(denominator=denominator,maximum_radius=maximum,failures=failures))
        if not failures:
            fractions=[Fraction(int(round(v*denominator)),denominator) for v in parameters]
            return fractions,trials
    return None,trials


def review(matrices,parameters):
    samples=[]
    for n,original in sorted(matrices.items()):
        for kind in (41,42):
            parts=blocks(original,np.array(parameters,float),kind)
            standalone=spectrum_sample(parts,0.,None,n)
            standalone.update(intervals=n,bctype=kind)
            samples.append(standalone)
            if not standalone['passes']:return samples,standalone
            for nu in (0.,1e-6,1e-5,1e-4,1/1600):
                for step in range(21):
                    sample=spectrum_sample(parts,nu,step/100,n)
                    sample.update(intervals=n,bctype=kind)
                    samples.append(sample)
                    if not sample['passes']:return samples,sample
    return samples,None


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--design',type=Path,required=True)
    parser.add_argument('--native',type=Path,required=True)
    parser.add_argument('--extra-native',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    matrices={}
    for directory in (args.native,args.extra_native):
        for path in sorted(directory.glob('filter_*.bin')):
            n=int(path.stem.split('_')[-1])
            matrices[n]=np.fromfile(path,'<f8').reshape(n+1,n+1,order='F')
    design=json.loads(args.design.read_text())
    parameters,trials=select_dyadic(matrices,design)
    if parameters is None:
        report=dict(status='rejected: no dyadic representation passed the original core spectrum gate',
            design=str(args.design),quantization_trials=trials,
            selection_training=dict(intervals=design['training_intervals'],bctype=design['training_bctype'],
                sound_cfl=design['training_sound_cfl']))
        with (args.output/'selection_failure.json').open('x') as stream:
            json.dump(report,stream,indent=2);stream.write('\n')
        print(report['status'],flush=True)
        raise SystemExit(1)
    report=dict(status='static design, not admitted',parameters=[str(p) for p in parameters],
        rows=exact_rows(parameters),quantization_trials=trials,
        selection_training=dict(intervals=design['training_intervals'],bctype=design['training_bctype'],
            sound_cfl=design['training_sound_cfl']),
        scope='41 normal and D tangent; common 42/421 normal with D/N tangents, uniform rest base',
        planned_intervals=sorted(matrices),viscosities=[0.,1e-6,1e-5,1e-4,1/1600],
        sound_cfl='0 to 0.20 at 0.01 spacing; no proof between samples',
        original_gate='radius <= 1 + 100*eps*max(1,block infinity norms)')
    with (args.output/'coefficient_selection.json').open('x') as stream:
        json.dump(report,stream,indent=2);stream.write('\n')
    print('dyadic_parameters',report['parameters'],flush=True)
    samples,failure=review(matrices,parameters)
    report.update(samples=samples,first_failure=failure,all_sampled_pass=failure is None)
    if failure is not None and failure['viscosity']==0. and failure['sound_cfl'] is not None:
        parts=blocks(matrices[failure['intervals']],np.array(parameters,float),failure['bctype'])
        part=next(p for p in parts if p[0]==failure['worst_block'])
        report['small_cfl_diagnostic']=small_cfl_diagnostic(part[1],part[2])
    with (args.output/'review.json').open('x') as stream:
        json.dump(report,stream,indent=2);stream.write('\n')
    print('samples',len(samples),'all_sampled_pass',failure is None,flush=True)
    if failure is not None:print('FIRST_FAILURE',json.dumps(failure),flush=True)


if __name__=='__main__':main()
