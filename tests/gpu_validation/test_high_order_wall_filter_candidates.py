"""HB6 design experiments only; no solver kernels or CFD advancement."""
import argparse
from fractions import Fraction
import json
from math import comb
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import block_diag,eig,eigvals
import sympy as sp

from test_high_order_adiabatic_wall_candidate import adiabatic_cns_matrix
from test_high_order_explicit_wall_candidates import dynamic_wall_cns_matrix,state_projection
from test_high_order_wall_filter_coupling import native_filter_matrices,rk3_filtered_map


def boundary_order_candidate(original,order):
    n=len(original)-1
    if n<order:
        raise ValueError('The proposed one-sided stencil does not fit this line')
    result=original.copy()
    difference=np.array([(-1)**j*comb(order,j) for j in range(order+1)])/2**order
    for row in range(1,5):
        result[row]=0.;result[row,row]=1.
        result[row,:order+1]+=(-1)**(row+1)*difference
        result[n-row]=result[row,::-1]
    return result


def constraints(n):
    weights=np.array(sp.finite_diff_weights(1,list(range(8)),0)[1][-1],float)
    result=np.zeros((2,n+1))
    result[0,:8]=weights;result[1,-8:]=weights[::-1]
    return result


def reduced_candidate(original,variant):
    n=len(original)-1
    velocity=state_projection(n,'dirichlet');thermal=state_projection(n,'neumann')
    f=original
    if variant=='order8':f=boundary_order_candidate(original,8)
    if variant=='order10':f=boundary_order_candidate(original,10)
    c=constraints(n)
    if variant=='increment-projection':
        lift=c.T.copy();lift[[0,-1]]=0.
        f=f-lift@np.linalg.solve(c@lift,c@(f-np.eye(n+1)))
    temperature=f@thermal
    if variant=='temperature-projection':
        temperature-=c.T@np.linalg.solve(c@c.T,c@temperature)
    return block_diag(f,f[1:-1]@velocity,temperature[1:-1])


def spectral_diagnostic(matrix):
    values,left,right=eig(matrix,left=True,right=True)
    j=int(np.argmax(abs(values)));u=left[:,j];v=right[:,j];value=values[j]
    norm=float(np.linalg.norm(matrix,np.inf))
    condition=float(np.linalg.norm(u)*np.linalg.norm(v)/abs(np.vdot(u,v)))
    return dict(spectral_radius=float(abs(value)),eigenvalue=[float(value.real),float(value.imag)],
        eigenpair_residual=float(np.linalg.norm(matrix@v-value*v,np.inf)),
        eigenvalue_condition=condition,
        conditioned_roundoff_scale=float(np.finfo(float).eps*norm*condition),
        original_roundoff_gate=float(100*np.finfo(float).eps*max(1.,norm)))


def compare_candidates(matrices):
    rows=[]
    for n,original in matrices.items():
        for variant in ('original','temperature-projection','increment-projection','order8','order10'):
            filtering=reduced_candidate(original,variant)
            worst=None
            for nu in (0.,1e-5,1/1600):
                a=adiabatic_cns_matrix(n,nu,'neumann7-projection')
                for cfl in (0.,.05,.1,.15,.2):
                    mapping=rk3_filtered_map(a,filtering,cfl/(n*np.sqrt(1.4)))
                    radius=float(np.max(abs(eigvals(mapping))))
                    if worst is None or radius>worst['spectral_radius']:
                        worst=dict(viscosity=nu,sound_cfl=cfl,**spectral_diagnostic(mapping))
            rows.append(dict(intervals=n,variant=variant,worst_sample=worst))
    return dict(status='design experiments, no candidate admitted',
        scope='42/421 frozen normal block at rest only; no CFD, positivity, MPI or physical acceptance',
        variants=rows,
        caveat='Raw near-unit eigenvalues and conditioning are retained; no gate has been relaxed.')


def candidate_moments(original,order):
    f=boundary_order_candidate(original,order);n=len(f)-1
    assert np.array_equal(f[5:n-4],original[5:n-4])
    assert np.array_equal(f[0],original[0]) and np.array_equal(f[-1],original[-1])
    for row in (*range(1,5),*range(n-4,n)):
        delta=f[row].copy();delta[row]-=1.
        for power in range(order):
            assert sum(Fraction.from_float(float(delta[j]))*(j-row)**power
                for j in np.flatnonzero(delta))==0
    nyquist=(-1.)**np.arange(n+1)
    assert np.array_equal((f@nyquist)[1:-1],np.zeros(n-1))


def sixth_order_family(original,parameters):
    """Nine-node rows, exact degree-five moments and zero alternating mode."""
    result=original.copy();n=len(result)-1
    difference=np.array([(-1)**j*comb(6,j) for j in range(7)])/64
    for row,(a0,a2) in enumerate(np.asarray(parameters).reshape(4,2),start=1):
        sign=(-1)**(row+1)
        a1=a0+a2-sign
        delta=np.zeros(9)
        for shift,weight in enumerate((a0,a1,a2)):
            delta[shift:shift+7]+=weight*difference
        result[row]=0.;result[row,row]=1.;result[row,:9]+=delta
        result[n-row]=result[row,::-1]
    return result


def search_design(matrices,row4_mode='free',refine_from=None,training_scope='initial'):
    from scipy.optimize import differential_evolution,minimize
    training=(12,16,24,32) if row4_mode=='free' else (8,12,16,24,32)
    kinds=(42,);cfl_values=(.05,.1,.15,.2)
    if training_scope!='initial':
        training=(8,9,10,11,12,16,24,32)
        kinds=(41,42);cfl_values=(.01,.025,.05,.1,.15,.2)
    blocks={}
    for n in training:
        for kind in kinds:
            a=dynamic_wall_cns_matrix(n,0.) if kind==41 else adiabatic_cns_matrix(n,0.,'neumann7-projection')
            blocks[n,kind]=(state_projection(n,'dirichlet'),
                state_projection(n,'dirichlet' if kind==41 else 'neumann'),
                [(np.eye(len(a))+cfl/(n*np.sqrt(1.4))*a) for cfl in cfl_values],a)
    if training_scope=='asymptotic':
        from review_high_order_wall_filter_design import slow_filter_coefficients
    def reduced_parameters(parameters):
        if row4_mode=='fixed':return np.asarray(parameters)[:6]
        if row4_mode=='symmetric':return np.r_[parameters[:6],np.mean(parameters[6:])]
        return np.asarray(parameters)

    baseline=reduced_parameters(np.array([1.,0.,-1.,0.,1.,0.,-.25,-.25]))
    cache={}

    def full_parameters(parameters):
        if row4_mode=='fixed':return np.r_[parameters,-.25,-.25]
        if row4_mode=='symmetric':return np.r_[parameters[:6],parameters[6],parameters[6]]
        return np.asarray(parameters)

    def objective(parameters):
        key=np.asarray(parameters).tobytes()
        if key not in cache:
            worst=1.
            for n in training:
                f=sixth_order_family(matrices[n],full_parameters(parameters))
                for kind in kinds:
                    pd,thermal,eulers,a=blocks[n,kind]
                    filtering=block_diag(f,f[1:-1]@pd,f[1:-1]@thermal)
                    worst=max(worst,float(np.max(abs(eigvals(filtering)))))
                    if training_scope=='asymptotic' and n in (8,12):
                        slow=slow_filter_coefficients(filtering,a)
                        worst=max(worst,1+slow['slow_maximum_real']/(n*np.sqrt(1.4)))
                        for mode in slow['modes']:
                            worst=max(worst,1+mode['amplitude_dt2']/(n*n*1.4))
                    for b in eulers:
                        stage1=b@filtering
                        stage2=.75*filtering+.25*stage1@stage1
                        mapping=filtering/3+(2/3)*stage1@stage2
                        worst=max(worst,float(np.max(abs(eigvals(mapping)))))
            cache[key]=worst-1.
        return cache[key]

    generation=0
    def progress(parameters,convergence):
        nonlocal generation
        generation+=1
        score=objective(parameters)
        if generation%5==0:
            print('generation',generation,'objective_violation',score,flush=True)
        # This is an optimizer stopping rule, not a numerical acceptance gate.
        return score<1e-12

    if refine_from is None:
        result=differential_evolution(objective,[(-2.,2.)]*len(baseline),seed=20261028,
            popsize=5,maxiter=40,atol=0.,tol=1e-6,polish=False,x0=baseline,callback=progress,workers=1)
        settings=dict(method='differential_evolution',seed=20261028,maximum_generations=40)
    else:
        previous=json.loads(refine_from.read_text())
        start=reduced_parameters(np.array(previous['floating_parameters']))
        def refine_progress(parameters):
            progress(parameters,0.)
        result=minimize(objective,start,method='Powell',bounds=[(-2.,2.)]*len(baseline),
            callback=refine_progress,options=dict(maxfev=3000,xtol=1e-10,ftol=1e-12))
        settings=dict(method='Powell',maximum_function_evaluations=3000,refined_from=str(refine_from))
    fractions=[Fraction(float(value)).limit_denominator(4096) for value in full_parameters(result.x)]
    rational=np.array([float(value) for value in fractions])
    validation=[]
    for n in (64,128):
        f=sixth_order_family(matrices[n],rational)
        pd=state_projection(n,'dirichlet');pn=state_projection(n,'neumann')
        filtering=block_diag(f,f[1:-1]@pd,f[1:-1]@pn)
        for nu in (0.,1e-5,1/1600):
            a=adiabatic_cns_matrix(n,nu,'neumann7-projection')
            for cfl in (0.,.025,.075,.125,.175,.2):
                mapping=rk3_filtered_map(a,filtering,cfl/(n*np.sqrt(1.4)))
                validation.append(dict(intervals=n,viscosity=nu,sound_cfl=cfl,
                    **spectral_diagnostic(mapping)))
    return dict(status='coefficient design research only; no solver change or admission',
        training_intervals=list(training),training_viscosity=0.,training_sound_cfl=list(cfl_values),
        training_bctype=list(kinds),training_scope=training_scope,
        small_cfl_design_condition=training_scope=='asymptotic',
        coefficient_basis='three shifted sixth differences per row; a1=a0+a2-(-1)^(row+1)',
        row4_mode=row4_mode,
        parameter_bounds=[-2,2],search_settings=settings,
        objective_definition='max radius excess and normalized slow linear/quadratic amplitude growth' if training_scope=='asymptotic' else 'max filter/RK spectral radius excess',
        baseline_objective_violation=objective(baseline),best_objective_violation=float(result.fun),
        floating_parameters=full_parameters(result.x).tolist(),rational_parameters=[str(v) for v in fractions],
        rational_objective_violation=objective(reduced_parameters(rational)),generations=generation,
        objective_evaluations=len(cache),held_out_samples=validation,
        caveat='Finite frozen spectra, not common-energy contraction or CFD acceptance; original gates unchanged.')


@pytest.fixture(scope='module')
def native_small(tmp_path_factory):
    return native_filter_matrices(tmp_path_factory.mktemp('hb6_candidates')/'native',intervals=(8,12))


@pytest.mark.parametrize('order',[8,10])
def test_design_boundary_order_moments_and_unchanged_interior(native_small,order):
    candidate_moments(native_small[12],order)


def test_cpu_two_physical_faces_preserve_constant_at_eight_intervals(native_small):
    result=native_small[8]@np.ones(9)
    assert np.array_equal(result,np.ones(9)),result


def test_sixth_order_family_reproduces_native_coefficients(native_small):
    baseline=[1.,0.,-1.,0.,1.,0.,-.25,-.25]
    assert np.array_equal(sixth_order_family(native_small[12],baseline),native_small[12])


def test_sixth_order_family_retains_moments_and_nyquist(native_small):
    original=native_small[12]
    f=sixth_order_family(original,[1.,.125,-1.,.25,.75,-.25,-.25,.125])
    assert np.array_equal(f[5:8],original[5:8])
    assert np.array_equal(f[[0,-1]],original[[0,-1]])
    for row in (*range(1,5),*range(8,12)):
        delta=f[row].copy();delta[row]-=1.
        for power in range(6):
            assert sum(Fraction.from_float(float(delta[j]))*(j-row)**power
                for j in np.flatnonzero(delta))==0
    assert np.array_equal((f@((-1.)**np.arange(13)))[1:-1],np.zeros(11))


def test_fixed_row4_family_preserves_short_line_reflection(native_small):
    for n,original in native_small.items():
        f=sixth_order_family(original,[1.,.125,-1.,.25,.75,-.25,-.25,-.25])
        assert np.array_equal(f,f[::-1,::-1])
        assert np.array_equal(f[[4,n-4]],original[[4,n-4]])
        assert np.array_equal(f@np.ones(n+1),np.ones(n+1))


def test_symmetric_row4_family_preserves_short_line_reflection(native_small):
    for n,original in native_small.items():
        f=sixth_order_family(original,[-1.,-1.,-2.,1.,0.,-1.,1.5,1.5])
        assert np.array_equal(f,f[::-1,::-1])
        assert np.array_equal(f@np.ones(n+1),np.ones(n+1))


def test_rk_filter_taylor_reconstructs_the_actual_stage_map():
    from review_high_order_wall_filter_design import rk_filter_taylor
    a=np.array([[0.,1.,0.],[-2.,0.,1.],[0.,-1.,0.]])
    f=np.array([[1.,0.,0.],[.125,.75,.125],[0.,0.,1.]])
    coefficients=rk_filter_taylor(a,f)
    for dt in (.001,.1,.2):
        actual=rk3_filtered_map(a,f,dt)
        expansion=sum(dt**power*matrix for power,matrix in enumerate(coefficients))
        assert np.max(abs(actual-expansion))<1e-14
    identity=rk_filter_taylor(a,np.eye(3))
    assert np.max(abs(identity[1]-a))<1e-14
    assert np.max(abs(identity[2]-a@a/2))<1e-14


def test_rejected_dyadic_candidate_has_small_cfl_growth(native_small):
    from review_high_order_wall_filter_design import blocks,small_cfl_diagnostic
    parameters=np.array([-1.,-1.,-.5,.5,1.,-1.,1.5,1.5])
    part=blocks(native_small[8],parameters,41)[0]
    report=small_cfl_diagnostic(part[1],part[2])
    maximum=max(m['amplitude_dt2'] for m in report['modes'])
    assert 1.79<maximum<1.81
    assert abs(report['shrinking_dt'][-1]['log_amplitude_over_dt2']-maximum)<.001
    assert report['unit_right_residual']<1e-12 and report['unit_left_residual']<1e-12


def test_amplitude_quadratic_coefficient_with_nonzero_linear_growth():
    from review_high_order_wall_filter_design import slow_filter_coefficients
    a=np.array([[.2,-1.],[1.,.2]])
    report=slow_filter_coefficients(np.eye(2),a)
    assert abs(report['modes'][0]['amplitude_dt2']-.2**2/2)<1e-14


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--large-native',type=Path,required=True)
    parser.add_argument('--small-native',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--search',action='store_true',help='Bounded frozen-spectrum coefficient design only')
    parser.add_argument('--row4-mode',choices=('free','fixed','symmetric'),default='free')
    parser.add_argument('--refine-from',type=Path,help='Refine a saved design result using the same constraints')
    parser.add_argument('--training-scope',choices=('initial','joint','asymptotic'),default='initial')
    parser.add_argument('--extra-native',type=Path,help='Additional immutable native matrices for design review')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    matrices={}
    for n in (12,16,24,32,64,128):
        directory=args.small_native if n<32 else args.large_native
        matrices[n]=np.fromfile(directory/f'filter_{n}.bin','<f8').reshape(n+1,n+1,order='F')
    if args.search:
        if args.extra_native is not None:
            for path in args.extra_native.glob('filter_*.bin'):
                n=int(path.stem.split('_')[-1])
                matrices[n]=np.fromfile(path,'<f8').reshape(n+1,n+1,order='F')
        if args.row4_mode!='free':
            matrices[8]=np.fromfile(args.small_native/'filter_8.bin','<f8').reshape(9,9,order='F')
            assert np.array_equal(matrices[8]@np.ones(9),np.ones(9))
        report=search_design(matrices,args.row4_mode,args.refine_from,args.training_scope)
        with (args.output/'coefficient_search.json').open('x') as stream:
            json.dump(report,stream,indent=2);stream.write('\n')
        print('best_objective_violation',report['best_objective_violation'],flush=True)
        print('held_out_maximum_radius',max(s['spectral_radius'] for s in report['held_out_samples']),flush=True)
        raise SystemExit(0)
    for original in matrices.values():
        for order in (8,10):candidate_moments(original,order)
    report=compare_candidates(matrices)
    overlap=np.fromfile(args.small_native/'filter_8.bin','<f8').reshape(9,9,order='F')
    constant=overlap@np.ones(9)
    report['independent_cpu_defect']=dict(intervals=8,ntype=4,
        constant_result=constant.tolist(),constant_maximum_error=float(np.max(abs(constant-1))),
        action='historical input matrix; do not infer current source status from this artifact')
    with (args.output/'candidate_comparison.json').open('x') as stream:
        json.dump(report,stream,indent=2);stream.write('\n')
    for variant in ('original','temperature-projection','increment-projection','order8','order10'):
        worst=max((r for r in report['variants'] if r['variant']==variant),
            key=lambda r:r['worst_sample']['spectral_radius'])
        print(variant,json.dumps(worst),flush=True)
    print('CPU constant overlap:',constant.tolist(),flush=True)
