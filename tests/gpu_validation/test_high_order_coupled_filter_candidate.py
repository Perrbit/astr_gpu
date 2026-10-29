"""HB6 coupled-coordinate design; no production implementation or CFD march."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import block_diag,eig,null_space,orth

from test_high_order_adiabatic_wall_candidate import adiabatic_cns_matrix
from test_high_order_boundary_closures import derivative_matrix
from test_high_order_explicit_wall_candidates import dynamic_wall_cns_matrix,state_projection
from test_high_order_wall_filter_candidates import boundary_order_candidate,native_small
from review_high_order_wall_filter_design import spectrum_sample


def algebraic_metric(a,allow_decay=False):
    """A quadratic metric for this diagonalizable finite frozen matrix only."""
    values,vectors=eig(a)
    tolerance=100*np.finfo(float).eps*max(1.,np.linalg.norm(a,np.inf))
    assert np.max(values.real if allow_decay else abs(values.real))<tolerance
    columns=[null_space(a,rcond=100*np.finfo(float).eps)]
    for j,value in enumerate(values):
        if value.imag>tolerance:
            columns.extend((vectors[:,j:j+1].real,vectors[:,j:j+1].imag))
        elif allow_decay and abs(value.imag)<=tolerance and value.real<-tolerance:
            columns.append(vectors[:,j:j+1].real)
    basis=np.column_stack(columns)
    assert basis.shape==a.shape
    inverse=np.linalg.inv(basis)
    return inverse.T@inverse


def constrained_reference(n,kind):
    """Small closed-line projection reference, not a large-domain stencil."""
    size=n+1
    pd=state_projection(n,'dirichlet')
    pt=state_projection(n,'dirichlet' if kind==41 else 'neumann')
    injection=block_diag(np.eye(size),pd,pt)
    selection=block_diag(np.eye(size),np.eye(size)[1:-1],np.eye(size)[1:-1])
    a=dynamic_wall_cns_matrix(n,0.) if kind==41 else adiabatic_cns_matrix(n,0.,'neumann7-projection')
    metric=algebraic_metric(a)
    complement=null_space(injection.T)
    transform=np.column_stack((injection,complement));inverse=np.linalg.inv(transform)
    full_metric=inverse.T@block_diag(metric,np.eye(4))@inverse
    reflection=block_diag(np.eye(size)[::-1],-np.eye(size)[::-1],np.eye(size)[::-1])
    full_metric=(full_metric+reflection.T@full_metric@reflection)/2
    constraints=np.zeros((6,3*size))
    constraints[0,size]=1.;constraints[1,2*size-1]=1.
    if kind==41:
        constraints[2,2*size]=1.;constraints[3,-1]=1.
    else:
        from test_high_order_wall_filter_candidates import constraints as neumann_constraints
        constraints[2:4,2*size:]=neumann_constraints(n)
    constraints[4,0]=1.;constraints[5,n]=1.
    polynomials=block_diag(*([np.polynomial.legendre.legvander(np.linspace(-1,1,size),5)]*3))
    raw=np.column_stack((polynomials,np.linalg.solve(full_metric,constraints.T)))
    raw=raw/np.linalg.norm(raw,axis=0)
    preserved=orth(raw,rcond=1e-10)
    projection=preserved@np.linalg.solve(preserved.T@full_metric@preserved,preserved.T@full_metric)
    reduced=selection@projection@injection
    reduced_metric=injection.T@full_metric@injection
    report=dict(intervals=n,bctype=kind,
        scope='full small closed-line algebraic projection; not an admitted local boundary stencil',
        preserved_full_dimension=preserved.shape[1],
        minimum_metric_eigenvalue=float(np.min(np.linalg.eigvalsh(reduced_metric))),
        metric_condition=float(np.linalg.cond(reduced_metric)),
        skew_residual=float(np.linalg.norm(a.T@reduced_metric+reduced_metric@a,np.inf)),
        polynomial_residual=float(np.max(abs(projection@polynomials-polynomials))),
        constraint_residual=float(np.max(abs(constraints@projection-constraints))),
        projection_residual=float(np.max(abs(projection@projection-projection))),
        reflection_residual=float(np.max(abs(reflection@projection@reflection-projection))),
        reduced_projection_residual=float(np.max(abs(reduced@reduced-reduced))),
        contraction_maximum_eigenvalue=float(np.max(np.linalg.eigvalsh(
            reduced.T@reduced_metric@reduced-reduced_metric))))
    return reduced,reduced_metric,projection,report


def embedded_projection(original,kind,template):
    """Seven near-wall rows, nine input nodes; overlap is not yet designed."""
    n=len(original)-1
    if n<16:raise ValueError('Overlapping two-sided embedded strips are not designed')
    size=n+1;full=block_diag(original,original,original)
    parity=(1.,-1.,1.)
    for component in range(3):
        for row in range(8):
            low=component*size+row;high=component*size+n-row
            full[low]=0.;full[high]=0.
            for source in range(3):
                coefficients=template[component*9+row,source*9:source*9+9]
                full[low,source*size:source*size+9]=coefficients
                full[high,source*size+n-np.arange(9)]=parity[component]*parity[source]*coefficients
    pd=state_projection(n,'dirichlet')
    pt=state_projection(n,'dirichlet' if kind==41 else 'neumann')
    injection=block_diag(np.eye(size),pd,pt)
    selection=block_diag(np.eye(size),np.eye(size)[1:-1],np.eye(size)[1:-1])
    return selection@full@injection,full


def polynomial_lift_reference(n,kind,viscosity=0.):
    """Constraint-compatible polynomial lift with a reduced-state projection."""
    from test_high_order_wall_filter_candidates import constraints as neumann_constraints
    size=n+1;x=np.linspace(0,1,size)
    pd=state_projection(n,'dirichlet')
    pt=state_projection(n,'dirichlet' if kind==41 else 'neumann')
    injection=block_diag(np.eye(size),pd,pt)
    selection=block_diag(np.eye(size),np.eye(size)[1:-1],np.eye(size)[1:-1])
    a=dynamic_wall_cns_matrix(n,viscosity) if kind==41 else adiabatic_cns_matrix(n,viscosity,'neumann7-projection')
    h=algebraic_metric(a,allow_decay=viscosity>0.)
    jr=block_diag(np.eye(size)[::-1],-np.eye(n-1)[::-1],np.eye(n-1)[::-1])
    h=(h+jr.T@h@jr)/2
    velocity=np.column_stack([(x*(1-x)*x**j)[1:-1] for j in range(4)])
    temperature=velocity if kind==41 else np.column_stack([np.ones(n-1)]+[
        (x**(j+2)/(j+2)-x**(j+3)/(j+3))[1:-1] for j in range(3)])
    polynomials=block_diag(np.polynomial.legendre.legvander(2*x-1,5),velocity,temperature)
    density=np.zeros((2,len(a)));density[0,0]=1.;density[1,n]=1.
    raw=np.column_stack((polynomials,np.linalg.solve(h,density.T)))
    raw=raw/np.linalg.norm(raw,axis=0)
    q=orth(raw,rcond=1e-10)
    reduced=q@np.linalg.solve(q.T@h@q,q.T@h)
    c=np.zeros((4,3*size));c[0,size]=1.;c[1,2*size-1]=1.
    lift=np.zeros((3*size,4))
    lift[size:2*size,0]=1-x;lift[size:2*size,1]=x
    if kind==41:
        c[2,2*size]=1.;c[3,-1]=1.
        lift[2*size:,2]=1-x;lift[2*size:,3]=x
    else:
        c[2:4,2*size:]=neumann_constraints(n)
        # The constants only fix the auxiliary lift gauge, not the field mean.
        lift[2*size:,2]=n*(x-x*x/2-1/3)
        lift[2*size:,3]=n*(1/6-x*x/2)
    full=lift@c+injection@reduced@selection@(np.eye(3*size)-lift@c)
    transform=np.column_stack((injection,lift));inverse=np.linalg.inv(transform)
    hf=inverse.T@block_diag(h,np.eye(4))@inverse
    basis=block_diag(*([np.polynomial.legendre.legvander(2*x-1,5)]*3))
    reflection=block_diag(np.eye(size)[::-1],-np.eye(size)[::-1],np.eye(size)[::-1])
    report=dict(intervals=n,bctype=kind,viscosity=viscosity,scope='small closed-line polynomial-lift reference only',
        preserved_reduced_dimension=q.shape[1],minimum_metric_eigenvalue=float(min(np.linalg.eigvalsh(h))),
        lift_residual=float(np.max(abs(c@lift-np.eye(4)))),
        polynomial_residual=float(np.max(abs(full@basis-basis))),
        density_residual=float(np.max(abs(density@reduced-density))),
        constraint_residual=float(np.max(abs(c@full-c))),
        projection_residual=float(np.max(abs(full@full-full))),
        reflection_residual=float(np.max(abs(reflection@full@reflection-full))),
        generator_energy_maximum_eigenvalue=float(max(np.linalg.eigvalsh(a.T@h+h@a))),
        contraction_maximum_eigenvalue=float(max(np.linalg.eigvalsh(full.T@hf@full-hf))))
    return reduced,h,full,report


def modal_filter(pressure,entropy,velocity,kind):
    """p'=rho'+T', s'=T'-(gamma-1)rho' at rho=RT=1 and zero velocity."""
    n=len(pressure)-1
    pd=state_projection(n,'dirichlet')
    thermal=state_projection(n,'dirichlet' if kind==41 else 'neumann')
    difference=pressure-entropy
    # Difference form recovers the original scalar filter exactly when P=S.
    rho=entropy+difference/1.4
    temperature=entropy+(.4/1.4)*difference
    zero=np.zeros((n+1,n-1));small_zero=np.zeros((n-1,n-1))
    return np.block([
        [rho,zero,(difference@thermal)/1.4],
        [np.zeros((n-1,n+1)),velocity[1:-1]@pd,small_zero],
        [(.4/1.4)*difference[1:-1],small_zero,temperature[1:-1]@thermal]])


def normal_parts(original,fp,fs,fv,kind):
    n=len(original)-1
    filtering=modal_filter(fp,fs,fv,kind)
    a0=dynamic_wall_cns_matrix(n,0.) if kind==41 else adiabatic_cns_matrix(n,0.,'neumann7-projection')
    a1=dynamic_wall_cns_matrix(n,1.) if kind==41 else adiabatic_cns_matrix(n,1.,'neumann7-projection')
    result=[('normal',filtering,a0,a1-a0)]
    d=n*derivative_matrix(n,True)
    for name,mode in (('tangent_D','dirichlet'),('tangent_N','neumann')):
        if kind==41 and mode=='neumann':continue
        p=state_projection(n,mode);f=fv[1:-1]@p
        result.append((name,f,np.zeros_like(f),(d@d@p)[1:-1]))
    return result


def modal_parts(original,choices,kind):
    operators={6:original,8:boundary_order_candidate(original,8)}
    return normal_parts(original,*(operators[choice] for choice in choices),kind)


def blend_parts(original,parameters,kind):
    difference=boundary_order_candidate(original,8)-original
    return normal_parts(original,*(original+weight*difference for weight in parameters),kind)


def reference_samples(f,a0,a1):
    samples=[]
    for nu in (0.,1e-5,1/1600):
        for cfl in (0.,.01,.025,.05,.1,.15,.2):
            sample=spectrum_sample([('normal',f,a0,a1-a0)],nu,cfl,8)
            samples.append(sample)
            if not sample['passes']:return samples,sample
    return samples,None


def search_blend(matrices):
    from scipy.linalg import eigvals
    from scipy.optimize import differential_evolution
    from review_high_order_wall_filter_design import slow_filter_coefficients
    prepared={}
    for n in (8,9,10,11,12,16,24,32):
        f0=matrices[n];f8=boundary_order_candidate(f0,8)
        for kind in (41,42):
            a=dynamic_wall_cns_matrix(n,0.) if kind==41 else adiabatic_cns_matrix(n,0.,'neumann7-projection')
            base=modal_filter(f0,f0,f0,kind)
            changes=(modal_filter(f8,f0,f0,kind)-base,modal_filter(f0,f8,f0,kind)-base,
                modal_filter(f0,f0,f8,kind)-base)
            prepared[n,kind]=(a,[np.eye(len(a))+c/(n*np.sqrt(1.4))*a for c in (.01,.025,.05,.1,.15,.2)],base,changes)
    cache={}
    def objective(parameters):
        key=np.asarray(parameters).tobytes()
        if key in cache:return cache[key]
        worst=1.
        for (n,kind),(a,eulers,base,changes) in prepared.items():
            f=base+sum(weight*change for weight,change in zip(parameters,changes))
            worst=max(worst,float(np.max(abs(eigvals(f)))))
            if n in (8,12):
                slow=slow_filter_coefficients(f,a)
                worst=max(worst,1+slow['slow_maximum_real']/(n*np.sqrt(1.4)))
                for mode in slow['modes']:
                    worst=max(worst,1+mode['amplitude_dt2']/(n*n*1.4))
            for b in eulers:
                first=b@f;second=.75*f+.25*first@first
                mapping=f/3+(2/3)*first@second
                worst=max(worst,float(np.max(abs(eigvals(mapping)))))
        cache[key]=worst-1.
        return cache[key]
    generation=0
    def progress(parameters,convergence):
        nonlocal generation
        generation+=1
        value=objective(parameters)
        if generation%5==0:print('blend_generation',generation,'objective_violation',value,flush=True)
        return value<1e-12
    result=differential_evolution(objective,[(-1.,2.)]*3,seed=20261028,
        maxiter=30,popsize=8,polish=False,atol=0.,tol=1e-6,x0=np.zeros(3),callback=progress,workers=1)
    return dict(status='coupled modal blend design only; not admitted',
        definition='F_mode=F_original+alpha_mode*(F_near8-F_original)',
        parameters=result.x.tolist(),best_objective_violation=float(result.fun),
        objective='max filter/RK radius excess and normalized small-CFL linear/quadratic growth',
        training_intervals=[8,9,10,11,12,16,24,32],training_bctype=[41,42],
        training_sound_cfl=[.01,.025,.05,.1,.15,.2],viscosity=0.,
        small_cfl_intervals=[8,12],bounds=[-1,2],seed=20261028,
        maximum_generations=30,generations=generation,evaluations=len(cache))


@pytest.mark.parametrize('kind',[41,42])
def test_modal_equal_filters_recover_the_native_scalar_map(native_small,kind):
    for n,f in native_small.items():
        pd=state_projection(n,'dirichlet')
        thermal=state_projection(n,'dirichlet' if kind==41 else 'neumann')
        reference=block_diag(f,f[1:-1]@pd,f[1:-1]@thermal)
        assert np.array_equal(modal_filter(f,f,f,kind),reference)


@pytest.mark.parametrize('kind',[41,42])
def test_modal_coupling_preserves_density_endpoints_and_deep_rows(native_small,kind):
    for n,original in native_small.items():
        f8=boundary_order_candidate(original,8)
        actual=modal_filter(f8,original,f8,kind)
        reference=modal_filter(original,original,original,kind)
        assert np.array_equal(actual[[0,n]],reference[[0,n]])
        for row in range(5,n-4):
            indices=[row,n+row,2*n+row-1]
            assert np.array_equal(actual[indices],reference[indices])
        reflection=block_diag(np.eye(n+1)[::-1],-np.eye(n-1)[::-1],np.eye(n-1)[::-1])
        assert np.max(abs(reflection@actual@reflection-actual))<1e-13


@pytest.mark.parametrize('kind',[41,42])
def test_modal_filter_preserves_admissible_degree_five_polynomials(native_small,kind):
    for n,original in native_small.items():
        x=np.linspace(0,1,n+1)
        rho=np.column_stack([x**j for j in range(6)])
        velocity=np.column_stack([(x*(1-x)*x**j)[1:-1] for j in range(4)])
        if kind==41:
            temperature=velocity
        else:
            temperature=np.column_stack([np.ones(n-1)]+[
                (x**(j+2)/(j+2)-x**(j+3)/(j+3))[1:-1] for j in range(3)])
        basis=block_diag(rho,velocity,temperature)
        f8=boundary_order_candidate(original,8)
        actual=modal_filter(f8,original,f8,kind)
        assert np.max(abs(actual@basis-basis))<1e-13


@pytest.mark.parametrize('kind',[41,42])
def test_modal_blend_zero_recovers_original_and_leaves_row4(native_small,kind):
    for n,original in native_small.items():
        reference=modal_filter(original,original,original,kind)
        assert np.array_equal(blend_parts(original,[0.,0.,0.],kind)[0][1],reference)
        actual=blend_parts(original,[.25,.5,.75],kind)[0][1]
        for row in range(4,n-3):
            indices=[row,n+row,2*n+row-1]
            assert np.array_equal(actual[indices],reference[indices])


@pytest.mark.parametrize('kind',[41,42])
def test_small_coupled_reference_preserves_constraints_and_polynomials(kind):
    _,_,_,report=constrained_reference(8,kind)
    assert report['minimum_metric_eigenvalue']>0
    for name in ('polynomial_residual','constraint_residual','projection_residual',
                 'reflection_residual','reduced_projection_residual'):
        assert report[name]<2e-10,(name,report)
    assert report['skew_residual']<2e-10
    assert report['contraction_maximum_eigenvalue']<2e-10


@pytest.mark.parametrize('kind',[41,42])
def test_embedded_coupled_template_preserves_moments_and_deep_map(kind):
    n=32
    # Only the near strip is inspected here; deep rows are arbitrary sentinels.
    original=np.diag(np.arange(n+1,dtype=float)+1)
    _,_,template,_=constrained_reference(8,kind)
    _,full=embedded_projection(original,kind,template)
    native=block_diag(original,original,original)
    polynomial=np.polynomial.legendre.legvander(np.linspace(-1,1,n+1),5)
    basis=block_diag(polynomial,polynomial,polynomial)
    for component in range(3):
        low=component*(n+1)+np.arange(8)
        high=component*(n+1)+n-np.arange(8)
        for rows in (low,high):
            assert np.max(abs((full@basis-basis)[rows]))<2e-10
        deep=component*(n+1)+np.arange(8,n-7)
        assert np.array_equal(full[deep],native[deep])


@pytest.mark.parametrize('kind',[41,42])
def test_polynomial_lift_reference_preserves_the_full_moment_contract(kind):
    _,_,_,report=polynomial_lift_reference(8,kind)
    assert report['minimum_metric_eigenvalue']>0
    for key in ('lift_residual','polynomial_residual','density_residual','constraint_residual',
                'projection_residual','reflection_residual','contraction_maximum_eigenvalue'):
        assert report[key]<2e-10,(key,report)


def test_reference_gate_stops_on_first_failed_map():
    samples,failure=reference_samples(2*np.eye(3),np.zeros((3,3)),np.zeros((3,3)))
    assert len(samples)==1 and failure is samples[0]
    assert not failure['passes']


@pytest.mark.parametrize('kind',[41,42])
@pytest.mark.parametrize('viscosity',[0.,1e-5,1/1600])
def test_viscous_reference_matches_its_own_frozen_state(kind,viscosity):
    f,h,_,report=polynomial_lift_reference(8,kind,viscosity)
    a=dynamic_wall_cns_matrix(8,viscosity) if kind==41 else adiabatic_cns_matrix(8,viscosity,'neumann7-projection')
    scale=max(1.,np.linalg.norm(a,np.inf)*np.linalg.norm(h,np.inf))
    assert report['generator_energy_maximum_eigenvalue']<=100*np.finfo(float).eps*scale
    for cfl in (0.,.01,.025,.05,.1,.15,.2):
        sample=spectrum_sample([('normal',f,a,np.zeros_like(a))],0.,cfl,8)
        assert sample['passes'],sample


def compare(matrices):
    results=[]
    for choices in itertools.product((6,8),repeat=3):
        records=[];failure=None
        for n,original in sorted(matrices.items()):
            for kind in (41,42):
                parts=modal_parts(original,choices,kind)
                standalone=spectrum_sample(parts,0.,None,n)
                standalone.update(intervals=n,bctype=kind)
                records.append(standalone)
                if not standalone['passes']:
                    failure=standalone;break
                for cfl in (0.,.01,.025,.05,.1,.15,.2):
                    sample=spectrum_sample(parts,0.,cfl,n)
                    sample.update(intervals=n,bctype=kind);records.append(sample)
                    if not sample['passes']:
                        failure=sample;break
                if failure is not None:break
            if failure is not None:break
        results.append(dict(pressure=choices[0],entropy=choices[1],velocity=choices[2],
            samples=records,first_failure=failure,all_sampled_pass=failure is None))
        print('modal_choices',choices,'samples',len(records),'passed',failure is None,
            'first_radius',None if failure is None else failure['spectral_radius'],flush=True)
    return dict(status='coupled design only; no solver or admission change',
        coordinate_scope='linear pressure/entropy perturbations at uniform rest, gamma=1.4, rho=RT=1',
        note='6 denotes the existing mixed 0-6-6-6-8-10 filter; 8 denotes near-end eighth-order rows',
        candidates=results)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--native',type=Path,required=True)
    parser.add_argument('--extra-native',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--search-blend',action='store_true')
    parser.add_argument('--reference',choices=('full-constraints','polynomial-lift','frozen-viscous'),default='full-constraints')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    matrices={}
    for directory in (args.native,args.extra_native):
        for path in directory.glob('filter_*.bin'):
            n=int(path.stem.split('_')[-1])
            matrices[n]=np.fromfile(path,'<f8').reshape(n+1,n+1,order='F')
    if args.search_blend:
        report=search_blend(matrices)
        with (args.output/'modal_blend_search.json').open('x') as stream:
            json.dump(report,stream,indent=2);stream.write('\n')
        print('modal_blend_result',json.dumps(report),flush=True)
        raise SystemExit(0)
    if args.reference=='frozen-viscous':
        cases=[]
        for kind in (41,42):
            for number,nu in enumerate((0.,1e-5,1/1600)):
                f,h,full,case=polynomial_lift_reference(8,kind,nu)
                a=dynamic_wall_cns_matrix(8,nu) if kind==41 else adiabatic_cns_matrix(8,nu,'neumann7-projection')
                filename=f'viscous_reference_{kind}_{number}.npz'
                np.savez(args.output/filename,reduced=f,metric=h,full=full)
                case['matrix_file']=filename;case['samples']=[]
                for cfl in (0.,.01,.025,.05,.1,.15,.2):
                    sample=spectrum_sample([('normal',f,a,np.zeros_like(a))],0.,cfl,8)
                    case['samples'].append(sample)
                    if not sample['passes']:break
                case['all_sampled_pass']=all(s['passes'] for s in case['samples'])
                cases.append(case)
                print('frozen_viscous_reference',kind,nu,'passed',case['all_sampled_pass'],flush=True)
        report=dict(status='viscosity-dependent small-line references only; no local production stencil',
            cases=cases,limitations='No common viscosity-independent metric, nonlinear extension or MPI ownership accepted')
        with (args.output/'frozen_viscous_references.json').open('x') as stream:
            json.dump(report,stream,indent=2);stream.write('\n')
        raise SystemExit(0)
    report=compare(matrices)
    factory=constrained_reference if args.reference=='full-constraints' else polynomial_lift_reference
    report['reference_construction']=args.reference
    references=[]
    for kind in (41,42):
        f,h,template,reference=factory(8,kind)
        filename=f'reference_{kind}.npz'
        np.savez(args.output/filename,reduced=f,metric=h,full=template)
        reference['matrix_file']=filename
        a0=dynamic_wall_cns_matrix(8,0.) if kind==41 else adiabatic_cns_matrix(8,0.,'neumann7-projection')
        a1=dynamic_wall_cns_matrix(8,1.) if kind==41 else adiabatic_cns_matrix(8,1.,'neumann7-projection')
        reference['samples'],reference['first_failure']=reference_samples(f,a0,a1)
        reference['all_sampled_pass']=reference['first_failure'] is None
        references.append(reference)
        print('coupled_reference',kind,'maximum_radius',max(s['spectral_radius'] for s in reference['samples']),flush=True)
    report['small_closed_line_references']=references
    embedded=[]
    for kind in (41,42):
        reference=next(r for r in references if r['bctype']==kind)
        if not reference['all_sampled_pass']:
            embedded.append(dict(bctype=kind,status='not run: reference spectrum failed',samples=[],
                all_sampled_pass=False))
            print('embedded_reference',kind,'not run: reference spectrum failed',flush=True)
            continue
        _,_,template,_=factory(8,kind)
        records=[];failure=None
        for n,original in sorted(matrices.items()):
            if n<16:continue
            f,_=embedded_projection(original,kind,template)
            a0=dynamic_wall_cns_matrix(n,0.) if kind==41 else adiabatic_cns_matrix(n,0.,'neumann7-projection')
            a1=dynamic_wall_cns_matrix(n,1.) if kind==41 else adiabatic_cns_matrix(n,1.,'neumann7-projection')
            for nu in (0.,1e-5,1/1600):
                for cfl in (0.,.01,.025,.05,.1,.15,.2):
                    sample=spectrum_sample([('normal',f,a0,a1-a0)],nu,cfl,n)
                    sample.update(intervals=n,bctype=kind);records.append(sample)
                    if not sample['passes']:
                        failure=sample;break
                if failure is not None:break
            if failure is not None:break
        embedded.append(dict(bctype=kind,samples=records,first_failure=failure,
            all_sampled_pass=failure is None,
            limitation='N=9..15 overlap and MPI ownership are pending; no size admission changed'))
        print('embedded_reference',kind,'samples',len(records),'passed',failure is None,
            'first_radius',None if failure is None else failure['spectral_radius'],flush=True)
    report['embedded_references']=embedded
    with (args.output/'modal_comparison.json').open('x') as stream:
        json.dump(report,stream,indent=2);stream.write('\n')
