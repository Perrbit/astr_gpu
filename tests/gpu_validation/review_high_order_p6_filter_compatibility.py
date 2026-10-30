"""HB6 frozen design review on the current P7 / P6-T7 space blocks, not CFD."""
import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag,eigvals,eigvalsh,orth

from test_high_order_adiabatic_pressure_candidate import normal_model
from test_high_order_boundary_closures import derivative_matrix
from test_high_order_coupled_filter_candidate import modal_filter,algebraic_metric
from test_high_order_explicit_wall_candidates import state_projection
from test_high_order_wall_filter_candidates import boundary_order_candidate,spectral_diagnostic
from test_high_order_wall_filter_coupling import native_filter_matrices,rk3_filtered_map


def current_normal(n,viscosity,kind):
    if kind not in (41,42,421):
        raise ValueError('The frozen review covers 41, 42 and 421 only')
    return normal_model(n,viscosity,degree=7 if kind==41 else 6,kind=kind)


def acoustic_filter(original,kind,acoustic_strength,entropy_strength,cross_strength):
    """Frozen p +/- sqrt(gamma) v filter; no nonlinear solver implementation."""
    n=len(original)-1;identity=np.eye(n+1);delta=original-identity
    near=sorted(set(range(1,5))|set(range(n-4,n)))
    acoustic=original.copy();entropy=original.copy()
    acoustic[near]=identity[near]+acoustic_strength*delta[near]
    entropy[near]=identity[near]+entropy_strength*delta[near]
    cross=np.zeros_like(original)
    for row in range(1,5):
        if row==n-row:continue
        cross[row]=cross_strength*delta[row]
        cross[n-row]=-cross[row,::-1]
    velocity=state_projection(n,'dirichlet')
    thermal=state_projection(n,'dirichlet' if kind==41 else 'neumann')
    gamma=1.4;sound=np.sqrt(gamma);difference=acoustic-entropy
    return np.block([
        [entropy+difference/gamma,(sound/gamma)*cross@velocity,difference@thermal/gamma],
        [cross[1:-1]/sound,(acoustic@velocity)[1:-1],(cross@thermal)[1:-1]/sound],
        [(.4/gamma)*difference[1:-1],(.4*sound/gamma)*(cross@velocity)[1:-1],
            ((entropy+(.4/gamma)*difference)@thermal)[1:-1]]])


def parts(original,choices,kind):
    n=len(original)-1
    operators={6:original,8:boundary_order_candidate(original,8)}
    filtering=modal_filter(*(operators[order] for order in choices),kind)
    a0=current_normal(n,0.,kind)
    diffusion=current_normal(n,1.,kind)-a0
    result=[('normal',filtering,a0,diffusion)]
    derivative=n*derivative_matrix(n,True)
    modes=('dirichlet','neumann') if kind==421 else ('dirichlet',)
    for mode in modes:
        projection=state_projection(n,mode)
        result.append(('tangent_'+mode,operators[choices[2]][1:-1]@projection,
            np.zeros((n-1,n-1)),(derivative@derivative@projection)[1:-1]))
    return result


def sample(blocks,viscosity,cfl,n):
    entries=[];worst=None
    for name,filtering,a0,diffusion in blocks:
        matrix=filtering if cfl is None else rk3_filtered_map(a0+viscosity*diffusion,
            filtering,cfl/(n*np.sqrt(1.4)))
        radius=float(np.max(abs(eigvals(matrix))))
        tolerance=float(100*np.finfo(float).eps*max(1.,np.linalg.norm(matrix,np.inf)))
        entry=dict(block=name,spectral_radius=radius,roundoff_reference=tolerance,
            passes=bool(radius<=1+tolerance))
        entries.append(entry)
        if worst is None or radius-1-tolerance>worst[0]:
            worst=(radius-1-tolerance,matrix)
    passed=all(entry['passes'] for entry in entries)
    return dict(viscosity=viscosity,sound_cfl=cfl,blocks=entries,passes=passed,
        diagnostic=None if passed else spectral_diagnostic(worst[1]))


def scan(matrices,build):
    samples=[]
    for n,original in sorted(matrices.items()):
        blocks=build(original)
        standalone=sample(blocks,0.,None,n)
        standalone['intervals']=n;samples.append(standalone)
        if not standalone['passes']:return samples,standalone
        for viscosity in (0.,1e-6,1e-5,1e-4,1/1600):
            for cfl in (0.,.001,.005,.01,.025,.05,.1,.15,.2):
                entry=sample(blocks,viscosity,cfl,n)
                entry['intervals']=n;samples.append(entry)
                if not entry['passes']:return samples,entry
    return samples,None


def review(matrices,kind,choices):
    return scan(matrices,lambda original:parts(original,choices,kind))


def review_acoustic(matrices,kind,parameters):
    def build(original):
        blocks=parts(original,(6,6,6),kind)
        name,_,a0,diffusion=blocks[0]
        blocks[0]=(name,acoustic_filter(original,kind,*parameters),a0,diffusion)
        return blocks
    return scan(matrices,build)


def energy_reference(n,kind,reference_viscosity=None):
    """Dense small-line diagnostic, deliberately not a local production filter."""
    a=current_normal(n,0.,kind)
    diffusion=current_normal(n,1.,kind)-a
    reference=a if reference_viscosity is None else a+reference_viscosity*diffusion
    metric=algebraic_metric(reference,allow_decay=reference_viscosity is not None and reference_viscosity>0.)
    indices=np.r_[np.arange(n,-1,-1),np.arange(2*n-1,n,-1),np.arange(3*n-2,2*n-1,-1)]
    signs=np.r_[np.ones(n+1),-np.ones(n-1),np.ones(n-1)]
    metric=(metric+signs[:,None]*metric[np.ix_(indices,indices)]*signs[None,:])/2
    x=np.linspace(0,1,n+1)
    density=np.polynomial.legendre.legvander(2*x-1,5)
    velocity=np.column_stack([(x*(1-x)*x**j)[1:-1] for j in range(4)])
    thermal=velocity if kind==41 else np.column_stack([np.ones(n-1)]+[
        (x**(j+2)/(j+2)-x**(j+3)/(j+3))[1:-1] for j in range(3)])
    polynomials=block_diag(density,velocity,thermal)
    endpoints=np.eye(len(a))[[0,n]].T
    raw=np.column_stack((polynomials,np.linalg.solve(metric,endpoints)))
    basis=orth(raw/np.linalg.norm(raw,axis=0))
    filtering=basis@np.linalg.solve(basis.T@metric@basis,basis.T@metric)
    samples=[]
    viscosities=(0.,1e-5,1/1600) if reference_viscosity is None else (reference_viscosity,)
    for viscosity in viscosities:
        for cfl in (0.,.001,.01,.05,.1,.2):
            samples.append(sample([('normal',filtering,a,diffusion)],viscosity,cfl,n))
    failed=[entry for entry in samples if not entry['passes']]
    report=dict(status='rejected' if failed else 'small dense reference only; not admitted',
        intervals=n,bctype=kind,space='P7' if kind==41 else 'P6/T7',
        scope='dense H-orthogonal projection on one small global line; no deep-interior or locality claim',
        reference_viscosity=reference_viscosity,viscosities_tested=viscosities,
        preserved_dimension=basis.shape[1],metric_condition=float(np.linalg.cond(metric)),
        metric_minimum_eigenvalue=float(np.min(eigvalsh(metric))),
        polynomial_residual=float(np.max(abs(filtering@polynomials-polynomials))),
        endpoint_residual=float(np.max(abs(filtering[[0,n]]-np.eye(len(a))[[0,n]]))),
        contraction_maximum_eigenvalue=float(np.max(eigvalsh(filtering.T@metric@filtering-metric))),
        inviscid_energy_residual=float(np.linalg.norm(a.T@metric+metric@a,np.inf)),
        matched_generator_energy_maximum_eigenvalue=float(np.max(eigvalsh(reference.T@metric+metric@reference))),
        generator_roundoff_reference=float(100*np.finfo(float).eps*max(1.,
            2*np.linalg.norm(reference,np.inf)*np.linalg.norm(metric,np.inf))),
        viscous_energy_maximum_eigenvalue=float(np.max(eigvalsh(diffusion.T@metric+metric@diffusion))),
        samples=samples,first_failure=failed[0] if failed else None)
    return filtering,metric,report


def load_native(directory):
    matrices={}
    for path in sorted(Path(directory).glob('filter_*.bin')):
        n=int(path.stem.split('_')[-1])
        matrix=np.fromfile(path,'<f8').reshape(n+1,n+1,order='F')
        if not np.array_equal(matrix@np.ones(n+1),np.ones(n+1)):
            raise ValueError(f'Native filter does not preserve constants: {path}; do not reuse the pre-fix N8 matrix')
        matrices[n]=matrix
    return matrices


def endpoint_dual_locality(original,kind,viscosity,width):
    """Necessary invariant for this numerical H, not exclusion of every H/filter."""
    n=len(original)-1
    _,metric,_=energy_reference(n,kind,viscosity)
    base=modal_filter(original,original,original,kind)
    endpoints=np.eye(len(base))[[0,n]].T
    dual=np.linalg.solve(metric,endpoints)
    scale=np.max(abs(dual),axis=0);dual=dual/scale
    residual=float(np.linalg.norm(metric@dual-endpoints/scale,np.inf))
    inverse_norm=float(np.linalg.norm(np.linalg.inv(metric),np.inf))
    delta=base-np.eye(len(base))
    uncertainty=float(np.linalg.norm(delta,np.inf)*inverse_norm*residual+
        100*np.finfo(float).eps*max(1.,np.linalg.norm(base,np.inf)*np.linalg.norm(dual,np.inf)))
    physical=np.arange(width+1,n-width)
    rows=np.r_[physical,n+physical,2*n-1+physical]
    defect=float(np.max(abs((delta@dual)[rows])))
    return dict(intervals=n,bctype=kind,viscosity=viscosity,near_output_width=width,
        scope='endpoint-dual necessary condition for this numerical H and fixed scalar deep rows only',
        immutable_deep_defect=defect,numerical_uncertainty_estimate=uncertainty,
        linear_solve_residual=residual,metric_condition=float(np.linalg.cond(metric)),
        compatible_with_fixed_deep_rows=bool(defect<=uncertainty),
        caveat='Does not rule out a different viscosity-dependent metric or a noncontractive stable filter')


def baseline_counterexample(original):
    n=len(original)-1
    a=current_normal(n,0.,42)
    filtering=parts(original,(6,6,6),42)[0][1]
    dt=.2/(n*np.sqrt(1.4))
    return dict(intervals=n,bctype=42,viscosity=0.,sound_cfl=.2,
        space='approved P6/T7; not the historical P7 absolute-pressure block',
        spatial_maximum_real=float(np.max(eigvals(a).real)),
        filter_alone_radius=float(np.max(abs(eigvals(filtering)))),
        unfiltered_rk_radius=float(np.max(abs(eigvals(rk3_filtered_map(a,np.eye(len(a)),dt))))),
        filtered_rk=spectral_diagnostic(rk3_filtered_map(a,filtering,dt)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--family',choices=('modal','acoustic','energy-reference','matched-energy-reference',
        'endpoint-dual-locality'),default='modal')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    if args.family in ('energy-reference','matched-energy-reference'):
        for kind in (41,42):
            for n in (8,9):
                references=(None,) if args.family=='energy-reference' else (0.,1e-5,1/1600)
                for index,viscosity in enumerate(references):
                    filtering,metric,report=energy_reference(n,kind,viscosity)
                    name=f'bc{kind}_N{n}'+('' if viscosity is None else f'_nu{index}')
                    np.savez(args.output/f'{name}.npz',filtering=filtering,metric=metric)
                    with (args.output/f'{name}.json').open('x') as stream:
                        json.dump(report,stream,indent=2);stream.write('\n')
                    print(name,report['status'],flush=True)
        return
    matrices=load_native(args.native)
    if args.family=='endpoint-dual-locality':
        for n in (16,32):
            for kind in (41,42):
                for width in (4,7):
                    report=endpoint_dual_locality(matrices[n],kind,1e-5,width)
                    name=f'bc{kind}_N{n}_width{width}'
                    with (args.output/f'{name}.json').open('x') as stream:
                        json.dump(report,stream,indent=2);stream.write('\n')
                    print(name,report['compatible_with_fixed_deep_rows'],report['immutable_deep_defect'],flush=True)
        return
    required=(8,9,10,11,12,16,24,32,64,128)
    missing=tuple(n for n in required if n not in matrices)
    if missing:matrices.update(native_filter_matrices(args.output/'native_missing',missing))
    matrices={n:matrices[n] for n in required}
    baseline=baseline_counterexample(matrices[8])
    with (args.output/'baseline.json').open('x') as stream:
        json.dump(baseline,stream,indent=2);stream.write('\n')
    summaries=[]
    for kind in (41,42,421):
        if args.family=='modal':
            candidates=list(product((6,8),repeat=3))
        else:
            candidates=[(acoustic,entropy,cross) for acoustic,entropy in
                product((.125,.25,.5,.75),(.125,.5,1.)) for cross in
                (-min(acoustic,1-acoustic),0.,min(acoustic,1-acoustic))]
        for index,choices in enumerate(candidates):
            samples,failure=(review(matrices,kind,choices) if args.family=='modal' else
                review_acoustic(matrices,kind,choices))
            label=f'bc{kind}_'+(''.join(map(str,choices)) if args.family=='modal' else f'acoustic_{index:03d}')
            report=dict(status='rejected' if failure else 'sampled spectrum only; not admitted',
                bctype=kind,space='P7' if kind==41 else 'P6/T7',
                family=args.family,parameters=choices,samples=samples,first_failure=failure,
                caveat='No nonlinear or production implementation; a spectral pass alone is not an energy or transient bound')
            with (args.output/f'{label}.json').open('x') as stream:
                json.dump(report,stream,indent=2);stream.write('\n')
            summaries.append(dict(case=label,status=report['status'],first_failure=failure))
            print(label,report['status'],'' if failure is None else
                f'N={failure["intervals"]} nu={failure["viscosity"]} CFL={failure["sound_cfl"]}',flush=True)
    with (args.output/'summary.json').open('x') as stream:
        json.dump(dict(scope='finite modal/acoustic design families on current space blocks; not an exhaustive design search',
            family=args.family,parameter_definition=('pressure/entropy/normal-velocity map order' if args.family=='modal' else
                'near-wall acoustic/entropy/cross strengths; tangential components keep the original map'),
            original_native=str(args.native),intervals=required,variants=summaries),stream,indent=2)
        stream.write('\n')


if __name__=='__main__':main()
