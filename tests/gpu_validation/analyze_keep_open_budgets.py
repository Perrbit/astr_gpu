"""Read native KEEP open-boundary RK/filter changes without evolving or repairing a field."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from analyze_keep_rk_budgets import RK,SOURCE_WEIGHTS,STATE_WEIGHTS,DEFECT_WEIGHTS,NAMES
from compare_q_validation_snapshots import read_q_snapshot


def analyze(case,dt,filtered):
    def state(label,stage):
        path=case/'outdat'/f'rhs.{label}.step00000000.rk{stage:02d}.rank00000000.bin'
        snapshot=read_q_snapshot(path)
        im,jm,km,hm,nq=snapshot.header
        if nq!=5:
            raise ValueError('Open-boundary budget requires five conservative components')
        shape=(im+2*hm+1,jm+2*hm+1,km+2*hm+1,nq)
        return snapshot.values.reshape(shape,order='F')[hm:hm+im+1,hm:hm+jm+1,hm:hm+km,:]

    pre=[state('pre_rhs',s) for s in (1,2,3)]
    post=[state('post_update',s) for s in (1,2,3)]
    initial=state('initialized',1)
    with h5py.File(case/'outdat/new/resources/geometry.h5') as geometry:
        jac=geometry['q0004'][:].transpose(2,1,0)[:,:,:-1]
        metric=np.empty((*jac.shape,3,3))
        for axis in range(3):
            for component in range(3):
                metric[...,axis,component]=geometry[f'q{5+axis+3*component:04d}'][:].transpose(2,1,0)[:,:,:-1]
    if jac.shape!=pre[0].shape[:3] or not np.isfinite(jac).all() or not (jac>0).all():
        raise ValueError('Open-boundary Jacobian or physical-node layout differs')
    weights=np.ones(jac.shape)
    weights[[0,-1],:,:]*=.5;weights[:,[0,-1],:]*=.5
    volume=float(np.sum(weights*jac))
    mean=lambda values:np.sum((weights*jac)[...,None]*values,axis=(0,1,2))/volume
    integrate_rhs=lambda values:np.sum(weights[...,None]*values,axis=(0,1,2))/volume

    def thermo(q):
        rho=q[...,0];u=q[...,1:4]/rho[...,None]
        kinetic=.5*np.sum(q[...,1:4]*u,axis=-1)
        internal=q[...,4]-kinetic;p=.4*internal
        if not np.isfinite(q).all() or not (rho>0).all() or not (p>0).all():
            raise ValueError('Budget input contains nonfinite or nonpositive physical states')
        entropy=(np.log(p)-1.4*np.log(rho))/.4
        return rho,u,kinetic,internal,p,entropy

    def quantities(q):
        rho,u,k,e,p,s=thermo(q)
        return mean(np.concatenate([q,k[...,None],e[...,None],(rho*s)[...,None]],axis=-1))

    def rates(q,mapped):
        rho,u,k,e,p,s=thermo(q);physical=mapped/jac[...,None]
        kr=np.sum(u*physical[...,1:4],axis=-1)-.5*np.sum(u*u,axis=-1)*physical[...,0]
        er=physical[...,4]-kr;sr=(s-1.4/.4)*physical[...,0]+rho*er/p
        return mean(np.concatenate([physical,kr[...,None],er[...,None],sr[...,None]],axis=-1))

    def rhs(label,stage):
        path=case/'outdat'/f'rhs.{label}.step00000000.rk{stage:02d}.rank00000000.bin'
        with path.open('rb') as stream:
            im,jm,km,nq=np.fromfile(stream,np.int32,4)
            values=np.fromfile(stream,np.float64).reshape((im+1,jm+1,km+1,nq),order='F')[:,:,:-1,:]
        if values.shape!=pre[0].shape or not np.isfinite(values).all():
            raise ValueError('Native RHS differs from the physical-node layout')
        return values

    def boundary_reference(q):
        rho,u,k,e,p,s=thermo(q)
        flux=[]
        for axis in (0,1):
            normal=np.sum(metric[...,axis,:]*u,axis=-1)
            mapped=q*normal[...,None]
            mapped[...,1:4]+=p[...,None]*metric[...,axis,:]
            mapped[...,4]+=p*normal
            mapped*=jac[...,None]
            face_weights=np.ones(jac.shape[1:] if axis==0 else (jac.shape[0],jac.shape[2]))
            face_weights[[0,-1],:]*=.5
            net=np.sum(face_weights[...,None]*(np.take(mapped,-1,axis=axis)-np.take(mapped,0,axis=axis)),axis=(0,1))
            flux.append(net)
        return dict(x_outward_flux=flux[0].tolist(),y_outward_flux=flux[1].tolist(),
                    normalized_negative_outward_flux=(-(flux[0]+flux[1])/volume).tolist())

    before=[state('pre_filter',s) for s in (1,2,3)] if filtered else [initial,post[0],post[1]]
    after=[state('post_filter',s) for s in (1,2,3)] if filtered else before
    filtering=[];preparation=[];convection=[];remaining=[];defects=[];stages=[]
    base=pre[0]
    for i,(a,b,c) in enumerate(RK):
        cr=rhs('conv',i+1);fr=rhs('full',i+1)
        expected=a*base+b*pre[i]+c*dt*fr/jac[...,None]
        error=float(np.max(abs(expected-post[i])/np.maximum(1,np.maximum(abs(expected),abs(post[i])))))
        chain=float(np.max(abs(before[i]-(initial if i==0 else post[i-1]))))
        if max(error,chain)>2e-10:
            raise AssertionError(f'Native open RK/filter recurrence differs at stage {i+1}: {error,chain}')
        f=quantities(after[i])-quantities(before[i]);p=quantities(pre[i])-quantities(after[i])
        rc=rates(pre[i],cr);rr=rates(pre[i],fr-cr)
        defect=quantities(post[i])-a*quantities(base)-b*quantities(pre[i])-c*dt*(rc+rr)
        filtering.append(f);preparation.append(p);convection.append(rc);remaining.append(rr);defects.append(defect)
        boundary=boundary_reference(pre[i])
        boundary['convective_mean_rate']=integrate_rhs(cr).tolist()
        boundary['volume_minus_surface_residual']=(integrate_rhs(cr)-np.array(boundary['normalized_negative_outward_flux'])).tolist()
        stages.append(dict(stage=i+1,recurrence_max_scaled=error,filter_input_chain_max_abs=chain,
            physical_surface_reference=boundary,preparation_change=p.tolist(),filter_change=f.tolist(),
            convective_rates=rc.tolist(),remaining_rhs_rates=rr.tolist(),rk_nonlinear_defect=defect.tolist()))
    terms=dict(convection=dt*SOURCE_WEIGHTS@np.array(convection),
               remaining_rhs=dt*SOURCE_WEIGHTS@np.array(remaining),
               filtering=STATE_WEIGHTS@np.array(filtering),
               boundary_preparation=STATE_WEIGHTS@np.array(preparation),
               rk_nonlinear_defect=DEFECT_WEIGHTS@np.array(defects))
    change=quantities(post[-1])-quantities(initial);residual=change-sum(terms.values())
    if np.max(abs(residual))>2e-10 or np.max(abs(np.array(defects)[:,:5]))>2e-10:
        raise AssertionError('Native open RK conservative recurrence or bookkeeping does not close')
    return dict(case=str(case),quantities=NAMES,dt=dt,filtered=filtered,volume=volume,
        measure='recorded J; trapezoidal physical x/y endpoints, unique periodic z; not an SBP norm',
        initial=quantities(initial).tolist(),final=quantities(post[-1]).tolist(),change=change.tolist(),
        terms={key:value.tolist() for key,value in terms.items()},closure_residual=residual.tolist(),stages=stages,
        interpretation='preparation includes boundary state/filter/halo reconciliation; full-minus-conv is reported separately; volume/surface residual is measured, not forced to zero or interpreted as an energy-stability proof')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu',type=Path,required=True)
    parser.add_argument('--gpu',type=Path,required=True)
    parser.add_argument('--dt',type=float,required=True)
    parser.add_argument('--filtered',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not np.isfinite(args.dt) or args.dt<=0:
        raise ValueError('dt must be positive and match the archived run')
    a,b=[analyze(case.resolve(),args.dt,args.filtered) for case in (args.cpu,args.gpu)]
    differences={name:float(np.max(abs(np.array(a['terms'][name])-np.array(b['terms'][name]))/
        np.maximum(1,np.maximum(abs(np.array(a['terms'][name])),abs(np.array(b['terms'][name])))))) for name in a['terms']}
    if max(differences.values())>2e-10:
        raise AssertionError(f'CPU/GPU open RK terms differ: {differences}')
    with args.output.open('x') as stream:
        json.dump(dict(status='passed_native_recurrence_and_backend_terms',cases=[a,b],term_max_scaled=differences),stream,indent=2)
        stream.write('\n')
    print('maximum term difference:',max(differences.values()),flush=True)


if __name__=='__main__':
    main()
