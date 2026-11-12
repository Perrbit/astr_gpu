"""Read-only complete RK3/filter balance from native periodic FP64 snapshots.

The nonlinear RK defect is measured, not assumed zero. No PDE is advanced and no
state is corrected. The declared dt must match the archived run.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from compare_q_validation_snapshots import read_q_snapshot

NAMES=['rho','rho_u','rho_v','rho_w','total_energy','kinetic_energy','internal_energy','entropy_density']
RK=((1.,0.,1.),(.75,.25,.25),(1/3,2/3,2/3))
SOURCE_WEIGHTS=np.array([1/6,1/6,2/3])
STATE_WEIGHTS=np.array([1.,1/6,2/3])
DEFECT_WEIGHTS=np.array([1/6,2/3,1.])


def analyze(case,dt,filtered):
    def state(label,stage):
        path=case/'outdat'/f'rhs.{label}.step00000000.rk{stage:02d}.rank00000000.bin'
        snap=read_q_snapshot(path)
        im,jm,km,hm,nq=snap.header
        if nq!=5:
            raise ValueError('RK balance requires the nonreacting five-variable state')
        shape=(im+2*hm+1,jm+2*hm+1,km+2*hm+1,nq)
        return snap.values.reshape(shape,order='F')[hm:hm+im,hm:hm+jm,hm:hm+km,:]

    pre=[state('pre_rhs',stage) for stage in (1,2,3)]
    post=[state('post_update',stage) for stage in (1,2,3)]
    with h5py.File(case/'outdat/new/resources/geometry.h5') as geometry:
        jac=geometry['q0004'][:-1,:-1,:-1].transpose(2,1,0)
    if jac.shape!=pre[0].shape[:3] or not np.isfinite(jac).all() or not (jac>0).all():
        raise ValueError('RK balance geometry differs from the periodic unique-node layout')
    volume=float(jac.sum())
    mean=lambda values:np.sum(jac[...,None]*values,axis=(0,1,2))/volume

    def thermodynamic(q):
        if not np.isfinite(q).all() or not (q[...,0]>0).all():
            raise ValueError('nonfinite state or nonpositive density in RK budget')
        rho=q[...,0];u=q[...,1:4]/rho[...,None]
        kinetic=.5*np.sum(q[...,1:4]*u,axis=-1)
        internal=q[...,4]-kinetic
        pressure=.4*internal
        if not (pressure>0).all():
            raise ValueError('nonpositive pressure in RK budget')
        entropy=(np.log(pressure)-1.4*np.log(rho))/.4
        return rho,u,kinetic,internal,pressure,entropy

    def quantities(q):
        rho,u,kinetic,internal,p,entropy=thermodynamic(q)
        return mean(np.concatenate([q,kinetic[...,None],internal[...,None],(rho*entropy)[...,None]],axis=-1))

    def rates(q,mapped):
        rho,u,kinetic,internal,p,entropy=thermodynamic(q)
        physical=mapped/jac[...,None]
        krate=np.sum(u*physical[...,1:4],axis=-1)-.5*np.sum(u*u,axis=-1)*physical[...,0]
        irate=physical[...,4]-krate
        srate=(entropy-1.4/.4)*physical[...,0]+rho*irate/p
        return mean(np.concatenate([physical,krate[...,None],irate[...,None],srate[...,None]],axis=-1))

    def rhs(label,stage):
        path=case/'outdat'/f'rhs.{label}.step00000000.rk{stage:02d}.rank00000000.bin'
        with path.open('rb') as stream:
            header=np.fromfile(stream,np.int32,4)
            values=np.fromfile(stream,np.float64)
        im,jm,km,nq=map(int,header)
        result=values.reshape((im+1,jm+1,km+1,nq),order='F')[:-1,:-1,:-1]
        if result.shape!=pre[0].shape or not np.isfinite(result).all():
            raise ValueError('RK balance RHS layout differs or contains nonfinite values')
        return result

    def scaled(a,b):
        return float(np.max(abs(a-b)/np.maximum(1,np.maximum(abs(a),abs(b)))))

    before_filter=[state('pre_filter',s) for s in (1,2,3)] if filtered else [pre[0],post[0],post[1]]
    after_filter=[state('post_filter',s) for s in (1,2,3)] if filtered else before_filter
    initial=before_filter[0];base=pre[0]
    filter_delta=[];preparation_delta=[];conv=[];visc=[];defects=[];stages=[]
    for index,(a,b,c) in enumerate(RK):
        stage=index+1
        conv_rhs=rhs('conv',stage);full_rhs=rhs('full',stage)
        expected=(a*base*jac[...,None]+b*pre[index]*jac[...,None]+c*dt*full_rhs)/jac[...,None]
        recurrence=scaled(expected,post[index])
        chain=0. if index==0 else scaled(before_filter[index],post[index-1])
        preparation=scaled(pre[index],after_filter[index])
        if max(recurrence,chain,preparation)>2e-10:
            raise AssertionError(f'RK/filter state transition mismatch: {case}: stage={stage}, {recurrence,chain,preparation}')
        f=quantities(after_filter[index])-quantities(before_filter[index])
        p=quantities(pre[index])-quantities(after_filter[index])
        rc=rates(pre[index],conv_rhs);rv=rates(pre[index],full_rhs-conv_rhs)
        defect=quantities(post[index])-a*quantities(base)-b*quantities(pre[index])-c*dt*(rc+rv)
        filter_delta.append(f);preparation_delta.append(p);conv.append(rc);visc.append(rv);defects.append(defect)
        stages.append(dict(stage=stage,recurrence_max_scaled=recurrence,
                           previous_update_to_filter_max_scaled=chain,periodic_preparation_max_scaled=preparation,
                           filter_change=f.tolist(),preparation_change=p.tolist(),
                           convective_rates=rc.tolist(),viscous_rates=rv.tolist(),rk_nonlinear_defect=defect.tolist()))
    terms=dict(convection=dt*SOURCE_WEIGHTS@np.array(conv),viscosity=dt*SOURCE_WEIGHTS@np.array(visc),
               filtering=STATE_WEIGHTS@np.array(filter_delta),preparation=STATE_WEIGHTS@np.array(preparation_delta),
               rk_nonlinear_defect=DEFECT_WEIGHTS@np.array(defects))
    change=quantities(post[-1])-quantities(initial)
    residual=change-sum(terms.values())
    if np.max(abs(residual))>2e-10:
        raise AssertionError(f'Complete RK3 budget does not close: {case}: {residual}')
    if max(np.max(abs(d[:5])) for d in defects)>2e-10:
        raise AssertionError('Conservative variables have a non-roundoff RK recurrence defect')
    return dict(case=str(case),quantities=NAMES,dt=dt,filter=filtered,
                rule='J-weighted means over unique periodic nodes; gamma=1.4',volume=volume,
                initial=quantities(initial).tolist(),final=quantities(post[-1]).tolist(),change=change.tolist(),
                terms={name:value.tolist() for name,value in terms.items()},closure_residual=residual.tolist(),
                stages=stages,interpretation='full-minus-conv is viscosity only for the declared source-free TGV; nonlinear RK and filter effects are measured, not forced to zero')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',type=Path,nargs='+',required=True)
    parser.add_argument('--dt',type=float,required=True)
    parser.add_argument('--filtered',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not np.isfinite(args.dt) or args.dt<=0:
        raise ValueError('dt must be finite and positive')
    report=dict(status='passed_transition_and_balance_identities',
                scope='source-free periodic FP64 TGV, one archived complete RK3 step, no physical-boundary stability claim',
                cases=[analyze(case.resolve(),args.dt,args.filtered) for case in args.cases])
    if len(report['cases'])==2:
        a,b=report['cases']
        comparison={name:float(np.max(abs(np.array(a['terms'][name])-np.array(b['terms'][name]))/
                    np.maximum(1,np.maximum(abs(np.array(a['terms'][name])),abs(np.array(b['terms'][name]))))))
                    for name in a['terms']}
        if max(comparison.values())>2e-10:
            raise AssertionError(f'CPU/GPU RK contributions differ: {comparison}')
        report['two_case_term_max_scaled']=comparison
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    for item in report['cases']:
        print(item['case'],'closure',max(map(abs,item['closure_residual'])),flush=True)


if __name__=='__main__':
    main()
