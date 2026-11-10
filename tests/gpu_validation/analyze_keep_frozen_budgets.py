"""Read-only periodic five-variable balances from native q, primitive and RHS snapshots.

No time integration or correction is performed. All gas/transport parameters are
explicit inputs and must match the archived case. Derivative symbols are those
of the periodic D6/global Pade operators, not physical-boundary closures.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from compare_q_validation_snapshots import read_q_snapshot
from run_keep_metric_audit import periodic_derivative

def read_frozen(directory,stage):
    suffix=f'.step00000000.rk{stage:02d}.rank00000000.bin'
    snapshot=read_q_snapshot(directory/'outdat'/('rhs.pre_rhs'+suffix))
    im,jm,km,hm,nq=snapshot.header
    if nq!=5 or min(im,jm,km)<=0:
        raise ValueError('budget reference requires three active periodic directions and five components')
    shape=(im+2*hm+1,jm+2*hm+1,km+2*hm+1)
    unique=(slice(hm,hm+im),slice(hm,hm+jm),slice(hm,hm+km))
    q=snapshot.values.reshape((*shape,nq),order='F')[unique]
    with (directory/'outdat'/('rhs.pre_rhs_primitives'+suffix)).open('rb') as stream:
        header=np.fromfile(stream,dtype=np.int32,count=4)
        if tuple(header)!=(im,jm,km,hm):
            raise ValueError('primitive snapshot dimensions differ')
        primitive=np.fromfile(stream,dtype=np.float64).reshape((*shape,6),order='F')[unique]
    rhs=[]
    for label in ['conv','full']:
        with (directory/'outdat'/('rhs.'+label+suffix)).open('rb') as stream:
            header=np.fromfile(stream,dtype=np.int32,count=4)
            if tuple(header)!=(im,jm,km,nq):
                raise ValueError('RHS snapshot dimensions differ')
            values=np.fromfile(stream,dtype=np.float64).reshape((im+1,jm+1,km+1,nq),order='F')
            rhs.append(values[:-1,:-1,:-1])
    with h5py.File(directory/'outdat/new/resources/geometry.h5') as state:
        if int(state['numerical_kind_bytes'][0])!=8:
            raise ValueError('budget audit requires FP64 source artifacts')
        fields=[state[f'q{c:04d}'][:-1,:-1,:-1].transpose(2,1,0) for c in range(4,14)]
    jac=fields[0]
    inverse=np.stack([fields[1+a+3*b] for a in range(3) for b in range(3)],axis=-1).reshape(jac.shape+(3,3))
    if jac.shape!=q.shape[:3] or not (jac>0).all():
        raise ValueError('geometry ownership or positive-Jacobian condition differs')
    if not all(np.isfinite(value).all() for value in [q,primitive,inverse,jac,*rhs]):
        raise ValueError('nonfinite native budget input')
    return q,primitive,jac,inverse,rhs[0],rhs[1]-rhs[0]


def averaged_product_divergence(factors,axis,scheme):
    """Expand the four-factor symmetric KEEP internal-energy flux only."""
    result=np.zeros_like(factors[0])
    for mask in range(1,2**len(factors)):
        left=np.ones_like(result);right=np.ones_like(result)
        for bit,values in enumerate(factors):
            if mask&(1<<bit):
                right*=values
            else:
                left*=values
        result+=left*periodic_derivative(right,axis,scheme)/2**(len(factors)-1)
    return result


def analyze(directory,args):
    q,primitive,jac,inverse,convection,viscous=read_frozen(directory,args.stage)
    area=jac[...,None,None]*inverse
    measure=float(jac.sum())
    mean=lambda values:float(np.sum(jac*values)/measure)
    integrated=lambda values:np.sum(values,axis=(0,1,2))/measure
    rho=q[...,0];u=q[...,1:4]/rho[...,None]
    internal=q[...,4]-0.5*np.sum(q[...,1:4]*u,axis=-1)
    pressure=(args.gamma-1)*internal
    if not (rho>0).all() or not (pressure>0).all() or not (primitive[...,5]>0).all():
        raise ValueError('positive density, pressure and temperature are required')
    const2=args.gamma*args.mach**2
    expected_primitive=np.concatenate([rho[...,None],u,pressure[...,None],(const2*pressure/rho)[...,None]],axis=-1)
    primitive_error=float(np.max(abs(primitive-expected_primitive)/
        np.maximum(1,np.maximum(abs(primitive),abs(expected_primitive)))))
    if primitive_error>2e-10:
        raise ValueError(f'primitive cache, q or declared perfect-gas normalization differ: {directory}: {primitive_error}')
    entropy_specific=np.log(pressure/rho**args.gamma)/(args.gamma-1)
    velocity2=np.sum(u*u,axis=-1)

    def rates(mapped):
        physical=mapped/jac[...,None]
        kinetic=np.sum(u*physical[...,1:4],axis=-1)-0.5*velocity2*physical[...,0]
        energy_internal=physical[...,4]-kinetic
        p_rate=(args.gamma-1)*energy_internal
        entropy=(entropy_specific-args.gamma/(args.gamma-1))*physical[...,0] \
            +rho*p_rate/((args.gamma-1)*pressure)
        return physical,kinetic,energy_internal,entropy

    rc,kc,ic,sc=rates(convection)
    rv,kv,iv,sv=rates(viscous)
    pressure_force=np.zeros_like(u)
    split_dilatation=np.zeros_like(rho)
    internal_advection=np.zeros_like(rho)
    entropy_flux_divergence=np.zeros_like(rho)
    e_specific=internal/rho
    for a in range(3):
        derivative=lambda values:periodic_derivative(values,a,args.convection)
        pressure_force-=0.5*(derivative(pressure[...,None]*area[...,a,:])
            +pressure[...,None]*derivative(area[...,a,:])+area[...,a,:]*derivative(pressure)[...,None])/jac[...,None]
        for b in range(3):
            metric=area[...,a,b]
            split_dilatation+=0.5*(metric*derivative(u[...,b])+derivative(metric*u[...,b])
                -u[...,b]*derivative(metric))/jac
            internal_advection-=averaged_product_divergence([rho,u[...,b],metric,e_specific],a,args.convection)/jac
        entropy_flux_divergence+=derivative(np.sum(area[...,a,:]*u,axis=-1)*rho*entropy_specific)/jac
    pressure_work=mean(np.sum(u*pressure_force,axis=-1))
    pressure_dilatation=mean(pressure*split_dilatation)
    expected_internal=internal_advection-pressure*split_dilatation
    internal_error=float(np.max(abs(ic-expected_internal)/np.maximum(1,np.maximum(abs(ic),abs(expected_internal)))))
    conservative_rates=integrated(convection)
    kinetic_exchange=mean(kc)-pressure_work
    internal_exchange=mean(ic)+pressure_dilatation
    pressure_adjoint=pressure_work-pressure_dilatation
    if np.max(abs(conservative_rates))>2e-10 or \
       max(abs(kinetic_exchange),abs(internal_exchange),abs(pressure_adjoint),internal_error)>2e-10:
        raise AssertionError(f'KEEP frozen conservation/pressure/internal-energy identity failed: {directory}')

    def gradient(values):
        return sum(inverse[...,a,:]*periodic_derivative(values,a,args.nodal_reference)[...,None] for a in range(3))

    cached_u=primitive[...,1:4];temperature=primitive[...,5]
    grad_u=np.stack([gradient(cached_u[...,b]) for b in range(3)],axis=-2)
    trace=np.trace(grad_u,axis1=-2,axis2=-1)
    s=args.sutherland_k/args.ref_temperature
    mu=temperature*np.sqrt(temperature)*(1+s)/(temperature+s)/args.reynolds
    kappa=(mu/args.prandtl)/((args.gamma-1)*args.mach**2)
    stress=mu[...,None,None]*(grad_u+np.swapaxes(grad_u,-1,-2)-(2/3)*trace[...,None,None]*np.eye(3))
    grad_t=gradient(temperature)
    hflux=kappa[...,None]*grad_t+np.einsum('...i,...ib->...b',cached_u,stress)
    dissipation=np.sum(stress*grad_u,axis=(-2,-1))
    temperature_q=const2*pressure/rho
    entropy_sbp=const2*mean(-np.sum(gradient(1/temperature_q)*hflux,axis=-1)
        +sum(np.sum(gradient(u[...,b]/temperature_q)*stress[...,b,:],axis=-1) for b in range(3)))
    entropy_continuum=const2*mean(dissipation/temperature_q+kappa*np.sum(grad_t*grad_t,axis=-1)/temperature_q**2)
    viscous_conservative=integrated(viscous)
    viscous_ke_balance=mean(kv)+mean(dissipation)
    entropy_sbp_error=mean(sv)-entropy_sbp
    if args.viscous_form=='standard' and (np.max(abs(viscous_conservative))>2e-10 or
        max(abs(viscous_ke_balance),abs(entropy_sbp_error))>2e-10):
        raise AssertionError(f'standard nodal viscous discrete work/entropy SBP identity failed: {directory}')

    return dict(source=str(directory),phase=dict(internal_step=0,rk_stage=args.stage),
        primitive_current_q_max_scaled=primitive_error,
        measure=dict(rule='sum J over unique periodic nodes; mapped RHS divided by sum J',volume=measure,
                     nodes=list(jac.shape)),
        convection=dict(conservative_mean_rates=conservative_rates.tolist(),kinetic_mean_rate=mean(kc),
            internal_mean_rate=mean(ic),pressure_work_mean=pressure_work,split_pressure_dilatation_mean=pressure_dilatation,
            kinetic_exchange_residual=kinetic_exchange,internal_exchange_residual=internal_exchange,
            pressure_adjoint_residual=pressure_adjoint,internal_product_max_scaled=internal_error,
            entropy_mean_rate=mean(sc),entropy_chain_flux_residual_max=float(np.max(abs(sc+entropy_flux_divergence))),
            entropy_note='arithmetic KEEP is not required to have zero chain-rule entropy rate or residual'),
        viscosity=dict(conservative_mean_rates=viscous_conservative.tolist(),kinetic_mean_rate=mean(kv),
            internal_mean_rate=mean(iv),entropy_mean_rate=mean(sv),nodal_stress_dissipation_mean=mean(dissipation),
            nodal_kinetic_work_residual=viscous_ke_balance,entropy_nodal_sbp_reference=entropy_sbp,
            entropy_nodal_sbp_residual=entropy_sbp_error,entropy_continuum_production_reference=entropy_continuum,
            reference_policy='exact nodal SBP identities checked' if args.viscous_form=='standard' else
                'nodal references are diagnostic only for half-point ME6; nonzero variable-property totals are recorded, not corrected'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--convection',choices=['643e','643c'],required=True)
    parser.add_argument('--nodal-reference',choices=['643e','643c'],required=True,
        help='Actual nodal gradient for standard viscosity; diagnostic reference only for half-point ME6')
    parser.add_argument('--viscous-form',choices=['standard','me6_opti'],required=True)
    parser.add_argument('--stage',type=int,choices=[1,2,3],default=1)
    for name in ['gamma','mach','reynolds','prandtl','ref-temperature','sutherland-k']:
        parser.add_argument('--'+name,type=float,required=True)
    args=parser.parse_args()
    parameters=[args.gamma,args.mach,args.reynolds,args.prandtl,args.ref_temperature,args.sutherland_k]
    if not np.isfinite(parameters).all() or min(parameters)<=0 or args.gamma<=1:
        raise ValueError('finite positive gas/transport parameters and gamma > 1 are required')
    report=dict(parameters={name:value for name,value in vars(args).items() if name not in ['cases','output']},
                cases=[analyze(path.resolve(),args) for path in args.cases],status='passed_declared_identities')
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    for item in report['cases']:
        print(item['source'],json.dumps(dict(convection=item['convection'],viscosity=item['viscosity'])),flush=True)


if __name__=='__main__':
    main()
