"""Frozen FP64 standard-viscosity audit on archived KEEP 11/22/41/52 states.

Uses production-extracted physical-end derivative matrices and independent
periodic symbols. No PDE advancement, correction, or SBP claim is made.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from compare_q_validation_snapshots import read_q_snapshot
from run_keep_metric_audit import periodic_derivative


def scaled(a,b):
    return float(np.max(abs(a-b)/np.maximum(1,np.maximum(abs(a),abs(b)))))


def analyze(case,matrix_file,scheme,stage,gamma,mach,reynolds,prandtl,ref_temperature,sutherland_k):
    suffix=f'.step00000000.rk{stage:02d}.rank00000000.bin'
    snapshot=read_q_snapshot(case/'outdat'/('rhs.pre_rhs'+suffix))
    im,jm,km,hm,nq=snapshot.header
    if nq!=5 or im!=jm:
        raise ValueError('This archive audit requires five variables and the same x/y interval count')
    physical=(slice(hm,hm+im+1),slice(hm,hm+jm+1),slice(hm,hm+km))
    shape=(im+2*hm+1,jm+2*hm+1,km+2*hm+1)
    q=snapshot.values.reshape((*shape,nq),order='F')[physical]
    with (case/'outdat'/('rhs.pre_rhs_primitives'+suffix)).open('rb') as stream:
        header=tuple(np.fromfile(stream,np.int32,4))
        if header!=(im,jm,km,hm):
            raise ValueError('Primitive header differs from q')
        primitive=np.fromfile(stream,np.float64).reshape((*shape,6),order='F')[physical]
    with h5py.File(case/'outdat/new/resources/geometry.h5') as geometry:
        jac=geometry['q0004'][:].transpose(2,1,0)[:,:,:-1]
        inverse=np.empty((*jac.shape,3,3))
        for axis in range(3):
            for component in range(3):
                inverse[...,axis,component]=geometry[f'q{5+axis+3*component:04d}'][:].transpose(2,1,0)[:,:,:-1]
    if jac.shape!=q.shape[:3] or not np.isfinite(jac).all() or not (jac>0).all():
        raise ValueError('Physical-node geometry or positive Jacobian differs')
    if not np.isfinite(q).all() or not np.isfinite(primitive).all():
        raise ValueError('Nonfinite physical state or primitive cache')
    rho=q[...,0];u=q[...,1:4]/rho[...,None]
    internal=q[...,4]-.5*np.sum(q[...,1:4]*u,axis=-1)
    p=(gamma-1)*internal;temperature=gamma*mach**2*p/rho
    expected=np.concatenate([rho[...,None],u,p[...,None],temperature[...,None]],axis=-1)
    if min(float(rho.min()),float(p.min()),float(temperature.min()))<=0 or scaled(expected,primitive)>2e-10:
        raise AssertionError('Primitive cache and declared ideal-gas normalization differ')

    rows=np.loadtxt(matrix_file)
    derivative=np.zeros((im+1,im+1))
    derivative[rows[:,0].astype(int),rows[:,1].astype(int)]=rows[:,2 if scheme=='643e' else 3]
    if rows.shape[0]!=(im+1)**2:
        raise ValueError('Production derivative matrix resolution differs')

    def computational(values,axis):
        if axis==2:
            return periodic_derivative(values,axis,scheme)
        return np.moveaxis(np.tensordot(derivative,values,axes=(1,axis)),0,axis)

    def gradient(values):
        return sum(inverse[...,axis,:]*computational(values,axis)[...,None] for axis in range(3))

    gu=np.stack([gradient(primitive[...,1+c]) for c in range(3)],axis=-2)
    gt=gradient(primitive[...,5])
    t=primitive[...,5];vel=primitive[...,1:4]
    sutherland=sutherland_k/ref_temperature
    mu=t*np.sqrt(t)*(1+sutherland)/(t+sutherland)/reynolds
    kappa=mu/prandtl/((gamma-1)*mach**2)
    stress=mu[...,None,None]*(gu+np.swapaxes(gu,-1,-2)-(2/3)*np.trace(gu,axis1=-2,axis2=-1)[...,None,None]*np.eye(3))
    conduction=kappa[...,None]*gt
    work=np.einsum('...a,...ab->...b',vel,stress)
    energy=work+conduction
    dissipation=np.sum(stress*gu,axis=(-2,-1))
    entropy_production=gamma*mach**2*(dissipation/t+kappa*np.sum(gt*gt,axis=-1)/t**2)

    weights=np.ones(jac.shape)
    weights[[0,-1],:,:]*=.5;weights[:,[0,-1],:]*=.5
    volume=float(np.sum(weights*jac))
    integral=lambda values:np.sum(weights[...,None]*values,axis=(0,1,2))/volume
    mean=lambda values:float(np.sum(weights*jac*values)/volume)
    reconstructed=np.zeros_like(q);raw=np.zeros_like(q)
    mask_change=np.zeros_like(q)
    face_records={};surface=np.zeros(5);heat_surface=0.;work_surface=0.;entropy_surface=0.
    for axis in range(3):
        area=jac[...,None]*inverse[...,axis,:]
        mapped=np.zeros_like(q)
        mapped[...,1:4]=np.einsum('...ab,...b->...a',stress,area)
        mapped[...,4]=np.sum(energy*area,axis=-1)
        direction=computational(mapped,axis);raw+=direction
        submitted=direction.copy()
        # Match the existing per-direction accumulation ranges, not a PDE boundary model.
        if axis in (0,1):
            index=[slice(None)]*4;index[axis]=[0,-1];submitted[tuple(index)]=0
        reconstructed+=submitted;mask_change+=submitted-direction
        if axis==2:
            continue
        fw=np.ones(jac.shape[1:] if axis==0 else (jac.shape[0],jac.shape[2]))
        fw[[0,-1],:]*=.5
        for index,sign,name in ((0,-1.,'minus'),(-1,1.,'plus')):
            take=lambda values:np.take(values,index,axis=axis)
            flux=sign*np.sum(fw[...,None]*take(mapped),axis=(0,1))
            conductive=sign*float(np.sum(fw*take(np.sum(conduction*area,axis=-1))))
            viscous_work=sign*float(np.sum(fw*take(np.sum(work*area,axis=-1))))
            entropy=sign*float(np.sum(fw*take(gamma*mach**2*np.sum(conduction*area,axis=-1)/t)))
            face_records['xy'[axis]+'_'+name]=dict(viscous_outward_rhs_flux=flux.tolist(),
                outward_physical_heat_flux=-conductive,conductive_energy_input=conductive,
                viscous_work_input=viscous_work,entropy_diffusion_input=entropy)
            surface+=flux;heat_surface+=conductive;work_surface+=viscous_work;entropy_surface+=entropy
    native=[]
    for label in ('conv','full'):
        with (case/'outdat'/('rhs.'+label+suffix)).open('rb') as stream:
            header=tuple(np.fromfile(stream,np.int32,4))
            if header!=(im,jm,km,5):
                raise ValueError('RHS header differs from q')
            native.append(np.fromfile(stream,np.float64).reshape((im+1,jm+1,km+1,5),order='F')[:,:,:-1,:])
    observed=native[1]-native[0]
    error=scaled(reconstructed,observed)
    if not np.isfinite(observed).all() or error>2e-10:
        raise AssertionError(f'Frozen standard-viscous RHS differs from native full-minus-conv: {error}')
    rate=integral(observed);unmasked=integral(raw);omitted=integral(mask_change)
    spatial_defect=unmasked-surface/volume
    closure=rate-surface/volume-spatial_defect-omitted
    if np.max(abs(closure))>2e-10:
        raise AssertionError('Viscous surface/stencil/mask decomposition does not close')
    physical=observed/jac[...,None]
    kinetic=np.sum(vel*physical[...,1:4],axis=-1)
    internal_rate=physical[...,4]-kinetic
    entropy_rate=gamma*mach**2*internal_rate/t
    raw_kinetic=np.sum(vel*raw[...,1:4],axis=-1)/jac
    omitted_kinetic=np.sum(vel*mask_change[...,1:4],axis=-1)/jac
    raw_internal=raw[...,4]/jac-raw_kinetic
    omitted_internal=mask_change[...,4]/jac-omitted_kinetic
    balance_parts=dict(
        kinetic=dict(raw_operator_residual=mean(raw_kinetic)-work_surface/volume+mean(dissipation),
                     inactive_normal_row_contribution=mean(omitted_kinetic)),
        internal=dict(raw_operator_residual=mean(raw_internal)-heat_surface/volume-mean(dissipation),
                      inactive_normal_row_contribution=mean(omitted_internal)),
        entropy=dict(raw_operator_residual=mean(gamma*mach**2*raw_internal/t)-entropy_surface/volume-mean(entropy_production),
                     inactive_normal_row_contribution=mean(gamma*mach**2*omitted_internal/t)))
    return dict(case=str(case),stage=stage,viscosity=scheme,
        normalization=dict(gamma=gamma,mach=mach,reynolds=reynolds,prandtl=prandtl,
                           reference_temperature_kelvin=ref_temperature,sutherland_kelvin=sutherland_k),
        derivative_matrix=str(matrix_file),primitive_q_max_scaled=scaled(expected,primitive),
        native_viscous_rhs_max_scaled=error,volume=volume,
        measure='J; trapezoidal physical x/y endpoints and unique periodic z; not an SBP norm',
        conservative_mean_rate=rate.tolist(),surface_mean_rate=(surface/volume).tolist(),
        stencil_volume_surface_defect=spatial_defect.tolist(),inactive_normal_rows=omitted.tolist(),
        decomposition_residual=closure.tolist(),physical_faces=face_records,
        nonlinear_residual_parts=balance_parts,
        kinetic=dict(rate=mean(kinetic),surface_work_input=work_surface/volume,
                     stress_dissipation=mean(dissipation),
                     work_identity_residual=mean(kinetic)-work_surface/volume+mean(dissipation)),
        internal=dict(rate=mean(internal_rate),surface_conduction_input=heat_surface/volume,
                      stress_heating=mean(dissipation),
                      heating_identity_residual=mean(internal_rate)-heat_surface/volume-mean(dissipation)),
        entropy=dict(rate=mean(entropy_rate),surface_diffusion_input=entropy_surface/volume,
                     continuum_production=mean(entropy_production),
                     chain_rule_residual=mean(entropy_rate)-entropy_surface/volume-mean(entropy_production)),
        interpretation='surface fluxes, stencil defects and inactive normal rows remain distinct; KE/internal/entropy residuals are observations, not zero or stability gates; no body forcing, ME6, or NSCBC viscous-source projection included')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cpu','gpu','matrix','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--scheme',choices=['643e','643c'],required=True)
    for name in ('gamma','mach','reynolds','prandtl','ref-temperature','sutherland-k'):
        parser.add_argument('--'+name,type=float,required=True)
    args=parser.parse_args()
    parameters=[args.gamma,args.mach,args.reynolds,args.prandtl,args.ref_temperature,args.sutherland_k]
    if not np.isfinite(parameters).all() or min(parameters)<=0 or args.gamma<=1:
        raise ValueError('Invalid explicit gas/transport parameters')
    results={backend:[analyze(getattr(args,backend).resolve(),args.matrix.resolve(),args.scheme,stage,*parameters)
                     for stage in (1,2,3)] for backend in ('cpu','gpu')}
    compared={}
    for stage,(cpu,gpu) in enumerate(zip(results['cpu'],results['gpu'],strict=True),start=1):
        errors={name:scaled(np.array(cpu[name]),np.array(gpu[name])) for name in (
            'conservative_mean_rate','surface_mean_rate','stencil_volume_surface_defect','inactive_normal_rows')}
        for group in ('kinetic','internal','entropy'):
            errors[group]=scaled(np.array(list(cpu[group].values())),np.array(list(gpu[group].values())))
            errors[group+'_parts']=scaled(np.array(list(cpu['nonlinear_residual_parts'][group].values())),
                                         np.array(list(gpu['nonlinear_residual_parts'][group].values())))
        for face in cpu['physical_faces']:
            def face_values(record):
                return np.r_[record['viscous_outward_rhs_flux'],record['outward_physical_heat_flux'],
                    record['conductive_energy_input'],record['viscous_work_input'],record['entropy_diffusion_input']]
            errors[face]=scaled(face_values(cpu['physical_faces'][face]),face_values(gpu['physical_faces'][face]))
        if max(errors.values())>2e-10:
            raise AssertionError(f'CPU/GPU physical viscous terms differ at stage {stage}: {errors}')
        compared[stage]=errors
    with args.output.open('x') as stream:
        json.dump(dict(status='passed_frozen_standard_viscosity_reconstruction',cpu_gpu_max_scaled=compared,cases=results),stream,indent=2)
        stream.write('\n')
    print('frozen viscous reconstruction and physical terms compared:',args.scheme,flush=True)


if __name__=='__main__':
    main()
