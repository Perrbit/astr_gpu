"""Original short-line diagnosis and the approved native P6/T7 regression."""
import json

import h5py
import numpy as np
import pytest
from scipy.linalg import eig

from test_high_order_wall_runtime import launch,initial_fields,EXE
from test_high_order_explicit_wall_candidates import state_projection
from test_high_order_adiabatic_wall_candidate import adiabatic_cns_matrix
from test_high_order_adiabatic_pressure_candidate import normal_model
from run_native_precision_tgv_gate import physical_state,rhs_state

pytestmark=pytest.mark.skipif(not EXE,reason='Root-CMake FP64 CUDA executable required for native RHS diagnosis')


def frozen_short_line_report():
    rows=[]
    for n in (8,9,10,11,12,14,16,20,24,28,32,64,128):
        a=adiabatic_cns_matrix(n,1/1600,'neumann7-projection')
        values=eig(a,left=False,right=False)
        rows.append(dict(global_normal_intervals=n,viscosity=1/1600,
            maximum_real=float(np.max(values.real)),
            roundoff_reference=float(100*np.finfo(float).eps*np.linalg.norm(a,np.inf))))
    n=8;a0=adiabatic_cns_matrix(n,0.,'neumann7-projection')
    diffusion=adiabatic_cns_matrix(n,1.,'neumann7-projection')-a0
    values,left,right=eig(a0,left=True,right=True)
    modes=[]
    momentum=np.zeros_like(diffusion)
    momentum[n+1:2*n,n+1:2*n]=diffusion[n+1:2*n,n+1:2*n]
    thermal=diffusion-momentum
    for j,value in enumerate(values):
        if value.imag<=1e-6:continue
        denominator=np.vdot(left[:,j],right[:,j])
        vm=np.vdot(left[:,j],momentum@right[:,j])/denominator
        vt=np.vdot(left[:,j],thermal@right[:,j])/denominator
        if (vm+vt).real<=0:continue
        modes.append(dict(inviscid_eigenvalue=[float(value.real),float(value.imag)],
            d_lambda_d_nu_momentum=[float(vm.real),float(vm.imag)],
            d_lambda_d_nu_thermal=[float(vt.real),float(vt.imag)],
            d_lambda_d_nu_total=[float((vm+vt).real),float((vm+vt).imag)]))
    return dict(scope='unfiltered frozen 42 normal operator; 421 shares this rest-state block, native 421 not tested here',
        base='rho=RT=1, gamma=1.4, Pr=0.72',samples=rows,
        n8_inviscid_maximum_real=float(np.max(values.real)),
        n8_momentum_diffusion_maximum_real=float(np.max(eig(momentum[n+1:2*n,n+1:2*n],left=False,right=False).real)),
        n8_thermal_diffusion_maximum_real=float(np.max(eig(thermal[2*n:,2*n:],left=False,right=False).real)),
        viscosity_drift_modes=modes,
        caveat='Eigenvalue perturbation, not a physical energy budget; sampled sizes do not define an admission limit.')


def temperature8_candidate_matrix(n,viscosity):
    """Rejected static alternative; never connected to the native solver."""
    import sympy as sp
    from test_high_order_adiabatic_wall_candidate import neumann_gradient
    from test_high_order_boundary_closures import derivative_matrix
    weights=np.array(sp.finite_diff_weights(1,list(range(9)),0)[1][-1],float)
    thermal=np.zeros((n+1,n-1));thermal[1:-1]=np.eye(n-1)
    if n==8:
        thermal[[0,-1]]=np.linalg.solve(np.array([[weights[0],weights[8]],[weights[8],weights[0]]]),
            -np.array([weights[1:8],weights[1:8][::-1]]))
    else:
        thermal[0,:8]=-weights[1:]/weights[0]
        thermal[-1,-8:]=-weights[1:][::-1]/weights[0]
    derivative=n*derivative_matrix(n,True);pressure=neumann_gradient(n)
    velocity=state_projection(n,'dirichlet');gradient=derivative@velocity
    return np.block([
        [np.zeros((n+1,n+1)),-gradient,np.zeros((n+1,n-1))],
        [-pressure[1:-1],(4/3)*viscosity*(derivative@derivative@velocity)[1:-1],-pressure[1:-1]@thermal],
        [np.zeros((n-1,n+1)),-.4*gradient[1:-1],
            (1.4/.72)*viscosity*(derivative@derivative@thermal)[1:-1]]])


@pytest.mark.parametrize('wall_type',[42,421],ids=('bc42','bc421'))
@pytest.mark.parametrize('n',[8,9])
def test_native_pressure6_short_line_mode(launch,record_property,wall_type,n):
    epsilon=1e-4;t0=.126;st=11030/27315
    viscosity=t0**1.5*(1+st)/(t0+st)/3000
    matrix=normal_model(n,viscosity)
    values,left,right=eig(matrix,left=True,right=True)
    candidates=np.flatnonzero(values.imag>1e-6)
    index=candidates[np.argmax(values[candidates].imag)]
    eigenvalue=values[index]
    assert np.max(values.real)<=100*np.finfo(float).eps*np.linalg.norm(matrix,np.inf)
    assert eigenvalue.real<0
    vector=right[:,index]/np.max(abs(right[:,index]))
    direction=vector.real
    rho=direction[:n+1]
    velocity=state_projection(n,'dirichlet')@direction[n+1:2*n]
    temperature=state_projection(n,'neumann')@direction[2*n:]
    producers=[]
    for sign in (1,-1):
        def producer(grid,manufactured,wall_type,sign=sign):
            coordinates,_=initial_fields(grid,manufactured,wall_type)
            shape=coordinates[0].shape
            def tile(line):return np.broadcast_to(line[None,:,None],shape).copy()
            return coordinates,dict(ro=tile(1+sign*epsilon*rho),u1=np.zeros(shape),
                u2=tile(sign*epsilon*velocity),u3=np.zeros(shape),t=tile(t0*(1+sign*epsilon*temperature)))
        producers.append(producer)
    report=dict(scope='one-step native P6/T7 directional RHS regression; not long-window validation',
        bctype=wall_type,global_intervals=[8,n,8],epsilon=epsilon,temperature_base=t0,viscosity_base=viscosity,
        expected_eigenvalue=[float(eigenvalue.real),float(eigenvalue.imag)],backends={})
    raw={}
    for backend in ('cpu','gpu'):
        rhs=[];states=[];directories=[]
        for producer in producers:
            case,_=launch(backend,steps=1,grid=(8,n,8),wall_type=wall_type,dt=1e-6,initial_condition=producer)
            suffix='step00000000.rk01.rank00000000.bin'
            weighted=rhs_state(case/'diagnostics'/f'state.full.{suffix}','fp64')
            state=physical_state(case/'diagnostics'/f'state.pre_rhs.{suffix}','fp64')
            with h5py.File(case/'outdat/new/resources/geometry.h5') as f:
                jacobian=f['q0004'][:].transpose(2,1,0)
            rhs.append(weighted/jacobian[...,None]);states.append(state);directories.append(str(case))
        derivative=(rhs[0][0,:,0]-rhs[1][0,:,0])/(2*epsilon)
        observed=np.r_[derivative[:,0],derivative[1:-1,2],.4*derivative[1:-1,4]-derivative[1:-1,0]]
        projected=np.vdot(left[:,index],observed)/np.vdot(left[:,index],direction)
        report['backends'][backend]=dict(cases=directories,
            observed_eigenvalue=[float(projected.real),float(projected.imag)],
            eigenvalue_error=float(abs(projected-eigenvalue)),
            maximum_directional_rhs_error=float(np.max(abs(observed-matrix@direction))))
        raw[backend]=rhs
        path=case.parent/f'pressure6_short_line_bc{wall_type}_n{n}.json'
        path.write_text(json.dumps(report,indent=2)+'\n')
        # Retain the original 10% modal-rate attribution check for finite perturbations.
        assert abs(projected-eigenvalue)<abs(eigenvalue.real)/10,report
        assert projected.real<0,report
    parity=max(float(np.max(abs(a-b)/(1+abs(a)))) for a,b in zip(raw['cpu'],raw['gpu'],strict=True))
    report['backend_rhs_max_scaled']=parity
    path.write_text(json.dumps(report,indent=2)+'\n')
    record_property('native_pressure6_short_line',json.dumps(report))
    assert parity<=2e-10,report
