"""Approved HB1-C2 ASTR integration, independent of static matrix proposals."""
import json
import shutil

import h5py
import numpy as np
import pytest

from test_high_order_wall_runtime import launch,defined_extras,EXE,analytic_normal_rhs,DT,GRID,check_completed_wall_diagnostics
from run_native_precision_tgv_gate import rhs_state,physical_state
from run_output_restart_validation import compare_fields
from compare_flowstate import read_flowstate

pytestmark=pytest.mark.skipif(not EXE,reason='Set ASTR_HB1B_EXE to the root-CMake FP64 CUDA executable')


@pytest.mark.parametrize('topology',[(1,1,1),(2,1,1),(1,2,1),(1,1,2),(2,2,1),(1,4,1)])
def test_adiabatic_same_stage_fields(launch,topology,record_property,wall_type=42):
    cpu,a=launch('cpu',topology,wall_type=wall_type)
    gpu,b=launch('gpu',topology,wall_type=wall_type)
    errors={}
    with h5py.File(a/'state.h5') as x,h5py.File(b/'state.h5') as y:
        for name in [f'q{i:04d}' for i in range(1,12)]:
            p,q=x[name][:],y[name][:]
            errors[name]=float(np.max(np.abs(p-q)/(1+np.abs(p))))
            assert errors[name]<=2e-10,(name,errors[name])
        for rank,(p,q) in enumerate(zip(defined_extras(x),defined_extras(y),strict=True)):
            assert np.isfinite(p).all() and np.isfinite(q).all()
            errors[f'extras_rank{rank}']=float(np.max(np.abs(p-q)/(1+np.abs(p))))
            assert errors[f'extras_rank{rank}']<=2e-10,errors
    for rank in range(int(np.prod(topology))):
        suffix=f'step00000000.rk01.rank{rank:08d}.bin'
        for phase in ('conv','full'):
            p=rhs_state(cpu/'diagnostics'/f'state.{phase}.{suffix}','fp64')
            q=rhs_state(gpu/'diagnostics'/f'state.{phase}.{suffix}','fp64')
            errors[f'{phase}_rank{rank}']=float(np.max(np.abs(p-q)/(1+np.abs(p))))
            assert errors[f'{phase}_rank{rank}']<=2e-10,errors
    ha,sa=read_flowstate(cpu);hb,sb=read_flowstate(gpu)
    assert ha==hb and sa.shape==sb.shape
    assert np.max(abs(sa-sb)/(1+abs(sa)))<=2e-10
    (gpu.parent/f'adiabatic_comparison_{"x".join(map(str,topology))}.json').write_text(json.dumps(errors,indent=2)+'\n')
    record_property('maximum_scaled_difference',max(errors.values()))


@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_adiabatic_state_and_tangent_rhs(launch,backend,wall_type=42):
    import sympy as sp
    coefficients=np.array(sp.finite_diff_weights(1,list(range(8)),0)[1][-1],float)
    case,checkpoint=launch(backend,wall_type=wall_type)
    suffix='step00000000.rk01.rank00000000.bin'
    state=physical_state(case/'diagnostics'/f'state.pre_rhs.{suffix}','fp64')
    rhs=rhs_state(case/'diagnostics'/f'state.full.{suffix}','fp64')
    after=physical_state(case/'diagnostics'/f'state.post_update.{suffix}','fp64')
    with h5py.File(case/'outdat/new/resources/geometry.h5') as f:
        jacobian=f['q0004'][:].transpose(2,1,0)
    physical_rhs=rhs/jacobian[...,None]
    rho=state[...,0];vel=state[...,1:4]/rho[...,None]
    temperature=(state[...,4]/rho-.5*np.sum(vel**2,axis=-1))*.126/2.5
    temperature_rhs=(physical_rhs[...,4]-np.sum(vel*physical_rhs[...,1:4],axis=-1)+
        (np.sum(vel**2,axis=-1)-state[...,4]/rho)*physical_rhs[...,0])*.126/(2.5*rho)
    for j,side in ((0,1),(GRID[1],-1)):
        locations=j+side*np.arange(8)
        reconstructed=-np.einsum('j,ijk->ik',coefficients[1:]/coefficients[0],temperature[:,locations[1:],:])
        assert np.max(abs(temperature[:,j,:]-reconstructed))<2e-14
        projected_rate=-np.einsum('j,ijk->ik',coefficients[1:]/coefficients[0],temperature_rhs[:,locations[1:],:])
        assert np.max(abs(temperature_rhs[:,j,:]-projected_rate))<2e-12
        assert np.max(abs(after[:,j,:,0]-(state[:,j,:,0]+DT*physical_rhs[:,j,:,0])))<5e-13
    with h5py.File(checkpoint/'state.h5') as f:
        temperature=f['q0011'][:].transpose(2,1,0)
        final_q=np.stack([f[f'q{m:04d}'][:].transpose(2,1,0) for m in range(1,6)],axis=-1)
    for j,side in ((0,1),(GRID[1],-1)):
        line=temperature[:,j+side*np.arange(8),:]
        reconstructed=-np.einsum('j,ijk->ik',coefficients[1:]/coefficients[0],line[:,1:,:])
        assert np.max(abs(line[:,0,:]-reconstructed))<2e-14
    from test_high_order_adiabatic_pressure_candidate import hermite_data
    weights=np.array([list(row[0,:6]) for row in hermite_data(6)],float)
    for q in (state,after,final_q):
        pressure=(q[...,4]-.5*np.sum(q[...,1:4]**2,axis=-1)/q[...,0])/2.5
        for j,side in ((0,1),(GRID[1],-1)):
            line=pressure[:,j+side*np.arange(6),:]
            ghosts=np.einsum('gj,ijk->igk',weights[:,1:],line[:,1:,:]-line[:,0:1,:])+line[:,0:1,:]
            assert np.isfinite(ghosts).all() and np.min(ghosts)>0


def test_adiabatic_actual_rhs_spatial_convergence(launch,record_property,wall_type=42):
    results=[]
    for n in (32,64,128):
        row={'intervals':n}
        exact=analytic_normal_rhs(np.linspace(0,1,n+1),wall_type=wall_type)
        strip=np.r_[0:8,n-7:n+1]
        backend_rhs={}
        for backend in ('cpu','gpu'):
            case,_=launch(backend,steps=1,grid=(8,n,8),manufactured=True,wall_type=wall_type,wall_samples=True)
            suffix='step00000000.rk01.rank00000000.bin'
            weighted=rhs_state(case/'diagnostics'/f'state.full.{suffix}','fp64')
            with h5py.File(case/'outdat/new/resources/geometry.h5') as f:
                jacobian=f['q0004'][:].transpose(2,1,0)
            rhs=weighted/jacobian[...,None]
            backend_rhs[backend]=rhs
            row[backend]=np.max(np.abs(rhs[0,strip,0,:]-exact[strip]),axis=0).tolist()
            with (case/'diagnostics/wall.wall.step00000001.rank00000000.bin').open('rb') as f:
                assert f.read(8)==b'ASTRIW01'
                header=np.fromfile(f,'<i4',9);np.fromfile(f,'<f8',2)
                shape=tuple(header[3:6]);nodes=int(np.prod(shape))
                np.fromfile(f,'<f8',nodes*3)
                fields=np.fromfile(f,'<f8',nodes*4).reshape((*shape,4),order='F')
                row[backend+'_wall_heat_error']=float(np.max(abs(fields[...,2])))
        row['backend_scaled']=float(np.max(abs(backend_rhs['cpu']-backend_rhs['gpu'])/(1+abs(backend_rhs['cpu']))))
        assert row['backend_scaled']<=2e-10,row
        results.append(row)
    orders={backend:np.log2(np.array([r[backend] for r in results])[:-1]/
        np.array([r[backend] for r in results])[1:]).tolist() for backend in ('cpu','gpu')}
    heat_orders={backend:np.log2(np.array([r[backend+'_wall_heat_error'] for r in results])[:-1]/
        np.array([r[backend+'_wall_heat_error'] for r in results])[1:]).tolist() for backend in ('cpu','gpu')}
    report=dict(samples=results,orders=orders,wall_heat_orders=heat_orders)
    (case.parent/'adiabatic_spatial_convergence.json').write_text(json.dumps(report,indent=2)+'\n')
    record_property('spatial_convergence',json.dumps(report))
    for backend in ('cpu','gpu'):
        assert np.min(orders[backend])>=5.5,(backend,results,orders)
        assert np.min(heat_orders[backend])>=5.5,(backend,report)


@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_adiabatic_exact_restart(launch,backend,wall_type=42):
    _,whole=launch(backend,(2,1,1),12,wall_type=wall_type)
    _,part=launch(backend,(2,1,1),5,wall_type=wall_type)
    _,resumed=launch(backend,(2,1,1),12,restore=part,wall_type=wall_type)
    compare_fields(whole/'state.h5',resumed/'state.h5')
    assert (whole/'control.bin').read_bytes()==(resumed/'control.bin').read_bytes()


def test_adiabatic_algorithm_change_rejected(launch,wall_type=42):
    case,part=launch('cpu',(2,1,1),5,wall_type=wall_type)
    payload=(part/'control.bin').read_bytes()
    source=(b'channel-adiabatic-p6t7-v1' if wall_type==42 else b'channel-slip-p6t7-v1').ljust(128,b' ')
    target=(b'channel-adiabatic-h7-v1' if wall_type==42 else b'channel-slip-h7-v1').ljust(128,b' ')
    assert payload.count(source)==1
    copy=case/'wrong_algorithm'
    shutil.copytree(part.parents[1],copy)
    changed=copy/'checkpoints'/part.name
    (changed/'control.bin').write_bytes(payload.replace(source,target))
    launch('cpu',(2,1,1),12,restore=changed,wall_type=wall_type,
        reject='boundary/driver configuration mismatch')
    assert (part/'control.bin').read_bytes()==payload


def test_adiabatic_memory_safety(launch):
    launch('gpu',(1,2,1),steps=2,wall_type=42,wall_samples=True,sanitizer=True)


@pytest.mark.parametrize('topology',[(1,1,1),(1,2,1)])
def test_adiabatic_wall_diagnostics(launch,topology,record_property):
    check_completed_wall_diagnostics(launch,topology,'hermite7',record_property,wall_type=42)


@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_adiabatic_wall_diagnostic_isolation(launch,backend):
    _,plain=launch(backend,steps=2,wall_type=42)
    _,sampled=launch(backend,steps=2,wall_type=42,wall_samples=True)
    compare_fields(plain/'state.h5',sampled/'state.h5')
