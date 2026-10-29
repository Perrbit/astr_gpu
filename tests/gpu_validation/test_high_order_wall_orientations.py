"""Rigid coordinate rotations of the approved Cartesian 41/42 closures."""
import json
import shutil

import h5py
import numpy as np
import pytest

from test_high_order_wall_runtime import launch,EXE,GRID,DT,defined_extras,analytic_normal_rhs
from run_native_precision_tgv_gate import rhs_state,physical_state
from run_output_restart_validation import compare_fields
from compare_flowstate import read_flowstate
from test_high_order_boundary_closures import derivative_rows

pytestmark=pytest.mark.skipif(not EXE,reason='Set ASTR_HB1B_EXE to the root-CMake FP64 CUDA executable')
ORIENTATIONS=((1,41),(1,42),(3,41))
PERMUTATIONS={1:(1,2,0),2:(0,1,2),3:(2,0,1)}


def fields(path):
    with h5py.File(path) as f:
        return np.stack([f[f'q{m:04d}'][:].transpose(2,1,0) for m in range(1,12)],axis=-1)


def gradients(path):
    with path.open('rb') as f:
        header=np.fromfile(f,'<i4',5)
        shape=tuple(header[:3]+1);nodes=int(np.prod(shape))
        velocity=np.fromfile(f,'<f8',9*nodes).reshape((*shape,3,3),order='F')
        temperature=np.fromfile(f,'<f8',3*nodes).reshape((*shape,3),order='F')
    assert np.isfinite(velocity).all() and np.isfinite(temperature).all()
    return velocity,temperature


@pytest.mark.parametrize('axis,kind',ORIENTATIONS)
@pytest.mark.parametrize('topology',[(1,1,1),(2,1,1),(1,2,1),(1,1,2),(2,2,1)])
def test_rotated_wall_backend_fields(launch,axis,kind,topology,record_property):
    cases={};errors={}
    for backend in ('cpu','gpu'):
        cases[backend]=launch(backend,topology,normal_axis=axis,wall_type=kind,flow_type='tgv',gradients=True)
    cpu,a=cases['cpu'];gpu,b=cases['gpu']
    x,y=fields(a/'state.h5'),fields(b/'state.h5')
    errors['fields']=float(np.max(abs(x-y)/(1+abs(x))))
    with h5py.File(a/'state.h5') as p,h5py.File(b/'state.h5') as q:
        for rank,(x,y) in enumerate(zip(defined_extras(p,axis),defined_extras(q,axis),strict=True)):
            assert np.isfinite(x).all() and np.isfinite(y).all()
            errors[f'extras_{rank}']=float(np.max(abs(x-y)/(1+abs(x))))
    for rank in range(int(np.prod(topology))):
        suffix=f'step00000000.rk01.rank{rank:08d}.bin'
        for phase in ('conv','full'):
            x=rhs_state(cpu/'diagnostics'/f'state.{phase}.{suffix}','fp64')
            y=rhs_state(gpu/'diagnostics'/f'state.{phase}.{suffix}','fp64')
            errors[f'{phase}_{rank}']=float(np.max(abs(x-y)/(1+abs(x))))
        for component,(x,y) in enumerate(zip(gradients(cpu/'diagnostics'/f'state.gradients.{suffix}'),
            gradients(gpu/'diagnostics'/f'state.gradients.{suffix}'),strict=True)):
            errors[f'gradient_{rank}_{component}']=float(np.max(abs(x-y)/(1+abs(x))))
    ha,sa=read_flowstate(cpu);hb,sb=read_flowstate(gpu)
    assert ha==hb and sa.shape==sb.shape
    errors['statistics']=float(np.max(abs(sa-sb)/(1+abs(sa))))
    report=dict(axis=axis,bctype=kind,topology=topology,errors=errors)
    (gpu.parent/f'orientation_{axis}_{kind}_{"x".join(map(str,topology))}.json').write_text(json.dumps(report,indent=2)+'\n')
    record_property('maximum_scaled_difference',max(errors.values()))
    assert max(errors.values())<=2e-10,report


def original_frame(values,axis):
    inverse=np.argsort(PERMUTATIONS[axis])
    back=values.transpose((*inverse,3))
    components=[0,*(inverse+1),4,5,*(inverse+6),9,10]
    return back[...,components]


@pytest.mark.parametrize('axis,kind',ORIENTATIONS)
@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_same_backend_coordinate_covariance(launch,axis,kind,backend,record_property):
    _,reference=launch(backend,normal_axis=2,wall_type=kind,flow_type='tgv',gradients=True)
    _,rotated=launch(backend,normal_axis=axis,wall_type=kind,flow_type='tgv',gradients=True)
    x=fields(reference/'state.h5');y=original_frame(fields(rotated/'state.h5'),axis)
    difference=float(np.max(abs(x-y)/(1+abs(x))))
    record_property('same_backend_rotation_scaled_difference',difference)
    assert difference<=2e-10,(axis,kind,backend,difference)


@pytest.mark.parametrize('axis,kind',ORIENTATIONS)
def test_rotated_actual_rhs_convergence(launch,axis,kind,record_property):
    permutation=PERMUTATIONS[axis]
    columns=[0,*(np.array(permutation)+1),4]
    rows=[]
    for n in (32,64,128):
        row={'normal_intervals':n};samples={}
        normal=np.linspace(0,1,n+1)
        reference=analytic_normal_rhs(normal,wall_type=kind)[:,columns]
        strip=np.r_[0:8,n-7:n+1]
        for backend in ('cpu','gpu'):
            case,_=launch(backend,normal_axis=axis,wall_type=kind,flow_type='tgv',
                gradients=True,grid=(8,n,8),manufactured=True,steps=1)
            suffix='step00000000.rk01.rank00000000.bin'
            rhs=rhs_state(case/'diagnostics'/f'state.full.{suffix}','fp64')
            with h5py.File(case/'outdat/new/resources/geometry.h5') as f:
                jacobian=f['q0004'][:].transpose(2,1,0)
            physical=rhs/jacobian[...,None];samples[backend]=physical
            line=np.moveaxis(physical,axis-1,0)[:,0,0,:]
            row[backend+'_rhs_error']=np.max(abs(line[strip]-reference[strip]),axis=0).tolist()
            velocity,temperature=gradients(case/'diagnostics'/f'state.gradients.{suffix}')
            actual=np.moveaxis(velocity,axis-1,0)[:,0,0,:,:]
            expected=np.zeros_like(actual)
            ref_gradient=np.stack([.1*np.pi*np.cos(np.pi*normal),.02*np.pi*np.cos(2*np.pi*normal),
                .02*np.pi*np.cos(np.pi*normal)],axis=-1)
            expected[:,:,axis-1]=ref_gradient[:,permutation]
            row[backend+'_gradient_error']=float(np.max(abs(actual[strip]-expected[strip])))
            thermal=np.moveaxis(temperature,axis-1,0)[:,0,0,axis-1]
            row[backend+'_temperature_gradient_error']=float(np.max(abs(thermal[strip]-.02*np.pi*np.sin(2*np.pi*normal[strip]))))
            row[backend+'_wall_temperature_gradient']=float(np.max(abs(thermal[[0,n]])))
        row['backend_scaled']=float(np.max(abs(samples['cpu']-samples['gpu'])/(1+abs(samples['cpu']))))
        assert row['backend_scaled']<=2e-10,row
        rows.append(row)
    orders={}
    for backend in ('cpu','gpu'):
        errors=np.array([r[backend+'_rhs_error'] for r in rows])
        orders[backend+'_rhs']=np.log2(errors[:-1]/errors[1:]).tolist()
        gradient=np.array([r[backend+'_gradient_error'] for r in rows])
        orders[backend+'_gradient']=np.log2(gradient[:-1]/gradient[1:]).tolist()
        thermal=np.array([r[backend+'_temperature_gradient_error'] for r in rows])
        orders[backend+'_temperature_gradient']=np.log2(thermal[:-1]/thermal[1:]).tolist()
    report=dict(axis=axis,bctype=kind,samples=rows,orders=orders)
    (case.parent/f'orientation_convergence_{axis}_{kind}.json').write_text(json.dumps(report,indent=2)+'\n')
    record_property('spatial_convergence',json.dumps(report))
    for backend in ('cpu','gpu'):
        assert np.min(orders[backend+'_rhs'])>=5.5,report
        assert np.min(orders[backend+'_gradient'])>=5.5,report
        assert np.min(orders[backend+'_temperature_gradient'])>=5.5,report


@pytest.mark.parametrize('axis,kind',ORIENTATIONS)
@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_rotated_exact_restart(launch,axis,kind,backend):
    topology=tuple(2 if d==axis else 1 for d in (1,2,3))
    _,whole=launch(backend,topology,normal_axis=axis,wall_type=kind,flow_type='tgv')
    _,part=launch(backend,topology,normal_axis=axis,wall_type=kind,flow_type='tgv',steps=5)
    _,resumed=launch(backend,topology,normal_axis=axis,wall_type=kind,flow_type='tgv',restore=part)
    compare_fields(whole/'state.h5',resumed/'state.h5')
    assert (whole/'control.bin').read_bytes()==(resumed/'control.bin').read_bytes()


@pytest.mark.parametrize('axis,kind',ORIENTATIONS)
def test_rotated_wall_memory_safety(launch,axis,kind):
    topology=tuple(2 if d==axis else 1 for d in (1,2,3))
    launch('gpu',topology,normal_axis=axis,wall_type=kind,flow_type='tgv',steps=2,sanitizer=True)


@pytest.mark.parametrize('axis,kind',ORIENTATIONS)
@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_rotated_wall_lifecycle(launch,axis,kind,backend):
    import sympy as sp
    case,checkpoint=launch(backend,normal_axis=axis,wall_type=kind,flow_type='tgv',gradients=True)
    suffix='step00000000.rk01.rank00000000.bin'
    q=physical_state(case/'diagnostics'/f'state.pre_rhs.{suffix}','fp64')
    after=physical_state(case/'diagnostics'/f'state.post_update.{suffix}','fp64')
    weighted=rhs_state(case/'diagnostics'/f'state.full.{suffix}','fp64')
    with h5py.File(case/'outdat/new/resources/geometry.h5') as f:
        jacobian=f['q0004'][:].transpose(2,1,0)
        metric=f[f'q{5+4*(axis-1):04d}'][:].transpose(2,1,0)
    rhs=weighted/jacobian[...,None]
    rho=q[...,0];v=q[...,1:4]/rho[...,None]
    t=(q[...,4]/rho-.5*np.sum(v**2,axis=-1))*.126/2.5
    tdot=(rhs[...,4]-np.sum(v*rhs[...,1:4],axis=-1)+
        (np.sum(v**2,axis=-1)-q[...,4]/rho)*rhs[...,0])*.126/(2.5*rho)
    a=np.array(sp.finite_diff_weights(1,list(range(8)),0)[1][-1],float)
    projection=-a[1:]/a[0];weights=np.array(derivative_rows(True)[0],float)
    qn=np.moveaxis(q,axis-1,0);an=np.moveaxis(after,axis-1,0)
    rn=np.moveaxis(rhs,axis-1,0);jn=np.moveaxis(jacobian,axis-1,0)
    mn=np.moveaxis(metric,axis-1,0);wn=np.moveaxis(weighted,axis-1,0)
    tn=np.moveaxis(t,axis-1,0);tdn=np.moveaxis(tdot,axis-1,0)
    for j,side in ((0,1),(GRID[1],-1)):
        locations=j+side*np.arange(8)
        expected=-side*np.einsum('s,sjk->jk',weights,jn[locations]*qn[locations,...,axis]*mn[locations])
        assert np.max(abs(wn[j,...,0]-expected))<5e-13
        assert np.max(abs(an[j,...,0]-qn[j,...,0]-DT*rn[j,...,0]))<5e-13
        assert np.max(abs(an[j,...,1:4]))==0
        if kind==41:
            assert np.max(abs(tn[j]-1))<2e-14
            assert np.max(abs(tdn[j]))<2e-12
        else:
            assert np.max(abs(tn[j]-np.einsum('s,sjk->jk',projection,tn[locations[1:]])))<2e-14
            assert np.max(abs(tdn[j]-np.einsum('s,sjk->jk',projection,tdn[locations[1:]])))<2e-12


@pytest.mark.parametrize('axis,kind',ORIENTATIONS)
def test_rotated_algorithm_identity(launch,axis,kind):
    topology=tuple(2 if d==axis else 1 for d in (1,2,3))
    case,part=launch('cpu',topology,normal_axis=axis,wall_type=kind,flow_type='tgv',steps=5)
    tag=f'cartesian-axis{axis}-'+('isothermal-h7-v1' if kind==41 else 'adiabatic-p6t7-v1')
    payload=(part/'control.bin').read_bytes();encoded=tag.encode().ljust(128,b' ')
    assert payload.count(encoded)==1
    clone=case/'wrong_wall_identity';shutil.copytree(part.parents[1],clone)
    changed=clone/'checkpoints'/part.name
    old_tag=f'cartesian-axis{axis}-adiabatic-h7-v1' if kind==42 else 'channel-wall-h7-v1'
    (changed/'control.bin').write_bytes(payload.replace(encoded,old_tag.encode().ljust(128,b' ')))
    launch('cpu',topology,normal_axis=axis,wall_type=kind,flow_type='tgv',restore=changed,
        reject='boundary/driver configuration mismatch')
    assert (part/'control.bin').read_bytes()==payload
