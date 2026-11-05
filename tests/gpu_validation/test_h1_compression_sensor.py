"""Approved compression-only H1 candidate: native formula, phase and lifecycle."""
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

import test_high_order_boundary_runtime as runtime
from analyze_h1_hcw_sensitivity import native_rhs,rank_rhs
from analyze_m12_sensor_diagnostics import replay
from check_air5_numq11_shock_tube import read_sensor_mask
from run_native_precision_curve_gate import read_gradients
from run_h1_hcw_span_grid import clock
from test_h1_hcw_inlet_compatible import state

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'tests/gpu_validation/out/h1_cfl03_20261105/window/test_independent_cfl03_physica0/cpu'
pytestmark=pytest.mark.skipif(not runtime.EXE or runtime.H1_VARIANT not in ('inlet-compression','inlet-compression-metric'),
    reason='Select the approved compression-only test variant and root executable')


def values(case,steps):
    identity=b'sbli-grcbc-h1mv' if runtime.H1_VARIANT=='inlet-compression-metric' else b'sbli-grcbc-h1sv'
    return state(case,steps,identity=identity)


def test_frozen415_same_state_and_sensor_inputs(tmp_path):
    field=next((SOURCE/'outdat/new/fields').glob('segment*/step000000000415/data.h5'))
    cases={};end={};sensors={};reports={}
    for backend in ('cpu','gpu'):
        case=tmp_path/backend;cases[backend]=case
        runtime.launch_hcw(case,backend,steps=1,field_source=field,diagnostics=True,
                           validation_gradients=True,automatic_cfl=.5)
        end[backend]=values(case,1)
        rows=[];sensors[backend]=[]
        for rank in (0,1):
            stem=f'step00000000.rk01.rank{rank:08d}.bin'
            gradient,_=read_gradients(case/'diagnostics'/f'state.gradients.{stem}','fp64')
            with (case/'diagnostics'/f'state.pre_rhs_primitives.{stem}').open('rb') as file:
                header=np.fromfile(file,np.int32,4)
                storage=np.fromfile(file,np.float64).reshape((*tuple(header[:3]+2*header[3]+1),6),order='F')
            g=gradient[...,:9].reshape((*gradient.shape[:3],3,3),order='F')
            parts=replay(np.r_[header,1 if rank==0 else 2,4,0],g,storage[...,4])
            raw,mask=read_sensor_mask(case/'diagnostics'/f'state.sensor.{stem}')
            expected=np.where(parts['div']<0,parts['sensor'],0.)
            error=float(abs(raw-expected).max())
            assert error<=2e-10,(backend,rank,error)
            assert np.all(raw[parts['div']>=0]==0)
            sensors[backend].append((raw,mask))
            rows.append(dict(rank=rank,input_replay_error=error,
                             expanding_nodes=int(np.count_nonzero(parts['div']>=0))))
        reports[backend]=rows
    with h5py.File(cases['cpu']/'outdat/new/resources/geometry.h5') as file:
        jac=file['q0004'][...].transpose(2,1,0)
    rhs={b:native_rhs(c,'full')/jac[...,None] for b,c in cases.items()}
    field_error=float((abs(end['gpu']-end['cpu'])/(1+abs(end['cpu']))).max())
    rhs_error=float(abs(rhs['gpu']-rhs['cpu']).max())
    weights={b:[rank_rhs(c,'h1_strength',rank=rank)[...,0] for rank in (0,1)] for b,c in cases.items()}
    for b,w in weights.items():np.testing.assert_array_equal(w[0][-1],w[1][0])
    report=dict(input_replay=reports,field_error=field_error,physical_rhs_error=rhs_error,
        sensor_backend_error=max(float(abs(a[0]-b[0]).max()) for a,b in zip(sensors['cpu'],sensors['gpu'])),
        mask_differences=sum(int(np.count_nonzero(a[1]!=b[1])) for a,b in zip(sensors['cpu'],sensors['gpu'])),
        weight_backend_error=max(float(abs(a-b).max()) for a,b in zip(weights['cpu'],weights['gpu'])))
    (tmp_path/'frozen415.json').write_text(json.dumps(report,indent=2)+'\n')
    assert field_error<=2e-10 and rhs_error<=2e-10,report
    assert report['weight_backend_error']<=2e-10,report


def test_twelve_steps_and_exact_restart(tmp_path):
    ends={};stats={}
    for backend in ('cpu','gpu'):
        full,part,tail=[tmp_path/f'{backend}_{name}' for name in ('full','part','tail')]
        options=dict(automatic_cfl=.5,checkpoint_interval=5)
        runtime.launch_hcw(full,backend,steps=12,**options)
        runtime.launch_hcw(part,backend,steps=5,**options)
        runtime.launch_hcw(tail,backend,steps=12,restore=part/'outdat/new/checkpoints/step000000000005',**options)
        ends[backend]=values(full,12)
        np.testing.assert_array_equal(ends[backend],values(tail,12))
        np.testing.assert_array_equal(clock(full)[5:],clock(tail))
        stats[backend]=runtime.read_flowstate(full)
    error=float((abs(ends['gpu']-ends['cpu'])/(1+abs(ends['cpu']))).max())
    assert stats['cpu'][0]==stats['gpu'][0]
    stat_error=float(abs(stats['cpu'][1][:,2:]-stats['gpu'][1][:,2:]).max())
    (tmp_path/'short.json').write_text(json.dumps(dict(steps=12,field_error=error,
        statistics_error=stat_error,exact_5plus7=True),indent=2)+'\n')
    assert error<=2e-10 and stat_error<=2e-10,(error,stat_error)


def test_variable_pressure_compressive_rhs_convergence(tmp_path):
    errors=[];cases=[]
    for n in (32,64,128):
        rows={};raw={};worst=0.
        for backend in ('cpu','gpu'):
            case=tmp_path/f'{backend}_{n}'
            _,raw[backend],error=runtime.run_manufactured(case,backend,n,n,pressure_compression=True)
            worst=max(worst,float(error.max()))
            sensor,_=read_sensor_mask(case/'diagnostics/state.sensor.step00000000.rk01.rank00000000.bin')
            weight=rank_rhs(case,'h1_strength')[...,0]
            rows[backend]=dict(rhs_component_errors=error.tolist(),raw_sensor_max=float(sensor.max()),
                               blend_weight_max=float(weight.max()))
            assert sensor.max()>0 and weight.max()>0,rows
        parity=float((abs(raw['gpu']-raw['cpu'])/(1+abs(raw['cpu']))).max())
        assert parity<=2e-10,parity
        errors.append(worst);cases.append(dict(n=n,backend=rows,raw_rhs_backend_scaled=parity))
    orders=np.log2(np.array(errors[:-1])/errors[1:])
    report=dict(scope='Final full RHS, simultaneous x/y refinement, compressive velocity, variable rho/p and T=1',
                errors=errors,orders=orders.tolist(),cases=cases)
    (tmp_path/'pressure_convergence.json').write_text(json.dumps(report,indent=2)+'\n')
    assert np.all(orders>=5.5),report


@pytest.mark.parametrize('previous_variant',['inlet-compatible','inlet-compression'])
def test_compatible_identity_rejected(tmp_path,monkeypatch,previous_variant):
    if previous_variant==runtime.H1_VARIANT:pytest.skip('Same algorithm is covered by exact continuation')
    with monkeypatch.context() as context:
        context.setattr(runtime,'H1_VARIANT',previous_variant)
        runtime.launch_hcw(tmp_path/'old','cpu',steps=1)
    with pytest.raises(AssertionError):
        runtime.launch_hcw(tmp_path/'reject','cpu',steps=2,restore=tmp_path/'old/outdat/new/checkpoints/step000000000001')
    assert 'boundary/driver configuration mismatch' in (tmp_path/'reject/run.log').read_text()


def test_native_memcheck(tmp_path):
    runtime.launch_hcw(tmp_path/'gpu','gpu',steps=2,memcheck=True,automatic_cfl=.5)
