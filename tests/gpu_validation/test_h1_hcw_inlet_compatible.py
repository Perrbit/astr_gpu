"""Approved coefficient candidate: native short gates only, not a long-window pass."""
from pathlib import Path
import json

import h5py
import numpy as np
import pytest

import test_high_order_boundary_runtime as runtime
from analyze_h1_hcw_sensitivity import native_rhs
from test_h1_hcw_inlet_compatibility import EVIDENCE

pytestmark=pytest.mark.skipif(not runtime.EXE or runtime.H1_VARIANT!='inlet-compatible',
    reason='Select the explicit inlet-compatible test candidate and current executable')


def state(case,steps,identity=b'sbli-grcbc-h1cv'):
    batch=case/f'outdat/new/checkpoints/step{steps:012d}'
    with h5py.File(batch/'state.h5') as file:
        values=np.stack([file[f'q{c:04d}'][...] for c in range(1,12)],axis=-1)
    assert identity in (batch/'control.bin').read_bytes()
    assert np.isfinite(values).all()
    assert np.all(values[...,[0,5,9,10]]>0)
    internal=values[...,4]-.5*np.sum(values[...,1:4]*(values[...,1:4]/values[...,:1]),axis=-1)
    assert internal.min()>0
    return values


def test_native_candidate_same_state(tmp_path):
    field=next((EVIDENCE/'dense840_to1260_cpu/outdat/new/fields').glob(
        'segment*/step000000001000/data.h5'))
    cases={};values={}
    for backend in ('cpu','gpu'):
        case=tmp_path/backend;cases[backend]=case
        runtime.launch_hcw(case,backend,steps=1,field_source=field,diagnostics=True)
        values[backend]=state(case,1)
    error=float(np.max(abs(values['gpu']-values['cpu'])/(1+abs(values['cpu']))))
    with h5py.File(cases['cpu']/'outdat/new/resources/geometry.h5') as geometry:
        jacobian=geometry['q0004'][...].transpose(2,1,0)
    rhs={backend:native_rhs(case,'full')/jacobian[...,None] for backend,case in cases.items()}
    rhs_error=float(np.max(abs(rhs['gpu']-rhs['cpu'])))
    report=dict(same_input_field_scaled_error=error,same_input_physical_rhs_absolute_error=rhs_error,
        scope='one native candidate step on identical supplied primitives; not long-trajectory stability')
    (tmp_path/'same_state.json').write_text(json.dumps(report,indent=2)+'\n')
    assert error<=2e-10 and rhs_error<=2e-10,report


def test_previous_algorithm_exact_identity_is_rejected(tmp_path,monkeypatch):
    with monkeypatch.context() as context:
        context.setattr(runtime,'H1_VARIANT','admissible-filter')
        runtime.launch_hcw(tmp_path/'previous','cpu',steps=1)
    source=tmp_path/'previous/outdat/new/checkpoints/step000000000001'
    with pytest.raises(AssertionError):runtime.launch_hcw(tmp_path/'reject','cpu',steps=2,restore=source)
    text=(tmp_path/'reject/run.log').read_text()
    assert 'boundary/driver configuration mismatch' in text
    assert 'H1_STAGE_STATE_PASS' not in text


def test_native_candidate_twelve_steps(tmp_path):
    values={};statistics={}
    for backend in ('cpu','gpu'):
        case=tmp_path/backend
        runtime.launch_hcw(case,backend,steps=12)
        values[backend]=state(case,12)
        statistics[backend]=runtime.read_flowstate(case)
    error=float(np.max(abs(values['gpu']-values['cpu'])/(1+abs(values['cpu']))))
    assert statistics['cpu'][0]==statistics['gpu'][0]
    np.testing.assert_array_equal(statistics['cpu'][1][:,0],statistics['gpu'][1][:,0])
    statistics_error=float(np.max(abs(statistics['cpu'][1]-statistics['gpu'][1])))
    (tmp_path/'short.json').write_text(json.dumps(dict(steps=12,maximum_scaled_field_error=error,
        maximum_absolute_statistics_error=statistics_error),indent=2)+'\n')
    assert error<=2e-10 and statistics_error<=2e-10,(error,statistics_error)


def test_selector_off_preserves_previous_short_trajectory(tmp_path,monkeypatch):
    reference=Path(__file__).parent/'out/h1_inlet_compatibility_20261104/extended_packet_current/test_hcw_native_short_window0'
    reports={}
    with monkeypatch.context() as context:
        context.setattr(runtime,'H1_VARIANT','admissible-filter')
        for backend in ('cpu','gpu'):
            case=tmp_path/backend
            runtime.launch_hcw(case,backend,steps=12)
            ending='outdat/new/checkpoints/step000000000012/state.h5'
            assert b'sbli-grcbc-h1pv' in (case/Path(ending).parent/'control.bin').read_bytes()
            differences=[]
            with h5py.File(case/ending) as actual,h5py.File(reference/backend/ending) as old:
                for c in range(1,12):
                    value=actual[f'q{c:04d}'][...];previous=old[f'q{c:04d}'][...]
                    differences.append(float(np.max(abs(value-previous)/(1+abs(previous)))))
            reports[backend]=dict(cross_build_component_scaled_errors=differences)
            assert max(differences)<=2e-10,reports
        # Bitwise repeatability applies to the same executable, not a rebuilt kernel.
        runtime.launch_hcw(tmp_path/'gpu_repeat','gpu',steps=12)
        with h5py.File(tmp_path/'gpu'/ending) as actual,h5py.File(tmp_path/'gpu_repeat'/ending) as repeated:
            for c in range(1,12):np.testing.assert_array_equal(actual[f'q{c:04d}'][...],repeated[f'q{c:04d}'][...])
    reports['gpu']['same_executable_repeat_exact']=True
    (tmp_path/'selector_off.json').write_text(json.dumps(reports,indent=2)+'\n')
