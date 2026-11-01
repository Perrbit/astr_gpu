"""Native H1 repair gate, not a flow-growth gate.

Production H1 remains known-red. Select a candidate executable explicitly when
evaluating a repair; a passing continuity gate alone does not admit that candidate.
"""
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from run_native_precision_tgv_gate import physical_state, rhs_state
from test_grcbc_oblique_shock import EXE, launch
from test_sbli_joint_nscbc import fields

VARIANT=os.environ.get('ASTR_H1_REPAIR_VARIANT')
ACTIVE_VARIANT=VARIANT or 'continuous'

EVIDENCE = Path(os.environ.get('ASTR_H1_THRESHOLD_EVIDENCE',
    Path(__file__).parent / 'out/h1_spanwise_diagnosis_20261031'))


@pytest.mark.skipif(not EXE, reason='Set ASTR_SBLI_GRCBC_EXE to a root-CMake executable')
@pytest.mark.parametrize('backend', ['gpu', 'cpu'])
def test_sensor_threshold_rhs_is_continuous(tmp_path, backend):
    sources = [EVIDENCE / f'threshold_{side}_input.h5' for side in ['below', 'above']]
    if not all(source.is_file() for source in sources):
        pytest.skip('Captured H1 threshold pair is required; synthetic states are not substituted')
    inputs = []
    for source in sources:
        with h5py.File(source) as file:
            inputs.append(np.stack([file[f'q{m:04d}'][:] for m in range(1, 6)], axis=-1))
    center = (inputs[0] + inputs[1]) / 2
    delta = (inputs[1] - inputs[0]) / 2
    rows = []
    jacobian = 1 / 192**2 / 64
    suffix = 'step00000000.rk01.rank00000000.bin'
    for scale in [1., .01]:
        q_values, rates = [], []
        for sign in [-1, 1]:
            source = tmp_path / f'input_{scale:g}_{sign}.h5'
            with h5py.File(source, 'w') as file:
                for m in range(5):
                    file[f'q{m+1:04d}'] = (center + sign * scale * delta)[..., m]
            case = launch(tmp_path / f'{backend}_{scale:g}_{sign}', backend, n=192, steps=1,
                field_source=source, diagnostics=True,h1_variant=VARIANT)
            q_values.append(physical_state(case / 'diagnostics' / f'state.pre_rhs.{suffix}', 'fp64'))
            rates.append(rhs_state(case / 'diagnostics' / f'state.full.{suffix}', 'fp64') / jacobian)
        rows.append(dict(scale=scale, input_linf=float(abs(q_values[1] - q_values[0]).max()),
                         rhs_jump=float(abs(rates[1][72, 191, 0] - rates[0][72, 191, 0]).max())))
    (tmp_path / 'continuity.json').write_text(json.dumps(rows, indent=2) + '\n')
    # A 100-fold input contraction must contract the response, not retain a finite jump.
    # The absolute term only accommodates FP64 evaluation noise, not flow perturbation growth.
    assert rows[1]['input_linf'] < .02 * rows[0]['input_linf'], rows
    assert rows[1]['rhs_jump'] <= .1 * rows[0]['rhs_jump'] + 1e-10, rows


@pytest.mark.skipif(not EXE, reason='Set ASTR_SBLI_GRCBC_EXE to a root-CMake executable')
def test_h1_split_cache_matches_uncached(tmp_path):
    source=EVIDENCE/'threshold_above_input.h5'
    if not source.is_file():
        pytest.skip('Captured H1 threshold field is required')
    states=[]
    for mode in ['off','on']:
        case=launch(tmp_path/mode,'gpu',n=192,steps=2,topology=(2,1,1),field_source=source,split_cache=mode,
                    h1_variant=VARIANT)
        states.append(fields(case,2)['q'])
    error=float(np.max(abs(states[1]-states[0])/(1+abs(states[0]))))
    (tmp_path/'cache.json').write_text(json.dumps(dict(maximum_scaled_q_error=error),indent=2)+'\n')
    assert error<=2e-10,error


@pytest.mark.skipif(not EXE, reason='Set ASTR_SBLI_GRCBC_EXE to a root-CMake executable')
@pytest.mark.parametrize('topology',[(2,1,1),(1,2,1),(1,1,2)])
def test_continuous_shared_strength(tmp_path,topology):
    cases=[launch(tmp_path/backend,backend,n=32,steps=2,topology=topology,
                  diagnostics=True,h1_variant=ACTIVE_VARIANT,z_intervals=16) for backend in ['cpu','gpu']]
    axis=topology.index(2)
    errors=[]
    for stage in [1,2,3]:
        weights=[]
        for case in cases:
            ranks=[]
            for rank in [0,1]:
                path=case/'diagnostics'/f'state.h1_strength.step00000000.rk{stage:02d}.rank{rank:08d}.bin'
                values=rhs_state(path,'fp64')[...,0]
                assert values.min()>=0 and values.max()<=1
                ranks.append(values)
            np.testing.assert_array_equal(np.take(ranks[0],-1,axis=axis),np.take(ranks[1],0,axis=axis))
            weights.append(ranks)
        errors.append(max(float(abs(a-b).max()) for a,b in zip(*weights)))
    (tmp_path/'strength.json').write_text(json.dumps(dict(cpu_gpu_errors=errors),indent=2)+'\n')
    assert max(errors)<=2e-10,errors
    state=[fields(case,2)['q'] for case in cases]
    assert float(np.max(abs(state[1]-state[0])/(1+abs(state[0]))))<=2e-10


@pytest.mark.skipif(not EXE,reason='Set ASTR_SBLI_GRCBC_EXE to a root-CMake executable')
@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_stage_observer_catches_known_failure(tmp_path,backend):
    source=EVIDENCE.parent/'h1_spanwise_repair_20261031/failure224_input.h5'
    if not source.is_file():
        pytest.skip('Captured positive input before candidate-v1 failure is required')
    case=tmp_path/backend
    with pytest.raises(AssertionError):
        launch(case,backend,n=192,steps=1,field_source=source,h1_variant='both',check_stages=True)
    text=(case/'run.log').read_text()
    assert f'H1_STAGE_STATE_FAILURE {backend.upper()}' in text,text[-3000:]


@pytest.mark.skipif(not EXE,reason='Set ASTR_SBLI_GRCBC_EXE to a root-CMake executable')
@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_continuous_exact_restart(tmp_path,backend):
    from run_output_restart_validation import compare_fields
    options=dict(n=32,topology=(2,1,1),h1_variant=ACTIVE_VARIANT,check_stages=True)
    full=launch(tmp_path/'full',backend,steps=12,**options)
    part=launch(tmp_path/'part',backend,steps=5,**options)
    source=part/'outdat/new/checkpoints/step000000000005'
    tail=launch(tmp_path/'tail',backend,steps=12,restore=source,**options)
    final=Path('outdat/new/checkpoints/step000000000012')
    compare_fields(full/final/'state.h5',tail/final/'state.h5')
    identity=b'sbli-grcbc-h1pc' if ACTIVE_VARIANT=='admissible-filter' else b'sbli-grcbc-h1w'
    assert identity in (full/final/'control.bin').read_bytes()
    assert (full/'run.log').read_text().count('H1_STAGE_STATE_PASS ')==36
    assert (tail/'run.log').read_text().count('H1_STAGE_STATE_PASS ')==21


@pytest.mark.skipif(not EXE,reason='Set ASTR_SBLI_GRCBC_EXE to a root-CMake executable')
def test_continuous_memcheck(tmp_path):
    launch(tmp_path/'gpu','gpu',n=32,steps=3,topology=(2,1,1),h1_variant=ACTIVE_VARIANT,
           check_stages=True,memcheck=True,z_intervals=16)


@pytest.mark.skipif(not EXE,reason='Set ASTR_SBLI_GRCBC_EXE to a root-CMake executable')
def test_continuous_rejects_legacy_exact_restart(tmp_path):
    part=launch(tmp_path/'legacy','gpu',n=32,steps=5,topology=(2,1,1),h1_variant='legacy')
    source=part/'outdat/new/checkpoints/step000000000005'
    case=tmp_path/'continuous'
    with pytest.raises(AssertionError):
        launch(case,'gpu',n=32,steps=12,topology=(2,1,1),restore=source,h1_variant=ACTIVE_VARIANT)
    text=(case/'run.log').read_text()
    assert 'boundary/driver configuration mismatch' in text,text[-3000:]
    assert 'The job is done!' not in text


@pytest.mark.skipif(not EXE,reason='Set ASTR_SBLI_GRCBC_EXE to a root-CMake executable')
def test_continuous_captured_positive_state(tmp_path):
    source=EVIDENCE.parent/'h1_spanwise_repair_20261031/failure224_input.h5'
    if not source.is_file():
        pytest.skip('Captured positive pre-failure state is required')
    launch(tmp_path/'gpu','gpu',n=192,steps=1,topology=(2,1,1),field_source=source,
           h1_variant=ACTIVE_VARIANT,check_stages=True,diagnostics=True)


@pytest.mark.skipif(not EXE,reason='Set ASTR_SBLI_GRCBC_EXE to a root-CMake executable')
def test_continuous_original_window225(tmp_path):
    from run_h1_limiter_factorial import frames
    source=EVIDENCE.parent/'h1_spanwise_repair_20261031/replay128/perturbation_0.0001.h5'
    if not source.is_file():
        pytest.skip('Original H1 failure replay source is required')
    case=launch(tmp_path/'gpu','gpu',n=192,steps=225,topology=(2,1,1),field_source=source,
                h1_variant=ACTIVE_VARIANT,check_stages=True,volume_interval=8,checkpoint_interval=32)
    timeline=frames(case)
    (tmp_path/'monitor.json').write_text(json.dumps(timeline,indent=2)+'\n')
    assert all(row['rho_min']>0 and row['temperature_min']>0 for row in timeline)
    assert (case/'run.log').read_text().count('H1_STAGE_STATE_PASS GPU')==675
