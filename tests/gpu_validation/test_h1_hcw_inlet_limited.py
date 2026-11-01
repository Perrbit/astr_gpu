"""Approved inlet-interface attribution experiment, not a production repair."""
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from analyze_h1_hcw_sensitivity import native_q
from compare_flowstate import read_flowstate
from test_high_order_boundary_runtime import EXE, H1_VARIANT, launch_hcw

pytestmark = pytest.mark.skipif(not EXE or H1_VARIANT != 'inlet-limited',
                               reason='Explicit current executable and inlet-limited test variant required')
EVIDENCE = Path(os.environ.get('ASTR_H1_HCW_EVIDENCE',
    Path(__file__).parent/'out/h1_hcw_long_window_20261101')).resolve()


def state(case, steps):
    checkpoint = case/f'outdat/new/checkpoints/step{steps:012d}'
    with h5py.File(checkpoint/'state.h5') as file:
        values = np.stack([file[f'q{i:04d}'][...] for i in range(1, 12)], axis=-1)
    assert b'sbli-grcbc-h1iv' in (checkpoint/'control.bin').read_bytes()
    assert np.isfinite(values).all()
    assert np.all(values[..., [0, 5, 9, 10]] > 0)
    internal = values[..., 4]-.5*np.sum(values[..., 1:4]*(values[..., 1:4]/values[..., :1]), axis=-1)
    assert internal.min() > 0
    return values


def test_frozen1000_error_direction(tmp_path):
    results, initial, cases = {}, {}, {}
    reports = {}
    for backend in ('cpu', 'gpu'):
        for source in ('cpu', 'gpu'):
            path = next((EVIDENCE/f'dense840_to1260_{source}/outdat/new/fields').glob(
                'segment*/step000000001000/data.h5'))
            case = tmp_path/f'{source}_on_{backend}'
            launch_hcw(case, backend, steps=1, field_source=path, diagnostics=True)
            cases[(source, backend)] = case
            initial[(source, backend)] = native_q(case)
            results[(source, backend)] = state(case, 1).transpose(2, 1, 0, 3)
        with h5py.File(cases[('cpu', backend)]/'outdat/new/resources/geometry.h5') as file:
            weight = file['q0004'][...].transpose(2, 1, 0)[1:4, 1:-1, :-1, None]
        delta = (initial[('gpu', backend)]-initial[('cpu', backend)])[1:4, 1:-1, :-1]
        advanced = (results[('gpu', backend)]-results[('cpu', backend)])[1:4, 1:-1, :-1, :5]
        reports[backend] = float(np.sqrt(np.sum(weight*advanced**2)/np.sum(weight*delta**2)))
    parity = {source: float(np.max(abs(results[(source, 'gpu')]-results[(source, 'cpu')])/
                                   (1+abs(results[(source, 'cpu')])))) for source in ('cpu', 'gpu')}
    report = dict(scope='Jacobian-weighted all5 diagnostic norm, inlet columns1..3; not physical energy',
                  native_step_error_factors=reports, same_input_backend_scaled_errors=parity)
    (tmp_path/'frozen1000.json').write_text(json.dumps(report, indent=2)+'\n')
    assert max(parity.values()) <= 2e-10, report
    assert max(reports.values()) < 1., report


def test_reject_original_exact_identity(tmp_path):
    source = EVIDENCE/'gpu_early_prefix/outdat/new/checkpoints/step000000000840'
    with pytest.raises(AssertionError):
        launch_hcw(tmp_path/'reject', 'gpu', steps=841, restore=source)
    text = (tmp_path/'reject/run.log').read_text()
    assert 'boundary/driver configuration mismatch' in text, text[-3000:]
    assert 'H1_STAGE_STATE_PASS' not in text


def test_inlet_limited_memcheck(tmp_path):
    launch_hcw(tmp_path/'gpu', 'gpu', steps=2, memcheck=True)


def test_original_420_step_trajectory(tmp_path):
    values, statistics = {}, {}
    for backend in ('cpu', 'gpu'):
        case = tmp_path/backend
        launch_hcw(case, backend, steps=420)
        values[backend] = state(case, 420)
        statistics[backend] = read_flowstate(case)
    errors = [float(np.max(abs(values['gpu'][..., i]-values['cpu'][..., i])/
                           (1+abs(values['cpu'][..., i])))) for i in range(11)]
    assert statistics['cpu'][0] == statistics['gpu'][0]
    np.testing.assert_array_equal(statistics['cpu'][1][:, 0], statistics['gpu'][1][:, 0])
    report = dict(complete_steps=420, scaled_field_errors=errors, threshold=2e-10,
        maximum_statistics_absolute_difference=float(abs(statistics['gpu'][1]-statistics['cpu'][1]).max()))
    (tmp_path/'trajectory420.json').write_text(json.dumps(report, indent=2)+'\n')
    assert max(errors) <= 2e-10, report
