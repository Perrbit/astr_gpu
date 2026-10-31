from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from air5_radau_reference import Air5RadauReference
from check_air5_actual_error import compare_case, error_values, g1_scales, require_g1
from run_air5_reference_convergence import synthetic_states


def test_cpu_does_not_mislabel_a_source_only_run(tmp_path):
    script=Path(__file__).with_name('check_air5_actual_error.py')
    result=subprocess.run([sys.executable,str(script),'--reference',str(tmp_path/'missing.json'),
        '--output',str(tmp_path/'out'),'--backend','cpu','--source-evaluation','source-only'],
        capture_output=True,text=True)
    assert result.returncode!=0
    assert 'CPU probe supports joint source evaluation only' in result.stderr
    assert not (tmp_path/'out').exists()


@pytest.fixture
def fixture():
    model = Air5RadauReference(Path(__file__).resolve().parents[2] / 'chemMech/air5_kimjo12.json')
    case = list(synthetic_states(model))[4]  # Hot V-T-only, all five species present.
    case['endpoints'] = [case['initial'].tolist()] * 3
    return model, case


def test_component_norm_preserves_near_zero_absolute_and_relative_errors():
    result = error_values([1e-13, 1. + 1e-8], [0., 1.], [1e-12, 1e-12], [1e-6, 1e-6])
    assert result['relative_error'][0] is None
    assert result['absolute_error'][0] == 1e-13
    assert result['normalized_error'][0] == pytest.approx(.1)
    assert result['normalized_error'][1] < 1.


@pytest.mark.parametrize('actual,absolute,relative', [
    ([np.nan], [1.], [0.]), ([1., 2.], [1.], [0.]),
    ([1.], [0.], [0.]), ([1.], [1.], [-1.]),
])
def test_invalid_comparison_is_not_a_passing_diagnostic(actual, absolute, relative):
    with pytest.raises(ValueError):
        error_values(actual, [1.], absolute, relative)


def test_exact_endpoint_and_reference_margin(fixture):
    model, case = fixture
    result = compare_case(model, case, case['initial'], np.zeros(6))
    assert result['within_g1_reference_margin']
    for row in result['comparisons'].values():
        assert row['within_g1_error'] and row['within_g1_invariants']
        assert max(row['normalized_error']) == 0.
    require_g1(result)
    absolute, _ = g1_scales(case)
    assert absolute[5] == 1e-10 * max(abs(case['initial'][5] / case['rho']), 1.)


def test_unresolved_reference_is_distinguished_from_candidate_error(fixture):
    model, case = fixture
    medium = np.asarray(case['endpoints'][-2]).copy()
    medium[5] *= 1. + 1e-6
    case['endpoints'][-2] = medium.tolist()
    result = compare_case(model, case, case['initial'], np.zeros(6))
    assert not result['within_g1_reference_margin']
    assert result['comparisons']['q']['within_g1_error']
    with pytest.raises(ValueError, match='unresolved independent reference'):
        require_g1(result)


@pytest.mark.parametrize('gate,message', [('within_g1_invariants', 'physical invariants'),
                                        ('within_g1_error', 'error budget')])
def test_failed_g1_stops_candidate_release(fixture, gate, message):
    model, case = fixture
    result = compare_case(model, case, case['initial'], np.zeros(6))
    result['comparisons']['q-minus-carry'][gate] = False
    with pytest.raises(ValueError, match=message):
        require_g1(result)


def test_carry_is_checked_not_absorbed_by_fp64_state_rounding(fixture):
    model, case = fixture
    carry = np.zeros(6)
    carry[5] = np.spacing(case['initial'][5]) / 4
    assert case['initial'][5] - carry[5] == case['initial'][5]
    result = compare_case(model, case, case['initial'], carry)
    assert result['comparisons']['q']['absolute_error'][5] == 0.
    assert result['comparisons']['q-minus-carry']['absolute_error'][5] == abs(carry[5]) / case['rho']


@pytest.mark.parametrize('kind', ['negative-high', 'negative-represented', 'nonfinite-carry'])
def test_invalid_state_cannot_be_hidden_by_compensation(fixture, kind):
    model, case = fixture
    high, carry = case['initial'].copy(), np.zeros(6)
    if kind == 'negative-high':
        high[2], carry[2] = -1e-30, -2e-30
    elif kind == 'negative-represented':
        high[2], carry[2] = 0., 1e-30
    else:
        carry[2] = np.nan
    with pytest.raises(ValueError):
        compare_case(model, case, high, carry)
