"""Replay an observed adaptive CPU clock for attribution, not GPU auto acceptance."""
import json
import os
from pathlib import Path
import re

import numpy as np
import pytest

from compare_flowstate import read_flowstate
from run_adaptive_tgv_gate import STEP
from test_high_order_boundary_runtime import EXE,H1_VARIANT,launch_hcw
from test_h1_hcw_inlet_compatible import state

REFERENCE=Path(os.environ.get('ASTR_H1_COMMON_CLOCK_REFERENCE',
    Path(__file__).parent/'out/h1_inlet_compatible_long_20261105/adaptive_early/test_hcw_early_trajectory_gate0/cpu')).resolve()
LIBRARY=os.environ.get('ASTR_H1_CONTROL_REPLAY_LIBRARY')
pytestmark=pytest.mark.skipif(not EXE or H1_VARIANT!='inlet-compatible' or not LIBRARY
    or os.environ.get('ASTR_H1_HCW_LONG')!='1',reason='Select the approved common-clock attribution experiment')


def clocks(case):
    return np.array([[int(match[1]),*map(float,match.groups()[1:])]
        for match in STEP.finditer((case/'run.log').read_text())])


def replay(case,backend,reference):
    launch_hcw(case,backend,steps=len(reference),replay_clocks=reference,replay_library=LIBRARY)
    actual=clocks(case)
    np.testing.assert_array_equal(actual,reference)
    reads=[int(value) for value in re.findall(r'ASTR_CONTROLLER_TEST_READ completed_step=(\d+)',
                                             (case/'run.log').read_text())]
    assert reads==list(range(len(reference)+1)),reads
    return state(case,len(reference))


def test_common_clock_prefix(tmp_path):
    reference=clocks(REFERENCE)[:12]
    values={}
    for backend in ('cpu','gpu'):
        values[backend]=replay(tmp_path/backend,backend,reference)
    names,source=read_flowstate(REFERENCE)
    other,replayed=read_flowstate(tmp_path/'cpu')
    assert names==other
    np.testing.assert_array_equal(replayed,source[:len(replayed)])
    error=float((abs(values['gpu']-values['cpu'])/(1+abs(values['cpu']))).max())
    (tmp_path/'prefix.json').write_text(json.dumps(dict(steps=12,
        cpu_source_statistics_exact=True,common_clocks_exact=True,maximum_scaled_field_error=error),indent=2)+'\n')
    assert error<=2e-10,error


def compare_common_clock_420(case):
    reference=clocks(REFERENCE)
    np.testing.assert_array_equal(reference[:,0],np.arange(1,421))
    expected=state(REFERENCE,420)
    actual=state(case/'gpu',420)
    np.testing.assert_array_equal(clocks(case/'gpu'),reference)
    errors=[float((abs(actual[...,c]-expected[...,c])/(1+abs(expected[...,c]))).max()) for c in range(11)]
    names,source=read_flowstate(REFERENCE)
    other,replayed=read_flowstate(case/'gpu')
    assert names==other
    # CPU listwrite and GPU ES24.16 print different time precision; STEP is exact above.
    np.testing.assert_array_equal(source[:,0],replayed[:,0])
    statistics_error=float(abs(source[:,2:]-replayed[:,2:]).max())
    (case/'common_clock420.json').write_text(json.dumps(dict(
        scope='CPU automatic dt sequence replayed through test-only controller inputs; not independent GPU auto',
        reference=str(REFERENCE),steps=420,final_time=float(reference[-1,2]),common_clocks_exact=True,
        statistics_time_text_difference=float(abs(source[:,1]-replayed[:,1]).max()),
        scaled_field_errors=errors,maximum_statistics_absolute_difference=statistics_error),indent=2)+'\n')
    assert max(errors)<=2e-10,errors
    assert statistics_error<=2e-10,statistics_error


def test_common_clock_420(tmp_path):
    replay(tmp_path/'gpu','gpu',clocks(REFERENCE))
    compare_common_clock_420(tmp_path)
