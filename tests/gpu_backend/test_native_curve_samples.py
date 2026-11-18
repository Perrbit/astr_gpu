import numpy as np
import pytest

from run_cpp_native_curve_gate import validate_scalar_samples


def samples():
    # These are the two actual statistic formats for the same FP32 time.
    cpu=np.array([[1.,float('0.1999999949476E-04'),.125,.375,.001]])
    gpu=np.array([[1.,float('1.9999999494757503E-05'),.125,.375,.001]])
    clocks=np.array([[1.,0.,float(np.float32(2e-5)),float(np.float32(2e-5)),float(np.float32(2e-5))]])
    return cpu,gpu,clocks


def test_native_time_identity_survives_statistic_text_format():
    cpu,gpu,clocks=samples()
    assert cpu[0,1]!=gpu[0,1]
    validate_scalar_samples(cpu,gpu,clocks)


def test_sample_step_difference_is_not_text_rounding():
    cpu,gpu,clocks=samples();cpu[0,0]=2
    with pytest.raises(AssertionError):validate_scalar_samples(cpu,gpu,clocks)


def test_different_native_time_is_rejected():
    cpu,gpu,clocks=samples();cpu[0,1]=float(np.float32(3e-5))
    with pytest.raises(AssertionError):validate_scalar_samples(cpu,gpu,clocks)


def test_both_tables_must_match_complete_step_clock():
    cpu,gpu,clocks=samples();cpu[0,1]=gpu[0,1];clocks[0,2]=float(np.float32(3e-5))
    with pytest.raises(AssertionError):validate_scalar_samples(cpu,gpu,clocks)
