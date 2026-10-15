"""T1 scale calculation uses source estimates, never species clipping."""
from types import SimpleNamespace
import subprocess

import numpy as np
import pytest

import run_air5_trace_scale_probe as probe
from run_air5_trace_scale_probe import absolute_tolerances, trace_scales


def test_birth_scale_uses_second_source_and_preserves_exact_zero_input():
    initial = np.array([.75, .25, 0., 0., 0., 100.])
    case = dict(initial=initial, rho=1., momentum=np.zeros(3), q5=1000., duration=1e-5)
    sources = iter((np.array([-2., -3., 2., 3., 0., 4.]),
                    np.array([-2., -3., 2., 3., 1e-8, 4.])))
    model = SimpleNamespace(rhs=lambda *args: next(sources))
    actual = trace_scales(model, case)
    np.testing.assert_allclose(actual, [.75, .25, 2e-5, 3e-5, 1e-13, 100.], rtol=1e-15)
    np.testing.assert_array_equal(initial, [.75, .25, 0., 0., 0., 100.])


def test_no_birth_uses_existing_density_scale_not_positive_species_floor():
    initial = np.array([.75, .25, 0., 0., 0., 100.])
    case = dict(initial=initial, rho=1., momentum=np.zeros(3), q5=1000., duration=1e-5)
    model = SimpleNamespace(rhs=lambda *args: np.zeros(6))
    np.testing.assert_array_equal(trace_scales(model, case), [.75, .25, 1., 1., 1., 100.])
    assert np.all(initial[2:5] == 0.)


def test_invalid_predictor_is_not_clipped():
    case = dict(initial=np.array([.75, .25, 0., 0., 0., 100.]), rho=1.,
                momentum=np.zeros(3), q5=1000., duration=1e-5)
    model = SimpleNamespace(rhs=lambda *args: np.array([0., 0., -1., 0., 0., 0.]))
    with pytest.raises(ValueError, match='predictor is not admissible'):
        trace_scales(model, case)


def test_underflowed_absolute_tolerance_is_not_replaced_with_floor():
    with pytest.raises(ValueError, match='no floor'):
        absolute_tolerances(np.full(6, np.nextafter(0., 1.)), 1e-13)


def test_failed_probe_retains_native_status_without_accepting_rolled_back_endpoint(tmp_path):
    path = tmp_path/'failed.out'
    rows = np.zeros((2, 19))
    rows[:, 0] = [1, 2]
    rows[0, 1:6] = [11, 199990, 10, 400000, 200000]
    np.savetxt(path, rows)
    failure = probe.failure_record(path, subprocess.CalledProcessError(1, ['native-probe']))
    assert failure['returncode'] == 1
    assert failure['native_identity_status_counts'][0] == [1, 11, 199990, 10, 400000, 200000]
    assert failure['endpoints_not_accepted'] is True
