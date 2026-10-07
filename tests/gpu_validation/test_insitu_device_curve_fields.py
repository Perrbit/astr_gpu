"""Actual periodic CURVE private GPU gradients/Q against controlled references.

These explicit diagnostic downloads do not admit CURVE device rendering.
"""
import json
import re

import numpy as np
import pytest

from run_output_restart_validation import run_case, compare_fields
from run_insitu_gpu_derivatives import configuration
from test_insitu_curve_derivatives import ROOT, FINAL, LIBRARY, DERIVED, index_coordinates, reference
from test_insitu_device_curve_wall_fields import current_arguments
from test_insitu_is4 import compare_numerical


@pytest.mark.parametrize('ranks,axis', [(1, 'x'), (2, 'x'), (2, 'y'), (2, 'z')],
                         ids=['single', 'x', 'y', 'z'])
@pytest.mark.parametrize('mode', ['pinned', 'device-aware'])
def test_real_periodic_curve_device_fields(tmp_path, ranks, axis, mode, record_property):
    args = current_arguments(tmp_path, axis=axis, samples=False)
    config = configuration('gpu', LIBRARY, ROOT / 'tests/gpu_validation/insitu_diagnostic_capture.py')
    kwargs = dict(grid='32,32,32', tgv_mapping='periodic', checkpoint_interval=1)
    actual, _ = run_case(args, ROOT, 'gpu', ranks, 'fields', 4, insitu_config=config,
                         device_sample_transport=mode, **kwargs)
    errors = re.findall(r'ASTR_INSITU_DEVICE_SAMPLE_CHECK rank=\d+ backend=\S+ maxabs=\s*(\S+)',
                        (actual / 'run.log').read_text())
    assert len(errors) == ranks * 3
    assert all(np.isfinite(float(error)) and float(error) <= 2e-10 for error in errors)
    independent = reference(actual, 4)
    maxima = {}
    for rank in range(ranks):
        idx = index_coordinates(ranks, axis, rank)
        with np.load(actual / f'outdat/render/fields.step00000004.rank{rank:08d}.npz') as fields:
            for name, expected in zip(DERIVED, independent):
                error = float(np.max(abs(fields[name] - expected[idx[2], idx[1], idx[0]])))
                assert error <= 2e-10, (rank, name, error)
                maxima[name] = max(maxima.get(name, 0.), error)
    cpu, _ = run_case(current_arguments(tmp_path, 'cpu', axis, samples=False), ROOT,
                      'cpu', ranks, 'cpu', 4, **kwargs)
    record_property('cpu_gpu_state_maxabs', compare_numerical(cpu / FINAL / 'state.h5', actual / FINAL / 'state.h5'))
    record_property('same_state_private_gradient_q_maxabs', max(map(float, errors)))
    record_property('independent_errors', json.dumps(maxima, sort_keys=True))
    source = actual / 'outdat/new/checkpoints/step000000000003'
    resumed, _ = run_case(args, ROOT, 'gpu', ranks, 'restart', 4, restore=source,
                          insitu_config=config, device_sample_transport=mode, **kwargs)
    compare_fields(actual / FINAL / 'state.h5', resumed / FINAL / 'state.h5')
    for name in ('control.bin', 'insitu_control.bin'):
        assert (actual / FINAL / name).read_bytes() == (resumed / FINAL / name).read_bytes()
    for rank in range(ranks):
        name = f'outdat/render/fields.step00000004.rank{rank:08d}.npz'
        with np.load(actual / name) as a, np.load(resumed / name) as b:
            assert set(a.files) == set(b.files)
            for field in a.files:
                np.testing.assert_array_equal(a[field], b[field])
    off, _ = run_case(args, ROOT, 'gpu', ranks, 'off', 4, **kwargs)
    compare_fields(actual / FINAL / 'state.h5', off / FINAL / 'state.h5')
