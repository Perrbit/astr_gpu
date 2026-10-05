"""Actual RK fields: device sampling against the controlled-host GPU reference.

This bounded diagnostic explicitly downloads its oracle; it does not qualify
the still-pending device geometry/rendering bridge or a zero-volume-transfer run.
"""
import re

import h5py
import numpy as np
import pytest

from run_output_restart_validation import run_case
from test_output_insitu_restart import ROOT, arguments, configuration


@pytest.mark.parametrize('ranks,axis', [(1, 'x'), (2, 'x'), (2, 'y'), (2, 'z')])
@pytest.mark.parametrize('mode', ['pinned', 'device-aware'])
@pytest.mark.parametrize('statistics', [False, True])
def test_completed_step_device_sample(tmp_path, ranks, axis, mode, statistics):
    args = arguments(tmp_path, axis)
    args.statistics = statistics
    args.directory_budget_bytes = 256 * 1024**2
    profile = 'all' if statistics else 'q_surface'
    config = configuration(statistics=statistics, interval=1).replace('render=.true.,',
        f"render=.true.,derivative_backend='gpu',products='{profile}',")
    case, _ = run_case(args, ROOT, 'gpu', ranks, 'device_sample', 2, grid='32,32,32',
                       insitu_config=config, device_sample_transport=mode)
    log = (case / 'run.log').read_text()
    errors = re.findall(r'ASTR_INSITU_DEVICE_SAMPLE_CHECK rank=\d+ backend=\S+ maxabs=\s*(\S+)', log)
    assert len(errors) == ranks * 2, log
    assert all(np.isfinite(float(error)) and float(error) <= 2e-10 for error in errors)
    if statistics:
        mean_errors = re.findall(r'ASTR_INSITU_DEVICE_MEAN_CHECK rank=\d+ maxabs=\s*(\S+)', log)
        assert len(mean_errors) == ranks * 2, log
        assert all(np.isfinite(float(error)) and float(error) <= 2e-10 for error in mean_errors)
    control, _ = run_case(args, ROOT, 'gpu', ranks, 'no_diagnostic', 2, grid='32,32,32',
                          insitu_config=config)
    files = ['state.h5', 'statistics.h5'] if statistics else ['state.h5']
    for filename in files:
        final = 'outdat/new/checkpoints/step000000000002/' + filename
        with h5py.File(case / final) as actual, h5py.File(control / final) as expected:
            names = []
            actual.visititems(lambda name, value: names.append(name) if isinstance(value, h5py.Dataset) else None)
            for name in names:
                np.testing.assert_array_equal(actual[name][...], expected[name][...], err_msg=name)


@pytest.mark.parametrize('mode', ['pinned', 'device-aware'])
def test_device_means_memory_and_zero_coverage(tmp_path, mode):
    args = arguments(tmp_path, 'x')
    args.directory_budget_bytes = 256 * 1024**2
    config = configuration(statistics=True, interval=1).replace('render=.true.,',
        "render=.true.,derivative_backend='gpu',").replace('initial_frame=f','initial_frame=t')
    case, _ = run_case(args, ROOT, 'gpu', 2, 'mean_memory', 2, grid='32,32,32',
                       insitu_config=config, device_sample_transport=mode, memcheck=True)
    log = (case / 'run.log').read_text()
    assert log.count('ASTR_INSITU_DEVICE_MEAN_CHECK uncovered') == 1, log
    reports = list(case.glob('memcheck.*.log'))
    assert len(reports) == 2
    for report in reports:
        assert 'ERROR SUMMARY: 0 errors' in report.read_text(), report


def test_formal_device_transport_prebind_is_independent(tmp_path):
    args = arguments(tmp_path, 'x')
    args.directory_budget_bytes = 256 * 1024**2
    config = configuration(statistics=False, interval=1).replace('render=.true.,',
        "render=.true.,derivative_backend='gpu',processing_backend='device',postprocess_transport='device-aware',")
    cache = args.executable.parent.parent / 'CMakeCache.txt'
    enabled = 'ASTR_WITH_INSITU_DEVICE:BOOL=ON' in cache.read_text()
    message = ('device products require 32^3 periodic Cartesian FP64 643e TGV NP=1/2' if enabled else
               'IS8 device processing is not connected to the native product bridge yet')
    case, _ = run_case(args, ROOT, 'gpu', 2, 'independent_prebind', 1, grid='16,16,16',
        insitu_config=config, postprocess_transport='device-aware', reject=message)
    log = (case / 'run.log').read_text()
    assert 'ASTR_GPU_PRE_MPI_DEVICE=0 local_rank=0' in log, log
    assert 'ASTR_GPU_PRE_MPI_DEVICE=1 local_rank=1' in log, log
