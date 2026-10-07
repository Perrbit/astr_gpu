"""Real CURVE bc41 fields against independent geometry and completed-state oracles.

The diagnostic explicitly downloads bounded reference arrays; it does not admit
CURVE rendering or claim zero-readback production statistics.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from run_output_restart_validation import run_case, compare_fields
from test_insitu_curve_derivatives import arguments, ROOT, FINAL
from test_insitu_curve_walls import check_walls
from test_insitu_device_channel_fields import checks
from test_insitu_is4 import compare_numerical


def current_arguments(output, backend='gpu', axis='x', samples=True):
    args = arguments(output, backend, axis)
    args.executable = Path(os.environ.get('ASTR_OUTPUT_INSITU_EXE', ROOT / 'build_insitu_device_render/bin/astr'))
    args.wall_samples = samples
    return args


@pytest.mark.parametrize('ranks,axis', [(1, 'x'), (2, 'x'), (2, 'y'), (2, 'z')],
                         ids=['single', 'x', 'y', 'z'])
@pytest.mark.parametrize('mode', ['pinned', 'device-aware'])
def test_real_curve_bc41_device_fields(tmp_path, ranks, axis, mode, record_property):
    args = current_arguments(tmp_path, axis=axis)
    kwargs = dict(grid='32,32,32', tgv_mapping='y-wavy', checkpoint_interval=1)
    gpu, _ = run_case(args, ROOT, 'gpu', ranks, 'fields', 4,
                      device_sample_transport=mode, **kwargs)
    error = checks(gpu, ranks, 4, mode)
    actual, independent_error = check_walls(gpu, ranks, 4)
    cpu, _ = run_case(current_arguments(tmp_path, 'cpu', axis), ROOT,
                      'cpu', ranks, 'cpu', 4, **kwargs)
    expected, cpu_oracle_error = check_walls(cpu, ranks, 4)
    field_error = float(np.max(abs(actual - expected)))
    assert field_error <= 2e-10
    record_property('same_state_device_wall_maxabs', error)
    record_property('independent_wall_maxabs', max(independent_error, cpu_oracle_error))
    record_property('cpu_gpu_wall_maxabs', field_error)
    record_property('cpu_gpu_state_maxabs', compare_numerical(cpu / FINAL / 'state.h5', gpu / FINAL / 'state.h5'))
    source = gpu / 'outdat/new/checkpoints/step000000000003'
    before = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    restarted, _ = run_case(args, ROOT, 'gpu', ranks, 'restart', 4,
                            restore=source, device_sample_transport=mode, **kwargs)
    checks(restarted, ranks, 1, mode)
    for name in ('state.h5', 'control.bin', 'insitu_control.bin'):
        if name.endswith('.h5'):
            compare_fields(gpu / FINAL / name, restarted / FINAL / name)
        else:
            assert (gpu / FINAL / name).read_bytes() == (restarted / FINAL / name).read_bytes()
    for rank in range(ranks):
        name = f'sample.wall.step00000004.rank{rank:08d}.bin'
        assert (gpu / 'outdat' / name).read_bytes() == (restarted / 'outdat' / name).read_bytes()
    assert before == {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    off, _ = run_case(current_arguments(tmp_path, axis=axis, samples=False), ROOT,
                      'gpu', ranks, 'off', 4, **kwargs)
    compare_fields(gpu / FINAL / 'state.h5', off / FINAL / 'state.h5')
    assert not list((off / 'outdat').glob('sample.wall.*'))


def test_real_curve_bc41_device_fields_memcheck(tmp_path):
    args = current_arguments(tmp_path, axis='x')
    case, _ = run_case(args, ROOT, 'gpu', 2, 'memcheck', 2,
                       grid='32,32,32', tgv_mapping='y-wavy', checkpoint_interval=1,
                       device_sample_transport='device-aware', memcheck=True)
    checks(case, 2, 2, 'device-aware')
    reports = list(case.glob('memcheck.*.log'))
    assert len(reports) == 2
    for report in reports:
        assert 'ERROR SUMMARY: 0 errors' in report.read_text(), report
