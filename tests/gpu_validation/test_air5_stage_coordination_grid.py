"""Host-only grid and output-budget plumbing for the U1/U4 AIR5 fixture."""
import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import run_air5_performance_diagnosis as performance
import run_output_air5_restart_validation as restart
from run_output_restart_validation import controlled_checkpoint_buffers


@pytest.mark.parametrize('grid', [(32, 32, 32), (64, 64, 64), (32, 16, 24)])
def test_prepared_grid_matches_initial_field_without_changing_physics(tmp_path, grid):
    case = tmp_path / 'case'
    restart.prepare(case, 'gpu', 24, incident=True, reconstruction=3, grid=grid)
    lines = (case / 'datin/input.air5_c4').read_text().splitlines()
    marker = next(i for i, line in enumerate(lines) if 'im,jm,km' in line)
    assert lines[marker + 1] == ','.join(map(str, grid))
    initial = (case / 'datin/air5_hbl_initial_field.dat').read_text().splitlines()
    assert initial[2] == f'{grid[0]+1} {grid[1]+1}'
    values = np.loadtxt(case / 'datin/air5_hbl_initial_field.dat', skiprows=3)
    assert values.shape == ((grid[0]+1)*(grid[1]+1), 13)
    np.testing.assert_array_equal(np.unique(values[:, 0]), np.linspace(0., restart.DOMAIN[0], grid[0]+1))
    np.testing.assert_array_equal(np.unique(values[:, 1]), np.linspace(0., restart.DOMAIN[1], grid[1]+1))
    assert np.all(values[:, 2] == restart.REFERENCE_DENSITY)
    assert (case / 'datin/air5_hbl_domain.dat').read_text().splitlines()[1] == ' '.join(map(str, restart.DOMAIN))
    assert '1.d-10' in (case / 'datin/controller').read_text()


def test_omitting_grid_preserves_existing_fixture(tmp_path):
    default, explicit = tmp_path / 'default', tmp_path / 'explicit'
    restart.prepare(default, 'cpu', 2, incident=True, reconstruction=3)
    restart.prepare(explicit, 'cpu', 2, incident=True, reconstruction=3, grid=(16, 16, 16))
    files = lambda root: {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert files(default) == files(explicit)


def test_launch_passes_grid_and_explicit_host_budget(tmp_path, monkeypatch):
    args = SimpleNamespace(output=tmp_path, case='sbli', reconstruction=3, mean_statistics=False,
        sample_interval=2, mode='steps', initial_restart=False, axis='x',
        top_mode='characteristic', filter_workspace='scalar', executable='unused', mpiexec='unused',
        grid=(64, 64, 64))
    calls = []

    class NoSolver:
        def __init__(self, command, **kwargs):
            calls.append(command)

        def wait(self, timeout):
            return 0

    monkeypatch.setattr(restart.subprocess, 'Popen', NoSolver)
    case, _ = restart.launch(args, 'cpu', 1, 'host_only', 24, host_budget_bytes=256*1024**2,
                             directory_budget_bytes=1024**3)
    config = (case / 'datin/input.output').read_text()
    assert 'host_budget_bytes=268435456' in config
    assert (case / 'datin/air5_hbl_initial_field.dat').read_text().splitlines()[2] == '65 65'
    assert len(calls) == 1


def test_checkpoint_estimator_uses_declared_budget(tmp_path):
    path = tmp_path / 'synthetic.h5'
    with h5py.File(path, 'w') as state:
        identity = np.zeros(13, dtype=np.int64)
        identity[6], identity[12] = 35, 7
        state['identity'] = identity
        state['metadata'] = [1, 1]
        state['partitions'] = [0, 0, 0, 64, 64, 64, 5, 1]
    with pytest.raises(AssertionError, match='host budget exceeded'):
        controlled_checkpoint_buffers(path, 0, 'gpu')
    bounds = controlled_checkpoint_buffers(path, 0, 'gpu', budget_bytes=256*1024**2)
    assert bounds == [75**3*35*8 + 8 + 128 + 65**3*12*8]
    assert 64*1024**2 < bounds[0] < 256*1024**2


@pytest.mark.parametrize('compensated,components,auxiliary', [(0, 35, 1), (1, 34, 11)])
def test_gpu_auxiliary_staging_counts_history_independently_of_compensation(tmp_path, compensated, components, auxiliary):
    path = tmp_path / 'auxiliary.h5'
    with h5py.File(path, 'w') as state:
        identity = np.zeros(13, dtype=np.int64)
        identity[6], identity[12] = components, 7
        state['identity'], state['metadata'] = identity, [1, compensated]
        state['partitions'] = [0, 0, 0, 16, 16, 16, 5, 1]
    assert controlled_checkpoint_buffers(path, 0, 'gpu') == [27**3*components*8 + 8 + 128 + 17**3*auxiliary*8]


@pytest.mark.parametrize('from_initial', [False, True])
def test_timing_driver_records_grid_and_forwards_budget_without_running_solver(tmp_path, monkeypatch, from_initial):
    binary = tmp_path / 'unused'
    binary.touch()
    args = SimpleNamespace(output=tmp_path / 'matrix', executable=binary, mpiexec=binary,
        nsys=False, phase_timing=False, repeats=5, warmup=4, steps=20, report_interval=4,
        case='sbli', chemistry_diagnostics='every-step', source_evaluation='joint',
        chemistry_tolerance='baseline', step_history='off',
        chemistry_block_size=128, axis='x', topology=None, top_mode='characteristic',
        filter_workspace='scalar', compare_to=None, cases=['gpu_np1'], grid=(64, 64, 64),
        host_budget_bytes=256*1024**2, device_budget_bytes=128*1024**2,
        directory_budget_bytes=1024**3, start_from_initial=from_initial)
    calls = []

    def stop_before_solver(*positional, **kwargs):
        calls.append((positional, kwargs))
        raise RuntimeError('host-only test stops before solver')

    monkeypatch.setattr(performance, 'launch', stop_before_solver)
    with pytest.raises(RuntimeError, match='host-only test'):
        performance.run(args)
    summary = json.loads((args.output / 'timing_summary.json').read_text())
    assert summary['grid_intervals'] == [64, 64, 64]
    assert summary['host_budget_bytes'] == args.host_budget_bytes
    assert summary['device_budget_bytes'] == args.device_budget_bytes
    assert summary['directory_budget_bytes'] == args.directory_budget_bytes
    assert summary['timing_start_step'] == (0 if from_initial else 2)
    assert args.initial_restart == from_initial
    assert calls[0][0][-1] == (1 if from_initial else 2)
    assert calls[0][1]['host_budget_bytes'] == args.host_budget_bytes
    assert calls[0][1]['device_budget_bytes'] == args.device_budget_bytes
    assert calls[0][1]['directory_budget_bytes'] == args.directory_budget_bytes


def test_one_initial_update_has_explicitly_unemitted_scalar_statistics(tmp_path):
    actual, expected = tmp_path / 'actual', tmp_path / 'expected'
    actual.mkdir()
    expected.mkdir()
    assert performance.compare_scalar_statistics(actual, expected, single_initial_update=True) == 'not-emitted-in-first-rk-call'


@pytest.mark.parametrize('single_initial_update', [False, True])
def test_present_scalar_statistics_must_still_match(tmp_path, single_initial_update):
    actual, expected = tmp_path / 'actual', tmp_path / 'expected'
    actual.mkdir()
    expected.mkdir()
    for case in (actual, expected):
        (case / 'flowstate.dat').write_text('same scalar records\n')
    assert performance.compare_scalar_statistics(actual, expected, single_initial_update) == 'equal'
    (actual / 'flowstate.dat').write_text('changed scalar records\n')
    with pytest.raises(ValueError, match='scalar statistics differ'):
        performance.compare_scalar_statistics(actual, expected, single_initial_update)


@pytest.mark.parametrize('one_present', [False, True])
def test_missing_regular_or_unpaired_seed_statistics_remain_errors(tmp_path, one_present):
    actual, expected = tmp_path / 'actual', tmp_path / 'expected'
    actual.mkdir()
    expected.mkdir()
    if one_present:
        (actual / 'flowstate.dat').write_text('unpaired records\n')
    with pytest.raises(FileNotFoundError):
        performance.compare_scalar_statistics(actual, expected, single_initial_update=one_present)
