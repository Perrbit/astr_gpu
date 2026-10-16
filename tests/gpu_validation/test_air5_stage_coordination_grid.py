"""Host-only grid and output-budget plumbing for the U1/U4 AIR5 fixture."""
import json
import struct
from pathlib import Path
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
@pytest.mark.parametrize('filter_mode', ['legacy', 'conservative-flux'])
def test_timing_driver_records_grid_and_forwards_budget_without_running_solver(tmp_path, monkeypatch, from_initial, filter_mode):
    binary = tmp_path / 'unused'
    binary.touch()
    args = SimpleNamespace(output=tmp_path / 'matrix', executable=binary, mpiexec=binary,
        nsys=False, phase_timing=False, repeats=5, warmup=4, steps=20, report_interval=4,
        case='sbli', chemistry_diagnostics='every-step', source_evaluation='joint',
        chemistry_tolerance='baseline', step_history='off',
        chemistry_block_size=128, axis='x', topology=None, top_mode='characteristic',
        filter_workspace='scalar', compare_to=None, cases=['gpu_np1'], grid=(64, 64, 64),
        host_budget_bytes=256*1024**2, device_budget_bytes=128*1024**2,
        directory_budget_bytes=1024**3, start_from_initial=from_initial, air5_filter_mode=filter_mode)
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
    assert summary['air5_filter_mode'] == filter_mode
    assert calls[0][0][0].air5_filter_mode == filter_mode
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


@pytest.fixture
def qualified_flow(tmp_path):
    binary = tmp_path/'frozen.astr'
    binary.write_bytes(b'host-only fake executable')
    args = SimpleNamespace(output=tmp_path/'timing', executable=binary, mpiexec=binary,
        nsys=False, phase_timing=False, repeats=5, warmup=4, steps=20, report_interval=4,
        case='sbli', chemistry_diagnostics='every-step', source_evaluation='joint',
        chemistry_tolerance='baseline', step_history='off', chemistry_block_size=128,
        axis='x', topology=None, top_mode='characteristic', filter_workspace='scalar',
        compare_to=None, cases=['gpu_np1'], grid=(64, 64, 64),
        host_budget_bytes=256*1024**2, device_budget_bytes=256*1024**2,
        directory_budget_bytes=1024**3, start_from_initial=True,
        air5_filter_mode='conservative-flux', flow_control=tmp_path/'flow_control.json')
    payload = b'ASTRA502'+bytes(1632)+b'joint'.ljust(128)+struct.pack('<2q', 0, 0)
    payload += b'ASTRF001'+struct.pack('<5q', *performance.FILTER_IDENTITY)

    def batch(case, step):
        path = case/f'outdat/new/checkpoints/step{step:012d}'
        path.mkdir(parents=True)
        (path/'COMPLETE').touch()
        (path/'state.h5').write_bytes(b'host-only checkpoint identity')
        (path/'air5_config.bin').write_bytes(payload)
        return path

    case = tmp_path/'qualified_gpu_np1'
    initial, final = batch(case, 0), batch(case, 24)
    (case/'flowstate.dat').write_text('qualified scalar sequence\n')
    row = dict(backend='gpu', ranks=1, tolerance='baseline', history='off', case=str(case),
        state=str(final/'state.h5'), executable_sha256=performance.digest(binary),
        batch_files_sha256=performance.batch_identity(final),
        error=dict(within_reference_margin=True, within_candidate_error=True))
    references = {str(level): dict(state=str(batch(tmp_path/f'reference{level}', 24)/'state.h5'))
                  for level in (11, 12)}
    for ref in references.values():
        ref['batch_files_sha256'] = performance.batch_identity(Path(ref['state']).parent)
    report = dict(status='local-flow-control-error-passed-not-performance',
        executable_sha256=performance.digest(binary), grid_intervals=[64]*3, dt=restart.DT,
        complete_steps=24, air5_filter_mode='conservative-flux', cases=[row],
        mechanism_sha256=performance.digest(restart.ROOT/'chemMech/air5_kimjo12.json'),
        checker_sha256=performance.digest(restart.ROOT/'tests/gpu_validation/check_air5_flow_actual_error.py'),
        reference_summary=dict(initial=str(initial/'state.h5'),
            initial_batch_sha256=performance.batch_identity(initial), references=references))
    args.flow_control.write_text(json.dumps(report))
    return args, report, row


def test_qualified_flow_reuses_the_selected_immutable_case(qualified_flow):
    args, _, row = qualified_flow
    assert performance.qualified_flow_case(args, 'gpu_np1', performance.digest(args.executable)) == row


@pytest.mark.parametrize('change', ['binary', 'filter', 'dt', 'failed', 'control', 'checkpoint', 'reference', 'source'])
def test_qualified_flow_rejects_changed_evidence_or_protocol(qualified_flow, change):
    args, report, row = qualified_flow
    if change == 'binary':
        report['executable_sha256'] = 'changed'
    elif change == 'filter':
        args.air5_filter_mode = 'legacy'
    elif change == 'dt':
        report['dt'] *= 2
    elif change == 'failed':
        row['error']['within_candidate_error'] = False
    elif change == 'control':
        args.chemistry_tolerance = 'tight'
    elif change == 'checkpoint':
        Path(row['state']).write_bytes(b'changed candidate')
    elif change == 'reference':
        Path(report['reference_summary']['references']['12']['state']).write_bytes(b'changed reference')
    elif change == 'source':
        args.source_evaluation = 'source-only'
    args.flow_control.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        performance.qualified_flow_case(args, 'gpu_np1', performance.digest(args.executable))


@pytest.mark.parametrize('phase_timing', [False, True])
def test_qualified_timing_runs_only_one_verification_and_five_output_free_rounds(qualified_flow, monkeypatch, phase_timing):
    args, _, row = qualified_flow
    args.phase_timing = phase_timing
    calls = []

    def no_solver(options, backend, ranks, name, steps, **kwargs):
        calls.append((name, kwargs))
        case = options.output/f'{backend}_np{ranks}_{name}'
        case.mkdir()
        (case/'run.log').write_text('\n'.join(
            f'ASTR_COMPLETE_STEP_TIMING {step} 0 0.1' for step in range(24)))
        (case/'flowstate.dat').write_text('qualified scalar sequence\n')
        return case, 1024

    monkeypatch.setattr(performance, 'launch', no_solver)
    monkeypatch.setattr(performance, 'state_sanity', lambda path: dict(checked=True))
    monkeypatch.setattr(performance, 'compare_fields', lambda left, right: [f'dataset{i}' for i in range(38)])
    monkeypatch.setattr(performance, 'phase_summary', lambda *args: {})
    performance.run(args)
    assert [name for name, _ in calls] == ['verification', 'r1', 'r2', 'r3', 'r4', 'r5']
    assert calls[0][1]['complete_step_timing'] is True
    assert calls[0][1]['air5_phase_timing'] is phase_timing
    assert all(kwargs['checkpoint_enabled'] is False for _, kwargs in calls[1:])
    summary = json.loads((args.output/'timing_summary.json').read_text())
    assert summary['status'] == ('instrumented-attribution-not-performance' if phase_timing else 'bounded-timing-collected')
    reused = summary['cases']['gpu_np1']['qualified_flow_reuse']
    assert reused['source_case'] == row['case']
    assert len(reused['timer_and_restart_equal_datasets']) == 38
    assert reused['verification_is_performance_evidence'] is False


def test_timing_campaign_reserves_disk_before_launch(tmp_path, monkeypatch):
    args = SimpleNamespace(campaign_root=tmp_path, campaign_budget_bytes=8, directory_budget_bytes=9)
    monkeypatch.setattr(performance, 'launch', lambda *a, **k: pytest.fail('must not start solver'))
    with pytest.raises(RuntimeError, match='per-run disk reservation'):
        performance.budgeted_launch(args, 'gpu', 1, 'unused', 24)


def test_initial_state_comparison_uses_full_window_statistics_not_seed_run_length(qualified_flow, monkeypatch):
    args, _, row = qualified_flow
    Path(row['case'],'run.log').write_text('AIR5_CHEMISTRY half=1 qualified-window\n')

    def no_solver(options, backend, ranks, name, steps, **kwargs):
        case=options.output/f'{backend}_np{ranks}_{name}'
        case.mkdir()
        log='\n'.join(f'ASTR_COMPLETE_STEP_TIMING {step} 0 0.1' for step in range(steps))
        if steps>1:
            log+='\nAIR5_CHEMISTRY half=1 qualified-window\n'
            (case/'flowstate.dat').write_text('qualified scalar sequence\n')
        (case/'run.log').write_text(log)
        if name in ('seed','timing_check'):
            batch=case/'outdat/new/checkpoints/step000000000000'
            batch.mkdir(parents=True)
            (batch/'state.h5').write_bytes(b'host-only initial state')
            (batch/'COMPLETE').touch()
        return case,1024

    monkeypatch.setattr(performance,'launch',no_solver)
    monkeypatch.setattr(performance,'state_sanity',lambda path: dict(checked=True))
    monkeypatch.setattr(performance,'compare_fields',lambda left,right: ['state'])
    performance.run(args)
    candidate=SimpleNamespace(**vars(args))
    candidate.output=args.output.parent/'fresh_engineering'
    candidate.flow_control=None
    candidate.compare_to=args.output
    candidate.source_evaluation='source-only'
    performance.run(candidate)
    summary=json.loads((candidate.output/'timing_summary.json').read_text())
    assert summary['cases']['gpu_np1']['baseline_scalar_statistics']=={'verification':'equal'}


def test_timing_campaign_checks_actual_growth(tmp_path, monkeypatch):
    args = SimpleNamespace(campaign_root=tmp_path, campaign_budget_bytes=8, directory_budget_bytes=4)

    def too_large(*args, **kwargs):
        (tmp_path/'unexpected_growth').write_bytes(b'123456789')
        return tmp_path, 9

    monkeypatch.setattr(performance, 'launch', too_large)
    with pytest.raises(RuntimeError, match='disk budget exceeded'):
        performance.budgeted_launch(args, 'gpu', 1, 'unused', 24)


@pytest.mark.parametrize('one_present', [False, True])
def test_missing_regular_or_unpaired_seed_statistics_remain_errors(tmp_path, one_present):
    actual, expected = tmp_path / 'actual', tmp_path / 'expected'
    actual.mkdir()
    expected.mkdir()
    if one_present:
        (actual / 'flowstate.dat').write_text('unpaired records\n')
    with pytest.raises(FileNotFoundError):
        performance.compare_scalar_statistics(actual, expected, single_initial_update=one_present)
