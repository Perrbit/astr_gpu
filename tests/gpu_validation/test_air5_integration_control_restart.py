"""U3 full-field control state: advisory history and exact same-topology restart."""
import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from run_air5_performance_diagnosis import batch_identity, digest
from air5_checkpoint_contract import read_air5_contract
from run_output_air5_restart_validation import launch, state_sanity, DT
from run_output_restart_validation import compare_fields
from test_air5_source_evaluation_restart import modified_batch
from test_air5_stage_coordination_holdout import check_statistics_finite
from test_output_air5_conservation import arguments, FINAL
from test_output_series_repair import seal_file_records, repair


CASES = [
    (backend, ranks, axis, tolerance, history, workspace, combined)
    for backend in ('cpu', 'gpu')
    for ranks, axis, tolerance, history, workspace, combined in (
        (1, 'x', 'baseline', 'previous', 'scalar', False),
        (2, 'x', 'loose', 'previous', 'scalar', False),
        (2, 'y', 'tight', 'previous', 'full', False),
        (2, 'z', 'baseline', 'previous', 'scalar', False),
        (1, 'x', 'loose', 'off', 'scalar', False),
    )
] + [('gpu', 1, 'x', 'loose', 'previous', 'scalar', True)]


def history_summary(path):
    with h5py.File(path) as state:
        values = state['q0035'][:]
        assert np.isfinite(values).all() and np.all(values >= 0)
        assert np.all(values <= .5 * DT)
        nondefault = np.count_nonzero((values > 0) & (values < .5 * DT))
        assert nondefault > 0, 'restart must exercise history different from the initial half duration'
        # The upper y physical face is advanced by the dynamic-top chemistry launch.
        assert np.count_nonzero(values[:, -1, :]) > 0
        return dict(nondefault_nodes=int(nondefault), positive_nodes=int(np.count_nonzero(values)),
                    minimum=float(values.min()), maximum=float(values.max()))


@pytest.fixture(scope='module', params=CASES,
    ids=['-'.join(map(str, row)) for row in CASES])
def reference(request, tmp_path_factory):
    backend, ranks, axis, tolerance, history, workspace, combined = request.param
    args = arguments(tmp_path_factory.mktemp('integration'), case='sbli', axis=axis, workspace=workspace)
    args.chemistry_tolerance, args.step_history = tolerance, history
    options = {}
    if combined:
        args.source_evaluation, args.chemistry_block_size, args.report_interval = 'source-only', 32, 4
        options['chemistry_diagnostics'] = 'report-steps'
    full, full_size = launch(args, backend, ranks, 'continuous', 12, **options)
    first, first_size = launch(args, backend, ranks, 'first', 5, **options)
    seed = first / 'outdat/new/checkpoints/step000000000005'
    if history == 'previous':
        history_summary(seed / 'state.h5')
    return args, backend, ranks, full, seed, options, [full_size, first_size]


def test_full_field_control_exact_restart(reference, tmp_path, record_property):
    args, backend, ranks, full, seed, options, sizes = reference
    args = SimpleNamespace(**vars(args))
    args.output = tmp_path
    source = batch_identity(seed)
    resumed, size = launch(args, backend, ranks, 'resumed', 12, restore=seed, **options)
    equal = {}
    for name in ('state.h5', 'statistics.h5'):
        equal[name] = compare_fields(full / FINAL / name, resumed / FINAL / name)
    check_statistics_finite(resumed / FINAL / 'statistics.h5')
    engineering_equal = {}
    if options:
        control = SimpleNamespace(**vars(args))
        control.source_evaluation, control.chemistry_block_size = 'joint', 128
        joint, joint_size = launch(control, backend, ranks, 'engineering_control', 12)
        for name in ('state.h5', 'statistics.h5'):
            engineering_equal[name] = compare_fields(full / FINAL / name, joint / FINAL / name)
        assert (full / 'flowstate.dat').read_bytes() == (joint / 'flowstate.dat').read_bytes()
        sizes = [*sizes, joint_size]
    config = (resumed / FINAL / 'air5_config.bin').read_bytes()
    assert config == (full / FINAL / 'air5_config.bin').read_bytes()
    assert len(config) == 1784
    assert read_air5_contract(config)['integration_control'] == (
        {'baseline': 0, 'loose': 1, 'tight': 2}[args.chemistry_tolerance],
        int(args.step_history == 'previous'))
    full_stats = np.loadtxt(full / 'flowstate.dat', skiprows=1, ndmin=2)
    resumed_stats = np.loadtxt(resumed / 'flowstate.dat', skiprows=1, ndmin=2)
    assert full_stats[full_stats[:, 0] >= 5].tobytes() == resumed_stats.tobytes()
    history = history_summary(resumed / FINAL / 'state.h5') if args.step_history == 'previous' else None
    if history is None:
        with h5py.File(resumed / FINAL / 'state.h5') as state:
            assert 'q0035' not in state
    assert source == batch_identity(seed)
    evidence = dict(scope='12-step same-backend/same-topology lifecycle, not flow accuracy qualification',
        executable_sha256=digest(args.executable), backend=backend, ranks=ranks, axis=args.axis,
        tolerance=args.chemistry_tolerance, step_history=args.step_history, engineering_combination=bool(options),
        filter_workspace=args.filter_workspace, exact_datasets=equal, restart_step=5, history=history,
        engineering_equal_datasets=engineering_equal,
        source_checkpoint_sha256=source, directory_bytes=[*sizes, size],
        sanity=state_sanity(resumed / FINAL / 'state.h5'))
    (tmp_path / 'integration_control.json').write_text(json.dumps(evidence, indent=2, allow_nan=False) + '\n')
    record_property('integration_control', json.dumps(evidence, allow_nan=False))


@pytest.fixture(scope='module')
def fault_seed(tmp_path_factory):
    args = arguments(tmp_path_factory.mktemp('integration_faults'), case='sbli')
    args.mean_statistics = False
    args.step_history = 'previous'
    first, _ = launch(args, 'gpu', 1, 'first', 5)
    return args, first / 'outdat/new/checkpoints/step000000000005'


@pytest.mark.parametrize('tolerance,history', [('loose', 'previous'), ('baseline', 'off')])
def test_control_change_rejected(fault_seed, tmp_path, tolerance, history):
    original, seed = fault_seed
    args = SimpleNamespace(**vars(original))
    args.output, args.chemistry_tolerance, args.step_history = tmp_path, tolerance, history
    launch(args, 'gpu', 1, 'changed', 6, restore=seed, reject='AIR5 integration control contract mismatch')


@pytest.mark.parametrize('fault', ['missing_config', 'partial_config', 'missing_field', 'negative'])
def test_history_state_not_silently_reinitialized(fault_seed, tmp_path, fault):
    original, seed = fault_seed
    args = SimpleNamespace(**vars(original))
    args.output = tmp_path
    config = (seed / 'air5_config.bin').read_bytes()
    payload = config[:-16] if fault == 'missing_config' else config[:-1] if fault == 'partial_config' else config
    batch = modified_batch(seed, tmp_path / 'modified', payload)
    if fault in ('missing_field', 'negative'):
        with h5py.File(batch / 'state.h5', 'r+') as state:
            if fault == 'missing_field':
                del state['q0035']
            else:
                values = state['q0035'][:]
                values[tuple(n // 2 for n in values.shape)] = -1.
                state['q0035'][:] = values
        names = [row.split()[0] for row in (batch / 'MANIFEST').read_text().splitlines()[2:]]
        (batch / 'MANIFEST').write_text(seal_file_records(batch, names, 'ASTR_CHECKPOINT_BUNDLE 1'))
        size, crc = repair.fingerprint(batch / 'MANIFEST')
        (batch / 'COMPLETE').write_text(f'ASTR_COMPLETE_1 {size} {crc:016X}\n')
    message = {'missing_config': 'AIR5 integration control contract mismatch',
               'partial_config': 'AIR5 configuration size/tail',
               'missing_field': 'open dataset q0035',
               'negative': 'AIR5 invalid restored step history'}[fault]
    launch(args, 'gpu', 1, fault, 6, restore=batch, reject=message)


@pytest.mark.parametrize('backend', ['cpu', 'gpu'])
def test_history_without_compensation(backend, tmp_path):
    args = arguments(tmp_path, case='sbli')
    args.mean_statistics = False
    args.step_history, args.compensation = 'previous', 'off'
    full, _ = launch(args, backend, 1, 'continuous', 2)
    first, _ = launch(args, backend, 1, 'first', 1)
    seed = first / 'outdat/new/checkpoints/step000000000001'
    resumed, _ = launch(args, backend, 1, 'resumed', 2, restore=seed)
    relative = 'outdat/new/checkpoints/step000000000002/state.h5'
    compare_fields(full / relative, resumed / relative)
    history_summary(resumed / relative)


def test_full_field_history_memcheck(tmp_path):
    args = arguments(tmp_path, case='sbli', axis='y', workspace='full')
    args.chemistry_tolerance, args.step_history = 'loose', 'previous'
    args.source_evaluation, args.chemistry_block_size = 'source-only', 32
    args.report_interval = 4
    options = dict(chemistry_diagnostics='report-steps')
    first, first_size = launch(args, 'gpu', 2, 'first', 5, **options)
    seed = first / 'outdat/new/checkpoints/step000000000005'
    history_summary(seed / 'state.h5')
    resumed, size = launch(args, 'gpu', 2, 'memcheck', 6, restore=seed, memcheck=True, **options)
    history_summary(resumed / 'outdat/new/checkpoints/step000000000006/state.h5')
    (tmp_path / 'memcheck.json').write_text(json.dumps(dict(executable_sha256=digest(args.executable),
        backend='gpu', ranks=2, topology=[1, 2, 1], filter_workspace='full',
        tolerance='loose', history='previous', source_evaluation='source-only', block_size=32,
        directory_bytes=[first_size, size], memcheck_errors=0), indent=2) + '\n')
