"""U7 engineering holdout: unchanged chemistry, later window and filter paths."""
import json

import h5py
import numpy as np
import pytest

from run_air5_performance_diagnosis import batch_identity, digest
from run_output_air5_restart_validation import launch, state_sanity
from run_output_restart_validation import compare_fields
from test_output_air5_conservation import arguments


def check_statistics_finite(path):
    with h5py.File(path) as state:
        def check(name, value):
            if isinstance(value, h5py.Dataset):
                assert np.isfinite(value[:]).all(), name
        state.visititems(check)


@pytest.mark.parametrize('value', [0., np.nan])
def test_nonfinite_statistics_are_not_exact_acceptance(tmp_path, value):
    path = tmp_path / 'statistics.h5'
    with h5py.File(path, 'w') as state:
        state['means'] = [value]
    if np.isfinite(value):
        check_statistics_finite(path)
    else:
        with pytest.raises(AssertionError, match='means'):
            check_statistics_finite(path)


def reports(case, start, stop, interval):
    rows = [s for s in (case / 'run.log').read_text().splitlines()
            if s.startswith('AIR5_CHEMISTRY half=')]
    expected = [(step, half) for step in range(start, stop)
                if step == 0 or step % interval == 0 for half in (1, 2)]
    assert len(rows) == len(expected)
    assert all(row.startswith(f'AIR5_CHEMISTRY half={half} ')
               for row, (_, half) in zip(rows, expected))
    return dict(zip(expected, rows))


@pytest.mark.parametrize('ranks,axis,case,top,workspace', [
    (1, 'x', 'sbli', 'characteristic', 'scalar'),
    (2, 'x', 'sbli', 'characteristic', 'scalar'),
    (2, 'z', 'sbli', 'characteristic', 'full'),
    (2, 'y', 'hbl', 'prescribed', 'scalar'),
    (2, 'x', 'hbl', 'prescribed', 'full'),
], ids=['np1-incident-scalar', 'np2-x-incident-scalar', 'np2-z-incident-full',
        'np2-y-prescribed-scalar', 'np2-x-prescribed-full'])
def test_combination_later_window_exact_restart(tmp_path, record_property, ranks, axis, case, top, workspace):
    args = arguments(tmp_path, case=case, axis=axis, workspace=workspace)
    args.top_mode, args.report_interval = top, 4
    binary_hash = digest(args.executable)
    baseline, baseline_size = launch(args, 'gpu', ranks, 'baseline', 24, interval=24)
    args.source_evaluation, args.chemistry_block_size = 'source-only', 32
    options = dict(chemistry_diagnostics='report-steps')
    candidate, candidate_size = launch(args, 'gpu', ranks, 'combined', 24, interval=24, **options)
    first, first_size = launch(args, 'gpu', ranks, 'first', 17, interval=17, **options)
    seed = first / 'outdat/new/checkpoints/step000000000017'
    identity = batch_identity(seed)
    with h5py.File(seed / 'state.h5') as state:
        nonzero_carry = sum(np.count_nonzero(state[f'q{m:04}'][:]) for m in range(12, 23))
        assert nonzero_carry > 0
    resumed, resumed_size = launch(args, 'gpu', ranks, 'resumed', 24, restore=seed,
                                   interval=24, override=True, **options)
    relative = 'outdat/new/checkpoints/step000000000024'
    equal = {}
    for name in ('state.h5', 'statistics.h5'):
        equal[name] = compare_fields(baseline / relative / name, candidate / relative / name)
        compare_fields(candidate / relative / name, resumed / relative / name)
    for actual in (baseline, candidate, resumed):
        check_statistics_finite(actual / relative / 'statistics.h5')
    assert (baseline / 'flowstate.dat').read_bytes() == (candidate / 'flowstate.dat').read_bytes()
    full = np.loadtxt(candidate / 'flowstate.dat', skiprows=1, ndmin=2)
    tail = np.loadtxt(resumed / 'flowstate.dat', skiprows=1, ndmin=2)
    assert full[full[:, 0] >= 17].tobytes() == tail.tobytes()
    expected_reports = reports(baseline, 0, 24, 4)
    assert reports(candidate, 0, 24, 4) == expected_reports
    assert reports(resumed, 17, 24, 4) == {key: row for key, row in expected_reports.items() if key[0] >= 17}
    assert (candidate / relative / 'air5_config.bin').read_bytes() == (
        resumed / relative / 'air5_config.bin').read_bytes()
    assert batch_identity(seed) == identity and digest(args.executable) == binary_hash
    sizes = [baseline_size, candidate_size, first_size, resumed_size]
    assert max(sizes) <= 64 * 1024**2
    evidence = dict(scope='24-step engineering holdout; not production or physical validation',
        executable_sha256=binary_hash, ranks=ranks, axis=axis, case=case,
        top_mode=top, filter_workspace=workspace, continuous_steps=24, restart_step=17,
        baseline='joint/every-step/128', candidate='source-only/report-steps/32',
        exact_datasets=equal, nonzero_restart_carry=int(nonzero_carry),
        source_checkpoint_sha256=identity, directory_bytes=sizes,
        sanity=state_sanity(resumed / relative / 'state.h5'))
    (tmp_path / 'holdout.json').write_text(json.dumps(evidence, indent=2, allow_nan=False) + '\n')
    record_property('holdout_evidence', json.dumps(evidence, allow_nan=False))
