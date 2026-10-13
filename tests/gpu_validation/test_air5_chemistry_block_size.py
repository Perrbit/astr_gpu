"""One-warp blocks preserve per-node chemistry and exact continuation."""
import numpy as np
import pytest

from run_output_air5_restart_validation import launch
from run_output_restart_validation import compare_fields
from test_output_air5_conservation import arguments


@pytest.mark.parametrize('ranks,axis', [(1, 'x'), (2, 'x'), (2, 'y'), (2, 'z')],
                         ids=['np1', 'np2-x', 'np2-y', 'np2-z'])
def test_block_size_preserves_state_reports_and_exact_restart(tmp_path, ranks, axis):
    args = arguments(tmp_path, case='sbli', axis=axis)
    args.mean_statistics = False
    full, _ = launch(args, 'gpu', ranks, 'baseline', 8)
    first, _ = launch(args, 'gpu', ranks, 'first', 3)
    args.chemistry_block_size = 32
    candidate, _ = launch(args, 'gpu', ranks, 'block32', 8)
    seed = first / 'outdat/new/checkpoints/step000000000003'
    resumed, _ = launch(args, 'gpu', ranks, 'resumed32', 8, restore=seed)
    relative = 'outdat/new/checkpoints/step000000000008'
    for actual in (candidate, resumed):
        compare_fields(full / relative / 'state.h5', actual / relative / 'state.h5')
        assert (full / relative / 'air5_config.bin').read_bytes() == (actual / relative / 'air5_config.bin').read_bytes()
        assert 'block_size=32' in (actual / 'run.log').read_text()
    assert (full / 'flowstate.dat').read_bytes() == (candidate / 'flowstate.dat').read_bytes()
    a = np.loadtxt(full / 'flowstate.dat', skiprows=1, ndmin=2)
    b = np.loadtxt(resumed / 'flowstate.dat', skiprows=1, ndmin=2)
    assert a[a[:, 0] >= 3].tobytes() == b.tobytes()
    reports = lambda path: [s for s in (path / 'run.log').read_text().splitlines()
                           if s.startswith('AIR5_CHEMISTRY half=')]
    assert reports(full) == reports(candidate)


def test_invalid_block_choice_is_rejected(tmp_path):
    args = arguments(tmp_path, case='sbli')
    args.mean_statistics = False
    args.chemistry_block_size = 17
    launch(args, 'gpu', 1, 'invalid', 1, checkpoint_enabled=False,
           reject='invalid or rank-inconsistent AIR5 chemistry performance options')
