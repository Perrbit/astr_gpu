"""Timing parser tests do not execute the solver or touch a GPU."""
import pytest

from run_insitu_curve_performance import later_output


def test_later_output_means_insitu_not_checkpoint_hook(tmp_path):
    (tmp_path/'run.log').write_text(
        'ASTR_INSITU_STAGE_TIMING completed_output_inclusive 4 0 .0001\n'
        'ASTR_INSITU_STAGE_TIMING completed_output_inclusive 4 1 .0002\n'
        'ASTR_INSITU_STAGE_TIMING insitu_sample_inclusive 4 0 2.5\n'
        'ASTR_INSITU_STAGE_TIMING insitu_sample_inclusive 4 1 2.6\n')
    assert later_output(tmp_path, 2) == 2.6


@pytest.mark.parametrize('rows', [[], ['0 1.'], ['0 1.', '0 2.'], ['0 1.', '2 2.']])
def test_incomplete_or_duplicate_ranks_rejected(tmp_path, rows):
    (tmp_path/'run.log').write_text(''.join('ASTR_INSITU_STAGE_TIMING insitu_sample_inclusive 4 '+row+'\n'
                                          for row in rows))
    with pytest.raises(ValueError):
        later_output(tmp_path, 2)
