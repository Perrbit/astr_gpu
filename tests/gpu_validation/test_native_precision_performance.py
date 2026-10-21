"""FP8 pairing and timing arithmetic; no solver or GPU is invoked here."""

import json

import pytest

from run_native_precision_performance import configurations, label, ordered_runs, summarize
from run_compact_execution import crashfix_summary, read_rank_output, window_times


def records(root):
    for case, scheme, filtering, ranks in configurations(['tgv']):
        for precision in ('fp64', 'fp32'):
            for repeat, factor in enumerate((3., 1., 2.)):
                path = root / 'timing' / f'{label(case, scheme, filtering, ranks, precision)}_r{repeat+1}'
                path.mkdir(parents=True)
                item = dict(numerical_precision=precision, profiled=False, phase_timing=False,
                    preflight=False, steps=40, warmup=10, statistics_interval=0, buffered_output=True,
                    case=case, scheme=scheme, filter=filtering,
                    topology=f'{ranks},1,1', grid_intervals=[256]*3,
                    executable_sha256=precision, workspace='scalar', compact_sync='explicit',
                    transport='pinned', batch_rhs=65535, deltat=2.5e-4, crashfix=False,
                    crashfix_repaired_node_visits=0, crashfix_nonzero_calls=0,
                    crashfix_max_pointwise_conservative_change=[0]*5,
                    rk_seconds_per_step=factor*(.002 if precision == 'fp64' else .001)/ranks)
                (path / 'summary.json').write_text(json.dumps(item))


def test_interleaving_keeps_every_configuration_and_both_precisions():
    expected = {(*item, precision) for item in configurations(['tgv', 'plate'])
                for precision in ('fp64', 'fp32')}
    for repeat in range(3):
        runs = ordered_runs(['tgv', 'plate'], repeat)
        assert len(runs) == len(expected) and set(runs) == expected
        for a, b in zip(runs[::2], runs[1::2]):
            assert a[:-1] == b[:-1]
            assert (a[-1], b[-1]) == (('fp32', 'fp64') if repeat % 2 else ('fp64', 'fp32'))


def test_repair_deltas_are_pointwise_maxima_not_conservation_sums():
    value = crashfix_summary('ASTR_GPU_CRASHFIX 0 1 2 1 2 3 4 5 .1 .2 .3 .4\n'
                            'ASTR_GPU_CRASHFIX 0 2 3 5 4 3 2 1 .2 .1 .4 .3\n')
    assert value['crashfix_repaired_node_visits'] == 5
    assert value['crashfix_nonzero_calls'] == 2
    assert value['crashfix_max_pointwise_conservative_change'] == [5, 4, 3, 4, 5]
    assert value['crashfix_reported_state_minima'] == [.1, .1, .3, .3]
    assert 'not summed' in value['crashfix_change_scope']


def test_disabled_repair_has_no_reported_state_minimum():
    value = crashfix_summary('unrelated output')
    assert value['crashfix_repaired_node_visits'] == value['crashfix_nonzero_calls'] == 0
    assert value['crashfix_max_pointwise_conservative_change'] == [0]*5
    assert value['crashfix_reported_state_minima'] is None


def test_buffered_repair_summary_preserves_the_same_quantities():
    unbuffered = crashfix_summary('ASTR_GPU_CRASHFIX 0 1 2 1 2 3 4 5 .1 .2 .3 .4\n'
                                'ASTR_GPU_CRASHFIX 0 2 3 5 4 3 2 1 .2 .1 .4 .3\n')
    buffered = crashfix_summary('ASTR_GPU_CRASHFIX_TOTAL 5 2 5 4 3 4 5 .1 .1 .3 .3\n')
    assert buffered == unbuffered


def test_rank_files_not_interleaved_mpi_console_are_the_timing_source(tmp_path):
    (tmp_path/'run.log').write_text('ASTR_GPU_RANK_RK_TIMING 0 0 0 0 AST')
    for rank, total in ((0, .2), (1, .3)):
        directory=tmp_path/f'rank_output/1/rank.{rank}'
        directory.mkdir(parents=True)
        (directory/'stdout').write_text(f'ASTR_GPU_RANK_RK_TIMING {rank} 0 0 0 {total}\n')
        (directory/'stderr').write_text('')
    sums, cost=window_times(read_rank_output(tmp_path,2),2,1,0)
    assert sums==[.2,.3] and cost==.3


def test_precision_and_scaling_use_three_run_medians(tmp_path):
    records(tmp_path)
    rows = summarize(tmp_path, ['tgv'])
    assert len(rows) == 8
    for row in rows:
        assert row['fp64_over_fp32'] == 2
        assert row['fp32_median_ms'] == 2 / row['np']
        assert row['fp32_repair_visits_samples'] == [0,0,0]
        assert row['fp32_repair_max_pointwise_change'] == [0]*5
        assert 'full 40 steps' in row['repair_scope']
        assert row['fp64_median_ms'] == 4 / row['np']
        for kind in ('fp32', 'fp64'):
            assert row[kind + '_speedup_np'] == row['np']
            assert row[kind + '_efficiency_np'] == 1
            assert row[kind + '_compact_over_explicit'] == 1
    assert (tmp_path / 'performance.csv').is_file()


@pytest.mark.parametrize('key,value,message', [
    ('preflight', True, 'measurement window'), ('warmup', 0, 'measurement window'),
    ('statistics_interval', 1, 'measurement window'), ('buffered_output', False, 'measurement window'),
    ('profiled', True, 'measurement window'), ('topology', '1,2,1', 'different case'),
    ('grid_intervals', [32]*3, 'different case'), ('workspace', 'full', 'runtime settings'),
])
def test_timing_does_not_mix_incompatible_measurements(tmp_path, key, value, message):
    records(tmp_path)
    path = next((tmp_path / 'timing').glob('*/summary.json'))
    record = json.loads(path.read_text())
    record[key] = value
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match=message):
        summarize(tmp_path, ['tgv'])
