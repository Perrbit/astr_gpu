import pytest

from run_air5_performance_diagnosis import step_samples, window_summary, phase_summary, batch_identity, check_no_field_output, run_log_text, startup_cost_summary
from summarize_air5_nsys import merged, duration, chemistry_cycle_window


def test_slowest_rank_and_restart_step():
    text = '\n'.join(['ASTR_COMPLETE_STEP_TIMING 1091 1 0.4',
                      'ASTR_COMPLETE_STEP_TIMING 1090 0 0.3',
                      'ASTR_COMPLETE_STEP_TIMING 1091 0 0.2',
                      'ASTR_COMPLETE_STEP_TIMING 1090 1 0.1'])
    assert step_samples(text, 2, 1090, 2) == [.3, .4]


@pytest.mark.parametrize('text', ['', 'ASTR_COMPLETE_STEP_TIMING 1090 0 nan',
    'ASTR_COMPLETE_STEP_TIMING 1090 0 -1', 'ASTR_COMPLETE_STEP_TIMING 1091 0 1',
    'ASTR_COMPLETE_STEP_TIMING 1090 1 1', 'ASTR_COMPLETE_STEP_TIMING 1090 0',
    'ASTR_COMPLETE_STEP_TIMING 1090 0 1\nASTR_COMPLETE_STEP_TIMING 1090 0 1'])
def test_invalid_or_missing_records_rejected(text):
    with pytest.raises(ValueError):
        step_samples(text, 1, 1090, 1)


def test_trace_union_does_not_double_count_compute_and_wait():
    busy = [(10, 20), (15, 30), (40, 50)]
    mpi = [(25, 45)]
    assert merged(busy) == [[10, 30], [40, 50]]
    assert duration(busy) == 30
    assert duration(busy+mpi)-duration(busy) == 10
    assert duration([]) == 0


def test_window_critical_rank_differs_from_stepwise_maxima():
    text = '\n'.join(f'ASTR_COMPLETE_STEP_TIMING {s} {r} {v}' for s, r, v in
        [(5, 0, 10), (5, 1, 10), (6, 0, .3), (6, 1, .1), (7, 0, .2), (7, 1, .4)])
    row = window_summary(text, 2, 5, 3, 1)
    assert row['seconds_per_step'] == .25
    assert row['rank_window_seconds'] == [.5, .5]
    assert row['supplemental_step_max_seconds'] == [.3, .4]


def test_four_rank_window_and_invalid_warmup():
    text = '\n'.join(f'ASTR_COMPLETE_STEP_TIMING {s} {r} {r+1}' for s in (8, 9) for r in range(4))
    assert window_summary(text, 4, 8, 2, 1)['critical_rank'] == 3
    assert window_summary(text, 4, 8, 2, 1)['seconds_per_step'] == 4
    for warmup in (-1, 2):
        with pytest.raises(ValueError):
            window_summary(text, 4, 8, 2, warmup)
    with pytest.raises(ValueError, match='incomplete'):
        window_summary(text.rsplit('\n', 1)[0], 4, 8, 2, 1)


def test_rank_stdout_replaces_interleaved_launcher_output(tmp_path):
    (tmp_path/'run.log').write_text('truncated launcher record')
    assert run_log_text(tmp_path, 2) == 'truncated launcher record'
    for rank in range(2):
        path=tmp_path/f'mpi_output/1/rank.{rank}'
        path.mkdir(parents=True)
        (path/'stdout').write_text(f'ASTR_COMPLETE_STEP_TIMING 0 {rank} 1\n')
    assert len(run_log_text(tmp_path, 2).splitlines()) == 3
    with pytest.raises(ValueError, match='incomplete per-rank'):
        run_log_text(tmp_path, 3)


def test_startup_cost_includes_warmup_and_separates_post_deadline():
    text='\n'.join(f'ASTR_COMPLETE_STEP_TIMING {100+step} {rank} {rank+1}'
                   for step in range(80) for rank in range(2))
    rows=startup_cost_summary(text, 2, 100, 80)
    assert rows['all_updates']['complete_steps'] == 80
    assert rows['all_updates']['rank_window_seconds'] == [80,160]
    assert rows['first_five']['startup_steps'] == [1,5]
    assert rows['through_deadline_after_warmup']['complete_steps'] == 59
    assert rows['post_deadline']['startup_steps'] == [65,80]
    assert rows['post_deadline']['seconds_per_step'] == 2
    short='ASTR_COMPLETE_STEP_TIMING 0 0 1'
    assert set(startup_cost_summary(short,1,0,1)) == {'all_updates','first_five'}


def test_reject_legacy_or_incomplete_batch(tmp_path):
    (tmp_path/'flowfield.h5').touch()
    with pytest.raises(ValueError, match='completed new'):
        batch_identity(tmp_path)
    (tmp_path/'state.h5').touch()
    (tmp_path/'COMPLETE').touch()
    assert 'state.h5' in batch_identity(tmp_path)


def test_field_output_not_permitted_in_timing(tmp_path):
    output = tmp_path/'outdat/new/checkpoints'
    output.mkdir(parents=True)
    check_no_field_output(tmp_path)
    (output/'state.h5').touch()
    with pytest.raises(ValueError, match='field/validation output'):
        check_no_field_output(tmp_path)


def test_complete_timer_includes_completed_step_monitor():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[2]/'src/mainloop.F90').read_text()
    assert text.index('call writemon(time+completed_step_dt)') < text.index('call end_complete_step_timing(nstep)')


def test_profile_window_includes_dynamic_top_once():
    rows = [dict(start=10*i, end=10*i+1, gridX=20 if i%2 == 0 else 2) for i in range(16)]
    assert chemistry_cycle_window(rows, True) == (40, 120)
    assert chemistry_cycle_window(rows[::2]) == (40, 120)
    rows[1]['gridX'] = 30
    with pytest.raises(ValueError, match='ambiguous'):
        chemistry_cycle_window(rows, True)


def test_air5_phase_parser_rejects_missing_and_invalid_records():
    with pytest.raises(ValueError, match='incomplete'):
        phase_summary('', 2, 5, 2, 1)
    for record in ('0 5 1 chem_kernels nan', '0 5 1 chem_kernels -1', '0 5 1 unknown .1'):
        with pytest.raises(ValueError, match='unexpected/duplicate/nonfinite'):
            phase_summary('ASTR_AIR5_PHASE_TIMING '+record, 2, 5, 2, 1)


def test_air5_phase_timer_does_not_introduce_gpu_fences():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[2]/'src_gpu/gpu_phase_timing.cuf').read_text()
    routine = text.split('  subroutine mark_air5_phase(')[1].split('  end subroutine mark_air5_phase')[0]
    assert 'sync_gpu' not in routine and 'cudaDeviceSynchronize' not in routine
