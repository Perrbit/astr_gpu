import pytest
from insitu_cfl_gate import CflGate


def record(step=0, upper='0.1', dt='2e-5',time='2e-5'):
    return (f'ASTR_CFL complete_step={step} state_time={time} dt={dt}\n'
            f'ASTR_CFL directional/local_sum/upper_bound= 0.03 0.03 0.04 0.09 {upper}\n')


def test_incremental_record(tmp_path):
    path = tmp_path/'run.log'
    text = record()
    path.write_text(text[:-3])
    gate = CflGate(path, .5, 2e-5, 0, 1)
    gate.check()
    assert gate.next_step == 0
    path.write_text(text)
    gate.check(final=True)
    assert gate.maximum == .1


@pytest.mark.parametrize('text', [record(upper='.500000000000001'), record(upper='nan'),
                                 record(dt='.001'), record(step=1), record().splitlines()[0]+'\n'])
def test_rejects_invalid_or_missing(tmp_path, text):
    path = tmp_path/'run.log'
    path.write_text(text)
    with pytest.raises(ValueError):
        CflGate(path, .5, 2e-5, 0, 1).check(final=True)


def test_report_cadence_is_explicit(tmp_path):
    path = tmp_path/'run.log'
    path.write_text(record(0)+record(100))
    gate = CflGate(path, .5, 2e-5, 0, 200, step_stride=100)
    gate.check(final=True)
    assert gate.next_step == 200 and gate.maximum == .1


def test_report_cadence_does_not_accept_missing_report(tmp_path):
    path = tmp_path/'run.log'
    path.write_text(record(0)+record(200))
    with pytest.raises(ValueError):
        CflGate(path, .5, 2e-5, 0, 300, step_stride=100).check(final=True)


def test_first_restored_dt_is_explicit(tmp_path):
    path = tmp_path / 'run.log'
    source_dt = 1.9999999494757503e-5
    path.write_text(record(5, dt=str(source_dt)) + record(6))
    CflGate(path, .5, 2e-5, 5, 7, first_dt=source_dt).check(final=True)
    path.write_text(record(5, dt=str(source_dt)) + record(6, dt=str(source_dt)))
    with pytest.raises(ValueError, match='timestep'):
        CflGate(path, .5, 2e-5, 5, 7, first_dt=source_dt).check(final=True)


def rk(step,start,end,used,next_dt):
    return (f'ASTR_TIMESTEP complete_step={step} time_start={start} time_end={end} '
            f'dt_used={used} dt_next={next_dt}\n')


def test_variable_dt_matches_actual_rk(tmp_path):
    path=tmp_path/'run.log'
    path.write_text(record()+rk(1,0,'2e-5','2e-5','2.2e-5')+
        record(1,dt='2.2e-5',time='4.2e-5')+rk(2,'2e-5','4.2e-5','2.2e-5','2.42e-5'))
    gate=CflGate(path,.5,2e-5,0,2,variable_dt=True)
    gate.check(final=True)
    assert gate.verified=={0,1} and gate.maximum==.1


@pytest.mark.parametrize('text',[
    record(dt='nan'),record(dt='0'),record(upper='.51'),
    record()+rk(1,0,'2e-5','2.2e-5','2.2e-5'),
    record()+rk(1,0,'4e-5','2e-5','2.2e-5'),record(),
])
def test_variable_dt_rejects_invalid_or_unmatched_clocks(tmp_path,text):
    path=tmp_path/'run.log';path.write_text(text)
    with pytest.raises(ValueError):CflGate(path,.5,2e-5,0,1,variable_dt=True).check(final=True)
