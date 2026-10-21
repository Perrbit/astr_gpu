import pytest
from insitu_cfl_gate import CflGate


def record(step=0, upper='0.1', dt='2e-5'):
    return (f'ASTR_CFL complete_step={step} state_time=2e-5 dt={dt}\n'
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
