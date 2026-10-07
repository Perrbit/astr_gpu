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
