from copy import deepcopy

import pytest

from check_air5_diffusion_probe import read_probe, validate_record


def fixture_record():
    return dict(base_SI=[1.]*11, negative_budget_SI=[0.]*6, raw_trial_SI=[1.]*11,
                raw_trial_admissible=True,
                **{'ratio budget/actual/relaxed': [1., 1., 1.]},
                faces=[dict(axis=a, side=s, ratio=1., thermal=1., correction=[0.]*11)
                       for a in range(1, 4) for s in range(1, 3)],
                shared=[dict(axis=a, theta=[1., 1.]) for a in range(1, 4)])


def test_face_budget_contract():
    valid = fixture_record()
    validate_record(valid)
    wrong = deepcopy(valid)
    wrong['faces'][0]['correction'][7] = -0.1
    with pytest.raises(ValueError, match='raw trial'):
        validate_record(wrong)
    wrong['raw_trial_SI'][7] = 0.9
    with pytest.raises(ValueError, match='negative budget'):
        validate_record(wrong)
    wrong['negative_budget_SI'][2] = -0.1
    validate_record(wrong)
    wrong['ratio budget/actual/relaxed'][1] = 0.5
    with pytest.raises(ValueError, match='actual ratio'):
        validate_record(wrong)


def test_reject_duplicate_or_incomplete_probe(tmp_path):
    wrong = fixture_record()
    wrong['faces'][0]['side'] = 2
    with pytest.raises(ValueError, match='distinct faces'):
        validate_record(wrong)
    log = tmp_path/'run.log'
    log.write_text('The job is done!\n')
    with pytest.raises(ValueError, match='no diffusion'):
        read_probe(log)


def test_trial_roundoff_scales_with_cancelling_operands():
    record = fixture_record()
    record['base_SI'][3] = 1e-15
    record['faces'][0]['correction'][3] = 1e-5
    record['faces'][1]['correction'][3] = -1e-5
    record['raw_trial_SI'][3] = 1e-15+1e-21
    validate_record(record)
    record['raw_trial_SI'][3] = 1e-10
    with pytest.raises(ValueError, match='raw trial'):
        validate_record(record)
