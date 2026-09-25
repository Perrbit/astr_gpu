from copy import deepcopy

import numpy as np
import pytest

from check_air5_convection_probe import validate_record, correction_by_axis, read_probe


def fixture_record():
    return dict(base_SI=[1.]*11, negative_budget_SI=[0.]*6,
        species_ratio=[1.]*5, roundoff_scale_SI=[1.]*5,
        face_other_and_actual_ratio=[1., 1.],
        faces=[dict(axis=a, side=s, correction=[0.]*11) for a in (1, 2, 3) for s in (1, 2)],
        shared=[dict(axis=a, theta=[1., 1.]) for a in (1, 2, 3)])


def test_operands_and_common_face_coefficient():
    record = fixture_record()
    record['species_ratio'][-1] = .5
    record['face_other_and_actual_ratio'][-1] = .5
    for face in record['shared']:
        face['theta'] = [.5, .5]
    record['faces'][2]['correction'][1] = 2.
    record['faces'][2]['correction'][9] = -.2
    record['negative_budget_SI'][4] = -.2
    validate_record(record)
    np.testing.assert_allclose(correction_by_axis(record)[1, [1, 9]], [-1., .1])
    wrong = deepcopy(record)
    wrong['negative_budget_SI'][4] = 0.
    with pytest.raises(ValueError, match='negative budget'):
        validate_record(wrong)
    wrong = deepcopy(record)
    wrong['shared'][0]['theta'][0] = .75
    with pytest.raises(ValueError, match='shared coefficient'):
        validate_record(wrong)


def test_reject_incomplete_or_nonfinite_probe(tmp_path):
    record = fixture_record()
    record['faces'].pop()
    with pytest.raises(ValueError, match='six distinct'):
        validate_record(record)
    record = fixture_record()
    record['species_ratio'][0] = float('nan')
    with pytest.raises(ValueError, match='invalid species_ratio'):
        validate_record(record)
    log = tmp_path/'run.log'
    log.write_text('The job is done!\n')
    with pytest.raises(ValueError, match='no convection'):
        read_probe(log)
