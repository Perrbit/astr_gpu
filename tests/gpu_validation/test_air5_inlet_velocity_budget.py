import numpy as np
import pytest
import json

import check_air5_inlet_velocity_budget as budget
from check_air5_inlet_velocity_budget import velocity_terms, WEIGHTS, UPDATE_WEIGHTS, RK


def test_velocity_budget_accounts_for_density_change():
    reference = np.array([2., 8.])
    candidate = np.array([3., 15.])
    terms = {'a': np.array([.25, 3.]), 'b': np.array([.75, 4.])}
    contributions = velocity_terms(terms, reference, candidate)
    assert np.isclose(sum(contributions.values()), 1.)
    assert np.isclose(contributions['a'], 2/3)


def test_ssprk3_snapshot_weights():
    propagation = [RK[1][1]*RK[2][1], RK[2][1], 1.]
    np.testing.assert_allclose(UPDATE_WEIGHTS, propagation)
    np.testing.assert_allclose(WEIGHTS, np.array(propagation)*np.array(RK)[:, 2])
    assert np.isclose(sum(WEIGHTS), 1.)


def test_shared_stress_coefficient_breaks_local_cancellation(monkeypatch):
    records = []
    for stage in range(1, 4):
        faces = [dict(axis=a, side=s, correction=[0.]*11)
                 for a in range(1, 4) for s in (1, 2)]
        if stage == 1:
            faces[2]['correction'][1] = -1.
            faces[3]['correction'][1] = 1.
        records.append(dict(step=1000, stage=stage, rank=0, i=1, j=35, k=1,
            base_SI=[2.]*11, faces=faces,
            shared=[dict(axis=a, theta=[1., .25 if a == 2 else 1.]) for a in range(1, 4)]))
    monkeypatch.setattr(budget, 'read_probe', lambda _: records)
    raw, limited, stages = budget.stress_budget(None, [1000], (1, 35, 1))
    np.testing.assert_allclose(raw, [0., 0.])
    np.testing.assert_allclose(limited, [0., -.125])
    assert stages[0]['equivalent_limiter_velocity_increment_m_s'] == -.375
    with pytest.raises(ValueError, match='incomplete'):
        budget.stress_budget(None, [1000, 1001], (1, 35, 1))
    records[0]['shared'][0]['theta'][1] = -1.
    with pytest.raises(ValueError, match='invalid shared'):
        budget.stress_budget(None, [1000], (1, 35, 1))


def test_pair_rejects_changed_checkpoint_or_time_window(tmp_path):
    coarse, fine = tmp_path/'coarse', tmp_path/'fine'
    coarse.mkdir()
    fine.mkdir()
    contract = dict(checkpoint_sha256='checkpoint', executable_sha256='binary',
        start_step=1090, start_time=2.18e-6, topology='2,1,1', backend='cpu',
        convection_limiter='species_budget', diffusion_limiter='layered',
        dt=2e-9, updates=1, target_time=2.182e-6)
    (coarse/'contract.json').write_text(json.dumps(contract))
    contract.update(dt=1e-9, updates=2, checkpoint_sha256='other')
    (fine/'contract.json').write_text(json.dumps(contract))
    with pytest.raises(ValueError, match='checkpoint_sha256'):
        budget.compare_pair(coarse, fine)
    contract.update(checkpoint_sha256='checkpoint', target_time=2.183e-6)
    (fine/'contract.json').write_text(json.dumps(contract))
    with pytest.raises(ValueError, match='matched-time'):
        budget.compare_pair(coarse, fine)
