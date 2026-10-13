from pathlib import Path

import numpy as np

from air5_radau_reference import Air5RadauReference
from run_air5_reference_convergence import synthetic_states, solve_case


def test_cold_zero_species_reference_resolves_production_without_clipping():
    root = Path(__file__).resolve().parents[2]
    model = Air5RadauReference(root / 'chemMech/air5_kimjo12.json')
    case = next(synthetic_states(model))
    initial = case['initial'].copy()
    assert np.all(initial[2:5] == 0.)
    assert np.all(case['reference_atol_scale'] > 0.)
    assert np.max(case['reference_atol_scale'][[2, 4]]) < 1e-45
    result = solve_case(model, case)
    np.testing.assert_array_equal(case['initial'], initial)
    assert np.all(np.asarray(result['endpoints']) >= 0.)
    assert np.all(np.asarray(result['endpoints'])[:, 2:5] > 0.)
    assert np.max(result['adjacent_absolute_differences'][-1][:5]) / case['rho'] < 1e-39
    assert max(result['tight_invariant_drifts']) < 1e-14
