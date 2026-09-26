import numpy as np
import pytest

from check_air5_mass_closure_stages import metrics


def test_mass_closure_diagnostic_preserves_zero_and_reports_drift():
    q = np.zeros((2, 1, 1, 11))
    q[..., 0] = 1.0
    q[..., 5] = 0.75
    q[..., 6] = 0.25
    q[1, 0, 0, 5] -= 256*np.finfo(float).eps
    before = q.copy()
    result = metrics(q, (1, 0, 0))
    assert result['transport_sum_violations'] == 1
    assert result['min_species_density'] == 0.0
    assert result['node_extended_mass_residual'] == -256*np.finfo(float).eps
    np.testing.assert_array_equal(q, before)


def test_rejects_flattened_or_nonfinite_snapshot():
    with pytest.raises(ValueError):
        metrics(np.ones(11), (0, 0, 0))
    q = np.ones((1, 1, 1, 11))
    q[0, 0, 0, 2] = np.nan
    with pytest.raises(ValueError):
        metrics(q, (0, 0, 0))


@pytest.mark.parametrize('node', [(-1, 0, 0), (1, 0, 0), (0, 0)])
def test_rejects_invalid_node(node):
    with pytest.raises(ValueError):
        metrics(np.ones((1, 1, 1, 11)), node)
