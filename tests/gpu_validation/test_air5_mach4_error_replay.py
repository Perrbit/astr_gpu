import json
from pathlib import Path

import numpy as np
import pytest

from check_air5_mach4_error_replay import COMPONENTS, difference, scalar


def test_component_names_match_fixed_mechanism():
    mechanism = Path(__file__).resolve().parents[2]/'chemMech/air5_kimjo12.json'
    species = json.loads(mechanism.read_text())['species_order']
    assert COMPONENTS[5:10] == tuple('rho_'+name for name in species)


def test_scalar_reader_removes_halo_and_rejects_truncation(tmp_path):
    filename = tmp_path/'ratio.bin'
    values = np.arange(5*6*4, dtype=float).reshape((5, 6, 4), order='F')
    with filename.open('wb') as stream:
        np.array([2, 3, 1, 1], dtype=np.int32).tofile(stream)
        values.ravel(order='F').tofile(stream)
    np.testing.assert_array_equal(scalar(filename), values[1:4, 1:5, 1:3])
    filename.write_bytes(filename.read_bytes()[:-8])
    with pytest.raises(ValueError, match='payload'):
        scalar(filename)


def test_difference_is_componentwise_and_fails_on_invalid_data():
    a = np.zeros((2, 11))
    b = a.copy()
    b[0, 1] = 3.
    result = difference(a, b)
    assert result['rho_u']['max_abs'] == 3.
    assert result['rho']['max_abs'] == 0.
    b[0, 2] = np.nan
    with pytest.raises(ValueError, match='nonfinite'):
        difference(a, b)
