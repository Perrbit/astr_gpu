"""Test the padding checker, not a model of the fluid solver."""
import h5py
import numpy as np
import pytest

from run_output_restart_validation import check_geometry_padding


def geometry_file(tmp_path, component, node, value):
    # Two cells per direction, one halo layer, one rank; Fortran i-fastest order.
    indices = np.indices((5, 5, 5))
    owner = np.all((indices >= 1) & (indices <= 3), axis=0).ravel(order="F")
    extras = np.zeros((13, np.count_nonzero(~owner)), dtype=np.float64)
    flat = np.ravel_multi_index(node, (5, 5, 5), order="F")
    extra_index = np.flatnonzero(np.flatnonzero(~owner) == flat)
    assert extra_index.size == 1
    extras[component - 1, extra_index[0]] = value
    path = tmp_path / "geometry.h5"
    with h5py.File(path, "w") as state:
        identity = np.zeros(13, dtype=np.int64)
        identity[3:7] = [3, 3, 3, 13]
        state["identity"] = identity
        state["partitions"] = np.array([0, 0, 0, 2, 2, 2, 1, extras.size], dtype=np.int64)
        state["rank_extras"] = extras.ravel()
    return path


@pytest.mark.parametrize("value", [np.float64(2.37e-322), -0.0, np.nan])
def test_undefined_wall_metric_rejected(tmp_path, value):
    path = geometry_file(tmp_path, 4, (2, 0, 2), value)
    with pytest.raises(AssertionError, match="noncanonical geometry padding"):
        check_geometry_padding(path, (True, False, True))


@pytest.mark.parametrize("component", [1, 4])
def test_edge_padding_rejected(tmp_path, component):
    path = geometry_file(tmp_path, component, (0, 0, 2), 1.0)
    with pytest.raises(AssertionError, match="noncanonical geometry padding"):
        check_geometry_padding(path, (True, True, True))


@pytest.mark.parametrize("component,homogeneous", [
    (1, (True, False, True)),  # Physical boundary coordinate halo is defined.
    (4, (True, True, True)),  # Periodic metric face halo is defined.
])
def test_defined_small_value_preserved(tmp_path, component, homogeneous):
    value = np.float64(2.37e-322)
    path = geometry_file(tmp_path, component, (2, 0, 2), value)
    assert check_geometry_padding(path, homogeneous) > 0
    with h5py.File(path) as state:
        assert np.count_nonzero(state["rank_extras"][:] == value) == 1
