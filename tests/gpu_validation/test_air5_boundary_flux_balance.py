import numpy as np
import pytest

from check_air5_boundary_flux_balance import balance


def manufactured(periodic=False):
    dims = (5, 4, 3)
    rhs = np.full(tuple(d+1 for d in dims)+(11,), 0. if periodic else -3.)
    faces = []
    for axis in range(3):
        shape = tuple(dims[a]+1 for a in range(3) if a != axis)+(12, 4)
        plane = np.zeros(shape)
        if periodic:
            plane[:] = 1.
        else:
            plane[..., :11, 1] = dims[axis]-1
        faces.append(plane)
    return [rhs], [faces]


def test_nonperiodic_flux_divergence_and_corruption():
    rhs, faces = manufactured()
    result = balance(rhs, faces, (1, 1, 1), (False, False, False))
    assert result['max_scaled_residual'] == 0
    assert result['plane_pairs'] == 0
    rhs[0][1, 1, 1, 0] += 1.
    with pytest.raises(ValueError, match='divergence mismatch'):
        balance(rhs, faces, (1, 1, 1), (False, False, False))


def test_periodic_bitwise_corruption():
    rhs, faces = manufactured(periodic=True)
    assert balance(rhs, faces, (1, 1, 1), (True, True, True))['plane_pairs'] == 6
    faces[0][0][0, 0, 0, 2] = np.nextafter(1., 2.)
    with pytest.raises(ValueError, match='bitwise'):
        balance(rhs, faces, (1, 1, 1), (True, True, True))
