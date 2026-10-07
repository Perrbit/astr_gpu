"""Mathematical reference checks, not GPU or production-scale acceptance."""
from fractions import Fraction

import numpy as np
import pytest

from insitu_curve_boundary_reference import lagrange_derivative_weights, wall_derivative


@pytest.mark.parametrize('node', range(7))
def test_exact_polynomial_moments(node):
    weights = lagrange_derivative_weights(node)
    for degree in range(7):
        actual = sum(weights[j]*j**degree for j in range(7))
        expected = Fraction(0 if degree == 0 else degree*node**(degree-1))
        assert actual == expected
    mirrored = lagrange_derivative_weights(6-node)
    assert weights == tuple(-value for value in reversed(mirrored))


@pytest.mark.parametrize('axis', range(3))
def test_all_boundary_rows_and_constant_preservation(axis):
    points = np.linspace(0., 1., 33)
    shape = [1, 1, 1]
    shape[axis] = len(points)
    for degree in range(7):
        data = np.broadcast_to((points**degree).reshape(shape), (33, 33, 33))
        expected = 0. if degree == 0 else (degree*points**(degree-1)).reshape(shape)
        actual = wall_derivative(data, axis, 1./32.)
        assert np.max(abs(actual-expected)) <= 2e-10
        if degree == 0:
            assert not np.count_nonzero(actual)


def test_invalid_reference_input():
    for values, spacing in ((np.zeros(6), 1.), (np.zeros(7), 0.), (np.full(7, np.nan), 1.)):
        with pytest.raises(ValueError):
            wall_derivative(values, 0, spacing)
