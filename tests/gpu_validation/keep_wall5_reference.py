"""Independent dense reference for one wall's five-node compact filter."""
from math import comb

import numpy as np


def wall5_system(values, alpha=0.49):
    """Return the 5x5 Schur complement and RHS for wall-oriented values[0:11]."""
    beta = 0.98
    matrix = np.eye(6) + alpha*(np.eye(6, k=1) + np.eye(6, k=-1))
    matrix[0, 1] = beta
    rhs = matrix @ values[:6]
    for j, order in enumerate((6, 6, 6, 6, 8, 10)):
        # The original RHS is the unfiltered LHS plus a high-order difference.
        denominator = 2**order
        strength = (1-beta if j == 0 else 1-2*alpha)*(-1)**(j+1)/denominator
        difference = np.array([(-1)**i*comb(order, i) for i in range(order+1)])
        rhs[j] += strength*(difference @ values[:order+1])
    # The j=6 value occurs on both sides; its identity contribution cancels.
    rhs[1] -= alpha*rhs[0]
    return matrix[1:, 1:]-np.outer(matrix[1:, 0], matrix[0, 1:]), rhs[1:]


def filter_wall5(values, alpha=0.49):
    matrix, rhs = wall5_system(values, alpha)
    return np.linalg.solve(matrix, rhs)


def self_check():
    constants = np.broadcast_to(np.arange(1., 6.), (11, 5))
    np.testing.assert_allclose(filter_wall5(constants), constants[1:6], atol=2e-14, rtol=0)
    impulse = np.eye(11)[:, -1]
    delta = filter_wall5(impulse)
    np.testing.assert_allclose(delta.sum(), 188727575/6470337996544, atol=1e-15, rtol=0)
    print(f'Constant test passed; q(10) unit impulse changes the five-node sum by {delta.sum():.15e}')


if __name__ == '__main__':
    self_check()
