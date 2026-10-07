"""Independent CPU reference for the approved private six-order wall gradient.

Coordinates are logical unit-spaced nodes. The caller supplies physical grid
spacing or contracts the resulting unit-index derivatives with grid metrics.
This helper is a small-grid diagnostic, not a production field supplier.
"""
from fractions import Fraction
from math import prod

import numpy as np


def completed_wall_fields(case, step):
    """Independent completed-state oracle: x/z periodic, y physically bounded."""
    import h5py
    with h5py.File(case/f'outdat/new/checkpoints/step{step:012d}/state.h5') as state:
        rho = state['q0001'][:-1, :, :-1]
        velocity = np.stack([state[f'q{c:04d}'][:-1, :, :-1]/rho for c in (2, 3, 4)], axis=-1)
    gradient = np.zeros(velocity.shape+(3,))
    with h5py.File(case/'outdat/new/resources/geometry.h5') as geometry:
        for computational, axis in enumerate((2, 1, 0)):
            if axis == 1:
                directional = wall_derivative(velocity, axis)
            else:
                directional = sum(w*(np.roll(velocity, -n, axis=axis)-np.roll(velocity, n, axis=axis))
                                  for n, w in ((1, 3/4), (2, -3/20), (3, 1/60)))
            for physical in range(3):
                metric = geometry[f'q{5+computational+3*physical:04d}'][:-1, :, :-1]
                if not np.isfinite(metric).all():
                    raise ValueError('Nonfinite independent grid metric')
                gradient[..., :, physical] += directional*metric[..., None]
    values = [gradient[..., c, d] for d in range(3) for c in range(3)]
    values += [-.5*np.einsum('...ij,...ji->...', gradient, gradient),
               np.trace(gradient, axis1=-2, axis2=-1),
               gradient[..., 2, 1]-gradient[..., 1, 2], gradient[..., 0, 2]-gradient[..., 2, 0],
               gradient[..., 1, 0]-gradient[..., 0, 1]]
    return [np.pad(value, [(0, 1), (0, 0), (0, 1)], mode='wrap') for value in values]


def lagrange_derivative_weights(node):
    if node not in range(7):
        raise ValueError('Seven-node derivative requires a node in 0..6')
    weights = []
    for j in range(7):
        denominator = prod(j-m for m in range(7) if m != j)
        numerator = sum(prod(node-m for m in range(7) if m not in (j, k))
                        for k in range(7) if k != j)
        weights.append(Fraction(numerator, denominator))
    return tuple(weights)


def wall_derivative(values, axis, spacing=1.):
    """Centered sixth-order interior and seven-node biased physical closure."""
    data = np.moveaxis(np.asarray(values, dtype=np.float64), axis, 0)
    if data.shape[0] < 7 or not np.isfinite(spacing) or spacing <= 0.:
        raise ValueError('Wall reference requires seven nodes and positive finite spacing')
    if not np.isfinite(data).all():
        raise ValueError('Nonfinite wall reference input')
    result = np.empty_like(data)
    result[3:-3] = (.75*(data[4:-2]-data[2:-4])
        -.15*(data[5:-1]-data[1:-5])+(data[6:]-data[:-6])/60.)/spacing
    for node in (0, 1, 2, 4, 5, 6):
        lower = 0 if node < 3 else data.shape[0]-7
        target = lower+node
        weights = np.array([float(value) for value in lagrange_derivative_weights(node)])
        result[target] = np.tensordot(weights, data[lower:lower+7]-data[target], axes=(0, 0))/spacing
    return np.moveaxis(result, 0, axis)
