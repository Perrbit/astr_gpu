import h5py
import numpy as np
import pytest

from check_air5_compensation_checkpoint import compare, read


def checkpoint(path, carry=0.0):
    q = np.zeros((2, 3, 4, 11))
    q[..., 0] = 1.0
    q[..., 5] = 0.75
    q[..., 6] = 0.25
    with h5py.File(path, 'w') as data:
        data['air5_compensation_version'] = 1
        data['nstep'] = 10
        data['time'] = 1e-8
        for i in range(1, 12):
            data[f'acq{i:02}'] = q[..., i-1]
            data[f'acc{i:02}'] = np.full(q.shape[:-1], carry)


def test_exact_comparison_includes_carry(tmp_path):
    a, b = tmp_path/'a.h5', tmp_path/'b.h5'
    checkpoint(a)
    checkpoint(b)
    assert compare(a, b, exact=True)['passed']
    checkpoint(b, carry=1e-30)
    assert not compare(a, b, exact=True)['passed']
    assert compare(a, b)['passed']


@pytest.mark.parametrize('fault', ['schema', 'missing', 'nan', 'closure', 'negative'])
def test_rejects_invalid_checkpoint(tmp_path, fault):
    path = tmp_path/'bad.h5'
    checkpoint(path)
    with h5py.File(path, 'a') as data:
        if fault == 'schema':
            data['air5_compensation_version'][...] = 2
        elif fault == 'missing':
            del data['acc11']
        elif fault == 'nan':
            data['acc11'][0, 0, 0] = np.nan
        elif fault == 'closure':
            data['acq06'][0, 0, 0] -= 256*np.finfo(float).eps
        else:
            data['acq08'][0, 0, 0] = -1e-30
    with pytest.raises((ValueError, KeyError)):
        read(path)


def test_rejects_phase_mismatch(tmp_path):
    a, b = tmp_path/'a.h5', tmp_path/'b.h5'
    checkpoint(a)
    checkpoint(b)
    with h5py.File(b, 'a') as data:
        data['nstep'][...] = 11
    with pytest.raises(ValueError, match='phase'):
        compare(a, b)
