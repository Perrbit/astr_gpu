"""The raw wall export distinguishes native-width payloads without changing FP64."""

import numpy as np
import pytest

from test_insitu_wall_scalar_statistics import read_statistics


@pytest.mark.parametrize('version,real', [(1, '<f8'), (2, '<f4')])
def test_wall_export_native_width(tmp_path, version, real):
    path = tmp_path/'wall.bin'
    with path.open('wb') as stream:
        stream.write(b'ASTRWS01')
        np.array([version, 4, 0, 1, 1, 1, 0, 0, 0, 3, 5], dtype='<i4').tofile(stream)
        np.array([.004, .0005, .004, .0035], dtype=real).tofile(stream)
        np.array([5], dtype='<i8').tofile(stream)
        np.zeros(3, dtype=real).tofile(stream)
        np.array([1, 2, 3, 4, 9, 16, 2, 3, 4], dtype=real).tofile(stream)
        np.array([1], dtype='<i4').tofile(stream)
    header, clock, samples, xyz, values, owned = read_statistics(path)
    assert header[0] == version and samples == 5
    assert clock.dtype == xyz.dtype == values.dtype == np.dtype(real)
    np.testing.assert_array_equal(values.ravel(), [1, 2, 3, 4, 9, 16, 2, 3, 4])
    assert owned.item() == 1
