"""Offline ownership must preserve walls and drop only duplicated seams."""

import numpy as np
import pytest

from report_native_precision_fields import ownership


@pytest.mark.parametrize('physical', [(False, False, False), (True, True, False)])
@pytest.mark.parametrize('axis', range(3))
def test_unique_periodic_or_wall_node_ownership(physical, axis):
    intervals = np.array([8, 6, 4])
    owners = np.zeros(tuple(intervals+np.asarray(physical)), dtype=int)
    for rank in range(2):
        cells = intervals.copy(); cells[axis] //= 2
        origin = np.zeros(3, dtype=int); origin[axis] = rank*cells[axis]
        row = np.r_[rank, 0, 0, 0, cells, origin]
        local, target = ownership(row, intervals, physical)
        assert all(selection.stop <= n+1 for selection, n in zip(local, cells))
        owners[target] += 1
    assert np.all(owners == 1)
