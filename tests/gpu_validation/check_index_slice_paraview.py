"""Compare a real numerical-slice time series in ParaView, including source dtype."""
from pathlib import Path
import sys

import numpy as np
from paraview import servermanager
from paraview.simple import Xdmf3ReaderS
from vtk.util.numpy_support import vtk_to_numpy


def leaf(data):
    if data.IsA('vtkMultiBlockDataSet'):
        assert data.GetNumberOfBlocks() == 1
        return leaf(data.GetBlock(0))
    if data.IsA('vtkMultiPieceDataSet'):
        assert data.GetNumberOfPieces() == 1
        return leaf(data.GetPiece(0))
    return data


segment = Path(sys.argv[1]).resolve()
reader = Xdmf3ReaderS(FileName=[str(segment/'series.xdmf')])
reader.UpdatePipeline()
with np.load(sys.argv[2]) as reference:
    coordinates = reference['coordinates']
    expected_times = reference['times'].tolist()
    for i, time in enumerate(expected_times):
        reader.UpdatePipeline(time=time)
        data = leaf(servermanager.Fetch(reader))
        points = vtk_to_numpy(data.GetPoints().GetData())
        assert points.dtype == coordinates.dtype
        np.testing.assert_array_equal(points, coordinates)
        for name in reference['names']:
            actual = vtk_to_numpy(data.GetPointData().GetArray(name))
            expected = reference[f'{i}_{name}']
            assert actual.dtype == expected.dtype
            np.testing.assert_array_equal(actual, expected, err_msg=name)
assert np.atleast_1d(reader.TimestepValues).tolist() == expected_times
print('PARAVIEW_INDEX_SLICE_PASS', len(expected_times))
