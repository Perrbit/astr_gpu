#!/usr/bin/env python3

import unittest

import numpy as np

from compare_flowfield_h5 import trim_boundary_axes


class CompareFlowfieldH5Tests(unittest.TestCase):
    def test_trims_requested_boundary_planes(self) -> None:
        values = np.arange(5 * 4 * 3).reshape(5, 4, 3)
        result = trim_boundary_axes({"ro": values}, [0, 2])
        np.testing.assert_array_equal(result["ro"], values[1:-1, :, 1:-1])

    def test_default_keeps_complete_field(self) -> None:
        values = np.arange(6).reshape(2, 3)
        self.assertIs(trim_boundary_axes({"ro": values}, [])["ro"], values)

    def test_rejects_invalid_axis(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid axes"):
            trim_boundary_axes({"ro": np.zeros((2, 2, 2))}, [3])


if __name__ == "__main__":
    unittest.main()
