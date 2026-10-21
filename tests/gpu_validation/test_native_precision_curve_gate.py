"""Native gradient payloads carry only physical derivatives and owned y faces."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.gpu_validation.run_native_precision_curve_gate import read_gradients


class GradientPayloadTests(unittest.TestCase):
    def test_explicit_kind_and_face_layout(self):
        for precision, dtype in (("fp32", np.float32), ("fp64", np.float64)):
            with self.subTest(precision=precision), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "gradient.bin"
                values = np.arange(2 * 3 * 2 * 12, dtype=dtype).reshape((2, 3, 2, 12), order="F")
                normal = np.ones((2, 2, 3), dtype=dtype)
                with path.open("wb") as stream:
                    np.array([1, 2, 1, 1, 0], dtype=np.int32).tofile(stream)
                    values.ravel(order="F").tofile(stream)
                    normal.ravel(order="F").tofile(stream)
                actual, normals = read_gradients(path, precision)
                np.testing.assert_array_equal(actual, values)
                self.assertEqual(actual.dtype, np.dtype(dtype))
                self.assertEqual(set(normals), {3})
                np.testing.assert_array_equal(normals[3], normal)

    def test_truncation_and_nonfinite_payload_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gradient.bin"
            for values in (np.ones(11, dtype=np.float32), np.full(12, np.nan, dtype=np.float32)):
                with path.open("wb") as stream:
                    np.zeros(5, dtype=np.int32).tofile(stream)
                    values.tofile(stream)
                with self.assertRaisesRegex(ValueError, "gradient payload"):
                    read_gradients(path, "fp32")


if __name__ == "__main__":
    unittest.main()
