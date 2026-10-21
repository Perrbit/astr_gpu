"""Typed primitive validation snapshots and report-only component scales."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.gpu_validation.run_native_precision_tgv_gate import component_report, primitive_state, frozen_rhs_report, cached_state_report


class NativePrecisionTgvGateTests(unittest.TestCase):
    def test_cached_density_is_an_exact_copy_in_both_precisions(self):
        for precision, dtype in (("fp32", np.float32), ("fp64", np.float64)):
            q = np.array([[[[1, 0, 0, 0, 2.5]]]], dtype=dtype)
            cached = np.array([[[[1, 0, 0, 0, 1, dtype(1.4) * dtype(0.1) ** 2]]]], dtype=dtype)
            cached_state_report(q, cached, precision, 0.1)
            cached[..., 0] = dtype(1.5)
            with self.assertRaisesRegex(ValueError, "cached density"):
                cached_state_report(q, cached, precision, 0.1)

    def test_fp32_cache_rounding_difference_is_report_only(self):
        q = np.array([[[[1, 0, 0, 0, 2.5]]]], dtype=np.float32)
        cached = np.array([[[[1, 0, 0, 0, 1, 0.0141]]]], dtype=np.float32)
        self.assertGreater(cached_state_report(q, cached, "fp32", 0.1)["temperature_max_abs"], 1e-5)

    def test_frozen_rhs_requires_identical_operator_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            roots = [Path(temporary) / name for name in ("cpu", "gpu")]
            suffix = "step00000000.rk01.rank00000000.bin"
            for root in roots:
                root.mkdir()
                with (root / ("state.pre_rhs." + suffix)).open("wb") as stream:
                    np.array([0, 0, 0, 1, 5], dtype=np.int32).tofile(stream)
                    np.ones(135, dtype=np.float32).tofile(stream)
                for label, value in (("conv", 1), ("full", 3)):
                    with (root / ("state." + label + "." + suffix)).open("wb") as stream:
                        np.array([0, 0, 0, 5], dtype=np.int32).tofile(stream)
                        np.full(5, value, dtype=np.float32).tofile(stream)
            report = frozen_rhs_report(*roots, "fp32", 1)
            self.assertTrue(report["ranks"]["0"]["q_and_axial_halos_bit_identical"])
            with (roots[1] / ("state.pre_rhs." + suffix)).open("r+b") as stream:
                stream.seek(20 + 13 * 4)
                np.array([2], dtype=np.float32).tofile(stream)
            with self.assertRaisesRegex(ValueError, "not frozen"):
                frozen_rhs_report(*roots, "fp32", 1)

    def test_primitive_payload_uses_explicit_precision(self):
        for precision, dtype in (("fp32", np.float32), ("fp64", np.float64)):
            with self.subTest(precision=precision), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "primitive.bin"
                values = np.ones((4, 3, 3, 6), dtype=dtype)
                values[..., 2] = -2
                with path.open("wb") as stream:
                    np.array([1, 0, 0, 1], dtype=np.int32).tofile(stream)
                    values.ravel(order="F").tofile(stream)
                actual = primitive_state(path, precision)
                self.assertEqual(actual.dtype, np.dtype(dtype))
                np.testing.assert_array_equal(actual, values[1:3, 1:2, 1:2])

    def test_truncated_payload_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "primitive.bin"
            with path.open("wb") as stream:
                np.zeros(4, dtype=np.int32).tofile(stream)
                np.ones(5, dtype=np.float32).tofile(stream)
            with self.assertRaisesRegex(ValueError, "primitive payload"):
                primitive_state(path, "fp32")

    def test_component_report_does_not_divide_by_near_zero(self):
        a = np.zeros((2, 1, 1, 2), dtype=np.float32)
        b = a.copy()
        b[1, 0, 0, 1] = 3
        rows = component_report(a, b)
        self.assertEqual(rows[0]["max_abs"], 0)
        self.assertEqual(rows[1]["max_scaled"], 3)
        self.assertEqual(rows[1]["local_ijk"], [1, 0, 0])

    def test_inactive_corner_nan_is_distinct_from_stencil_nan(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "primitive.bin"
            values = np.ones((4, 3, 3, 6), dtype=np.float32)
            values[0, 0, 0, 1] = np.nan
            for invalid_face in (False, True):
                if invalid_face:
                    values[0, 1, 1, 1] = np.nan
                with path.open("wb") as stream:
                    np.array([1, 0, 0, 1], dtype=np.int32).tofile(stream)
                    values.ravel(order="F").tofile(stream)
                if invalid_face:
                    with self.assertRaisesRegex(ValueError, "primitive stencil"):
                        primitive_state(path, "fp32")
                else:
                    self.assertTrue(np.all(np.isfinite(primitive_state(path, "fp32"))))

    def test_physical_end_has_no_external_halo_but_mpi_end_does(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "primitive.bin"
            values = np.ones((4, 3, 3, 6), dtype=np.float32)
            values[0, 1, 1, 1] = np.nan
            with path.open("wb") as stream:
                np.array([1, 0, 0, 1], dtype=np.int32).tofile(stream)
                values.ravel(order="F").tofile(stream)
            self.assertTrue(np.all(np.isfinite(primitive_state(path, "fp32",
                halo_sides=((False, True), (False, False), (True, True))))))
            with self.assertRaisesRegex(ValueError, "primitive stencil"):
                primitive_state(path, "fp32")


if __name__ == "__main__":
    unittest.main()
