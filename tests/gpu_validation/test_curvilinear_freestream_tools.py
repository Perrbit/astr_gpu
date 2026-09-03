#!/usr/bin/env python3
"""Unit tests for the curvilinear free-stream validation tools."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CurvilinearFreestreamToolsTest(unittest.TestCase):
    def test_x_wavy_mapping_has_positive_jacobian_and_noncartesian_x_faces(self) -> None:
        generator = load_module(
            "generate_curvilinear_tgv_grid",
            ROOT / "tests/gpu_validation/generate_curvilinear_tgv_grid.py",
        )
        x, y, z, jacobian = generator.mapped_grid(16, 18, 20, 0.15, "x-wavy")
        self.assertGreater(float(np.min(jacobian)), 0.0)
        self.assertGreater(float(np.ptp(x[0, :, :])), 0.25)
        self.assertTrue(np.allclose(y[0, :, :], y[-1, :, :]))
        self.assertTrue(np.allclose(z[0, :, :], z[-1, :, :]))

    def test_analytic_curvilinear_metrics_are_inverse_mapping(self) -> None:
        checker = load_module(
            "check_curvilinear_metrics",
            ROOT / "tests/gpu_validation/check_curvilinear_metrics.py",
        )
        jacobian, inverse = checker.analytic_metrics((16, 18, 20), 0.15)
        mapping = checker.index_mapping_derivatives((16, 18, 20), 0.15)
        product = np.einsum("...ij,...jk->...ik", inverse, mapping)
        identity = np.eye(3)[None, None, None, :, :]
        self.assertGreater(float(np.min(jacobian)), 0.0)
        self.assertLess(float(np.max(np.abs(product - identity))), 1.0e-12)

    def test_analytic_metrics_satisfy_discrete_identity(self) -> None:
        checker = load_module(
            "check_curvilinear_metrics_identity",
            ROOT / "tests/gpu_validation/check_curvilinear_metrics.py",
        )
        residuals = []
        for size in (16, 24, 32):
            jacobian, inverse = checker.analytic_metrics((size, size, size), 0.15)
            residual = checker.metric_identity_residual(jacobian, inverse)
            residuals.append(float(np.max(np.abs(residual))))
        self.assertTrue(all(np.isfinite(residuals)))
        self.assertGreater(residuals[0], residuals[1])
        self.assertGreater(residuals[1], residuals[2])

    def test_metric_convergence_requires_each_error_to_decrease(self) -> None:
        convergence = load_module(
            "check_curvilinear_metric_convergence",
            ROOT / "tests/gpu_validation/check_curvilinear_metric_convergence.py",
        )
        self.assertTrue(convergence.errors_decrease([1.0e-3, 2.0e-4, 4.0e-5]))
        self.assertFalse(convergence.errors_decrease([1.0e-3, 1.1e-3, 4.0e-5]))

    def test_metric_convergence_reads_component_linf(self) -> None:
        convergence = load_module(
            "check_curvilinear_metric_convergence_norm",
            ROOT / "tests/gpu_validation/check_curvilinear_metric_convergence.py",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            report = Path(tmpdir) / "metric.txt"
            report.write_text("dxi23 linf=1.2500000000000000e-06 l2=2.0e-07\n")
            self.assertEqual(convergence.report_norm_linf(report, "dxi23"), 1.25e-6)

    def test_set_ninit_updates_input_value(self) -> None:
        prepare = load_module(
            "prepare_tgv_case",
            ROOT / "tests/gpu_validation/prepare_tgv_case.py",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input.tgv"
            input_file.write_text("# ninit : Initial method\n0\n")
            prepare.set_ninit(input_file, 3)
            self.assertEqual(input_file.read_text(), "# ninit : Initial method\n3\n")

    def test_uniform_field_has_astr_datasets_and_shape(self) -> None:
        generator = load_module(
            "generate_uniform_flow_field",
            ROOT / "tests/gpu_validation/generate_uniform_flow_field.py",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "flowini3d.h5"
            generator.write_uniform_field(
                output,
                grid=(10, 12, 14),
                density=1.1,
                velocity=(0.7, -0.2, 0.1),
                temperature=0.9,
            )
            with h5py.File(output, "r") as handle:
                self.assertEqual(set(handle), {"ro", "u1", "u2", "u3", "t"})
                for name in handle:
                    self.assertEqual(handle[name].shape, (15, 13, 11))
                self.assertAlmostEqual(float(handle["ro"][3, 4, 5]), 1.1)
                self.assertAlmostEqual(float(handle["u2"][3, 4, 5]), -0.2)
                self.assertAlmostEqual(float(handle["t"][3, 4, 5]), 0.9)

    def test_freestream_checker_detects_field_drift(self) -> None:
        checker = load_module(
            "check_uniform_flowfield",
            ROOT / "tests/gpu_validation/check_uniform_flowfield.py",
        )
        expected = {
            "ro": 1.0,
            "u1": 0.7,
            "u2": -0.2,
            "u3": 0.1,
            "p": 1.0 / (1.4 * 0.1**2),
            "t": 1.0,
        }
        fields = {name: np.full((4, 3, 2), value) for name, value in expected.items()}
        lines, passed = checker.check_fields(fields, expected, atol=1.0e-12, rtol=1.0e-12)
        self.assertTrue(passed)
        self.assertTrue(any(line.startswith("ro ") for line in lines))

        fields["u1"][2, 1, 1] += 1.0e-6
        _, passed = checker.check_fields(fields, expected, atol=1.0e-12, rtol=1.0e-12)
        self.assertFalse(passed)


if __name__ == "__main__":
    unittest.main()
