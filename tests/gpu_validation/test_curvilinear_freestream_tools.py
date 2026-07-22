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
