#!/usr/bin/env python3
"""Unit tests for deterministic dynamic-inflow slice generation."""

from pathlib import Path
import subprocess
import tempfile
import unittest

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tests/gpu_validation/generate_dynamic_inflow_slices.py"


class DynamicInflowSliceGeneratorTests(unittest.TestCase):
    def test_writes_cpu_compatible_time_and_yz_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "inflow"
            subprocess.run(
                [
                    "python3",
                    str(SCRIPT),
                    "--output",
                    str(destination),
                    "--jm",
                    "4",
                    "--km",
                    "6",
                    "--count",
                    "5",
                    "--delta-time",
                    "0.25",
                ],
                check=True,
            )
            files = sorted(destination.glob("islice*.h5"))
            self.assertEqual([path.name for path in files], [f"islice{i:05d}.h5" for i in range(5)])
            with h5py.File(files[3], "r") as handle:
                self.assertAlmostEqual(float(handle["time"][()]), 0.75)
                for name in ("ro", "u1", "u2", "u3", "t"):
                    self.assertEqual(handle[name].shape, (7, 5))
                    self.assertTrue(np.isfinite(handle[name][...]).all())
                self.assertGreater(float(np.max(np.abs(handle["u1"][...]))), 0.0)

    def test_nonpolynomial_mode_cannot_be_reproduced_by_one_global_cubic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "inflow"
            subprocess.run(
                [
                    "python3",
                    str(SCRIPT),
                    "--output",
                    str(destination),
                    "--jm",
                    "4",
                    "--km",
                    "6",
                    "--count",
                    "8",
                    "--delta-time",
                    "0.25",
                    "--temporal-mode",
                    "nonpolynomial",
                ],
                check=True,
            )
            values = []
            for path in sorted(destination.glob("islice*.h5")):
                with h5py.File(path, "r") as handle:
                    values.append(float(handle["u1"][1, 2]))
            fourth_difference = np.diff(np.asarray(values), n=4)
            self.assertGreater(float(np.max(np.abs(fourth_difference))), 1.0e-8)


if __name__ == "__main__":
    unittest.main()
