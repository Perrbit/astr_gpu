import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from prepare_opensbli_reference_inputs import prepare


class OpenSBLIInputsTests(unittest.TestCase):
    def test_grid_initial_field_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "inputs"
            report = prepare(root, nx=41, ny=65, nz=9, prandtl=0.72)
            self.assertEqual(report["status"], "INPUT_ONLY_NOT_RUNNABLE")
            self.assertLess(abs(report["similarity_delta_star"] - 1.0), 1e-12)
            with h5py.File(root / "grid.h5") as grid, h5py.File(root / "initial.h5") as field:
                self.assertEqual(grid["x"].shape, (9, 65, 41))
                np.testing.assert_allclose(grid["y"][0, :, 0],
                    115.0 * np.sinh(5.0 * np.linspace(0, 1, 65)) / np.sinh(5.0))
                for name in ("ro", "u1", "u2", "u3", "t"):
                    values = field[name][:]
                    self.assertTrue(np.isfinite(values).all())
                    np.testing.assert_array_equal(values[:, :, 0], values[:, :, -1])
                    np.testing.assert_array_equal(values[0], values[-1])
                np.testing.assert_allclose(field["ro"][:] * field["t"][:], 1.0)
                np.testing.assert_allclose(field["t"][:, 0, :], 1.676194)
                np.testing.assert_allclose(field["u2"][:, 0, :], 0.0, atol=1e-14)
                inlet = np.loadtxt(root / "profile.dat", skiprows=4)
                np.testing.assert_allclose(inlet[:, 2], field["u2"][0, :, 0])
            self.assertFalse((root / "input").exists())
            boundary = (root / "conservative_boundary.nml").read_text()
            self.assertIn("&conservative_boundary", boundary)
            self.assertIn("schema=1", boundary)
            self.assertIn("q_left=", boundary)
            self.assertIn("q_right=", boundary)
            with self.assertRaises(FileExistsError):
                prepare(root)

    def test_invalid_parameters(self):
        for values in ({"nx": 1}, {"ny": 1}, {"nz": 1}, {"prandtl": float("nan")}):
            with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
                prepare(Path(directory) / "inputs", **values)


if __name__ == "__main__":
    unittest.main()
