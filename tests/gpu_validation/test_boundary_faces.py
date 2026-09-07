import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np


class BoundaryFacesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("ASTR_BOUNDARY_FACES_PROBE_EXE"):
            cls.exe = Path(os.environ["ASTR_BOUNDARY_FACES_PROBE_EXE"]).resolve()
            return
        root = Path(__file__).resolve().parents[2]
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.exe = Path(cls.temp.name) / "probe"
        result = subprocess.run(["gfortran", "-std=f2008", "-fcheck=all", "-ffpe-trap=invalid,zero,overflow",
                                 "-o", str(cls.exe), str(root / "src/perfect_gas_boundary.F90"),
                                 str(root / "src/conservative_boundary_faces.F90"),
                                 str(root / "tests/gpu_validation/boundary_faces_probe.F90")],
                                cwd=cls.temp.name, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)

    def probe(self, mode):
        result = subprocess.run([str(self.exe), str(mode)], check=True, capture_output=True, text=True)
        return np.fromstring(result.stdout, sep=" ").reshape(2, 13, 13, 5)

    def test_corners_halos_and_interior(self):
        q, initial = self.probe(15), self.probe(0)
        np.testing.assert_array_equal(q[:, 3:10, 3:10], initial[:, 3:10, 3:10])
        np.testing.assert_array_equal(q[0], q[1])
        for i in range(13):
            target = [1, .1, 0, 0, 3] if i-2<=4 else [1.2, .2, .1, 0, 4]
            np.testing.assert_array_equal(q[:, 10:, i], np.tile(target, (2, 3, 1)))
        np.testing.assert_array_equal(q[:, 2, :, 1:4], 0.)
        expected_density = initial[:, 2, :, 0].copy()
        expected_density[:, 10:] = initial[:, 2, 9, 0, None]
        np.testing.assert_array_equal(q[:, 2, :, 0], expected_density)
        for h in (1, 2):
            ghost = q[:, 2-h]
            pghost = .4*(ghost[..., 4]-.5*np.sum(ghost[..., 1:4]**2, axis=-1)/ghost[..., 0])
            pwall = q[:, 2, :, 0]*1.7/5.6
            np.testing.assert_allclose(pghost, pwall, rtol=1e-14)
            np.testing.assert_allclose(ghost[..., 1:4]/ghost[..., 0, None],
                                       -q[:, 2+h, :, 1:4]/q[:, 2+h, :, 0, None], rtol=1e-14, atol=1e-15)
        np.testing.assert_array_equal(q[:, 3:10, 10], q[:, 3:10, 9])
        np.testing.assert_array_equal(q[:, 3:10, 0, 4], q[:, 3:10, 2, 4])

    def test_no_x_physical_faces_preserves_x_states_away_from_y_faces(self):
        q, initial = self.probe(12), self.probe(0)
        np.testing.assert_array_equal(q[:, 3:10], initial[:, 3:10])

    def test_each_face_mask_leaves_other_regions_untouched(self):
        initial = self.probe(0)
        for side in range(4):
            with self.subTest(side=side):
                q = self.probe(1 << side)
                unchanged = np.ones(q.shape[:-1], dtype=bool)
                if side == 0:
                    unchanged[:, 2:11, :3] = False
                elif side == 1:
                    unchanged[:, 2:11, 10:] = False
                elif side == 2:
                    unchanged[:, :3, :] = False
                else:
                    unchanged[:, 10:, :] = False
                np.testing.assert_array_equal(q[unchanged], initial[unchanged])


if __name__ == "__main__":
    unittest.main()
