"""Independent algebra checks; no claim about the solver's RK or MPI integration."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


class PerfectGasBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("ASTR_BOUNDARY_PROBE_EXE"):
            cls.exe = Path(os.environ["ASTR_BOUNDARY_PROBE_EXE"]).resolve()
            return
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.exe = Path(cls.temp.name) / "probe"
        command = ["gfortran", "-std=f2008", "-fcheck=all", "-ffpe-trap=invalid,zero,overflow",
                   str(ROOT / "src/perfect_gas_boundary.F90")]
        if os.environ.get("ASTR_TEST_BOUNDARY_CUDA") == "1":
            command = ["nvfortran", "-cuda", "-Mfree", "-Mpreprocess", "-DTEST_CUDA",
                       str(ROOT / "src_gpu/perfect_gas_boundary_gpu.cuf")]
        result = subprocess.run([*command, "-o", str(cls.exe),
                                 str(ROOT / "tests/gpu_validation/boundary_contract_probe.F90")],
                                cwd=cls.temp.name, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)

    def probe(self, mode):
        result = subprocess.run([str(self.exe), str(mode)], check=True, capture_output=True, text=True)
        lines = result.stdout.splitlines()
        return int(lines[0]), np.array([[float(x) for x in line.split()] for line in lines[1:]])

    def test_subsonic_inlet_keeps_face_and_prescribed_halo_momentum(self):
        status, q = self.probe(1)
        self.assertEqual(status, 0)
        np.testing.assert_array_equal(q[0], [1, .5, 0, 0, 2.125])
        np.testing.assert_array_equal(q[1:], np.tile([1, .1, 0, 0, 2.125], (6, 1)))

    def test_supersonic_reverse_and_exact_sonic_copy_first_halo(self):
        for mode in (2, 3, 4):
            with self.subTest(mode=mode):
                status, q = self.probe(mode)
                self.assertEqual(status, 0)
                np.testing.assert_array_equal(q[0], q[1])
                np.testing.assert_allclose(q[1:, 4], 3 + .01*np.arange(1, 7), rtol=0, atol=1e-15)

    def test_wall_keeps_density_and_constructs_all_six_halos(self):
        status, q = self.probe(5)
        self.assertEqual(status, 0)
        tw, gamma, mach, rho = 1.676194, 1.4, 2., 1.3
        pwall = rho*tw/(gamma*mach**2)
        np.testing.assert_allclose(q[0], [rho, 0, 0, 0, pwall/(gamma-1)], rtol=1e-14)
        for h in range(1, 7):
            tghost = tw + .02*h
            rghost = pwall*gamma*mach**2/tghost
            velocity = -h*np.array([.1, .02, .03])
            expected = np.r_[rghost, rghost*velocity, pwall/(gamma-1)+rghost*np.dot(velocity, velocity)/2]
            np.testing.assert_allclose(q[h], expected, rtol=2e-14, atol=1e-15)

    def test_negative_ghost_temperature_rejects_without_partial_write(self):
        for mode in (6, 7):
            status, q = self.probe(mode)
            self.assertNotEqual(status, 0)
            np.testing.assert_array_equal(q[0], [1.3, .4, .2, .1, 2.])
            np.testing.assert_array_equal(q[1:], np.full((6, 5), -99.))

    def test_fixed_state_split_and_zero_order_outlet(self):
        for mode in (10, 11, 12, 13):
            status, q = self.probe(mode)
            self.assertEqual(status, 0)
            target = [1., .1, 0, 0, 3.] if mode in (10, 11, 13) else [1.2, .2, .1, 0, 4.]
            np.testing.assert_array_equal(q, np.tile(target, (7, 1)))


if __name__ == "__main__":
    unittest.main()
