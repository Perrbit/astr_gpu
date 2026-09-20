from __future__ import annotations

import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class Air5HblBoundaryStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        compiler = shutil.which("gfortran")
        if compiler is None:
            raise unittest.SkipTest("gfortran is required")
        out = ROOT / "tests/gpu_validation/out"
        out.mkdir(parents=True, exist_ok=True)
        cls.directory = tempfile.TemporaryDirectory(dir=out)
        directory = Path(cls.directory.name)
        cls.exe = directory / "air5_hbl_boundary_state_probe"
        subprocess.run(
            [
                compiler,
                "-std=f2008",
                "-O0",
                "-J",
                str(directory),
                str(ROOT / "src/chemistry_air5_data.F90"),
                str(ROOT / "src/chemistry_core.F90"),
                str(ROOT / "src/chemistry_properties.F90"),
                str(ROOT / "src/chemistry_boundary_state.F90"),
                str(ROOT / "tests/gpu_validation/air5_hbl_boundary_state_probe.F90"),
                "-o",
                str(cls.exe),
            ],
            check=True,
            cwd=ROOT,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    def run_probe(self, mode: str) -> tuple[int, int, list[float]]:
        output = subprocess.run(
            [str(self.exe), mode], check=True, capture_output=True, text=True
        ).stdout.split()
        return int(output[0]), int(output[1]), [float(value) for value in output[2:]]

    def test_wall_state_preserves_positive_first_inner_composition(self) -> None:
        status, state_status, values = self.run_probe("wall")

        self.assertEqual((status, state_status), (0, 0))
        self.assertTrue(all(math.isfinite(value) for value in values))
        rho, u, v, w, pressure, temperature, tv = values[:7]
        mass_fraction = values[7:12]
        self.assertGreater(rho, 0.0)
        self.assertEqual((u, v, w), (0.0, 0.0, 0.0))
        self.assertAlmostEqual(pressure, (4.0 * 101000.0 - 100000.0) / 3.0, delta=2.0e-8)
        self.assertAlmostEqual(temperature, 2925.0, delta=2.0e-10)
        self.assertAlmostEqual(tv, 2925.0, delta=2.0e-10)
        expected_y = [0.70, 0.20, 0.03, 0.04, 0.03]
        for actual, expected in zip(mass_fraction, expected_y):
            self.assertAlmostEqual(actual, expected, delta=2.0e-15)
        self.assertAlmostEqual(values[12], 1.0, delta=4.0e-15)

    def test_outflow_extrapolates_the_authoritative_q11_state(self) -> None:
        status, state_status, values = self.run_probe("outflow")

        self.assertEqual((status, state_status), (0, 0))
        self.assertTrue(all(math.isfinite(value) for value in values))
        self.assertGreater(values[0], 0.0)
        self.assertTrue(all(value >= 0.0 for value in values[7:12]))
        self.assertAlmostEqual(values[12], 1.0, delta=4.0e-15)
        self.assertLessEqual(values[13], 2.0e-12)

    def test_outflow_limits_trace_species_overshoot_without_clipping(self) -> None:
        status, state_status, values = self.run_probe("outflow-trace")

        self.assertEqual((status, state_status), (0, 0))
        self.assertTrue(all(math.isfinite(value) for value in values))
        self.assertAlmostEqual(values[12], 1.0, delta=4.0e-15)
        self.assertGreater(values[13], 0.0)
        self.assertLess(values[13], 1.0)
        self.assertLess(values[14], 0.0)
        self.assertGreaterEqual(values[15], 0.0)
        self.assertLessEqual(values[16], 2.0e-12)


if __name__ == "__main__":
    unittest.main()
