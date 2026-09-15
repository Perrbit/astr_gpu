from __future__ import annotations

import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from air5_htr_profile import (
    export_astr_air5_profile,
    load_htr_similarity_profile,
    map_htr_profile_to_air5,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tests/gpu_validation/data/htr_multispecies_tbl_mach6_similarity.dat"


class Air5HblFortranProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        compiler = shutil.which("gfortran")
        if compiler is None:
            raise unittest.SkipTest("gfortran is required")
        out = ROOT / "tests/gpu_validation/out"
        out.mkdir(parents=True, exist_ok=True)
        cls.directory = tempfile.TemporaryDirectory(dir=out)
        directory = Path(cls.directory.name)
        cls.profile_path = directory / "air5_hbl_profile.dat"
        source = load_htr_similarity_profile(SOURCE)
        mapped, metadata = map_htr_profile_to_air5(source, pressure=101325.0)
        export_astr_air5_profile(cls.profile_path, mapped, metadata)
        cls.exe = directory / "air5_hbl_profile_probe"
        subprocess.run(
            [
                compiler,
                "-std=f2008",
                "-O0",
                "-J",
                str(directory),
                str(ROOT / "src/chemistry_air5_data.F90"),
                str(ROOT / "src/chemistry_model.F90"),
                str(ROOT / "src/chemistry_state_layout.F90"),
                str(ROOT / "src/chemistry_thermo.F90"),
                str(ROOT / "src/chemistry_flow_state.F90"),
                str(ROOT / "src/chemistry_hbl_profile.F90"),
                str(ROOT / "tests/gpu_validation/air5_hbl_profile_probe.F90"),
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
            [str(self.exe), mode, str(self.profile_path)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.split()
        return int(output[0]), int(output[1]), [float(value) for value in output[2:]]

    def test_wall_endpoint_reconstructs_complete_air5_state(self) -> None:
        status, point_count, values = self.run_probe("lower")

        self.assertEqual(status, 0)
        self.assertEqual(point_count, 200)
        self.assertEqual(len(values), 16)
        y_min, y_max, y, rho, u, v, w, pressure, temperature, tv = values[:10]
        mass_fraction = values[10:15]
        self.assertEqual(y_min, 0.0)
        self.assertGreater(y_max, 0.0)
        self.assertEqual(y, 0.0)
        self.assertGreater(rho, 0.0)
        self.assertEqual((u, v, w), (0.0, 0.0, 0.0))
        self.assertAlmostEqual(pressure, 101325.0, delta=2.0e-8)
        self.assertAlmostEqual(temperature, 2925.0, delta=2.0e-10)
        self.assertAlmostEqual(tv, temperature, delta=2.0e-10)
        self.assertAlmostEqual(sum(mass_fraction), 1.0, delta=4.0e-15)
        self.assertAlmostEqual(values[15], 1.0, delta=4.0e-15)

    def test_midpoint_is_finite_and_within_endpoint_bounds(self) -> None:
        status, point_count, values = self.run_probe("midpoint")

        self.assertEqual(status, 0)
        self.assertEqual(point_count, 200)
        self.assertTrue(all(math.isfinite(value) for value in values))
        self.assertGreater(values[3], 0.0)
        self.assertGreater(values[4], 0.0)
        self.assertGreaterEqual(values[8], 450.0)
        self.assertLessEqual(values[8], 2925.0)
        self.assertAlmostEqual(values[9], values[8], delta=2.0e-10)
        self.assertAlmostEqual(sum(values[10:15]), 1.0, delta=4.0e-15)

    def test_farfield_endpoint_retains_mach_six_profile_values(self) -> None:
        status, _, values = self.run_probe("upper")

        self.assertEqual(status, 0)
        self.assertAlmostEqual(values[4], 2905.07, delta=2.0e-10)
        self.assertAlmostEqual(values[7], 101325.0, delta=2.0e-8)
        self.assertAlmostEqual(values[8], 450.0, delta=2.0e-10)
        self.assertAlmostEqual(values[9], 450.0, delta=2.0e-10)
        self.assertAlmostEqual(sum(values[10:15]), 1.0, delta=4.0e-15)


if __name__ == "__main__":
    unittest.main()
