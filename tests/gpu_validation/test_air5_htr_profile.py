from __future__ import annotations

import hashlib
import math
from pathlib import Path
import tempfile
import unittest

from air5_htr_profile import (
    HTR_PROFILE_SHA256,
    export_astr_air5_profile,
    load_htr_similarity_profile,
    map_htr_profile_to_air5,
)


ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "tests/gpu_validation/data/htr_multispecies_tbl_mach6_similarity.dat"


class Air5HtrProfileTests(unittest.TestCase):
    def test_vendored_profile_is_the_pinned_htr_release(self) -> None:
        digest = hashlib.sha256(PROFILE.read_bytes()).hexdigest()

        self.assertEqual(digest, HTR_PROFILE_SHA256)
        self.assertEqual(digest, "5bca0f7dbb844d81444e1033059c9a9a18c516dc4b9f42b067c96965c33a468a")

    def test_published_similarity_invariants_are_reconstructed(self) -> None:
        source = load_htr_similarity_profile(PROFILE)

        self.assertEqual(len(source), 200)
        self.assertTrue(all(b.eta > a.eta for a, b in zip(source, source[1:])))
        self.assertAlmostEqual(source[-1].u / source[-1].speed_of_sound, 6.0, delta=2.0e-5)
        self.assertAlmostEqual(source[0].temperature / source[-1].temperature, 6.5, delta=2.0e-12)
        self.assertAlmostEqual(source[0].re_x, 96203.8, delta=1.0e-8)
        for row in source:
            self.assertAlmostEqual(sum(row.mole_fraction), 1.0, delta=2.0e-6)

    def test_air5_mapping_preserves_pressure_and_species_order(self) -> None:
        source = load_htr_similarity_profile(PROFILE)
        mapped, metadata = map_htr_profile_to_air5(source, pressure=101325.0)

        self.assertEqual(len(mapped), len(source))
        self.assertAlmostEqual(metadata.re_delta_star, 4000.0, delta=2.0e-2)
        self.assertGreater(metadata.x_origin, 0.0)
        self.assertTrue(all(b.y > a.y for a, b in zip(mapped, mapped[1:])))
        for row in mapped:
            self.assertEqual(row.tv, row.temperature)
            self.assertEqual(row.w, 0.0)
            self.assertAlmostEqual(sum(row.mass_fraction), 1.0, delta=4.0e-15)
            self.assertTrue(all(value >= 0.0 for value in row.mass_fraction))
            self.assertTrue(all(math.isfinite(value) for value in row.as_columns()))
            self.assertAlmostEqual(row.pressure, 101325.0, delta=1.0e-12)

        # HTR columns are N2/O2/NO/N/O; ASTR stores N2/O2/N/O/NO.
        wall_x = source[0].mole_fraction
        wall_y = mapped[0].mass_fraction
        self.assertGreater(wall_y[3], wall_y[4])
        self.assertGreater(wall_y[4], wall_y[2])
        self.assertGreater(wall_y[2], 0.0)
        self.assertGreater(wall_x[4], wall_x[2])
        self.assertLess(metadata.max_source_density_relative_difference, 1.0e-3)
        self.assertGreater(metadata.max_source_density_relative_difference, 1.0e-6)

    def test_export_has_versioned_header_and_thirteen_columns(self) -> None:
        source = load_htr_similarity_profile(PROFILE)
        mapped, metadata = map_htr_profile_to_air5(source, pressure=101325.0)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "air5_hbl_profile.dat"
            export_astr_air5_profile(output, mapped, metadata)
            lines = output.read_text(encoding="ascii").splitlines()

        self.assertIn("HTR-solver commit 328f84270ce26b83e4c7a4e75c0681a129a48bda", lines[0])
        self.assertIn("columns: y rho u v w p T Tv Y_N2 Y_O2 Y_N Y_O Y_NO", lines[2])
        data_lines = [line for line in lines if line and not line.startswith("#")]
        self.assertEqual(len(data_lines), 200)
        self.assertTrue(all(len(line.split()) == 13 for line in data_lines))


if __name__ == "__main__":
    unittest.main()
