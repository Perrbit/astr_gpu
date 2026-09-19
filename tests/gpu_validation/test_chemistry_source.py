from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class ChemistrySourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.exe = Path(cls.directory.name) / "chemistry_source_probe"
        sources = [
            ROOT / "src/chemistry_air5_data.F90",
            ROOT / "src/chemistry_core.F90",
            ROOT / "src/chemistry_properties.F90",
            ROOT / "src/chemistry_kinetics.F90",
            ROOT / "tests/gpu_validation/chemistry_source_probe.F90",
        ]
        result = subprocess.run(
            ["gfortran", "-std=f2008", "-O0", "-g", "-fcheck=all",
             "-ffpe-trap=invalid,zero,overflow", *map(str, sources), "-o", str(cls.exe)],
            cwd=cls.directory.name, capture_output=True, text=True,
        )
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)

    def run_probe(self, mode):
        return subprocess.run([str(self.exe), mode], check=True,
                              capture_output=True, text=True)

    def test_sources_are_finite_conservative_and_vt_equilibrium_is_zero(self):
        result = self.run_probe("sources")
        conservation, maximum_source, equilibrium_vt = map(float, result.stdout.split())
        self.assertLessEqual(conservation, 5.0e-13)
        self.assertGreater(maximum_source, 0.0)
        self.assertEqual(equilibrium_vt, 0.0)

    def test_complete_analytic_jacobian_matches_centered_oracle(self):
        result = self.run_probe("jacobian")
        self.assertLessEqual(float(result.stdout), 1.0)

    def test_exact_zero_products_match_admissible_directional_oracle(self):
        result = self.run_probe("boundary_jacobian")
        self.assertLessEqual(float(result.stdout), 1.0)

    def test_invalid_composition_pressure_and_temperature_fail_closed(self):
        result = self.run_probe("invalid")
        actual, expected = [list(map(int, line.split())) for line in result.stdout.splitlines()]
        invalid_composition, out_of_domain, nonfinite = expected
        self.assertEqual(actual, [invalid_composition, out_of_domain, out_of_domain,
                                  out_of_domain, nonfinite])

    def test_coupled_source_equals_chemical_plus_vt_components(self):
        decomposition, vt_species, equilibrium_vt = map(
            float, self.run_probe("components").stdout.split()
        )
        self.assertLessEqual(decomposition, 5.0e-15)
        self.assertEqual(vt_species, 0.0)
        self.assertLessEqual(equilibrium_vt, 1.0e-6)

    def test_unknown_source_mode_fails_closed(self):
        actual, expected = map(int, self.run_probe("invalid_mode").stdout.split())
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
