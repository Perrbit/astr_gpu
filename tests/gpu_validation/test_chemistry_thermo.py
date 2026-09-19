from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class ChemistryThermoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.exe = Path(cls.directory.name) / "chemistry_thermo_probe"
        sources = [
            ROOT / "src/chemistry_air5_data.F90",
            ROOT / "src/chemistry_core.F90",
            ROOT / "src/chemistry_properties.F90",
            ROOT / "tests/gpu_validation/chemistry_thermo_probe.F90",
        ]
        result = subprocess.run(
            ["gfortran", "-std=f2008", "-O0", "-g", "-fcheck=all",
             "-ffpe-trap=invalid,zero,overflow", *map(str, sources), "-o", str(cls.exe)],
            cwd=cls.directory.name, capture_output=True, text=True,
        )
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)

    def run_probe(self, mode):
        return subprocess.run([str(self.exe), mode], check=True, capture_output=True, text=True)

    def test_ev_tv_and_complete_energy_round_trips(self):
        result = self.run_probe("roundtrip")
        ev_residual, tv_residual, q5_residual = map(float, result.stdout.split())
        self.assertLessEqual(ev_residual, 1.0e-12)
        self.assertLessEqual(tv_residual, 1.0e-12)
        self.assertLessEqual(q5_residual, 1.0e-12)

    def test_invalid_states_fail_closed_with_distinct_status(self):
        result = self.run_probe("invalid")
        actual, expected = [list(map(int, line.split())) for line in result.stdout.splitlines()]
        invalid_mechanism, invalid_density, invalid_composition, out_of_domain, no_vib, \
            nonfinite, ok = expected
        self.assertEqual(actual, [invalid_mechanism, invalid_density, invalid_composition,
                                  out_of_domain, out_of_domain, no_vib,
                                  out_of_domain, invalid_composition, nonfinite])
        self.assertEqual(ok, 0)

    def test_runtime_mechanism_identifier_is_fixed(self):
        result = self.run_probe("mechanism")
        actual, expected = [list(map(int, line.split())) for line in result.stdout.splitlines()]
        self.assertEqual(actual, expected)

    def test_implicit_temperature_derivatives_match_centered_differences(self):
        result = self.run_probe("derivatives")
        self.assertLess(float(result.stdout), 2.0e-8)


if __name__ == "__main__":
    unittest.main()
