from pathlib import Path
import json
import math
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class ChemistryFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("gfortran")
        if compiler is None:
            raise unittest.SkipTest("gfortran is required")
        out = ROOT / "tests/gpu_validation/out"
        out.mkdir(parents=True, exist_ok=True)
        cls.directory = tempfile.TemporaryDirectory(dir=out)
        cls.exe = Path(cls.directory.name) / "chemistry_flow_probe"
        subprocess.run(
            [compiler, "-std=f2008", "-O0", "-J", cls.directory.name,
             str(ROOT / "src/chemistry_air5_data.F90"),
             str(ROOT / "src/chemistry_core.F90"),
             str(ROOT / "src/chemistry_properties.F90"),
             str(ROOT / "tests/gpu_validation/chemistry_flow_probe.F90"),
             "-o", str(cls.exe)],
            check=True, cwd=ROOT,
        )

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def run_probe(self, mode):
        values = subprocess.run(
            [str(self.exe), mode], check=True, capture_output=True, text=True
        ).stdout.split()
        return int(values[0]), [float(value) for value in values[1:]]

    def test_conservative_primitive_roundtrip(self):
        status, values = self.run_probe("roundtrip")
        self.assertEqual(status, 0)
        self.assertEqual(int(values[0]), 0)
        for residual in values[1:7]:
            self.assertLessEqual(residual, 2.0e-9)
        self.assertGreater(values[7], 1.0e3)
        self.assertLess(values[7], 1.0e6)

    def test_transport_properties_are_finite_positive(self):
        status, values = self.run_probe("properties")
        self.assertEqual(status, 0)
        self.assertTrue(all(math.isfinite(value) for value in values))
        self.assertTrue(all(value > 0.0 for value in values))

        data = json.loads((ROOT / "chemMech/air5_kimjo12.json").read_text())
        temperature = 5000.0
        pressure = 2.0e5
        mass_fraction = [0.70, 0.20, 0.04, 0.03, 0.03]
        molar_mass = [entry["molecular_weight_kg_per_mol"] for entry in data["species"]]
        mole_amount = [y / molar for y, molar in zip(mass_fraction, molar_mass)]
        mole_fraction = [amount / sum(mole_amount) for amount in mole_amount]
        species_mu = []
        for entry in data["species"]:
            a, b, c = entry["blottner_coefficients"]
            log_t = math.log(temperature)
            species_mu.append(0.1 * math.exp(a * log_t**2 + b * log_t + c))
        phi = [[0.0] * 5 for _ in range(5)]
        for i in range(5):
            for j in range(5):
                phi[i][j] = (
                    1.0
                    + math.sqrt(species_mu[i] / species_mu[j])
                    * (molar_mass[j] / molar_mass[i]) ** 0.25
                ) ** 2 / math.sqrt(8.0 * (1.0 + molar_mass[i] / molar_mass[j]))
        expected_mu = sum(
            mole_fraction[i] * species_mu[i]
            / sum(mole_fraction[j] * phi[i][j] for j in range(5))
            for i in range(5)
        )
        coefficient = data["binary_diffusion_coefficients"][0][1]
        a, b, c, d = coefficient
        expected_d12 = (
            10.1325
            / pressure
            * math.exp(d)
            * temperature ** (a * math.log(temperature) ** 2 + b * math.log(temperature) + c)
        )
        self.assertAlmostEqual(values[0], expected_mu, delta=2.0e-14 * expected_mu)
        for actual, expected in zip(values[3:8], species_mu):
            self.assertAlmostEqual(actual, expected, delta=2.0e-14 * expected)
        self.assertAlmostEqual(values[8], expected_d12, delta=2.0e-14 * expected_d12)

    def test_corrected_species_flux_and_energy_bookkeeping_are_finite(self):
        status, values = self.run_probe("flux")
        self.assertEqual(status, 0)
        self.assertLessEqual(values[0], 2.0e-14)
        self.assertTrue(all(math.isfinite(value) for value in values))
        self.assertGreater(max(abs(value) for value in values[1:6]), 0.0)
        self.assertGreater(max(abs(value) for value in values[6:9]), 0.0)
        self.assertGreater(max(abs(value) for value in values[9:12]), 0.0)

    def test_pure_species_limit_is_finite_and_zero_flux(self):
        status, values = self.run_probe("pure")
        self.assertEqual(status, 0)
        self.assertTrue(all(math.isfinite(value) for value in values))
        self.assertEqual(values, [0.0, 0.0, 0.0, 0.0])

    def test_trace_species_are_not_truncated(self):
        status, values = self.run_probe("trace")
        self.assertEqual(status, 0)
        self.assertTrue(all(math.isfinite(value) for value in values))
        self.assertGreater(values[0], 0.0)
        self.assertGreater(values[1], 0.0)
        self.assertLessEqual(values[2], 2.0e-14)


if __name__ == "__main__":
    unittest.main()
