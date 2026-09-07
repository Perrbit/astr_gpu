import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class TransportConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.exe = Path(cls.directory.name) / "probe"
        subprocess.run(["gfortran", "-std=f2008", "-fcheck=all", "-o", str(cls.exe),
                        str(ROOT / "src/perfect_gas_transport.F90"),
                        str(ROOT / "tests/gpu_validation/transport_config_probe.F90")],
                       cwd=cls.directory.name, check=True, capture_output=True, text=True)

    def run_config(self, values):
        env = dict(os.environ)
        for key in ("ASTR_PERFECT_GAS_PRANDTL", "ASTR_SUTHERLAND_TEMPERATURE_K"):
            env.pop(key, None)
        env.update(values)
        return subprocess.run([str(self.exe)], env=env, capture_output=True, text=True)

    def test_defaults(self):
        result = self.run_config({})
        self.assertEqual(result.returncode, 0, result.stderr)
        pr, suth, changed = result.stdout.split()
        self.assertEqual((float(pr), float(suth), changed), (0.72, 110.3, "F"))

    def test_reference_overrides(self):
        result = self.run_config({"ASTR_PERFECT_GAS_PRANDTL": "0.71d0",
                                 "ASTR_SUTHERLAND_TEMPERATURE_K": "110.4"})
        self.assertEqual(result.returncode, 0, result.stderr)
        pr, suth, changed = result.stdout.split()
        self.assertEqual((float(pr), float(suth), changed), (0.71, 110.4, "T"))

    def test_invalid_values_fail(self):
        for key in ("ASTR_PERFECT_GAS_PRANDTL", "ASTR_SUTHERLAND_TEMPERATURE_K"):
            for value in ("0", "-1", "NaN", "Inf", "bad", "1" * 300, "", "1,2", "/"):
                with self.subTest(key=key, value=value):
                    self.assertNotEqual(self.run_config({key: value}).returncode, 0)


if __name__ == "__main__":
    unittest.main()
