import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
VALID = """&conservative_boundary
 schema=1, split_x=40,
 q_left=1.00000596004,1.00000268202,.00565001630205,0,.94644428042,
 q_right=1.129734572,1.0921171,-.058866065,0,1.0590824
/
"""


class ConservativeBoundaryConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.exe = cls.root / "probe"
        if os.environ.get("ASTR_BOUNDARY_CONFIG_PROBE_EXE"):
            cls.exe = Path(os.environ["ASTR_BOUNDARY_CONFIG_PROBE_EXE"]).resolve()
            return
        result = subprocess.run(["gfortran", "-std=f2008", "-fcheck=all", "-o", str(cls.exe),
                                 str(ROOT / "src/perfect_gas_boundary.F90"),
                                 str(ROOT / "src/conservative_boundary_config.F90"),
                                 str(ROOT / "tests/gpu_validation/conservative_boundary_config_probe.F90")],
                                cwd=cls.root, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)

    def parse(self, contents):
        filename = self.root / "boundary.nml"
        filename.write_text(contents)
        return subprocess.run([str(self.exe), str(filename)], capture_output=True, text=True)

    def test_empty_path_is_disabled(self):
        result = subprocess.run([str(self.exe)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines()[0].split(), ["0", "F"])

    def test_valid_lf_and_crlf_input(self):
        for contents in (VALID, VALID.replace("\n", "\r\n")):
            result = self.parse(contents)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines()[0].split(), ["0", "T"])
            self.assertEqual(float(result.stdout.splitlines()[1]), 40.)
            self.assertAlmostEqual(float(result.stdout.splitlines()[3].split()[2]), -.058866065)

    def test_missing_or_invalid_fields_are_not_defaulted(self):
        for contents in ("", VALID.replace("schema=1,", ""), VALID.replace("schema=1", "schema=2"),
                         VALID.replace("split_x=40,", ""), VALID.replace("split_x=40", "split_x=NaN"),
                         VALID.replace("1.129734572", "-1.0"),
                         VALID.replace("1.0590824", "0.0"),
                         VALID.replace("1.129734572,1.0921171,-.058866065,0,1.0590824", "1,0,0,0"),
                         VALID.replace("schema=1", "typo=1"), VALID + "unexpected trailing content\n"):
            with self.subTest(contents=contents):
                result = self.parse(contents)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout.splitlines()[0].split()[-1], "F")


if __name__ == "__main__":
    unittest.main()
