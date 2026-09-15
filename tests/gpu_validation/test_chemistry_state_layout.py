from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class ChemistryStateLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("gfortran")
        if compiler is None:
            raise unittest.SkipTest("gfortran is required")
        out = ROOT / "tests/gpu_validation/out"
        out.mkdir(parents=True, exist_ok=True)
        cls.directory = tempfile.TemporaryDirectory(dir=out)
        cls.exe = Path(cls.directory.name) / "chemistry_state_layout_probe"
        subprocess.run(
            [
                compiler,
                "-std=f2008",
                "-O0",
                "-J",
                cls.directory.name,
                str(ROOT / "src/chemistry_air5_data.F90"),
                str(ROOT / "src/chemistry_state_layout.F90"),
                str(ROOT / "tests/gpu_validation/chemistry_state_layout_probe.F90"),
                "-o",
                str(cls.exe),
            ],
            check=True,
            cwd=ROOT,
        )

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def run_probe(self, mode):
        return subprocess.run(
            [str(self.exe), mode], check=True, capture_output=True, text=True
        ).stdout.split()

    def test_named_conservative_indices_are_frozen(self):
        self.assertEqual(
            [int(value) for value in self.run_probe("indices")],
            [1, 2, 4, 5, 6, 10, 11, 11],
        )

    def test_valid_air5_layout_is_numq_11(self):
        values = [int(value) for value in self.run_probe("valid")]
        self.assertEqual(values, [1, 11, 0, 1, 11, 0, 0])

    def test_invalid_layouts_fail_before_modifying_outputs(self):
        for mode in (
            "invalid_species",
            "invalid_turbulence",
            "invalid_filter",
            "invalid_nondimensional",
        ):
            values = [int(value) for value in self.run_probe(mode)]
            self.assertEqual(values[:2], values[3:5], mode)
            self.assertEqual(values[2], values[6], mode)

    def test_disabled_air5_keeps_existing_layout(self):
        values = [int(value) for value in self.run_probe("disabled")]
        self.assertEqual(values, [2, 9, 0, 2, 9, 0, 0])


class ChemistryRuntimeActivationContractTests(unittest.TestCase):
    def test_air5_sources_include_state_layout(self):
        source = (ROOT / "src/CMakeLists.txt").read_text(encoding="utf-8")
        block = re.search(
            r"if\s*\(ASTR_WITH_AIR5_CHEMISTRY\)(.*?)endif\s*\(\)",
            source,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(block)
        self.assertIn("chemistry_state_layout.F90", block.group(1))

    def test_fixed_air5_input_assigns_and_broadcasts_runtime_lcomb(self):
        source = (ROOT / "src/readwrite.F90").read_text(encoding="utf-8")
        self.assertRegex(
            source,
            re.compile(
                r"#ifdef\s+ASTR_AIR5_CHEMISTRY.*?lcomb\s*=\s*lcomb_input.*?#endif",
                re.DOTALL,
            ),
        )
        self.assertRegex(source, r"call\s+bcast\s*\(\s*lcomb\s*\)")

    def test_refcal_applies_air5_runtime_layout(self):
        source = (ROOT / "src/solver.F90").read_text(encoding="utf-8")
        self.assertIn("call air5_configure_runtime_layout", source.lower())


if __name__ == "__main__":
    unittest.main()
