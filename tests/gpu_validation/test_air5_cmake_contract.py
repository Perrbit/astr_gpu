from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]


class Air5CMakeContractTests(unittest.TestCase):
    def test_air5_backend_has_an_independent_default_off_option(self):
        root_cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn(
            'option(ASTR_WITH_AIR5_CHEMISTRY "Build the fixed FP64 air5 chemistry backend" OFF)',
            root_cmake,
        )
        self.assertIsNotNone(re.search(
            r"if\s*\(ASTR_WITH_AIR5_CHEMISTRY AND CHEMISTRY\).*?FATAL_ERROR.*?endif",
            root_cmake,
            flags=re.DOTALL,
        ))

    def test_air5_sources_are_gated_without_cantera_linkage(self):
        source_cmake = (ROOT / "src/CMakeLists.txt").read_text(encoding="utf-8")
        match = re.search(
            r"if\s*\(ASTR_WITH_AIR5_CHEMISTRY\)(.*?)endif\s*\(\)",
            source_cmake,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(match)
        block = match.group(1)
        for name in (
            "chemistry_air5_data.F90",
            "chemistry_core.F90",
            "chemistry_properties.F90",
            "chemistry_kinetics.F90",
            "chemistry_boundary_state.F90",
            "chemistry_boundary.F90",
            "chemistry_runtime.F90",
            "chemistry_solver.F90",
        ):
            self.assertIn(name, block)
        self.assertNotRegex(block.lower(), r"cantera|comb|-lcantera")

    def test_untracked_legacy_chemistry_directory_is_ignored(self):
        patterns = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        self.assertIn("/chem/", patterns)

    def test_air5_gpu_sources_require_both_cuda_and_air5(self):
        source_cmake = (ROOT / "src/CMakeLists.txt").read_text(encoding="utf-8")
        match = re.search(
            r"if\s*\(ASTR_WITH_CUDA AND ASTR_WITH_AIR5_CHEMISTRY\)(.*?)endif\s*\(\)",
            source_cmake,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(match)
        block = match.group(1)
        for name in (
            "chemistry_core_gpu.cuf",
            "chemistry_flow_state_gpu.cuf",
            "chemistry_transport_gpu.cuf",
            "chemistry_relaxation_gpu.cuf",
            "chemistry_kinetics_gpu.cuf",
            "chemistry_boundary_gpu.cuf",
            "chemistry_solver_gpu.cuf",
            "chemistry_coupling_gpu.cuf",
        ):
            self.assertIn(name, block)

    def test_air5_gpu_source_probe_is_an_explicit_target(self):
        source_cmake = (ROOT / "src/CMakeLists.txt").read_text(encoding="utf-8")
        self.assertRegex(
            source_cmake,
            re.compile(
                r"if\s*\(ASTR_WITH_CUDA AND ASTR_WITH_AIR5_CHEMISTRY\).*?"
                r"add_executable\(chemistry_source_gpu_probe\s+EXCLUDE_FROM_ALL",
                re.DOTALL,
            ),
        )

    def test_air5_gpu_ros2_sources_require_both_cuda_and_air5(self):
        source_cmake = (ROOT / "src/CMakeLists.txt").read_text(encoding="utf-8")
        match = re.search(
            r"if\s*\(ASTR_WITH_CUDA AND ASTR_WITH_AIR5_CHEMISTRY\)(.*?)endif\s*\(\)",
            source_cmake,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(match)
        block = match.group(1)
        self.assertIn("chemistry_kinetics_gpu.cuf", block)

    def test_air5_gpu_ros2_probe_is_an_explicit_target(self):
        source_cmake = (ROOT / "src/CMakeLists.txt").read_text(encoding="utf-8")
        self.assertRegex(
            source_cmake,
            re.compile(
                r"if\s*\(ASTR_WITH_CUDA AND ASTR_WITH_AIR5_CHEMISTRY\).*?"
                r"add_executable\(chemistry_ros2_gpu_probe\s+EXCLUDE_FROM_ALL",
                re.DOTALL,
            ),
        )


if __name__ == "__main__":
    unittest.main()
