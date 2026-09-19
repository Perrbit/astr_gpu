import os
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]


class ChemistryFlowGpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        value = os.environ.get("ASTR_CHEMISTRY_FLOW_GPU_PROBE_EXE", "")
        if not value:
            raise unittest.SkipTest("ASTR_CHEMISTRY_FLOW_GPU_PROBE_EXE is not set")
        cls.exe = Path(value)
        if not cls.exe.is_file():
            raise RuntimeError(f"GPU chemistry flow probe does not exist: {cls.exe}")

    def test_fp64_state_and_transport_match_cpu_oracle(self):
        result = subprocess.run([str(self.exe)], check=True, capture_output=True, text=True)
        metrics = {}
        for line in result.stdout.splitlines():
            fields = line.split()
            if len(fields) == 2:
                metrics[fields[0]] = float(fields[1])
        for name in (
            "CHEMISTRY_FLOW_GPU_STATE_MAX_RATIO",
            "CHEMISTRY_FLOW_GPU_TRANSPORT_MAX_RATIO",
            "CHEMISTRY_FLOW_GPU_FLUX_MAX_RATIO",
        ):
            self.assertIn(name, metrics)
            self.assertLessEqual(metrics[name], 1.0)
        self.assertEqual(metrics["CHEMISTRY_FLOW_GPU_STATUS_MISMATCHES"], 0.0)


class ChemistryFlowGpuContractTests(unittest.TestCase):
    def test_production_and_probe_sources_are_registered(self):
        source = (ROOT / "src/CMakeLists.txt").read_text(encoding="utf-8")
        for name in (
            "chemistry_properties.F90",
            "chemistry_core_gpu.cuf",
            "chemistry_flow_state_gpu.cuf",
            "chemistry_transport_gpu.cuf",
            "chemistry_flow_gpu_probe",
        ):
            self.assertIn(name, source)


if __name__ == "__main__":
    unittest.main()
