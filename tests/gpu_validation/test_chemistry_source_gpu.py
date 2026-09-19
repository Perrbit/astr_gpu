import os
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]


class ChemistrySourceGpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        value = os.environ.get("ASTR_CHEMISTRY_GPU_PROBE_EXE", "")
        if not value:
            raise unittest.SkipTest("ASTR_CHEMISTRY_GPU_PROBE_EXE is not set")
        cls.exe = Path(value)
        if not cls.exe.is_file():
            raise RuntimeError(f"GPU chemistry probe does not exist: {cls.exe}")

    def test_fp64_batch_source_matches_cpu_oracle(self):
        result = subprocess.run(
            [str(self.exe)], check=True, capture_output=True, text=True
        )
        metrics = {}
        for line in result.stdout.splitlines():
            fields = line.split()
            if len(fields) == 2:
                metrics[fields[0]] = float(fields[1])
        for name in (
            "CHEMISTRY_GPU_SOURCE_MAX_RATIO",
            "CHEMISTRY_GPU_RATE_MAX_RATIO",
            "CHEMISTRY_GPU_JACOBIAN_MAX_RATIO",
            "CHEMISTRY_GPU_TOTAL_ENERGY_MAX_RATIO",
        ):
            self.assertIn(name, metrics)
            self.assertLessEqual(metrics[name], 1.0)
        self.assertEqual(metrics["CHEMISTRY_GPU_STATUS_MISMATCHES"], 0.0)
        self.assertEqual(metrics["CHEMISTRY_GPU_Q5_MAX_CHANGE"], 0.0)


class ChemistrySourceGpuContractTests(unittest.TestCase):
    def test_launcher_accepts_device_resident_arrays_and_forces_sync(self):
        source = (ROOT / "src_gpu/chemistry_kinetics_gpu.cuf").read_text(
            encoding="utf-8"
        ).lower()
        start = source.index("subroutine launch_air5_source_batch_gpu")
        end = source.index("end subroutine launch_air5_source_batch_gpu", start)
        launcher = source[start:end]
        self.assertGreaterEqual(launcher.count("device, intent(in)"), 2)
        self.assertGreaterEqual(launcher.count("device, intent(out)"), 3)
        self.assertIn("force_sync=.true.", launcher)
        self.assertNotIn("cudamemcpy", launcher)

    def test_batch_kernel_maps_one_thread_to_one_state(self):
        source = (ROOT / "src_gpu/chemistry_kinetics_gpu.cuf").read_text(
            encoding="utf-8"
        ).lower()
        self.assertIn("(blockidx%x-1)*blockdim%x+threadidx%x", source)
        self.assertIn("if (index > count) return", source)


if __name__ == "__main__":
    unittest.main()
