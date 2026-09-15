import os
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]


class ChemistryRos2GpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        value = os.environ.get("ASTR_CHEMISTRY_ROS2_GPU_PROBE_EXE", "")
        if not value:
            raise unittest.SkipTest("ASTR_CHEMISTRY_ROS2_GPU_PROBE_EXE is not set")
        cls.exe = Path(value)
        if not cls.exe.is_file():
            raise RuntimeError(f"GPU ROS-2 probe does not exist: {cls.exe}")

    def test_fp64_batch_trajectory_matches_cpu_oracle(self):
        result = subprocess.run(
            [str(self.exe)], check=True, capture_output=True, text=True
        )
        metrics = {}
        for line in result.stdout.splitlines():
            fields = line.split()
            if len(fields) == 2:
                metrics[fields[0]] = float(fields[1])
        for name in (
            "CHEMISTRY_GPU_ROS2_STATE_MAX_RATIO",
            "CHEMISTRY_GPU_ROS2_MASS_MAX_DRIFT",
            "CHEMISTRY_GPU_ROS2_NITROGEN_MAX_DRIFT",
            "CHEMISTRY_GPU_ROS2_OXYGEN_MAX_DRIFT",
        ):
            self.assertIn(name, metrics)
        self.assertLessEqual(metrics["CHEMISTRY_GPU_ROS2_STATE_MAX_RATIO"], 1.0)
        self.assertLessEqual(metrics["CHEMISTRY_GPU_ROS2_MASS_MAX_DRIFT"], 5.0e-12)
        self.assertLessEqual(metrics["CHEMISTRY_GPU_ROS2_NITROGEN_MAX_DRIFT"], 5.0e-12)
        self.assertLessEqual(metrics["CHEMISTRY_GPU_ROS2_OXYGEN_MAX_DRIFT"], 5.0e-12)
        self.assertEqual(metrics["CHEMISTRY_GPU_ROS2_STATUS_MISMATCHES"], 0.0)
        self.assertEqual(metrics["CHEMISTRY_GPU_ROS2_INVALID_MISMATCHES"], 0.0)
        self.assertEqual(metrics["CHEMISTRY_GPU_ROS2_DIAGNOSTIC_MISMATCHES"], 0.0)
        self.assertEqual(metrics["CHEMISTRY_GPU_ROS2_LINEAR_STATUS_MISMATCHES"], 0.0)
        self.assertEqual(metrics["CHEMISTRY_GPU_ROS2_Q5_MAX_CHANGE"], 0.0)
        self.assertEqual(metrics["CHEMISTRY_GPU_ROS2_INPUT_STATE_MAX_CHANGE"], 0.0)


class ChemistryRos2GpuContractTests(unittest.TestCase):
    def ros2_source(self):
        path = ROOT / "src_gpu/chemistry_ros2_gpu.cuf"
        self.assertTrue(path.is_file(), f"missing GPU ROS-2 implementation: {path}")
        return path.read_text(encoding="utf-8").lower()

    def test_launcher_uses_resident_arrays_and_forces_sync(self):
        source = self.ros2_source()
        start = source.index("subroutine launch_air5_ros2_batch_gpu")
        end = source.index("end subroutine launch_air5_ros2_batch_gpu", start)
        launcher = source[start:end]
        self.assertGreaterEqual(launcher.count("device, intent(in)"), 3)
        self.assertGreaterEqual(launcher.count("device, intent(out)"), 3)
        self.assertIn("force_sync=.true.", launcher)
        self.assertNotIn("cudamemcpy", launcher)

    def test_batch_kernel_maps_one_thread_to_one_cell(self):
        source = self.ros2_source()
        self.assertIn("(blockidx%x-1)*blockdim%x+threadidx%x", source)
        self.assertIn("if (index > count) return", source)
        self.assertIn("call air5_ros2_advance_gpu", source)

    def test_device_integrator_uses_analytic_source_and_fixed_lu(self):
        source = self.ros2_source()
        self.assertIn("air5_instantaneous_source_jacobian_gpu", source)
        self.assertIn("air5_lu_factor_6_gpu", source)
        self.assertIn("air5_lu_solve_6_gpu", source)
        self.assertNotIn("centered_difference", source)
        self.assertNotIn("admissible_difference", source)


if __name__ == "__main__":
    unittest.main()
