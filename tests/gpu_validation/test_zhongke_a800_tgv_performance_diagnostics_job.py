#!/usr/bin/env python3
"""Static contracts for the combined A800 performance diagnostic job."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
JOB = ROOT / "tests/gpu_validation/run_zhongke_a800_tgv_performance_diagnostics.sbatch"


class A800PerformanceDiagnosticsJobTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = JOB.read_text(encoding="ascii")

    def test_uses_four_a800_gpus_and_frozen_executable(self) -> None:
        self.assertIn("#SBATCH --gres=gpu:4", self.text)
        self.assertIn("#SBATCH --time=12:00:00", self.text)
        self.assertIn("EXPECTED_SOURCE_COMMIT=c801eebc8f14a84d045eb60db92e9c022ccce7b7", self.text)
        self.assertIn("EXPECTED_GPU_EXE_SHA256=7ed0e326e907143d9cb5b7abb045ea8cea6d0256c0f63cdd02f91ceed0b06755", self.text)

    def test_runs_conventional_nsys_on_representative_scaling_cases(self) -> None:
        for row in (
            "nsys_np1_111 1 1,1,1",
            "nsys_np2_211 2 2,1,1",
            "nsys_np4_122 4 1,2,2",
        ):
            self.assertIn(row, self.text)
        self.assertIn("--trace=cuda,nvtx,osrt,mpi", self.text)
        self.assertIn("cuda_gpu_kern_sum", self.text)
        self.assertIn("cuda_api_sum", self.text)
        self.assertIn("mpi_event_sum", self.text)

    def test_nsys_block_does_not_abort_internal_timing(self) -> None:
        self.assertIn("CUDA events unavailable; internal timing continues", self.text)
        self.assertIn("nsys_status=BLOCKED_BY_PLATFORM", self.text)
        self.assertIn('mkdir -p "$RUN_ROOT/phase"', self.text)

    def test_runs_topology_complete_phase_matrix(self) -> None:
        expected = [
            "strong_np1_111 1 1,1,1",
            "strong_np2_211 2 2,1,1",
            "strong_np2_121 2 1,2,1",
            "strong_np2_112 2 1,1,2",
            "strong_np4_411 4 4,1,1",
            "strong_np4_141 4 1,4,1",
            "strong_np4_114 4 1,1,4",
            "strong_np4_221 4 2,2,1",
            "strong_np4_212 4 2,1,2",
            "strong_np4_122 4 1,2,2",
            "weak_np1_111 1 1,1,1",
            "weak_np2_211 2 2,1,1",
            "weak_np4_221 4 2,2,1",
            "control_np2_overlap 2 2,1,1",
            "control_np4_overlap 4 1,2,2",
        ]
        for row in expected:
            self.assertIn(row, self.text)
        self.assertIn("PHASE_TIMING=1", self.text)
        self.assertIn("SYNC_MODE=\"$sync\"", self.text)
        self.assertIn("FILTER_WORKSPACE=full", self.text)

    def test_forbids_field_output_and_preflights_runtime(self) -> None:
        self.assertIn("ASTR_GPU_BENCHMARK_NO_FIELD_IO=1", self.text)
        self.assertIn("forbidden field HDF5 output", self.text)
        self.assertIn('ldd "$GPU_EXE"', self.text)
        self.assertIn("libhdf5_fortran", self.text)
        self.assertIn("run_compute_preflight", self.text)


if __name__ == "__main__":
    unittest.main()
