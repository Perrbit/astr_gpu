#!/usr/bin/env python3
"""Static contracts for the topology-complete A800 TGV scaling job."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
JOB = ROOT / "tests/gpu_validation/run_zhongke_a800_tgv_scaling_matrix.sbatch"


class ZhongkeA800TgvScalingMatrixJobTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = JOB.read_text(encoding="ascii")

    def test_requests_four_a800_gpus_for_two_days(self) -> None:
        self.assertIn("#SBATCH --nodes=1", self.text)
        self.assertIn("#SBATCH --ntasks=4", self.text)
        self.assertIn("#SBATCH --gres=gpu:4", self.text)
        self.assertIn("#SBATCH --time=2-00:00:00", self.text)

    def test_locks_authoritative_fp64_filter_configuration(self) -> None:
        self.assertIn("FILTER_WORKSPACE=full", self.text)
        self.assertIn("PRIMARY_HALO_TRANSPORT=pinned-pipeline", self.text)
        self.assertIn("PRIMARY_SYNC_MODE=explicit", self.text)
        self.assertIn("expected full GPU filter workspace", self.text)

    def test_targets_the_uploaded_git_checkout_and_dedicated_build(self) -> None:
        self.assertIn(
            "EXPECTED_SOURCE_COMMIT=c801eebc8f14a84d045eb60db92e9c022ccce7b7",
            self.text,
        )
        self.assertIn("astr_gpu_p4_c801eeb", self.text)
        self.assertIn("build_gpu_p4_c801eeb", self.text)

    def test_runs_gpu_only_strong_matrix_and_minimal_weak_matrix(self) -> None:
        expected = [
            "strong_np1_111 1 1,1,1 512,512,512",
            "strong_np2_211 2 2,1,1 512,512,512",
            "strong_np2_121 2 1,2,1 512,512,512",
            "strong_np2_112 2 1,1,2 512,512,512",
            "strong_np4_411 4 4,1,1 512,512,512",
            "strong_np4_141 4 1,4,1 512,512,512",
            "strong_np4_114 4 1,1,4 512,512,512",
            "strong_np4_221 4 2,2,1 512,512,512",
            "strong_np4_212 4 2,1,2 512,512,512",
            "strong_np4_122 4 1,2,2 512,512,512",
            "weak_np1_111 1 1,1,1 256,256,256",
            "weak_np2_211 2 2,1,1 512,256,256",
            "weak_np4_221 4 2,2,1 512,512,256",
        ]
        for row in expected:
            self.assertIn(row, self.text)
        omitted = [
            "weak_np2_121",
            "weak_np2_112",
            "weak_np4_411",
            "weak_np4_141",
            "weak_np4_114",
            "weak_np4_212",
            "weak_np4_122",
        ]
        for label in omitted:
            self.assertNotIn(label, self.text)
        self.assertNotIn("CPU_EXE", self.text)
        self.assertNotIn("cpu_baseline", self.text)

    def test_adds_controls_on_the_measured_fastest_topologies(self) -> None:
        self.assertIn("select_fastest_topology", self.text)
        self.assertIn("control_np2_pinned_overlap_explicit", self.text)
        self.assertIn("control_np2_pipeline_dependency", self.text)
        self.assertIn("control_np4_pinned_overlap_explicit", self.text)
        self.assertIn("control_np4_pipeline_dependency", self.text)

    def test_uses_stable_short_timing_and_case_level_fault_isolation(self) -> None:
        self.assertIn("MAXSTEP=21", self.text)
        self.assertIn("DISCARD_STEPS=2", self.text)
        self.assertIn("REPEATS=5", self.text)
        self.assertIn("DELTAT=1.0e-4", self.text)
        self.assertIn("$11+0 > 0.03", self.text)
        self.assertIn("$11+0 > 0.05", self.text)
        self.assertIn("run_case \"$label\"", self.text)
        self.assertNotIn("set -euo pipefail", self.text)

    def test_uses_supported_python_and_writes_combined_summary(self) -> None:
        self.assertIn('export PATH="$(dirname "$PYTHON_EXE"):', self.text)
        self.assertIn('$BENCH_DIR/summarize_tgv_scaling_matrix.py', self.text)
        self.assertIn('summary_script_sha256:', self.text)
        self.assertIn("matrix_manifest.tsv", self.text)
        self.assertIn("case_status.tsv", self.text)

    def test_forbids_large_field_output_but_keeps_statistics(self) -> None:
        self.assertIn("ASTR_GPU_BENCHMARK_NO_FIELD_IO=1", self.text)
        self.assertIn("forbidden HDF5 output", self.text)
        self.assertIn("statistics output", self.text)

    def test_preflights_runtime_libraries_math_flags_and_crlf(self) -> None:
        self.assertIn('ldd "$GPU_EXE"', self.text)
        self.assertIn("unresolved compute-node dynamic library dependency", self.text)
        self.assertIn("libhdf5_fortran", self.text)
        for flag in ("-ffast-math", "-Ofast", "-Mfprelaxed", "-Mfpapprox", "fastmath"):
            self.assertIn(flag, self.text)
        self.assertIn("CRLF or bare carriage return found", self.text)
        self.assertIn("run_compute_preflight", self.text)
        self.assertIn("compute-node preflight failed", self.text)


if __name__ == "__main__":
    unittest.main()
