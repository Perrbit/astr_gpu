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
        self.assertIn("HALO_TRANSPORT=pinned-overlap", self.text)
        self.assertIn("SYNC_MODE=explicit", self.text)
        self.assertIn("expected full GPU filter workspace", self.text)

    def test_runs_complete_strong_and_weak_topology_matrices(self) -> None:
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
            "weak_np2_121 2 1,2,1 256,512,256",
            "weak_np2_112 2 1,1,2 256,256,512",
            "weak_np4_411 4 4,1,1 1024,256,256",
            "weak_np4_141 4 1,4,1 256,1024,256",
            "weak_np4_114 4 1,1,4 256,256,1024",
            "weak_np4_221 4 2,2,1 512,512,256",
            "weak_np4_212 4 2,1,2 512,256,512",
            "weak_np4_122 4 1,2,2 256,512,512",
        ]
        for row in expected:
            self.assertIn(row, self.text)

    def test_uses_stable_short_timing_and_case_level_fault_isolation(self) -> None:
        self.assertIn("MAXSTEP=21", self.text)
        self.assertIn("DISCARD_STEPS=2", self.text)
        self.assertIn("REPEATS=5", self.text)
        self.assertIn("DELTAT=1.0e-4", self.text)
        self.assertIn("run_case \"$label\"", self.text)
        self.assertNotIn("set -euo pipefail", self.text)

    def test_uses_supported_python_and_writes_combined_summary(self) -> None:
        self.assertIn('export PATH="$(dirname "$PYTHON_EXE"):', self.text)
        self.assertIn('$BENCH_DIR/summarize_tgv_scaling_matrix.py', self.text)
        self.assertIn('summary_script_sha256:', self.text)
        self.assertIn("matrix_manifest.tsv", self.text)
        self.assertIn("case_status.tsv", self.text)


if __name__ == "__main__":
    unittest.main()
