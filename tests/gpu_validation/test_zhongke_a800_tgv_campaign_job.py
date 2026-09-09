#!/usr/bin/env python3
"""Static contracts for the fault-tolerant four-A800 TGV campaign."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
JOB = ROOT / "tests/gpu_validation/run_zhongke_a800_tgv_campaign.sbatch"
PRODUCTION = ROOT / "tests/gpu_validation/run_tgv_512_production.sh"


class ZhongkeA800TgvCampaignJobTests(unittest.TestCase):
    def test_job_requests_one_node_and_four_a800_gpus(self) -> None:
        text = JOB.read_text(encoding="ascii")
        self.assertIn("#SBATCH --partition=A800-N", text)
        self.assertIn("#SBATCH --nodes=1", text)
        self.assertIn("#SBATCH --ntasks=4", text)
        self.assertIn("#SBATCH --gres=gpu:4", text)

    def test_job_is_tgv_only_and_stays_in_approved_root(self) -> None:
        text = JOB.read_text(encoding="ascii")
        self.assertIn("WORK_ROOT=/data/user/hd56000/weiph", text)
        self.assertNotIn("/tmp/", text)
        self.assertNotIn("fang", text.lower())
        self.assertNotIn("opensbli", text.lower())

    def test_job_contains_full_evidence_matrix(self) -> None:
        text = JOB.read_text(encoding="ascii")
        for label in (
            "t0_cfl_memory_preflight",
            "t1_256_cpu_np1",
            "t1_256_gpu_np1",
            "t2_512_gpu_np1",
            "t3_512_gpu_np2_z",
            "t4_512_gpu_np4_z",
            "t5_512_gpu_np4_pageable",
            "t6_512_gpu_np4_nsys",
            "t7_512_gpu_np4_t20",
        ):
            self.assertIn(label, text)

    def test_case_failures_are_recorded_without_global_errexit(self) -> None:
        text = JOB.read_text(encoding="ascii")
        self.assertIn("set -uo pipefail", text)
        self.assertNotIn("set -euo pipefail", text)
        self.assertIn("case_status.tsv", text)
        self.assertIn("run_case", text)
        self.assertIn("SKIP", text)

    def test_production_uses_measured_cfl_and_segmented_restart(self) -> None:
        text = PRODUCTION.read_text(encoding="ascii")
        job_text = JOB.read_text(encoding="ascii")
        self.assertIn('TARGET_TIME="${TARGET_TIME:-20.0}"', text)
        self.assertIn('TARGET_CFL="${TARGET_CFL:-0.50}"', text)
        self.assertIn('SEGMENT_STEPS="${SEGMENT_STEPS:-2000}"', text)
        self.assertIn("time step for CFL=1", job_text)
        self.assertIn("CFL_DT_ONE_FILE", text)
        self.assertIn("set_restart", text)
        self.assertIn("MAX_RETRIES", text)
        self.assertIn("compare_tgv_dlr_reference.py", text)


if __name__ == "__main__":
    unittest.main()
