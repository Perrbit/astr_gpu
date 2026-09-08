#!/usr/bin/env python3
"""Static contracts for the four-A800 TGV scaling job."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
JOB = ROOT / "tests/gpu_validation/run_zhongke_a800_tgv_scaling.sbatch"


class ZhongkeA800TgvScalingJobTests(unittest.TestCase):
    def test_job_requests_one_node_and_four_a800_gpus(self) -> None:
        text = JOB.read_text(encoding="ascii")
        self.assertIn("#SBATCH --nodes=1", text)
        self.assertIn("#SBATCH --ntasks=4", text)
        self.assertIn("#SBATCH --gres=gpu:4", text)

    def test_job_stays_inside_approved_remote_root(self) -> None:
        text = JOB.read_text(encoding="ascii")
        self.assertIn("WORK_ROOT=/data/user/hd56000/weiph", text)
        self.assertNotIn("/tmp/", text)

    def test_job_runs_preflight_and_strong_scaling_matrix(self) -> None:
        text = JOB.read_text(encoding="ascii")
        self.assertIn("GRID=512,512,512", text)
        self.assertIn("--maxstep 0", text)
        self.assertIn("run_timing t2_np1 1 1,1,1", text)
        self.assertIn("run_timing t3_np2_z 2 1,1,2", text)
        self.assertIn("run_timing t4_np4_z 4 1,1,4", text)
        self.assertIn("run_timing t5_np4_pageable 4 1,1,4", text)

    def test_job_uses_twenty_timed_steps_and_five_repeats(self) -> None:
        text = JOB.read_text(encoding="ascii")
        self.assertIn("MAXSTEP=21", text)
        self.assertIn("DISCARD_STEPS=2", text)
        self.assertIn("REPEATS=5", text)


if __name__ == "__main__":
    unittest.main()
