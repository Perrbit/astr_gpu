#!/usr/bin/env python3
"""Contracts for the dynamic-inflow D2 restart and combination driver."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / "tests/gpu_validation/run_dynamic_inflow_d2_matrix.sh"
STAGE_DRIVER = ROOT / "tests/gpu_validation/run_dynamic_inflow_d2_stage_compare.sh"


class DynamicInflowD2DriverTests(unittest.TestCase):
    def test_driver_covers_restart_time_and_solver_combinations(self) -> None:
        text = DRIVER.read_text(encoding="utf-8")
        for label in ("slice_node", "cross_slice", "multi_slice", "filter_curve_mpi"):
            self.assertIn(label, text)
        self.assertIn("set_restart_input", text)
        self.assertIn("compare_flowfield_h5.py", text)
        self.assertIn("ASTR_VALIDATION_RK_SNAPSHOT", text)
        self.assertIn("TEMPORAL_MODE=nonpolynomial", text)
        self.assertIn('--lfilter "$lfilter"', text)
        self.assertIn("12 t 0.03 0.02", text)
        self.assertIn("--warp-x", text)
        self.assertIn("TOPOLOGY=2,2,1", text)

    def test_filtered_case_preserves_upstream_checkpoint_semantics(self) -> None:
        text = DRIVER.read_text(encoding="utf-8")
        self.assertIn("KNOWN_UPSTREAM_CHECKPOINT_SEMANTICS", text)
        self.assertIn("shared_checkpoint", text)
        self.assertIn('cp "$shared_checkpoint/flowfield.h5"', text)
        self.assertIn('cp "$shared_checkpoint/auxiliary.txt"', text)
        self.assertIn("cpu_gpu_shared_restart.txt", text)

    def test_driver_refuses_to_overwrite_evidence(self) -> None:
        text = DRIVER.read_text(encoding="utf-8")
        self.assertIn('if [[ -e "$OUT_DIR" ]]', text)
        self.assertIn("refusing to overwrite", text.lower())

    def test_stage_driver_covers_single_and_multi_rank_filtered_curve_cases(self) -> None:
        text = STAGE_DRIVER.read_text(encoding="utf-8")
        self.assertIn("dynamic_flat_np1", text)
        self.assertIn("dynamic_filter_curve_np4", text)
        self.assertIn("2,2,1", text)
        self.assertIn("ASTR_VALIDATION_RHS_PREFIX", text)
        self.assertIn("compare_q_validation_snapshots.py", text)
        self.assertIn("pre_rhs,post_update", text)


if __name__ == "__main__":
    unittest.main()
