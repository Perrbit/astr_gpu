#!/usr/bin/env python3
"""Contracts for the NSCBC farfield state observed by GPU statistics."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
GPU_RUNTIME = (ROOT / "src_gpu/gpu_runtime.cuf").read_text(encoding="utf-8")
GPU_MAINLOOP = (ROOT / "src_gpu/mainloop_gpu.cuf").read_text(encoding="utf-8")
GPU_ARRAYS = (ROOT / "src_gpu/commarray_gpu.cuf").read_text(encoding="utf-8")


def subroutine(source: str, name: str) -> str:
    match = re.search(
        rf"subroutine {name}\b(.*?)end subroutine {name}",
        source,
        flags=re.DOTALL,
    )
    if match is None:
        raise AssertionError(f"missing subroutine {name}")
    return match.group(1)


class P2SbliStatsPhaseContractTests(unittest.TestCase):
    def test_stats_snapshot_reuses_rk_save_storage(self) -> None:
        save = subroutine(GPU_ARRAYS, "save_q_stats_snapshot_gpu")
        restore = subroutine(GPU_ARRAYS, "restore_q_stats_snapshot_gpu")
        self.assertNotIn("qstats_snapshot_d", GPU_ARRAYS)
        self.assertIn("save_q_stats_snapshot_kernel", save)
        self.assertIn("restore_q_stats_snapshot_kernel", restore)
        self.assertIn("sync_after_kernel('save_q_stats_snapshot_kernel')", save)
        self.assertIn("sync_after_kernel('restore_q_stats_snapshot_kernel')", restore)

    def test_all_flatplate_nscbc_stats_states_are_snapshotted(self) -> None:
        body = subroutine(GPU_RUNTIME, "gpu_prepare_rkfirst_stats")
        self.assertIn("gpu_s1_flatplate_nscbc_farfield_case", body)
        self.assertIn(
            "stats_snapshot_ready = gpu_s1_flatplate_nscbc_farfield_case()", body
        )
        self.assertIn("if(stats_snapshot_ready) call save_q_stats_snapshot_gpu()", body)

    def test_stats_prepare_applies_farfield_filters_in_cpu_order(self) -> None:
        body = subroutine(GPU_MAINLOOP, "prepare_rkfirst_stats_gpu")
        start = body.index("if(flatplate_nscbc_farfield_case) then")
        staged = body[start:]
        operations = [
            "call apply_nscbc_farfield_y_upper_filter_x_gpu()",
            "call exchange_solution_halo_gpu()",
            "call apply_nscbc_farfield_y_upper_filter_z_gpu()",
            "call q_to_primitive_kernel",
            "call exchange_solution_halo_gpu()",
        ]
        position = 0
        for operation in operations:
            position = staged.index(operation, position) + len(operation)

    def test_restored_farfield_state_is_reprepared_for_rk(self) -> None:
        body = subroutine(GPU_MAINLOOP, "time_integration_rk_gpu")
        self.assertIn("if(flatplate_nscbc_farfield_case) prepared = .false.", body)


if __name__ == "__main__":
    unittest.main()
