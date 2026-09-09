#!/usr/bin/env python3
"""Contracts for single- and multi-rank TGV performance timing."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (ROOT / "tests/gpu_validation/run_tgv_256_performance_benchmark.sh").read_text(
    encoding="utf-8"
)


class TgvPerformanceDriverTests(unittest.TestCase):
    def test_driver_accepts_rank_topology_and_visible_gpu_list(self) -> None:
        self.assertIn('NP="${NP:-1}"', SCRIPT)
        self.assertIn('TOPOLOGY="${TOPOLOGY:-1,1,1}"', SCRIPT)
        self.assertIn('GPU_IDS="${GPU_IDS:-$GPU_ID}"', SCRIPT)
        self.assertIn('CUDA_VISIBLE_DEVICES="$GPU_IDS"', SCRIPT)
        self.assertIn('ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY"', SCRIPT)
        self.assertIn('mpirun -np "$NP"', SCRIPT)

    def test_driver_uses_slowest_rank_for_each_rk_step(self) -> None:
        self.assertIn('ASTR_GPU_RANK_RK_TIMING=1', SCRIPT)
        self.assertIn('summarize_rank_rk_timing.py', SCRIPT)
        self.assertIn('--ranks "$NP"', SCRIPT)

    def test_driver_requires_one_distinct_gpu_per_rank(self) -> None:
        self.assertIn("math.prod(topology) != np", SCRIPT)
        self.assertIn("len(devices) != np", SCRIPT)
        self.assertIn("len(set(devices)) != np", SCRIPT)

    def test_driver_records_halo_transport_backend(self) -> None:
        self.assertIn('HALO_TRANSPORT="${HALO_TRANSPORT:-pageable}"', SCRIPT)
        self.assertIn('ASTR_GPU_HALO_TRANSPORT="$HALO_TRANSPORT"', SCRIPT)
        self.assertIn("halo_transport=%s", SCRIPT)

    def test_driver_accepts_cfl_selected_time_step(self) -> None:
        self.assertIn('DELTAT="${DELTAT:-}"', SCRIPT)
        self.assertIn('args+=(--deltat "$DELTAT")', SCRIPT)


if __name__ == "__main__":
    unittest.main()
