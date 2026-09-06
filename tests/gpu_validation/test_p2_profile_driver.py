#!/usr/bin/env python3
"""Contracts for single- and multi-rank P2 profiling."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (ROOT / "tests/gpu_validation/run_p2_shock_performance_profile.sh").read_text(
    encoding="utf-8"
)


class P2ProfileDriverTests(unittest.TestCase):
    def test_driver_accepts_rank_topology_and_visible_gpu_list(self) -> None:
        self.assertIn('NP="${NP:-1}"', SCRIPT)
        self.assertIn('TOPOLOGY="${TOPOLOGY:-1,1,1}"', SCRIPT)
        self.assertIn('GPU_IDS="${GPU_IDS:-$GPU_ID}"', SCRIPT)
        self.assertIn('CUDA_VISIBLE_DEVICES="$GPU_IDS"', SCRIPT)
        self.assertIn('ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY"', SCRIPT)
        self.assertIn('mpirun -np "$NP"', SCRIPT)

    def test_nsys_driver_can_report_mpi_waits_and_message_sizes(self) -> None:
        self.assertIn('NSYS_TRACE="${NSYS_TRACE:-cuda}"', SCRIPT)
        self.assertIn('--trace="$NSYS_TRACE"', SCRIPT)
        self.assertIn('--report mpi_event_sum', SCRIPT)
        self.assertIn('mpi_event_sum.txt', SCRIPT)
        self.assertIn('--report mpi_msg_size_sum', SCRIPT)
        self.assertIn('mpi_msg_size_sum.txt', SCRIPT)
        self.assertEqual(SCRIPT.count('--force-export=true'), 3)


if __name__ == "__main__":
    unittest.main()
