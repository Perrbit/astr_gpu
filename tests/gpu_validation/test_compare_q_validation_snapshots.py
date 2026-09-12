#!/usr/bin/env python3
"""Unit tests for binary RK-stage q snapshot comparison."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.gpu_validation.compare_q_validation_snapshots import (
    compare_snapshot_sets,
    read_q_snapshot,
)


def write_snapshot(
    path: Path,
    header: tuple[int, int, int, int, int],
    values: np.ndarray,
) -> None:
    with path.open("wb") as stream:
        np.asarray(header, dtype=np.int32).tofile(stream)
        np.asarray(values, dtype=np.float64).tofile(stream)


class CompareQValidationSnapshotsTests(unittest.TestCase):
    def test_reads_header_and_complete_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "q.bin"
            header = (1, 0, 0, 1, 5)
            values = np.arange(180, dtype=np.float64)
            write_snapshot(path, header, values)

            snapshot = read_q_snapshot(path)

        self.assertEqual(snapshot.header, header)
        self.assertEqual(snapshot.values.size, 180)
        self.assertEqual(snapshot.values[-1], 179.0)

    def test_matching_stage_sets_pass_with_roundoff_difference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suffix = "pre_rhs.step00000000.rk01.rank00000000.bin"
            header = (0, 0, 0, 0, 5)
            cpu = np.arange(5, dtype=np.float64)
            gpu = cpu.copy()
            gpu[3] += 2.0e-14
            write_snapshot(root / f"cpu.{suffix}", header, cpu)
            write_snapshot(root / f"gpu.{suffix}", header, gpu)

            result = compare_snapshot_sets(
                root / "cpu",
                root / "gpu",
                labels=("pre_rhs",),
                atol=1.0e-12,
                rtol=0.0,
            )

        self.assertTrue(result.passed)
        self.assertEqual(result.file_count, 1)
        self.assertLess(result.max_abs, 1.0e-12)

    def test_value_outside_tolerance_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suffix = "post_update.step00000000.rk03.rank00000000.bin"
            header = (0, 0, 0, 0, 5)
            cpu = np.ones(5, dtype=np.float64)
            gpu = cpu.copy()
            gpu[4] += 1.0e-6
            write_snapshot(root / f"cpu.{suffix}", header, cpu)
            write_snapshot(root / f"gpu.{suffix}", header, gpu)

            result = compare_snapshot_sets(
                root / "cpu",
                root / "gpu",
                labels=("post_update",),
                atol=1.0e-10,
                rtol=0.0,
            )

        self.assertFalse(result.passed)
        self.assertAlmostEqual(result.max_abs, 1.0e-6)

    def test_header_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suffix = "pre_rhs.step00000000.rk01.rank00000000.bin"
            write_snapshot(root / f"cpu.{suffix}", (0, 0, 0, 0, 5), np.ones(5))
            write_snapshot(root / f"gpu.{suffix}", (0, 0, 0, 1, 5), np.ones(135))

            with self.assertRaisesRegex(ValueError, "header mismatch"):
                compare_snapshot_sets(
                    root / "cpu",
                    root / "gpu",
                    labels=("pre_rhs",),
                    atol=1.0e-10,
                    rtol=0.0,
                )

    def test_missing_stage_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suffix = "pre_rhs.step00000000.rk01.rank00000000.bin"
            write_snapshot(root / f"cpu.{suffix}", (0, 0, 0, 0, 5), np.ones(5))

            with self.assertRaisesRegex(ValueError, "snapshot file set mismatch"):
                compare_snapshot_sets(
                    root / "cpu",
                    root / "gpu",
                    labels=("pre_rhs",),
                    atol=1.0e-10,
                    rtol=0.0,
                )


if __name__ == "__main__":
    unittest.main()
