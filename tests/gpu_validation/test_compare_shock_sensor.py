#!/usr/bin/env python3
"""Unit tests for full-field shock-sensor dump comparison."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.gpu_validation.compare_shock_sensor import (
    compare_sensor_dumps,
    read_sensor_dump,
    read_sensor_dump_set,
    read_sensor_dump_rank_set,
    compare_sensor_dump_rank_sets,
)


class CompareShockSensorTests(unittest.TestCase):
    def test_reads_complete_dump_and_compares_raw_values_and_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cpu_path = Path(tmp) / "cpu.dat"
            gpu_path = Path(tmp) / "gpu.dat"
            cpu_path.write_text(
                "# shock_sensor 1 0 0\n"
                "0 0 0 1.0000000000000000e-01 1\n"
                "1 0 0 2.0000000000000000e-01 0\n",
                encoding="utf-8",
            )
            gpu_path.write_text(
                "# shock_sensor 1 0 0\n"
                "0 0 0 1.0000000000000010e-01 1\n"
                "1 0 0 2.0000000000000000e-01 0\n",
                encoding="utf-8",
            )

            cpu = read_sensor_dump(cpu_path)
            result = compare_sensor_dumps(cpu, read_sensor_dump(gpu_path), atol=1e-12, rtol=1e-12)

        self.assertEqual(cpu.shape, (2, 1, 1))
        self.assertTrue(result.passed)
        self.assertLess(result.max_abs, 1e-12)
        self.assertEqual(result.mask_mismatches, 0)

    def test_mask_mismatch_fails_even_when_raw_sensor_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cpu_path = Path(tmp) / "cpu.dat"
            gpu_path = Path(tmp) / "gpu.dat"
            cpu_path.write_text(
                "# shock_sensor 0 0 0\n0 0 0 1.0e-1 1\n",
                encoding="utf-8",
            )
            gpu_path.write_text(
                "# shock_sensor 0 0 0\n0 0 0 1.0e-1 0\n",
                encoding="utf-8",
            )

            result = compare_sensor_dumps(
                read_sensor_dump(cpu_path),
                read_sensor_dump(gpu_path),
                atol=1e-12,
                rtol=1e-12,
            )

        self.assertFalse(result.passed)
        self.assertEqual(result.mask_mismatches, 1)

    def test_relative_diagnostic_uses_absolute_tolerance_floor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cpu_path = Path(tmp) / "cpu.dat"
            gpu_path = Path(tmp) / "gpu.dat"
            cpu_path.write_text(
                "# shock_sensor 0 0 0\n0 0 0 0.0 0\n",
                encoding="utf-8",
            )
            gpu_path.write_text(
                "# shock_sensor 0 0 0\n0 0 0 1.0e-16 0\n",
                encoding="utf-8",
            )

            result = compare_sensor_dumps(
                read_sensor_dump(cpu_path),
                read_sensor_dump(gpu_path),
                atol=1e-12,
                rtol=1e-12,
            )

        self.assertTrue(result.passed)
        self.assertAlmostEqual(result.max_rel, 1e-4)

    def test_merges_rank_dumps_using_global_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "sensor"
            (Path(tmp) / "sensor.rank0").write_text(
                "# shock_sensor 1 0 0 0 0 0\n"
                "0 0 0 1.0e-1 0\n"
                "1 0 0 2.0e-1 1\n",
                encoding="utf-8",
            )
            (Path(tmp) / "sensor.rank1").write_text(
                "# shock_sensor 1 0 0 1 0 0\n"
                "0 0 0 2.0e-1 1\n"
                "1 0 0 3.0e-1 0\n",
                encoding="utf-8",
            )

            result = read_sensor_dump_set(base)

        self.assertEqual(result.shape, (3, 1, 1))
        self.assertAlmostEqual(result.sensor[2, 0, 0], 3.0e-1)
        self.assertEqual(int(result.mask[1, 0, 0]), 1)

    def test_rejects_inconsistent_rank_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "sensor"
            (Path(tmp) / "sensor.rank0").write_text(
                "# shock_sensor 0 0 0 0 0 0\n0 0 0 1.0e-1 0\n",
                encoding="utf-8",
            )
            (Path(tmp) / "sensor.rank1").write_text(
                "# shock_sensor 0 0 0 0 0 0\n0 0 0 2.0e-1 0\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "inconsistent overlapping raw sensor"):
                read_sensor_dump_set(base)

    def test_incomplete_dump_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "incomplete.dat"
            path.write_text(
                "# shock_sensor 1 0 0\n0 0 0 1.0e-1 1\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "missing 1 nodes"):
                read_sensor_dump(path)

    def test_rankwise_comparison_allows_cpu_owned_overlap_masks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cpu_base = Path(tmp) / "cpu"
            gpu_base = Path(tmp) / "gpu"
            for base, mask0, mask1 in ((cpu_base, 1, 0), (gpu_base, 1, 0)):
                (Path(tmp) / f"{base.name}.rank0").write_text(
                    f"# shock_sensor 0 0 0 0 0 0\n0 0 0 1.0e-1 {mask0}\n",
                    encoding="utf-8",
                )
                (Path(tmp) / f"{base.name}.rank1").write_text(
                    f"# shock_sensor 0 0 0 0 0 0\n0 0 0 1.0e-1 {mask1}\n",
                    encoding="utf-8",
                )

            result = compare_sensor_dump_rank_sets(
                read_sensor_dump_rank_set(cpu_base),
                read_sensor_dump_rank_set(gpu_base),
                atol=1e-12,
                rtol=1e-12,
            )

        self.assertTrue(result.passed)
        self.assertEqual(result.mask_mismatches, 0)


if __name__ == "__main__":
    unittest.main()
