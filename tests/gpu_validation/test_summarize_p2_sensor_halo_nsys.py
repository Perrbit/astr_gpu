#!/usr/bin/env python3
"""Tests for the P2 shock-sensor halo timeline summarizer."""

from pathlib import Path
import sqlite3
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tests/gpu_validation/summarize_p2_sensor_halo_nsys.py"


class SummarizeP2SensorHaloNsysTests(unittest.TestCase):
    def test_reports_multi_device_critical_path_fraction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            database = tmpdir / "profile.sqlite"
            log = tmpdir / "run.log"
            report = tmpdir / "report.md"
            connection = sqlite3.connect(database)
            connection.executescript(
                """
                CREATE TABLE StringIds (id INTEGER, value TEXT);
                CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                    start INTEGER, end INTEGER, deviceId INTEGER,
                    globalPid INTEGER, shortName INTEGER
                );
                CREATE TABLE MPI_P2P_EVENTS (
                    start INTEGER, end INTEGER, globalTid INTEGER, size INTEGER
                );
                INSERT INTO StringIds VALUES
                  (1, 'shock_sensor_gpu_raw_shock_sensor_kernel_'),
                  (2, 'shock_sensor_gpu_expand_shock_sensor_kernel_');
                INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
                  (0, 10, 0, 16777216, 1), (40, 50, 0, 16777216, 2),
                  (60, 70, 0, 16777216, 1), (110, 120, 0, 16777216, 2),
                  (0, 12, 1, 33554432, 1), (37, 50, 1, 33554432, 2),
                  (60, 75, 1, 33554432, 1), (105, 120, 1, 33554432, 2);
                INSERT INTO MPI_P2P_EVENTS VALUES
                  (15, 25, 16777217, 320), (16, 24, 33554434, 320),
                  (80, 90, 16777217, 640), (200, 220, 16777217, 320);
                """
            )
            connection.commit()
            connection.close()
            log.write_text(
                "ASTR_GPU_RK_TIMING 0 1.0E-8 4.0E-8 5.0E-8\n"
                "ASTR_GPU_RK_TIMING 1 1.0E-8 4.0E-8 5.0E-8\n",
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    "python3",
                    str(SCRIPT),
                    "--input",
                    str(database),
                    "--runtime-log",
                    str(log),
                    "--sensor-message-bytes",
                    "320",
                    "--report",
                    str(report),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            text = report.read_text(encoding="utf-8")
            self.assertIn("critical-path halo total: `70 ns`", text)
            self.assertIn("profiled complete-RK total: `100 ns`", text)
            self.assertIn("complete-RK upper-bound fraction: `70.000%`", text)
            self.assertIn("matching MPI time: `18 ns`", text)


if __name__ == "__main__":
    unittest.main()
