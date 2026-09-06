#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

from summarize_tgv_performance import read_timings, summarize


HEADER = (
    "label\trepeat\trk_samples\tmedian_rk_seconds\tmin_rk_seconds\t"
    "max_rk_seconds\twall_seconds\tmax_memory_mib\t"
    "max_utilization_percent\n"
)


class SummarizeTgvPerformanceTests(unittest.TestCase):
    def test_five_repeat_summary(self) -> None:
        rows = [
            f"baseline\t{repeat}\t10\t{seconds}\t0.9\t1.1\t12.0\t8000\t99\n"
            for repeat, seconds in enumerate((1.00, 1.01, 0.99, 1.00, 1.02), 1)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            timing_path = Path(tmp) / "timings.tsv"
            timing_path.write_text(HEADER + "".join(rows), encoding="ascii")
            lines = summarize(read_timings(timing_path), "256,256,256", 10, 1)
        rendered = "\n".join(lines)
        self.assertIn("GPU synchronization mode: `explicit`", rendered)
        self.assertIn("Median complete-RK time across runs: `1.000000000 s`", rendered)
        self.assertIn("Run-to-run relative spread: `3.000%`", rendered)

    def test_rejects_missing_repeat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            timing_path = Path(tmp) / "timings.tsv"
            timing_path.write_text(
                HEADER
                + "baseline\t1\t10\t1.0\t0.9\t1.1\t12.0\t8000\t99\n"
                + "baseline\t3\t10\t1.0\t0.9\t1.1\t12.0\t8000\t99\n",
                encoding="ascii",
            )
            rows = read_timings(timing_path)
        with self.assertRaisesRegex(ValueError, "not contiguous"):
            summarize(rows, "256,256,256", 10, 1)


if __name__ == "__main__":
    unittest.main()
