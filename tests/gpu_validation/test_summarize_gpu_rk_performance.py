#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

from summarize_gpu_rk_performance import read_timings, summarize


HEADER = (
    "label\trepeat\trk_samples\tmedian_rk_seconds\tmin_rk_seconds\t"
    "max_rk_seconds\twall_seconds\tmax_memory_mib\t"
    "max_utilization_percent\n"
)


class SummarizeGpuRkPerformanceTests(unittest.TestCase):
    def test_reports_case_grid_and_five_repeat_median(self) -> None:
        rows = [
            f"baseline\t{repeat}\t20\t{seconds}\t0.9\t1.1\t12.0\t8000\t99\n"
            for repeat, seconds in enumerate((1.00, 1.01, 0.99, 1.00, 1.02), 1)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            timing_path = Path(tmp) / "timings.tsv"
            timing_path.write_text(HEADER + "".join(rows), encoding="ascii")
            lines = summarize(
                read_timings(timing_path),
                case_name="shuosher",
                grid="256,64,32",
                maxstep=20,
                discard_steps=1,
            )

        rendered = "\n".join(lines)
        self.assertIn("# ASTR GPU RK Performance Benchmark", rendered)
        self.assertIn("case: `shuosher`", rendered)
        self.assertIn("grid: `256,64,32`", rendered)
        self.assertIn("Median complete-RK time across runs: `1.000000000 s`", rendered)


if __name__ == "__main__":
    unittest.main()
