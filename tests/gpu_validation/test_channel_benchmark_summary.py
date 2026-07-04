import tempfile
import unittest
from pathlib import Path

from tests.gpu_validation.channel_benchmark_summary import (
    compute_speedups,
    load_timings,
    write_summary,
)


class ChannelBenchmarkSummaryTest(unittest.TestCase):
    def test_computes_gpu_speedups_against_np1_cpu_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            timings = Path(tmp) / "benchmark_times.tsv"
            timings.write_text(
                "\n".join(
                    [
                        "case\trole\tnp\ttopology\tseconds\tout_dir",
                        "channel_np1_1x1x1\tcpu\t1\t1,1,1\t500.0\t/out/np1/cpu",
                        "channel_np1_1x1x1\tgpu\t1\t1,1,1\t12.5\t/out/np1/gpu",
                        "channel_np2_2x1x1\tcpu\t2\t2,1,1\t300.0\t/out/np2/cpu",
                        "channel_np2_2x1x1\tgpu\t2\t2,1,1\t10.0\t/out/np2/gpu",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            rows = load_timings(timings)
            speedups = compute_speedups(rows, baseline_case="channel_np1_1x1x1")

        self.assertEqual([row.case for row in speedups], ["channel_np1_1x1x1", "channel_np2_2x1x1"])
        self.assertEqual(speedups[0].gpu_seconds, 12.5)
        self.assertEqual(speedups[0].speedup, 40.0)
        self.assertEqual(speedups[1].gpu_seconds, 10.0)
        self.assertEqual(speedups[1].speedup, 50.0)

    def test_writes_markdown_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = Path(tmp) / "summary.md"
            write_summary(
                summary,
                baseline_seconds=500.0,
                speedups=compute_speedups(
                    load_timings_from_text(
                        "case\trole\tnp\ttopology\tseconds\tout_dir\n"
                        "channel_np1_1x1x1\tcpu\t1\t1,1,1\t500.0\t/out/np1/cpu\n"
                        "channel_np1_1x1x1\tgpu\t1\t1,1,1\t12.5\t/out/np1/gpu\n"
                    ),
                    baseline_case="channel_np1_1x1x1",
                ),
                grid="128,128,128",
                deltat="7.5d-4",
                maxstep="100",
            )
            text = summary.read_text(encoding="utf-8")

        self.assertIn("NP=1 CPU baseline: `500.000 s`", text)
        self.assertIn("| channel_np1_1x1x1 | 1 | 1,1,1 | 12.500 | 40.00x |", text)


def load_timings_from_text(text: str):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "timings.tsv"
        path.write_text(text, encoding="utf-8")
        return load_timings(path)


if __name__ == "__main__":
    unittest.main()
