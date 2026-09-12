import tempfile
import unittest
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from summarize_mp2_benchmark import summarize


def write_timings(path: Path, label: str) -> None:
    rows = [
        "label\trepeat\trk_samples\tmedian_rk_seconds\tmin_rk_seconds\tmax_rk_seconds\twall_seconds\tmax_memory_mib\tmax_utilization_percent"
    ]
    for repeat in range(1, 6):
        rows.append(
            f"{label}\t{repeat}\t4\t{1.0 + repeat * 0.01:.3f}\t1.0\t1.1\t2.0\t100\t90"
        )
    path.write_text("\n".join(rows) + "\n", encoding="ascii")


class SummarizeMp2BenchmarkTest(unittest.TestCase):
    def test_mp2_remains_the_default_phase(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fp64 = Path(tmp) / "fp64.tsv"
            candidate = Path(tmp) / "candidate.tsv"
            write_timings(fp64, "fp64")
            write_timings(candidate, "derivative")
            lines = summarize(fp64, candidate, 800, 400, 800, 400)

        self.assertEqual(lines[0], "# MP2 derivative Workspace Benchmark")
        self.assertFalse(any("GPU utilization" in line for line in lines))

    def test_mp3_phase_uses_candidate_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fp64 = Path(tmp) / "fp64.tsv"
            candidate = Path(tmp) / "candidate.tsv"
            write_timings(fp64, "fp64")
            write_timings(candidate, "characteristic_flux")
            lines = summarize(fp64, candidate, 800, 400, 800, 400, "MP3")

        self.assertEqual(lines[0], "# MP3 characteristic_flux Workspace Benchmark")
        self.assertIn("| Peak sampled GPU utilization | 90% | 90% |", lines)

    def test_rejects_unknown_phase(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fp64 = Path(tmp) / "fp64.tsv"
            candidate = Path(tmp) / "candidate.tsv"
            write_timings(fp64, "fp64")
            write_timings(candidate, "characteristic_flux")
            with self.assertRaisesRegex(ValueError, "unsupported mixed-precision phase"):
                summarize(fp64, candidate, 800, 400, 800, 400, "MP4")


if __name__ == "__main__":
    unittest.main()
