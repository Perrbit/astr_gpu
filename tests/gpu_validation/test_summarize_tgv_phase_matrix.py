#!/usr/bin/env python3
"""Tests for the A800 TGV phase-matrix summarizer."""

import csv
from pathlib import Path
import tempfile
import unittest

from summarize_tgv_phase_matrix import PHASE_ORDER, read_cases, render_summary, write_tsv


class TgvPhaseMatrixSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = self.root / "matrix_manifest.tsv"
        self.manifest.write_text(
            "label\tkind\tnp\ttopology\tgrid\tlocal_grid\thalo_transport\tsync_mode\n"
            "strong_np1_111\tstrong\t1\t1,1,1\t512,512,512\t512,512,512\tpinned\texplicit\n"
            "strong_np2_211\tstrong\t2\t2,1,1\t512,512,512\t256,512,512\tpinned-pipeline\texplicit\n",
            encoding="ascii",
        )
        self._write_case("strong_np1_111", 2.0, 0.01)
        self._write_case("strong_np2_211", 1.5, 0.02)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_case(self, label: str, rk_seconds: float, phase_seconds: float) -> None:
        root = self.root / label
        root.mkdir()
        with (root / f"{label}_timings.tsv").open(
            "w", encoding="ascii", newline=""
        ) as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(
                [
                    "label",
                    "repeat",
                    "rk_samples",
                    "median_rk_seconds",
                    "min_rk_seconds",
                    "max_rk_seconds",
                    "wall_seconds",
                    "max_memory_mib",
                    "max_utilization_percent",
                ]
            )
            for repeat in range(1, 6):
                writer.writerow(
                    [label, repeat, 20, rk_seconds, rk_seconds, rk_seconds, 10, 1, 100]
                )
        with (root / f"{label}_phase_summary.tsv").open(
            "w", encoding="ascii", newline=""
        ) as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(
                ["phase", "samples", "median_seconds", "min_seconds", "max_seconds"]
            )
            for phase in PHASE_ORDER:
                writer.writerow([phase, 100, phase_seconds, phase_seconds, phase_seconds])

    def test_reads_complete_cases_and_renders_scaling_ratios(self) -> None:
        cases = read_cases(self.manifest, self.root)
        self.assertEqual([case.label for case in cases], ["strong_np1_111", "strong_np2_211"])
        summary = render_summary(cases, 2)
        self.assertIn("completed matrix entries: 2/2", summary)
        self.assertIn("| strong | 2 | `2,1,1` | 1.3333", summary)
        self.assertIn("prepare is inclusive", summary)

    def test_writes_one_row_per_case_and_phase(self) -> None:
        cases = read_cases(self.manifest, self.root)
        output = self.root / "phase_results.tsv"
        write_tsv(output, cases)
        with output.open(encoding="ascii", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        self.assertEqual(len(rows), 2 * len(PHASE_ORDER))
        self.assertEqual(rows[0]["phase"], "prepare")
        self.assertEqual(rows[-1]["phase"], "rk_update")

    def test_skips_incomplete_case_without_masking_completed_data(self) -> None:
        (self.root / "strong_np2_211" / "strong_np2_211_phase_summary.tsv").unlink()
        cases = read_cases(self.manifest, self.root)
        self.assertEqual([case.label for case in cases], ["strong_np1_111"])


if __name__ == "__main__":
    unittest.main()
