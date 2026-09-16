#!/usr/bin/env python3
"""Tests for the A800 TGV strong/weak scaling matrix summary."""

from pathlib import Path
import tempfile
import unittest

from summarize_tgv_scaling_matrix import render_summary, summarize_matrix


HEADER = (
    "label\trepeat\trk_samples\tmedian_rk_seconds\tmin_rk_seconds\t"
    "max_rk_seconds\twall_seconds\tmax_memory_mib\tmax_utilization_percent\n"
)


def write_timing(root: Path, label: str, values: list[float]) -> None:
    case_dir = root / label
    case_dir.mkdir(parents=True)
    rows = [
        f"{label}\t{repeat}\t20\t{value}\t{value}\t{value}\t10.0\t1000\t99\n"
        for repeat, value in enumerate(values, start=1)
    ]
    (case_dir / f"{label}_timings.tsv").write_text(
        HEADER + "".join(rows), encoding="ascii"
    )


class TgvScalingMatrixSummaryTests(unittest.TestCase):
    def test_selects_fastest_topology_and_computes_efficiencies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "matrix_manifest.tsv"
            manifest.write_text(
                "label\tkind\tnp\ttopology\tgrid\tlocal_grid\thalo_transport\tsync_mode\n"
                "strong_np1_111\tstrong\t1\t1,1,1\t512,512,512\t512,512,512\tpinned\texplicit\n"
                "strong_np2_211\tstrong\t2\t2,1,1\t512,512,512\t256,512,512\tpinned-pipeline\texplicit\n"
                "strong_np2_112\tstrong\t2\t1,1,2\t512,512,512\t512,512,256\tpinned-pipeline\texplicit\n"
                "control_np2_dependency\tcontrol\t2\t1,1,2\t512,512,512\t512,512,256\tpinned-pipeline\tdependency\n"
                "weak_np1_111\tweak\t1\t1,1,1\t256,256,256\t256,256,256\tpinned\texplicit\n"
                "weak_np2_211\tweak\t2\t2,1,1\t512,256,256\t256,256,256\tpinned-pipeline\texplicit\n",
                encoding="ascii",
            )
            for label, values in {
                "strong_np1_111": [4.0] * 5,
                "strong_np2_211": [2.2] * 5,
                "strong_np2_112": [2.0] * 5,
                "control_np2_dependency": [1.9] * 5,
                "weak_np1_111": [1.0] * 5,
                "weak_np2_211": [1.1] * 5,
            }.items():
                write_timing(root, label, values)

            rows, strong, weak = summarize_matrix(manifest, root)
            summary = render_summary(rows, strong, weak, 6)

        self.assertEqual(len(rows), 6)
        self.assertEqual(strong[2].topology, "1,1,2")
        self.assertEqual(strong[2].halo_transport, "pinned-pipeline")
        self.assertEqual(strong[2].sync_mode, "explicit")
        self.assertEqual(strong[2].wall_seconds, 10.0)
        self.assertAlmostEqual(strong[2].speedup, 2.0)
        self.assertAlmostEqual(strong[2].efficiency, 1.0)
        self.assertAlmostEqual(weak[2].efficiency, 1.0 / 1.1)
        self.assertAlmostEqual(weak[2].throughput_ratio, 2.0 / 1.1)
        self.assertIn(
            "| 2 | `1,1,2` | `pinned-pipeline` | `explicit` | 2.000000000 |",
            summary,
        )
        self.assertIn("statistics remain enabled", summary)
        self.assertIn("## Best measured GPU configuration", summary)
        self.assertIn(
            "| 2 | `control_np2_dependency` | `1,1,2` | `pinned-pipeline` | "
            "`dependency` | 1.900000000 | 2.10526 | 105.26% |",
            summary,
        )

    def test_ignores_failed_or_missing_timing_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "matrix_manifest.tsv"
            manifest.write_text(
                "label\tkind\tnp\ttopology\tgrid\tlocal_grid\thalo_transport\tsync_mode\n"
                "strong_np1_111\tstrong\t1\t1,1,1\t512,512,512\t512,512,512\tpinned\texplicit\n"
                "strong_np2_211\tstrong\t2\t2,1,1\t512,512,512\t256,512,512\tpinned-pipeline\texplicit\n",
                encoding="ascii",
            )
            write_timing(root, "strong_np1_111", [4.0] * 5)

            rows, strong, weak = summarize_matrix(manifest, root)

        self.assertEqual([row.label for row in rows], ["strong_np1_111"])
        self.assertEqual(list(strong), [1])
        self.assertEqual(weak, {})


if __name__ == "__main__":
    unittest.main()
