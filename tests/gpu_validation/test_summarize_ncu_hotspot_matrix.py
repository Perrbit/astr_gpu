#!/usr/bin/env python3

import tempfile
import unittest
import math
from pathlib import Path
from summarize_ncu_hotspot_matrix import (
    bottleneck,
    parse_number,
    source_hotspots,
    stall_shares,
)


class SummarizeNcuHotspotMatrixTests(unittest.TestCase):
    def test_non_numeric_metadata_is_ignored(self) -> None:
        self.assertTrue(math.isnan(parse_number("astr")))

    def test_stall_shares_use_not_issued_samples(self) -> None:
        metrics = {
            "smsp__pcsamp_warps_issue_stalled_short_scoreboard_not_issued": 60.0,
            "smsp__pcsamp_warps_issue_stalled_long_scoreboard_not_issued": 30.0,
            "smsp__pcsamp_warps_issue_stalled_math_pipe_throttle_not_issued": 10.0,
        }
        shares = stall_shares(metrics)
        self.assertAlmostEqual(shares["short_sb"], 60.0)
        self.assertAlmostEqual(shares["long_sb"], 30.0)
        self.assertAlmostEqual(shares["math"], 10.0)

    def test_bottleneck_classification(self) -> None:
        self.assertEqual(bottleneck(80.0, 40.0), "compute")
        self.assertEqual(bottleneck(30.0, 70.0), "memory")
        self.assertEqual(bottleneck(60.0, 50.0), "mixed")

    def test_source_hotspots_rank_source_rows_only(self) -> None:
        content = (
            '"File Path","solver_gpu.cuf"\n'
            '"Function Name","kernel"\n'
            '"Line No","Source","Address","Source","# Samples"\n'
            '"12","slow statement","-","SASS one","40"\n'
            '"","","0x10","SASS statement","100"\n'
            '"14","second statement","-","SASS two","20"\n'
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.csv"
            path.write_text(content, encoding="ascii")
            result = source_hotspots(path, count=2)
        self.assertEqual(result, [(12, 40, "slow statement"), (14, 20, "second statement")])


if __name__ == "__main__":
    unittest.main()
