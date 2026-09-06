#!/usr/bin/env python3

import unittest

from compare_tgv_candidate_performance import compare
from summarize_tgv_performance import Timing


def rows(label: str, medians: tuple[float, ...], memory: int = 8000) -> list[Timing]:
    return [
        Timing(label, index, 10, median, median, median, 12.0, memory, 99)
        for index, median in enumerate(medians, 1)
    ]


class CompareTgvCandidatePerformanceTests(unittest.TestCase):
    def test_accepts_faster_stable_candidate(self) -> None:
        baseline = rows("baseline", (1.00, 1.01, 0.99, 1.00, 1.02))
        candidate = rows("candidate", (0.94, 0.95, 0.95, 0.96, 0.95), 8200)
        lines, passed = compare(baseline, candidate, 3.0, 5.0, 5.0)
        self.assertTrue(passed)
        self.assertIn("Overall status: `pass`", "\n".join(lines))

    def test_rejects_insufficient_speedup(self) -> None:
        baseline = rows("baseline", (1.0, 1.0, 1.0, 1.0, 1.0))
        candidate = rows("candidate", (0.99, 0.99, 0.99, 0.99, 0.99))
        _, passed = compare(baseline, candidate, 3.0, 5.0, 5.0)
        self.assertFalse(passed)

    def test_rejects_fewer_than_five_repeats(self) -> None:
        baseline = rows("baseline", (1.0, 1.0, 1.0, 1.0, 1.0))
        candidate = rows("candidate", (0.9, 0.9, 0.9, 0.9))
        with self.assertRaisesRegex(ValueError, "at least five repeats"):
            compare(baseline, candidate, 3.0, 5.0, 5.0)

    def test_rejects_mismatched_rk_sample_contract(self) -> None:
        baseline = rows("baseline", (1.0, 1.0, 1.0, 1.0, 1.0))
        candidate = [
            Timing("candidate", index, 9, 0.9, 0.9, 0.9, 12.0, 8000, 99)
            for index in range(1, 6)
        ]
        with self.assertRaisesRegex(ValueError, "RK sample counts differ"):
            compare(baseline, candidate, 3.0, 5.0, 5.0)


if __name__ == "__main__":
    unittest.main()
